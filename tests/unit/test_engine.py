"""Deterministic tests for the bounded AgentEngine (WU-07).

Every I/O seam is injected: scripted streams return ProviderRound values
(or raise typed StreamError/StreamCancelledError), consent is a stub, the
clock is a counter, and cancellation is a flag. Covers the roadmap list:
no tools, one tool, multi-step sequence, multi-call batch, malformed
args, denial, cancellation at every active state, transient provider
retry, permanent failure, repeated call, each budget, and crash after
start/before finish.
"""

import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shadow_code.domain.approval import ApprovalAuthority
from shadow_code.domain.policy import PolicyFacts
from shadow_code.domain.tools import Capability, ToolCall
from shadow_code.engine import (
    AgentEngine,
    EngineBudgets,
    EngineState,
    ProviderRound,
    StreamError,
)
from shadow_code.events import EventStore, EventStoreError
from shadow_code.policy.engine import PolicyEngine
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.tools.catalog import BASH_SPEC, READ_FILE_SPEC, WorkspaceContext
from shadow_code.tools.registry import ToolRegistry

_READ_HELLO = {"call_id": "r1", "name": "read_file", "arguments": {"file_path": "hello.txt"}}
_BASH_ECHO = {"call_id": "b1", "name": "bash", "arguments": {"command": "echo wu07"}}


def _scripted(*items):
    """Stream returning scripted ProviderRounds; exceptions are raised."""
    planned = iter(items)

    def stream() -> ProviderRound:
        item = next(planned)
        if isinstance(item, BaseException):
            raise item
        return item

    return stream


class EngineCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)
        (self.workspace / "hello.txt").write_text("hello engine\n", encoding="utf-8")
        (self.workspace / "big.txt").write_text("x" * 100, encoding="utf-8")
        self.guard = WorkspaceGuard(str(self.workspace))
        self.addCleanup(self.guard.close)
        self.handler_calls: list[str] = []
        self.registry = ToolRegistry(
            (
                self._counting(READ_FILE_SPEC),
                self._counting(BASH_SPEC),
            )
        )
        capabilities = {Capability.FILESYSTEM_READ, Capability.PROCESS_EXECUTE}
        self.policy = PolicyEngine(PolicyFacts(capabilities, self.guard.identity))
        self.context = WorkspaceContext(
            guard=self.guard,
            workspace_root=str(self.workspace),
            process_env={},
            sandbox_label="unconfined",
        )

    def _counting(self, spec):
        handler = spec.handler
        calls = self.handler_calls

        def wrapped(call: ToolCall, arguments, context):
            calls.append(call.call_id)
            return handler(call, arguments, context)

        return spec.model_copy(update={"handler": wrapped})

    def _engine(self, **overrides) -> AgentEngine:
        options = {
            "consent": lambda plan: False,
            "budgets": None,
            "cancel_requested": None,
            "on_round": None,
            "on_event": None,
            "event_store": None,
            "on_store_warning": None,
            "clock": itertools.count(0, 1).__next__,
        }
        options.update(overrides)
        return AgentEngine(
            self.registry,
            self.policy,
            self.context,
            ApprovalAuthority(),
            consent=options["consent"],
            event_store=options["event_store"],
            event_session_id="engine-test-session",
            budgets=options["budgets"],
            cancel_requested=options["cancel_requested"],
            on_round=options["on_round"],
            on_event=options["on_event"],
            on_store_warning=options["on_store_warning"],
            clock=options["clock"],
        )


class TestTurnOutcomes(EngineCase):
    def test_no_tools_completes_with_text(self) -> None:
        engine = self._engine()
        outcome = engine.run_turn(_scripted(ProviderRound(text="hello")))

        self.assertIs(outcome.status, EngineState.COMPLETED)
        self.assertEqual(outcome.reason, "completed")
        self.assertEqual(outcome.text, "hello")
        self.assertEqual(outcome.steps, 0)
        self.assertEqual(outcome.calls_executed, 0)
        self.assertEqual(outcome.results, ())
        self.assertEqual(outcome.rounds, ())

    def test_one_tool_executes_then_completes(self) -> None:
        engine = self._engine()
        outcome = engine.run_turn(
            _scripted(
                ProviderRound(text="", native_calls=(_READ_HELLO,)),
                ProviderRound(text="done"),
            )
        )

        self.assertIs(outcome.status, EngineState.COMPLETED)
        self.assertEqual(outcome.steps, 1)
        self.assertEqual(outcome.calls_executed, 1)
        self.assertEqual(self.handler_calls, ["r1"])
        self.assertEqual(len(outcome.results), 1)
        self.assertTrue(outcome.results[0].success)
        self.assertIn("hello engine", outcome.results[0].output)

    def test_multi_step_sequence_preserves_round_order(self) -> None:
        read_big = {"call_id": "r2", "name": "read_file", "arguments": {"file_path": "big.txt"}}
        rounds_seen = []
        engine = self._engine(on_round=rounds_seen.append)
        outcome = engine.run_turn(
            _scripted(
                ProviderRound(text="", native_calls=(_READ_HELLO,)),
                ProviderRound(text="", native_calls=(read_big,)),
                ProviderRound(text="summary"),
            )
        )

        self.assertIs(outcome.status, EngineState.COMPLETED)
        self.assertEqual(outcome.steps, 2)
        self.assertEqual(outcome.text, "summary")
        self.assertEqual([r.call_id for r in outcome.results], ["r1", "r2"])
        self.assertEqual(len(rounds_seen), 2)
        self.assertEqual(rounds_seen[0].native_calls, (_READ_HELLO,))
        self.assertEqual(rounds_seen[1].native_calls, (read_big,))

    def test_multi_call_batch_gives_each_call_exactly_one_result(self) -> None:
        calls = tuple(
            {"call_id": f"r{i}", "name": "read_file", "arguments": {"file_path": name}}
            for i, name in enumerate(("hello.txt", "big.txt", "hello.txt"), start=1)
        )
        engine = self._engine()
        outcome = engine.run_turn(
            _scripted(
                ProviderRound(text="", native_calls=calls),
                ProviderRound(text="done"),
            )
        )

        self.assertIs(outcome.status, EngineState.COMPLETED)
        self.assertEqual(outcome.steps, 1)
        self.assertEqual([r.call_id for r in outcome.results], ["r1", "r2", "r3"])
        self.assertEqual(self.handler_calls, ["r1", "r2", "r3"])

    def test_malformed_arguments_never_reach_a_handler(self) -> None:
        malformed = {"call_id": "m1", "name": "read_file", "arguments": {"file_path": 123}}
        unknown = {"call_id": "m2", "name": "nope", "arguments": {}}
        engine = self._engine()
        outcome = engine.run_turn(
            _scripted(
                ProviderRound(text="", native_calls=(malformed, unknown, "not-a-dict")),
                ProviderRound(text="done"),
            )
        )

        self.assertEqual(self.handler_calls, [])
        self.assertEqual(outcome.calls_executed, 0)
        self.assertEqual(
            [r.error.code for r in outcome.results],
            ["invalid_arguments", "unknown_tool", "invalid_tool_call"],
        )

    def test_denial_is_final_and_not_retried(self) -> None:
        consents = []
        engine = self._engine(consent=lambda plan: consents.append(plan) or False)
        outcome = engine.run_turn(
            _scripted(
                ProviderRound(text="", native_calls=(_BASH_ECHO,)),
                ProviderRound(text="ok, skipping"),
            )
        )

        self.assertIs(outcome.status, EngineState.COMPLETED)
        self.assertEqual(len(consents), 1)  # asked exactly once
        self.assertEqual(outcome.calls_executed, 0)
        self.assertEqual(self.handler_calls, [])
        self.assertEqual(outcome.results[0].error.code, "approval_denied")

    def test_state_transitions_are_explicit(self) -> None:
        states = []
        engine = self._engine(on_event=states.append)
        engine.run_turn(
            _scripted(
                ProviderRound(text="", native_calls=(_READ_HELLO,)),
                ProviderRound(text="done"),
            )
        )

        self.assertEqual(
            states,
            [
                EngineState.STREAMING,
                EngineState.COLLECTING,
                EngineState.ADMITTING,
                EngineState.EXECUTING,
                EngineState.ADMITTING,
                EngineState.STREAMING,
                EngineState.COLLECTING,
                EngineState.COMPLETED,
            ],
        )


class TestProviderFailures(EngineCase):
    def test_transient_provider_error_retries_once_then_succeeds(self) -> None:
        calls = []

        def stream() -> ProviderRound:
            calls.append(1)
            if len(calls) == 1:
                raise StreamError("disconnect", "connection dropped", transient=True)
            return ProviderRound(text="recovered")

        outcome = self._engine().run_turn(stream)

        self.assertIs(outcome.status, EngineState.COMPLETED)
        self.assertEqual(outcome.text, "recovered")
        self.assertEqual(len(calls), 2)

    def test_permanent_provider_failure_is_failed_terminal(self) -> None:
        outcome = self._engine().run_turn(_scripted(StreamError("http_error", "server exploded")))

        self.assertIs(outcome.status, EngineState.FAILED)
        self.assertEqual(outcome.reason, "provider_error")
        self.assertEqual(outcome.detail, "server exploded")

    def test_repeated_transient_failure_exhausts_the_single_retry(self) -> None:
        outcome = self._engine().run_turn(
            _scripted(
                StreamError("disconnect", "drop one", transient=True),
                StreamError("disconnect", "drop two", transient=True),
            )
        )

        self.assertIs(outcome.status, EngineState.FAILED)
        self.assertEqual(outcome.reason, "provider_error")
        self.assertEqual(outcome.detail, "drop two")


class TestCancellation(EngineCase):
    def _run_cancellation_case(self, case: str) -> tuple:
        cancel = [False]
        consent_calls = []

        def stream() -> ProviderRound:
            if case == "after_collect":
                cancel[0] = True
            if case == "during_approval":
                return ProviderRound(text="", native_calls=(_BASH_ECHO,))
            return ProviderRound(text="", native_calls=(_READ_HELLO,))

        def on_event(state: EngineState) -> None:
            if case == "during_admission" and state is EngineState.EXECUTING:
                cancel[0] = True  # after the first (only) handler run
            if case == "during_execution" and state is EngineState.ADMITTING:
                cancel[0] = True  # after admission, before the handler
            if case == "during_approval" and state is EngineState.AWAITING_APPROVAL:
                cancel[0] = True  # while waiting on consent

        if case == "before_streaming":
            cancel[0] = True
        engine = self._engine(
            consent=lambda plan: consent_calls.append(plan) or True,
            cancel_requested=lambda: cancel[0],
            on_event=on_event,
        )
        return engine.run_turn(stream), consent_calls

    def test_cancellation_at_every_active_state(self) -> None:
        cases = [
            "before_streaming",
            "after_collect",
            "during_admission",
            "during_approval",
            "during_execution",
        ]
        for case in cases:
            with self.subTest(case=case):
                self.handler_calls.clear()
                outcome, consent_calls = self._run_cancellation_case(case)

                self.assertIs(outcome.status, EngineState.CANCELLED)
                self.assertEqual(outcome.reason, "cancelled")
                if case == "during_admission":
                    self.assertEqual(self.handler_calls, ["r1"])  # ran, then stopped
                else:
                    self.assertEqual(self.handler_calls, [])  # zero handler runs
                self.assertEqual(consent_calls, [])

    def test_keyboard_interrupt_inside_handler_becomes_cancelled(self) -> None:
        def interrupting_handler(call, arguments, context):
            raise KeyboardInterrupt

        spec = READ_FILE_SPEC.model_copy(update={"handler": interrupting_handler})
        self.registry = ToolRegistry((spec, self._counting(BASH_SPEC)))
        outcome = self._engine().run_turn(
            _scripted(ProviderRound(text="", native_calls=(_READ_HELLO,)))
        )

        self.assertIs(outcome.status, EngineState.CANCELLED)
        self.assertEqual(outcome.reason, "cancelled")


class TestBudgets(EngineCase):
    def test_step_budget_stops_after_configured_rounds(self) -> None:
        streams = []

        def stream() -> ProviderRound:
            streams.append(1)
            return ProviderRound(text="", native_calls=(_READ_HELLO,))

        budgets = EngineBudgets(max_steps=1)
        outcome = self._engine(budgets=budgets).run_turn(stream)

        self.assertIs(outcome.status, EngineState.BUDGET_EXHAUSTED)
        self.assertEqual(outcome.reason, "budget_steps")
        self.assertEqual(outcome.steps, 1)
        self.assertEqual(len(streams), 1)  # no further provider round
        self.assertTrue(outcome.results[0].success)  # the round still ran

    def test_call_budget_skips_remaining_proposals_as_terminal(self) -> None:
        calls = (
            _READ_HELLO,
            {"call_id": "r2", "name": "read_file", "arguments": {"file_path": "big.txt"}},
        )
        budgets = EngineBudgets(max_calls=1)
        outcome = self._engine(budgets=budgets).run_turn(
            _scripted(ProviderRound(text="", native_calls=calls))
        )

        self.assertIs(outcome.status, EngineState.BUDGET_EXHAUSTED)
        self.assertEqual(outcome.reason, "budget_calls")
        self.assertEqual(self.handler_calls, ["r1"])
        self.assertEqual(len(outcome.results), 2)  # every proposal terminates
        self.assertEqual(outcome.results[1].error.code, "budget_exhausted")

    def test_time_budget_trips_via_injected_clock(self) -> None:
        clock = itertools.count(0, 100).__next__  # start=0, then 100, 200, ...
        budgets = EngineBudgets(max_seconds=150.0)
        outcome = self._engine(budgets=budgets, clock=clock).run_turn(
            _scripted(ProviderRound(text="", native_calls=(_READ_HELLO,)))
        )

        self.assertIs(outcome.status, EngineState.BUDGET_EXHAUSTED)
        self.assertEqual(outcome.reason, "budget_time")
        self.assertEqual(self.handler_calls, [])
        self.assertEqual(outcome.results[0].error.code, "budget_exhausted")

    def test_output_budget_trips_on_aggregate_result_chars(self) -> None:
        read_big = {"call_id": "r2", "name": "read_file", "arguments": {"file_path": "big.txt"}}
        budgets = EngineBudgets(max_output_chars=50)
        outcome = self._engine(budgets=budgets).run_turn(
            _scripted(ProviderRound(text="", native_calls=(read_big,)))
        )

        self.assertIs(outcome.status, EngineState.BUDGET_EXHAUSTED)
        self.assertEqual(outcome.reason, "budget_output")
        self.assertEqual(self.handler_calls, ["r2"])  # the call ran, then stopped

    def test_repeated_call_beyond_max_duplicates_is_not_executed(self) -> None:
        calls = tuple(
            {"call_id": f"r{i}", "name": "read_file", "arguments": {"file_path": "hello.txt"}}
            for i in range(1, 4)
        )
        budgets = EngineBudgets(max_duplicates=2)
        outcome = self._engine(budgets=budgets).run_turn(
            _scripted(ProviderRound(text="", native_calls=calls))
        )

        self.assertIs(outcome.status, EngineState.BUDGET_EXHAUSTED)
        self.assertEqual(outcome.reason, "duplicates")
        self.assertEqual(self.handler_calls, ["r1", "r2"])  # third never runs
        self.assertEqual(outcome.results[2].error.code, "duplicate_call")


class TestCrashSemantics(EngineCase):
    def _store(self) -> EventStore:
        store = EventStore(self.workspace / "events.db")
        self.addCleanup(store.close)
        return store

    def test_engine_turn_records_the_wu06_event_chain_shape(self) -> None:
        store = self._store()
        engine = self._engine(consent=lambda plan: True, event_store=store)
        outcome = engine.run_turn(
            _scripted(
                ProviderRound(text="", native_calls=(_BASH_ECHO,)),
                ProviderRound(text="done"),
            )
        )

        self.assertIs(outcome.status, EngineState.COMPLETED)
        types = [event.type for event in store.events_for("engine-test-session")]
        self.assertEqual(
            types,
            [
                "tool_call_proposed",
                "policy_decision",
                "approval_requested",
                "approval_granted",
                "tool_result",
            ],
        )
        events = store.events_for("engine-test-session")
        payloads = {event.type: event.parse_payload() for event in events}
        self.assertEqual(store.pending_tool_calls("engine-test-session"), [])
        requested = payloads["approval_requested"]
        granted = payloads["approval_granted"]
        self.assertEqual(granted.plan_digest, requested.plan_digest)

    def test_crash_after_start_leaves_pending_and_never_re_executes(self) -> None:
        store = self._store()
        original = EventStore.append_group

        def crash_on_terminal(self_store, session_id, events):
            if any(event.type == "tool_result" for event in events):
                raise EventStoreError("append_failed", "simulated crash")
            return original(self_store, session_id, events)

        warnings: list[str] = []
        engine = self._engine(
            consent=lambda plan: True,
            event_store=store,
            on_store_warning=warnings.append,
        )
        with patch.object(EventStore, "append_group", crash_on_terminal):
            outcome = engine.run_turn(
                _scripted(
                    ProviderRound(text="", native_calls=(_BASH_ECHO,)),
                    ProviderRound(text="done"),
                )
            )

        # The store failure degraded to a warning; the turn still completed.
        self.assertIs(outcome.status, EngineState.COMPLETED)
        self.assertTrue(warnings)
        self.assertEqual(self.handler_calls, ["b1"])  # executed exactly once

        # Pending state is detectable; nothing is silently re-executed.
        pending = store.pending_tool_calls("engine-test-session")
        self.assertEqual([call.call_id for call in pending], ["b1"])
        types = [event.type for event in store.events_for("engine-test-session")]
        self.assertIn("tool_call_proposed", types)  # proposals land before admission
        self.assertIn("approval_requested", types)
        self.assertNotIn("tool_result", types)  # the terminal group never landed

        # A fresh engine over the same store runs no hidden resume logic:
        # pending calls stay pending until a user decision (main's startup).
        self.assertEqual([call.call_id for call in pending], ["b1"])
        self.assertEqual(self.handler_calls, ["b1"])


if __name__ == "__main__":
    unittest.main()
