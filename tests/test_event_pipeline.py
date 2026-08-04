"""End-to-end tests for the event store runtime integration (WU-06).

Drives main() with a fake client and isolated XDG state, then reopens the
event store to verify transcript rebuild, prompt digests, pending-state
detection on resume, and the store-failure degradation path.
"""

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from shadow_code.conversation import Conversation
from shadow_code.events import (
    EventStore,
    EventStoreError,
    NewEvent,
    SessionStartedPayload,
    ToolCallProposedPayload,
)
from shadow_code.prompt_compiler import compile_prompt
from shadow_code.tools.catalog import BASH_SPEC, EDIT_FILE_SPEC, READ_FILE_SPEC, WRITE_FILE_SPEC
from shadow_code.tools.registry import ToolRegistry


def _runtime_registry() -> ToolRegistry:
    return ToolRegistry((READ_FILE_SPEC, WRITE_FILE_SPEC, EDIT_FILE_SPEC, BASH_SPEC))


class EventPipelineCase(unittest.TestCase):
    def setUp(self) -> None:
        import shadow_code.main as main_module

        self.main_module = main_module
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.state_home = root / "state"
        self.config_home = root / "config"
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.events_db = self.state_home / "shadow-code" / "events.db"
        self.inputs: list[str] = []
        self.responses: list[tuple[list[str], list[dict]]] = []
        self.conversation = Conversation()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _expected_prompt_digest(self) -> str:
        compiled = compile_prompt(
            user_path=self.config_home / "shadow-code" / "prompt.md",
            workspace_path=self.workspace / ".shadow-code" / "prompt.md",
            registry=_runtime_registry(),
        )
        return compiled.digest

    def _run_main(self) -> str:
        main_module = self.main_module
        client = MagicMock()
        client.health_check.return_value = (True, "OK")
        client.last_prompt_tokens = 0
        client.last_eval_tokens = 0
        client.last_tool_calls = []
        self.client = client

        responses = iter(self.responses or [(["ok"], [])])

        def chat_stream(messages, system, model=None, tools=None):
            chunks, client.last_tool_calls = next(responses, ([], []))
            return iter(chunks)

        client.chat_stream.side_effect = chat_stream

        from shadow_code.policy.workspace import WorkspaceGuard

        patches = [
            patch.object(main_module, "_RICH", False),
            patch.object(main_module, "_HAS_REPL", False),
            patch.object(main_module, "_HAS_DB", False),
            patch.object(main_module, "OllamaClient", return_value=client),
            patch.object(main_module, "Conversation", return_value=self.conversation),
            patch.object(
                main_module,
                "WorkspaceGuard",
                side_effect=lambda root, **kw: WorkspaceGuard(str(self.workspace)),
            ),
            patch.object(main_module, "_register_optional_tools"),
            patch.object(main_module.tool_reg, "register"),
            patch.object(main_module.signal, "signal"),
            patch("os.getcwd", return_value=str(self.workspace)),
            patch.dict(
                os.environ,
                {
                    "XDG_STATE_HOME": str(self.state_home),
                    "XDG_CONFIG_HOME": str(self.config_home),
                },
            ),
            patch("builtins.input", side_effect=list(self.inputs)),
            patch("sys.stdout", new_callable=io.StringIO),
        ]
        for patcher in patches[:-1]:
            patcher.start()
            self.addCleanup(patcher.stop)
        stdout = patches[-1].start()
        self.addCleanup(patches[-1].stop)
        main_module.main()
        return stdout.getvalue()

    def _prepopulate_interrupted_session(self) -> None:
        self.events_db.parent.mkdir(parents=True, exist_ok=True)
        with EventStore(self.events_db) as store:
            store.append_group(
                "interrupted-session",
                [
                    NewEvent(
                        "session_started",
                        SessionStartedPayload(model="test-model", cwd="/tmp"),
                    ),
                    NewEvent(
                        "tool_call_proposed",
                        ToolCallProposedPayload(
                            call_id="call-0",
                            name="bash",
                            arguments_json='{"command": "echo hi"}',
                        ),
                    ),
                ],
            )


class TestEventRecording(EventPipelineCase):
    def test_approved_bash_turn_rebuilds_identical_transcript(self) -> None:
        bash_call = {
            "call_id": "call-0",
            "name": "bash",
            "arguments": {"command": "echo wu06"},
        }
        self.responses = [([], [bash_call]), (["done"], [])]
        self.inputs = ["run echo", "y", "/exit"]

        self._run_main()

        with EventStore(self.events_db) as store:
            session = store.latest_session_id()
            assert session is not None
            rebuilt = store.rebuild_transcript(session)
            self.assertEqual(rebuilt, self.conversation.get_messages())
            self.assertEqual(store.pending_tool_calls(session), [])
            self.assertEqual(store.verify(), [])

            types = [event.type for event in store.events_for(session)]
            for expected in (
                "session_started",
                "user_message",
                "tool_call_proposed",
                "policy_decision",
                "approval_requested",
                "approval_granted",
                "tool_result",
                "assistant_text",
                "turn_completed",
                "session_ended",
            ):
                self.assertIn(expected, types)

            payloads = {event.type: event.parse_payload() for event in store.events_for(session)}
            turn_completed = payloads["turn_completed"]
            self.assertEqual(turn_completed.prompt_digest, self._expected_prompt_digest())
            decision = payloads["policy_decision"]
            self.assertEqual(decision.disposition, "require_approval")
            granted = payloads["approval_granted"]
            requested = payloads["approval_requested"]
            self.assertEqual(granted.plan_digest, requested.plan_digest)
            result = payloads["tool_result"]
            self.assertTrue(result.ok)
            self.assertIn("wu06", result.output)

    def test_plain_text_turn_records_minimal_chain(self) -> None:
        self.inputs = ["hello", "/exit"]

        self._run_main()

        with EventStore(self.events_db) as store:
            session = store.latest_session_id()
            types = [event.type for event in store.events_for(session)]
            self.assertEqual(
                types,
                [
                    "session_started",
                    "user_message",
                    "assistant_text",
                    "turn_completed",
                    "session_ended",
                ],
            )
            self.assertEqual(store.verify(), [])
            self.assertEqual(store.rebuild_transcript(session), self.conversation.get_messages())


class TestResumePendingReport(EventPipelineCase):
    def test_pending_calls_are_reported_and_abandoned_with_acknowledgment(self) -> None:
        self._prepopulate_interrupted_session()
        self.inputs = ["y", "/exit"]

        output = self._run_main()

        self.assertIn("unfinished tool call", output)
        self.assertIn("call-0", output)
        self.assertIn("bash", output)
        self.assertIn("nothing will be re-executed", output)
        self.client.chat_stream.assert_not_called()
        with EventStore(self.events_db) as store:
            started = [event for event in store.events_for("interrupted-session")]
            self.assertEqual(len(started), 2)  # untouched
            new_session = store.latest_session_id()
            self.assertNotEqual(new_session, "interrupted-session")
            types = [event.type for event in store.events_for(new_session)]
            self.assertIn("session_started", types)

    def test_declining_abandon_leaves_without_a_new_session(self) -> None:
        self._prepopulate_interrupted_session()
        self.inputs = ["n"]

        output = self._run_main()

        self.assertIn("pending record stays", output)
        self.client.chat_stream.assert_not_called()
        with EventStore(self.events_db) as store:
            self.assertEqual(store.latest_session_id(), "interrupted-session")

    def test_crash_mid_turn_leaves_detectable_pending_state(self) -> None:
        # Simulate a crash: the terminal (grant + result) group never lands.
        original = EventStore.append_group

        def crash_on_terminal(store, session_id, events):
            if any(event.type == "tool_result" for event in events):
                raise EventStoreError("append_failed", "simulated crash")
            return original(store, session_id, events)

        bash_call = {
            "call_id": "call-9",
            "name": "bash",
            "arguments": {"command": "echo crash"},
        }
        self.responses = [([], [bash_call]), (["done"], [])]
        self.inputs = ["run echo", "y", "/exit"]
        with patch.object(EventStore, "append_group", crash_on_terminal):
            output = self._run_main()
        self.assertIn("[events warning", output)

        # Next startup sees the interrupted session and asks.
        self.conversation = Conversation()
        self.responses = [(["ok"], [])]
        self.inputs = ["y", "/exit"]
        output = self._run_main()
        self.assertIn("unfinished tool call", output)
        self.assertIn("call-9", output)
        self.assertIn("plan=", output)


class TestEventsCommandAndDegradation(EventPipelineCase):
    def test_events_command_reports_ok(self) -> None:
        self.inputs = ["hello", "/events", "/exit"]

        output = self._run_main()

        self.assertIn("event store OK", output)
        self.assertIn("integrity verified", output)

    def test_store_failure_never_breaks_the_cli(self) -> None:
        self.inputs = ["hello", "/exit"]

        def broken_store(path):
            raise EventStoreError("open_failed", "disk gone")

        with patch.object(self.main_module, "EventStore", side_effect=broken_store):
            output = self._run_main()

        self.assertIn("[events warning", output)
        self.client.chat_stream.assert_called_once()
        self.assertIn("ok", output)
        self.assertFalse(self.events_db.exists())


if __name__ == "__main__":
    unittest.main()
