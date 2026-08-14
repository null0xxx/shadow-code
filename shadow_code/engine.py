# shadow_code/engine.py -- Bounded, UI-free agent engine (WU-07)
#
# Owns the multi-step native-tool turn: stream -> collect -> admit
# (validate -> policy -> approval -> execute) -> append results, looping
# until completion, a typed budget trip, a provider failure, or
# cancellation. Every proposed call reaches exactly one terminal result;
# no handler runs before stream completion, validation, and policy; a
# denial is final and never retried.
#
# The engine never prints and never reads input: every I/O seam is an
# injected callable (stream, consent, cancel_requested, on_round,
# on_event, on_store_warning). Provider internals never cross the
# boundary; the stream callable returns a ProviderRound of plain text
# plus raw call dictionaries.
#
# Event semantics are exactly those established by WU-06: the proposals
# group lands before admission, terminal groups append atomically, and an
# interrupted turn leaves detectable pending state -- nothing is ever
# silently re-executed.

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import ValidationError

from .config import MAX_NATIVE_TOOL_TURNS
from .domain.approval import (
    ActionPlan,
    ApprovalAuthority,
    build_action_plan,
    render_action_preview,
)
from .domain.policy import PolicyDisposition, WorkspaceAccessError
from .domain.tools import Capability, ToolError, ToolResult, ValidatedToolCall
from .events import (
    ApprovalDeniedPayload,
    ApprovalGrantedPayload,
    ApprovalRequestedPayload,
    EventStore,
    EventStoreError,
    NewEvent,
    PolicyDecisionPayload,
    ToolCallProposedPayload,
    ToolResultPayload,
)
from .executor import execute_validated_call
from .mutation import MutationError, MutationPlan, build_edit_plan, build_write_plan
from .policy.engine import PolicyEngine
from .process import classify_command, execution_facts
from .tools.catalog import EditFileArgs, WorkspaceContext, WriteFileArgs
from .tools.registry import ToolRegistry


class EngineState(str, Enum):
    """Explicit engine states; the last four are terminal outcomes."""

    IDLE = "idle"
    STREAMING = "streaming"
    COLLECTING = "collecting"
    ADMITTING = "admitting"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


TERMINAL_STATES = frozenset(
    {
        EngineState.COMPLETED,
        EngineState.CANCELLED,
        EngineState.FAILED,
        EngineState.BUDGET_EXHAUSTED,
    }
)


class StreamCancelledError(Exception):
    """Raised by the injected stream when the user cancels the provider round."""


class StreamError(Exception):
    """Typed provider-round failure raised by the injected stream callable.

    A transient error retries the provider round exactly once per turn; a
    second failure (or a permanent one) ends the turn FAILED with reason
    ``provider_error``.
    """

    def __init__(self, code: str, message: str, *, transient: bool = False) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.transient = transient


class _TurnCancelled(Exception):
    """Internal: the injected cancel_requested seam tripped mid-turn."""


@dataclass(frozen=True, slots=True)
class EngineBudgets:
    """Per-turn bounds; a trip ends the turn as BUDGET_EXHAUSTED.

    ``max_duplicates`` is the number of times the same (name, canonical
    arguments) call may appear per turn; the next occurrence is not
    executed and ends the turn with reason ``duplicates``.
    """

    max_steps: int = MAX_NATIVE_TOOL_TURNS
    max_calls: int = 32
    max_seconds: float = 600.0
    max_output_chars: int = 500_000
    max_duplicates: int = 2


@dataclass(frozen=True, slots=True)
class ProviderRound:
    """One completed provider round as returned by the injected stream."""

    text: str
    native_calls: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class EngineRound:
    """One admitted round: the raw proposed calls plus terminal results."""

    native_calls: tuple[Mapping[str, Any], ...]
    results: tuple[ToolResult, ...]


@dataclass(frozen=True, slots=True)
class CallEvent:
    """Additive per-call lifecycle notification for UI frontends (WU-10).

    The engine's decisions are unchanged by this seam; it only reports
    stages as they happen: ``round`` (a new admission batch starts),
    ``proposed`` (a raw call entered the batch), ``awaiting_approval``,
    ``executing``, and ``result`` (one terminal ToolResult, attached).
    Frontends that do not need it simply never pass the callback.
    """

    stage: str
    call_id: str = ""
    tool_name: str = ""
    arguments_json: str = ""
    step: int = 0
    result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Terminal outcome of one bounded turn."""

    status: EngineState
    steps: int
    calls_executed: int
    results: tuple[ToolResult, ...]
    reason: str
    rounds: tuple[EngineRound, ...] = ()
    text: str | None = None
    detail: str = ""


def _canonical_json(value: object) -> str:
    """Serialize tool-call arguments deterministically; tolerate odd values."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=False, default=str)


def _call_identity(raw_call: object) -> tuple[str, str, str]:
    """(call_id, name, canonical arguments JSON) for a raw proposed call."""
    if isinstance(raw_call, dict):
        return (
            str(raw_call.get("call_id") or "invalid-call"),
            str(raw_call.get("name") or "unknown"),
            _canonical_json(raw_call.get("arguments")),
        )
    return "invalid-call", "unknown", _canonical_json(None)


def _proposal_events(native_calls: Sequence[Mapping[str, Any]]) -> list[NewEvent]:
    """One tool_call_proposed event per raw call, in proposal order."""
    events = []
    for raw_call in native_calls:
        call_id, name, arguments_json = _call_identity(raw_call)
        events.append(
            NewEvent(
                "tool_call_proposed",
                ToolCallProposedPayload(
                    call_id=call_id,
                    name=name,
                    arguments_json=arguments_json,
                ),
            )
        )
    return events


def _result_event(result: ToolResult) -> NewEvent:
    """The terminal tool_result event for an admission outcome."""
    return NewEvent(
        "tool_result",
        ToolResultPayload(
            call_id=result.call_id,
            tool_name=result.tool_name,
            ok=result.success,
            output=result.output,
            error_code=result.error.code if result.error else None,
            error_message=result.error.message if result.error else None,
        ),
    )


def _decision_event(call_id: str, disposition: str, reason: str) -> NewEvent:
    return NewEvent(
        "policy_decision",
        PolicyDecisionPayload(call_id=call_id, disposition=disposition, reason=reason),
    )


def _invalid_result(raw_call: object, error: ToolError) -> ToolResult:
    """Terminal result for a call that failed envelope/argument validation."""
    fallback_id = raw_call.get("call_id") if isinstance(raw_call, dict) else None
    fallback_name = raw_call.get("name") if isinstance(raw_call, dict) else None
    return ToolResult(
        call_id=fallback_id or "invalid-call",
        tool_name=fallback_name or "unknown",
        error=error,
    )


def _skipped_result(raw_call: object, reason: str) -> ToolResult:
    """Terminal result for a proposal the exhausted budget never admitted."""
    fallback_id = raw_call.get("call_id") if isinstance(raw_call, dict) else None
    fallback_name = raw_call.get("name") if isinstance(raw_call, dict) else None
    return ToolResult(
        call_id=str(fallback_id or "invalid-call"),
        tool_name=str(fallback_name or "unknown"),
        error=ToolError(
            code="budget_exhausted",
            message=f"Turn budget exhausted ({reason}); the call was not executed.",
        ),
    )


def _mutation_plan_for_call(
    validated: ValidatedToolCall, context: WorkspaceContext
) -> MutationPlan | ToolError:
    """Build the pure mutation plan for an approval preview; fail typed.

    The preview is built from a snapshot taken at approval time. The handler
    re-plans and re-snapshots at execution time, and apply_mutation re-checks
    the snapshot immediately before the commit, so any drift between preview
    and execution aborts with the original intact.
    """
    try:
        raw_arguments = json.loads(validated.canonical_arguments_json())
        arguments = validated.spec.args_model.model_validate(raw_arguments, strict=True)
        if isinstance(arguments, WriteFileArgs):
            return build_write_plan(context.guard, arguments)
        if isinstance(arguments, EditFileArgs):
            return build_edit_plan(context.guard, arguments)
        return ToolError(
            code="unsupported_mutation",
            message=f"Tool '{validated.call.name}' has no mutation planner.",
        )
    except MutationError as error:
        return ToolError(code=error.code, message=str(error))
    except WorkspaceAccessError as error:
        return ToolError(code=error.reason.value, message=str(error))
    except ValidationError as error:
        return ToolError(code="invalid_arguments", message=str(error))


class AgentEngine:
    """Bounded engine for multi-step native-tool turns; UI-free by injection.

    ``consent`` renders an ActionPlan and returns the user's one-shot
    approval decision. ``cancel_requested`` is polled before every state
    transition point; a trip ends the turn CANCELLED with no further
    handler runs. ``on_round`` observes each admitted round (transcript
    mirroring), ``on_event`` observes state transitions, and
    ``on_store_warning`` receives event-store degradation messages.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy_engine: PolicyEngine,
        execution_context: WorkspaceContext,
        approval_authority: ApprovalAuthority,
        *,
        consent: Callable[[ActionPlan], bool],
        event_store: EventStore | None = None,
        event_session_id: str = "",
        budgets: EngineBudgets | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        on_round: Callable[[EngineRound], None] | None = None,
        on_event: Callable[[EngineState], None] | None = None,
        on_call_event: Callable[[CallEvent], None] | None = None,
        on_store_warning: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._policy_engine = policy_engine
        self._execution_context = execution_context
        self._approval_authority = approval_authority
        self._consent = consent
        self._event_store = event_store
        self._event_session_id = event_session_id
        self._budgets = budgets or EngineBudgets()
        self._cancel_requested = cancel_requested
        self._on_round = on_round
        self._on_event = on_event
        self._on_call_event = on_call_event
        self._on_store_warning = on_store_warning
        self._clock = clock
        self._state = EngineState.IDLE
        self._reset_turn()

    @property
    def state(self) -> EngineState:
        return self._state

    def _reset_turn(self) -> None:
        self._state = EngineState.IDLE
        self._start = self._clock()
        self._steps = 0
        self._calls_seen = 0
        self._calls_executed = 0
        self._output_chars = 0
        self._seen: dict[tuple[str, str], int] = {}
        self._results: list[ToolResult] = []
        self._rounds: list[EngineRound] = []
        self._pending_calls: tuple[Mapping[str, Any], ...] | None = None
        self._pending_results: list[ToolResult] = []

    # -- event emission -------------------------------------------------

    def _emit(self, events: list[NewEvent]) -> None:
        """Append events; a store failure degrades to an injected warning."""
        if self._event_store is None or not events:
            return
        try:
            if len(events) == 1:
                self._event_store.append(self._event_session_id, events[0])
            else:
                self._event_store.append_group(self._event_session_id, events)
        except (EventStoreError, OSError) as error:
            if self._on_store_warning is not None:
                self._on_store_warning(str(error))

    # -- state machine --------------------------------------------------

    def _transition(self, state: EngineState) -> None:
        self._state = state
        if self._on_event is not None:
            self._on_event(state)

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested is not None and self._cancel_requested():
            raise _TurnCancelled

    def _notify_call(self, event: CallEvent) -> None:
        """Additive UI seam (WU-10), unguarded like on_round/on_event."""
        if self._on_call_event is not None:
            self._on_call_event(event)

    # -- public entry points --------------------------------------------

    def admit_calls(self, native_calls: Sequence[Mapping[str, Any]]) -> list[ToolResult]:
        """Single admission pass over already-collected calls (compat path).

        No budgets, duplicate detection, or proposal emission: identical to
        the pre-engine ``_admit_native_calls`` behavior.
        """
        results = []
        for raw_call in native_calls:
            validated = self._registry.validate_call(raw_call)
            if isinstance(validated, ToolError):
                result = _invalid_result(raw_call, validated)
                self._emit([_result_event(result)])
                results.append(result)
                continue
            result, _executed = self._admit_validated(validated)
            results.append(result)
        return results

    def run_turn(self, stream: Callable[[], ProviderRound]) -> EngineResult:
        """Drive one bounded user turn to a terminal outcome."""
        self._reset_turn()
        try:
            return self._run(stream)
        except _TurnCancelled:
            return self._finish(EngineState.CANCELLED, "cancelled")
        except KeyboardInterrupt:
            # Handlers already kill their process groups (WU-03); the turn
            # ends as a cancellation, never as a silent re-execution.
            return self._finish(EngineState.CANCELLED, "cancelled")

    # -- turn loop --------------------------------------------------------

    def _run(self, stream: Callable[[], ProviderRound]) -> EngineResult:
        retried = False
        while True:
            self._raise_if_cancelled()
            trip = self._budget_trip_reason()
            if trip:
                return self._finish(EngineState.BUDGET_EXHAUSTED, trip)
            self._transition(EngineState.STREAMING)
            try:
                provider_round = stream()
            except StreamCancelledError:
                return self._finish(EngineState.CANCELLED, "cancelled")
            except StreamError as error:
                if error.transient and not retried:
                    retried = True  # one bounded retry of the provider round
                    continue
                return self._finish(EngineState.FAILED, "provider_error", detail=error.message)
            self._transition(EngineState.COLLECTING)
            native_calls = tuple(provider_round.native_calls)
            if not native_calls:
                return self._finish(EngineState.COMPLETED, "completed", text=provider_round.text)
            self._emit(_proposal_events(native_calls))
            step = self._steps + 1
            self._notify_call(CallEvent(stage="round", step=step))
            for raw_call in native_calls:
                call_id, name, arguments_json = _call_identity(raw_call)
                self._notify_call(
                    CallEvent(
                        stage="proposed",
                        call_id=call_id,
                        tool_name=name,
                        arguments_json=arguments_json,
                        step=step,
                    )
                )
            terminal = self._admit_round(native_calls)
            if terminal is not None:
                return terminal
            if self._steps >= self._budgets.max_steps:
                return self._finish(EngineState.BUDGET_EXHAUSTED, "budget_steps")

    def _admit_round(self, native_calls: tuple[Mapping[str, Any], ...]) -> EngineResult | None:
        """Admit one proposed batch sequentially; None when fully admitted."""
        self._pending_calls = native_calls
        self._pending_results = []
        pending = list(native_calls)
        while pending:
            self._raise_if_cancelled()
            trip = self._budget_trip_reason()
            if trip:
                self._skip_remaining(pending, trip)
                return self._terminate(trip)
            raw_call = pending.pop(0)
            self._transition(EngineState.ADMITTING)
            validated = self._registry.validate_call(raw_call)
            self._calls_seen += 1
            if isinstance(validated, ToolError):
                result = _invalid_result(raw_call, validated)
                self._emit([_result_event(result)])
                self._record_result(result)
                continue
            duplicate = self._register_duplicate(validated)
            if duplicate is not None:
                self._emit([_result_event(duplicate)])
                self._record_result(duplicate)
                remaining, pending = pending, []
                self._skip_remaining(remaining, "duplicates")
                return self._terminate("duplicates")
            result, executed = self._admit_validated(validated)
            self._calls_executed += int(executed)
            self._record_result(result)
            self._output_chars += len(result.output or "")
            if self._output_chars > self._budgets.max_output_chars:
                remaining, pending = pending, []
                self._skip_remaining(remaining, "budget_output")
                return self._terminate("budget_output")
        self._record_round()
        return None

    # -- admission (validate -> policy -> approval -> execute) ------------

    def _admit_validated(self, validated: ValidatedToolCall) -> tuple[ToolResult, bool]:
        """Admit one validated call; return (terminal result, executed).

        Fail-closed at every step: policy denials produce typed results and
        never reach a handler. Approval-required calls ask the injected
        consent seam for a one-shot, digest-bound token; denial or
        cancellation is final and the call is not retried.
        """
        call_id = validated.call.call_id
        decision = self._policy_engine.decide(validated)
        if decision.disposition is PolicyDisposition.ALLOW:
            self._emit([_decision_event(call_id, "allow", decision.reason.value)])
            self._raise_if_cancelled()
            self._transition(EngineState.EXECUTING)
            self._notify_call(
                CallEvent(stage="executing", call_id=call_id, tool_name=validated.call.name)
            )
            result = execute_validated_call(validated, self._execution_context)
            self._transition(EngineState.ADMITTING)
            self._emit([_result_event(result)])
            return result, True
        if decision.disposition is PolicyDisposition.DENY:
            result = ToolResult(
                call_id=call_id,
                tool_name=validated.call.name,
                error=ToolError(
                    code="policy_denied",
                    message=(
                        f"Policy denied execution ({decision.reason.value})."
                        " Do not retry this call; respond to the user instead."
                    ),
                ),
            )
            self._emit(
                [
                    _decision_event(call_id, "deny", decision.reason.value),
                    _result_event(result),
                ]
            )
            return result, False

        preview = render_action_preview(validated)
        facts = ""
        context = self._execution_context
        if validated.spec.capability is Capability.PROCESS_EXECUTE:
            facts = execution_facts(
                context.process_env,
                context.workspace_root,
                context.sandbox_label,
            )
            command = str(validated.arguments.get("command", ""))
            features = sorted(classify_command(command))
            sandbox_label = context.sandbox_label
            if sandbox_label == "unconfined":
                sandbox_line = "sandbox: unconfined (no sandbox helper available)"
            else:
                sandbox_line = f"sandbox: unconfined ({sandbox_label} available but not applied)"
            preview += (
                f"\n{sandbox_line}"
                f" (no confinement; approval is the only control)"
                f"\nfeatures: {', '.join(features) if features else 'none detected'}"
            )
        elif validated.spec.capability is Capability.FILESYSTEM_WRITE:
            mutation_plan = _mutation_plan_for_call(validated, context)
            if isinstance(mutation_plan, ToolError):
                # Plan-build failures (no_match, ambiguous_match, ...)
                # surface as typed results, never as approval prompts.
                result = ToolResult(
                    call_id=call_id,
                    tool_name=validated.call.name,
                    error=mutation_plan,
                )
                self._emit([_result_event(result)])
                return result, False
            strict_note = " [strict: patch export]" if context.mutation_mode == "export" else ""
            preview += (
                f"\nmutation: {mutation_plan.operation} "
                f"{mutation_plan.relative_path}{strict_note}\n{mutation_plan.preview}"
            )
        plan = build_action_plan(
            validated,
            registry_digest=self._registry.digest,
            workspace=context.guard.identity,
            preview=preview,
            execution_facts=facts,
        )
        plan_digest = plan.digest()
        self._emit(
            [
                _decision_event(call_id, "require_approval", decision.reason.value),
                NewEvent(
                    "approval_requested",
                    ApprovalRequestedPayload(
                        call_id=call_id, plan_digest=plan_digest, preview=preview
                    ),
                ),
            ]
        )
        self._transition(EngineState.AWAITING_APPROVAL)
        self._notify_call(
            CallEvent(stage="awaiting_approval", call_id=call_id, tool_name=validated.call.name)
        )
        self._raise_if_cancelled()
        if not self._consent(plan):
            result = ToolResult(
                call_id=call_id,
                tool_name=validated.call.name,
                error=ToolError(
                    code="approval_denied",
                    message=(
                        "User denied or cancelled the approval; the call was not executed."
                        " Do not retry this call; respond to the user instead."
                    ),
                ),
            )
            self._emit(
                [
                    NewEvent(
                        "approval_denied",
                        ApprovalDeniedPayload(call_id=call_id, plan_digest=plan_digest),
                    ),
                    _result_event(result),
                ]
            )
            self._transition(EngineState.ADMITTING)
            return result, False
        self._raise_if_cancelled()
        token = self._approval_authority.issue(plan)
        self._transition(EngineState.EXECUTING)
        self._notify_call(
            CallEvent(stage="executing", call_id=call_id, tool_name=validated.call.name)
        )
        result = execute_validated_call(
            validated,
            context,
            approval=token,
            authority=self._approval_authority,
            plan=plan,
        )
        self._emit(
            [
                NewEvent(
                    "approval_granted",
                    ApprovalGrantedPayload(call_id=call_id, plan_digest=plan_digest),
                ),
                _result_event(result),
            ]
        )
        self._transition(EngineState.ADMITTING)
        return result, True

    # -- budgets, duplicates, bookkeeping ---------------------------------

    def _budget_trip_reason(self) -> str:
        if self._clock() - self._start > self._budgets.max_seconds:
            return "budget_time"
        if self._calls_seen >= self._budgets.max_calls:
            return "budget_calls"
        return ""

    def _register_duplicate(self, validated: ValidatedToolCall) -> ToolResult | None:
        """Track (name, canonical arguments); trip past max_duplicates."""
        key = (validated.call.name, validated.canonical_arguments_json())
        seen = self._seen.get(key, 0) + 1
        self._seen[key] = seen
        if seen <= self._budgets.max_duplicates:
            return None
        return ToolResult(
            call_id=validated.call.call_id,
            tool_name=validated.call.name,
            error=ToolError(
                code="duplicate_call",
                message=(
                    f"Same call repeated {seen} times this turn "
                    f"(max {self._budgets.max_duplicates}); the call was not executed."
                ),
            ),
        )

    def _record_result(self, result: ToolResult) -> None:
        self._pending_results.append(result)
        self._results.append(result)
        self._notify_call(
            CallEvent(
                stage="result",
                call_id=result.call_id,
                tool_name=result.tool_name,
                result=result,
            )
        )

    def _skip_remaining(self, remaining: list[Mapping[str, Any]], reason: str) -> None:
        """Give every unadmitted proposal a terminal budget_exhausted result."""
        for raw_call in remaining:
            result = _skipped_result(raw_call, reason)
            self._emit([_result_event(result)])
            self._record_result(result)

    def _record_round(self) -> None:
        if self._pending_calls is None:
            return
        engine_round = EngineRound(
            native_calls=self._pending_calls,
            results=tuple(self._pending_results),
        )
        self._pending_calls = None
        self._pending_results = []
        self._rounds.append(engine_round)
        self._steps += 1
        if self._on_round is not None:
            self._on_round(engine_round)

    def _terminate(self, reason: str) -> EngineResult:
        self._record_round()
        return self._finish(EngineState.BUDGET_EXHAUSTED, reason)

    def _finish(
        self,
        status: EngineState,
        reason: str,
        *,
        text: str | None = None,
        detail: str = "",
    ) -> EngineResult:
        self._record_round()
        self._transition(status)
        return EngineResult(
            status=status,
            steps=self._steps,
            calls_executed=self._calls_executed,
            results=tuple(self._results),
            reason=reason,
            rounds=tuple(self._rounds),
            text=text,
            detail=detail,
        )
