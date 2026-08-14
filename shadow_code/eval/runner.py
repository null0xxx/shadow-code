# shadow_code/eval/runner.py -- Harness wiring, live runner, redaction (WU-11)
#
# One harness builds disposable scenario workspaces and wires the REAL
# engine with the real registry, policy, guard, and mutation pieces. The
# deterministic pytest suite drives it with the scenario's scripted
# transcript; the live CLI drives it with a real provider (or with the
# transcript for engine-facing invariants). The harness MEASURES: it never
# weakens validation, budgets, or approvals to fit an outcome.
#
# Live runs happen in disposable temporary workspaces only; everything is
# deleted after the scenario, and recorded events are redacted (no absolute
# home/tmp paths, no environment values) before they leave the machine.

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel

from ..domain.approval import ActionPlan, ApprovalAuthority
from ..domain.policy import PolicyFacts
from ..domain.tools import Capability, ToolCall, ToolHandler, ToolResult, ToolSpec
from ..engine import (
    AgentEngine,
    EngineBudgets,
    EngineResult,
    EngineRound,
    ProviderRound,
    StreamError,
)
from ..events import Event, EventStore
from ..ollama_client import render_ollama_tool_schemas
from ..policy.engine import PolicyEngine
from ..policy.workspace import WorkspaceGuard
from ..process import build_process_env
from ..prompt import render_system_prompt, render_tool_documentation
from ..provider import (
    OllamaProvider,
    ProviderError,
    TextDelta,
    ToolCallComplete,
    UsageUpdate,
    iter_events_sync,
    thaw_arguments,
)
from ..tools.catalog import (
    BASH_SPEC,
    EDIT_FILE_SPEC,
    READ_FILE_SPEC,
    WRITE_FILE_SPEC,
    WorkspaceContext,
)
from ..tools.registry import ToolRegistry
from .corpus import Scenario
from .report import ScenarioScore
from .scoring import CallRecord, RunTrace, ScoreOutcome, score_scenario

_LOCK_FILE = ".shadow-code.lock"

# Wall-clock cap for one scenario run: the engine's production default is
# 600s, but the slowest legitimate scenario observed finishes in ~39s, so a
# live scenario whose expected event never arrives (e.g. a model that never
# starts the cancellable long call) would otherwise burn 10 minutes looking
# hung. A trip ends the turn as budget_exhausted/budget_time, which scoring
# already records in the report notes.
LIVE_SCENARIO_MAX_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class ProviderOutput:
    """One provider round as measured by the harness."""

    text: str
    native_calls: tuple[Mapping[str, Any], ...]
    prompt_tokens: int = 0
    eval_tokens: int = 0


class EvalProvider(Protocol):
    """The only provider surface the live runner needs; fakes are trivial."""

    def round(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        tools: list[dict[str, Any]],
    ) -> ProviderOutput: ...


class OllamaEvalProvider:
    """Live provider over the real Ollama client (opt-in; CI never uses it)."""  # pragma: no cover

    def __init__(self, base_url: str, options: Mapping[str, Any]) -> None:  # pragma: no cover
        self._provider = OllamaProvider(base_url, dict(options))

    def round(  # pragma: no cover
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str,
        tools: list[dict[str, Any]],
    ) -> ProviderOutput:
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        prompt_tokens = 0
        eval_tokens = 0
        events = self._provider.stream(messages, system, model, tools)
        for event in iter_events_sync(events):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
            elif isinstance(event, ToolCallComplete):
                arguments = event.arguments
                calls.append(
                    {
                        "call_id": event.call_id,
                        "name": event.name,
                        "arguments": (
                            thaw_arguments(arguments)
                            if isinstance(arguments, Mapping)
                            else arguments
                        ),
                    }
                )
            elif isinstance(event, UsageUpdate):
                prompt_tokens = event.prompt_tokens
                eval_tokens = event.eval_tokens
            elif isinstance(event, ProviderError):
                raise StreamError(
                    event.code,
                    event.message,
                    transient=event.code in {"timeout", "disconnect"},
                )
        return ProviderOutput("".join(text_parts), tuple(calls), prompt_tokens, eval_tokens)


# -- workspace and harness construction --------------------------------------


def build_scenario_dirs(scenario: Scenario, base: Path) -> tuple[Path, Path]:
    """Create the disposable workspace and outside-sentinel directory."""
    workspace = base / "workspace"
    outside = base / "outside"
    workspace.mkdir(parents=True, exist_ok=True)
    outside.mkdir(parents=True, exist_ok=True)
    for relative, content in scenario.workspace_files.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for relative, content in scenario.outside_files.items():
        target = outside / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for link, target_text in scenario.symlinks.items():
        link_path = workspace / link
        link_path.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target_text, link_path)
    return workspace, outside


def snapshot_tree(root: Path) -> dict[str, str]:
    """Workspace-relative content map; symlinks recorded, never followed."""
    snapshot: dict[str, str] = {}
    if not root.is_dir():
        return snapshot
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == _LOCK_FILE:
            continue
        if path.is_symlink():
            snapshot[relative] = f"<symlink>{os.readlink(path)}"
        elif path.is_file():
            snapshot[relative] = path.read_text(encoding="utf-8", errors="replace")
    return snapshot


def policy_digest(facts: PolicyFacts) -> str:
    """Deterministic digest of the policy facts the engine classifies with."""
    encoded = json.dumps(
        {
            "capabilities": sorted(capability.value for capability in facts.granted_capabilities),
            "workspace_identity": [
                facts.workspace_identity.device if facts.workspace_identity else None,
                facts.workspace_identity.inode if facts.workspace_identity else None,
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _recording(spec: ToolSpec, runs: list[str]) -> ToolSpec:
    """Wrap a catalog handler to record which calls actually reached it."""
    handler = cast(ToolHandler, spec.handler)

    def wrapped(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
        runs.append(call.call_id)
        return handler(call, arguments, context)

    return spec.model_copy(update={"handler": wrapped})


@dataclass(slots=True)
class Harness:
    """A wired engine plus everything scoring needs from the environment."""

    engine: AgentEngine
    registry: ToolRegistry
    workspace: Path
    outside: Path
    guard: WorkspaceGuard
    policy_facts: PolicyFacts
    handler_runs: list[str] = field(default_factory=list)
    workspace_before: dict[str, str] = field(default_factory=dict)
    outside_before: dict[str, str] = field(default_factory=dict)
    event_store: EventStore | None = None
    session_id: str = ""

    @property
    def policy_digest(self) -> str:
        return policy_digest(self.policy_facts)

    def close(self) -> None:
        if self.event_store is not None:
            self.event_store.close()
        self.guard.close()


def build_harness(
    scenario: Scenario,
    base: Path,
    *,
    consent: Callable[[ActionPlan], bool],
    cancel_requested: Callable[[], bool] | None = None,
    on_round: Callable[[EngineRound], None] | None = None,
    event_store: EventStore | None = None,
    session_id: str = "",
    budgets: EngineBudgets | None = None,
) -> Harness:
    """Wire the real engine for one scenario inside a disposable workspace."""
    workspace, outside = build_scenario_dirs(scenario, base)
    handler_runs: list[str] = []
    registry = ToolRegistry(
        (
            _recording(READ_FILE_SPEC, handler_runs),
            _recording(WRITE_FILE_SPEC, handler_runs),
            _recording(EDIT_FILE_SPEC, handler_runs),
            _recording(BASH_SPEC, handler_runs),
        )
    )
    guard = WorkspaceGuard(workspace)
    capabilities = {
        Capability.FILESYSTEM_READ,
        Capability.FILESYSTEM_WRITE,
        Capability.PROCESS_EXECUTE,
    }
    facts = PolicyFacts(capabilities, guard.identity)
    context = WorkspaceContext(
        guard=guard,
        workspace_root=str(workspace),
        process_env=build_process_env(),
        mutation_mode=scenario.mutation_mode,
    )
    engine = AgentEngine(
        registry,
        PolicyEngine(facts),
        context,
        ApprovalAuthority(),
        consent=consent,
        event_store=event_store,
        event_session_id=session_id,
        cancel_requested=cancel_requested,
        on_round=on_round,
        budgets=budgets,
    )
    return Harness(
        engine=engine,
        registry=registry,
        workspace=workspace,
        outside=outside,
        guard=guard,
        policy_facts=facts,
        handler_runs=handler_runs,
        workspace_before=snapshot_tree(workspace),
        outside_before=snapshot_tree(outside),
        event_store=event_store,
        session_id=session_id,
    )


def harness_consent(scenario: Scenario, workspace: Path) -> Callable[[ActionPlan], bool]:
    """Default consent: deny unless the scenario opts into auto-approval.

    ``tamper_on_approval`` files are rewritten by the consent callback to
    simulate a concurrent writer racing the approval; the mutation must then
    abort typed instead of landing on stale content.
    """

    def consent(_plan: ActionPlan) -> bool:
        if not scenario.auto_approve:
            return False
        for relative, content in scenario.tamper_on_approval.items():
            (workspace / relative).write_text(content, encoding="utf-8")
        return True

    return consent


def harness_cancel(
    scenario: Scenario,
) -> tuple[Callable[[EngineRound], None] | None, Callable[[], bool] | None]:
    """(round counter, cancel seam) tripping after N admitted rounds."""
    if scenario.cancel_after_rounds <= 0:
        return None, None
    admitted = {"count": 0}

    def on_round(_engine_round: EngineRound) -> None:
        admitted["count"] += 1

    def cancel_requested() -> bool:
        return admitted["count"] >= scenario.cancel_after_rounds

    return on_round, cancel_requested


def scripted_stream(scenario: Scenario) -> Callable[[], ProviderRound]:
    """The scenario's fixed transcript as an engine stream callable."""
    planned = iter(scenario.transcript)

    def stream() -> ProviderRound:
        scripted = next(planned)
        return ProviderRound(text=scripted.text, native_calls=scripted.calls)

    return stream


# -- trace extraction ----------------------------------------------------------


def _derive_decision(executed: bool, error_code: str | None) -> str:
    if error_code == "policy_denied":
        return "deny"
    if error_code == "approval_denied":
        return "require_approval"
    if executed:
        return "allow"
    return ""


def trace_from_outcome(
    scenario: Scenario,
    outcome: EngineResult,
    harness: Harness,
    *,
    latency_ms: float = 0.0,
    peak_context_chars: int = 0,
    peak_prompt_tokens: int = 0,
) -> RunTrace:
    """Flatten an engine outcome plus filesystem state into a RunTrace."""
    calls: list[CallRecord] = []
    for engine_round in outcome.rounds:
        # A mid-round cancellation records the full proposal batch with only
        # partial results; unadmitted proposals simply carry no record.
        for raw_call, result in zip(engine_round.native_calls, engine_round.results, strict=False):
            arguments_json = ""
            if isinstance(raw_call, Mapping):
                arguments_json = json.dumps(
                    raw_call.get("arguments"),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            error_code = result.error.code if result.error else None
            executed = result.call_id in harness.handler_runs
            calls.append(
                CallRecord(
                    call_id=result.call_id,
                    name=result.tool_name,
                    arguments_json=arguments_json,
                    validated=error_code
                    not in {"invalid_tool_call", "unknown_tool", "invalid_arguments"},
                    decision=_derive_decision(executed, error_code),
                    executed=executed,
                    error_code=error_code,
                    output=result.output,
                )
            )
    return RunTrace(
        scenario_id=scenario.id,
        status=outcome.status.value,
        reason=outcome.reason,
        calls=tuple(calls),
        final_text=outcome.text or "",
        handler_runs=tuple(harness.handler_runs),
        workspace_before=harness.workspace_before,
        workspace_after=snapshot_tree(harness.workspace),
        outside_before=harness.outside_before,
        outside_after=snapshot_tree(harness.outside),
        latency_ms=latency_ms,
        peak_context_chars=peak_context_chars,
        peak_prompt_tokens=peak_prompt_tokens,
        policy_digest_before=harness.policy_digest,
        policy_digest_after=harness.policy_digest,
    )


# -- redaction ----------------------------------------------------------------


def _redaction_secrets(workspace: Path, outside: Path) -> list[str]:
    """Absolute paths and environment values that must never leave reports."""
    secrets = [str(workspace), str(outside), str(Path.home()), tempfile.gettempdir()]
    for value in build_process_env().values():
        if len(value) >= 8:
            secrets.append(value)
    return sorted({secret for secret in secrets if secret}, key=len, reverse=True)


def redact_text(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def redact_events(
    events: list[Event], workspace: Path, outside: Path
) -> tuple[list[dict[str, Any]], str]:
    """Redact recorded events and return (events, content digest)."""
    secrets = _redaction_secrets(workspace, outside)
    redacted = [
        {
            "seq": event.seq,
            "type": event.type,
            "payload": json.loads(redact_text(event.payload_json, secrets)),
        }
        for event in events
    ]
    encoded = json.dumps(redacted, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return redacted, hashlib.sha256(encoded.encode()).hexdigest()


# -- live / scripted scenario execution ----------------------------------------


def compile_system_prompt(registry: ToolRegistry) -> tuple[str, str]:
    """The exact system prompt text and its digest for provenance."""
    system = render_system_prompt(native_tools=True, legacy_markdown_tools=False)
    system += "\n" + render_tool_documentation(registry)
    return system, hashlib.sha256(system.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ScenarioRunResult:
    """Score, redacted evidence, and the measured trace of one run."""

    score: ScenarioScore
    outcome: ScoreOutcome
    trace: RunTrace
    redacted_events: tuple[dict[str, Any], ...]


def run_scenario(
    scenario: Scenario,
    provider: EvalProvider | None,
    model_tag: str,
    *,
    corpus_version: str,
    base_dir: Path | None = None,
) -> ScenarioRunResult:
    """Run one scenario in a disposable workspace and score the evidence.

    Model-driven scenarios stream from ``provider``; transcript-driven ones
    replay the corpus attack script. Either way the real engine, registry,
    policy, guard, and mutation paths execute, and the workspace is deleted
    when the run ends (unless the caller supplied ``base_dir``).
    """
    base = (
        Path(base_dir) if base_dir is not None else Path(tempfile.mkdtemp(prefix="shadow-eval-"))  # noqa: S108
    )
    base.mkdir(parents=True, exist_ok=True)
    harness: Harness | None = None
    try:
        round_counter, cancel = harness_cancel(scenario)
        workspace = base / "workspace"
        consent = harness_consent(scenario, workspace)
        store = EventStore(base / "events.db")
        session_id = f"eval-{scenario.id}"
        messages: list[dict[str, Any]] = [{"role": "user", "content": scenario.prompt}]

        def mirror(engine_round: EngineRound) -> None:
            native_calls = []
            for raw in engine_round.native_calls:
                call = dict(raw)
                arguments = call.get("arguments")
                if isinstance(arguments, Mapping):
                    call["arguments"] = thaw_arguments(arguments)
                native_calls.append(call)
            messages.append({"role": "assistant", "content": "", "tool_calls": native_calls})
            for result in engine_round.results:
                if result.success:
                    content = result.output or ""
                elif result.error is not None:
                    content = f"[{result.error.code}] {result.error.message}"
                else:  # pragma: no cover - ToolResult guarantees one payload
                    content = "[error]"
                messages.append({"role": "tool", "name": result.tool_name, "content": content})

        def on_round(engine_round: EngineRound) -> None:
            if round_counter is not None:
                round_counter(engine_round)
            mirror(engine_round)

        harness = build_harness(
            scenario,
            base,
            consent=consent,
            cancel_requested=cancel,
            on_round=on_round,
            event_store=store,
            session_id=session_id,
            budgets=EngineBudgets(max_seconds=LIVE_SCENARIO_MAX_SECONDS),
        )
        system, prompt_digest = compile_system_prompt(harness.registry)
        tools = render_ollama_tool_schemas(harness.registry)
        peaks = {"chars": len(system) + len(scenario.prompt), "tokens": 0}

        if scenario.live_driver == "transcript" or provider is None:
            stream = scripted_stream(scenario)
        else:

            def stream() -> ProviderRound:
                output = provider.round(list(messages), system, model_tag, tools)
                peaks["tokens"] = max(peaks["tokens"], output.prompt_tokens)
                peaks["chars"] = max(
                    peaks["chars"],
                    len(system) + sum(len(str(message)) for message in messages),
                )
                return ProviderRound(text=output.text, native_calls=output.native_calls)

        started = time.monotonic()
        outcome = harness.engine.run_turn(stream)
        latency_ms = (time.monotonic() - started) * 1000.0

        events = store.events_for(session_id)
        redacted, events_digest = redact_events(events, harness.workspace, harness.outside)
        trace = trace_from_outcome(
            scenario,
            outcome,
            harness,
            latency_ms=latency_ms,
            peak_context_chars=peaks["chars"],
            peak_prompt_tokens=peaks["tokens"],
        )
        scored = score_scenario(scenario, trace)
        score = ScenarioScore(
            scenario_id=scenario.id,
            category=scenario.category,
            passed=scored.passed,
            failure_classes=tuple(failure.value for failure in scored.failures),
            metrics=dict(scored.metrics),
            notes=scored.notes,
            latency_ms=trace.latency_ms,
            events_digest=events_digest,
            model_tag=model_tag,
            prompt_digest=prompt_digest,
            registry_digest=harness.registry.digest,
            corpus_version=corpus_version,
        )
        return ScenarioRunResult(
            score=score,
            outcome=scored,
            trace=trace,
            redacted_events=tuple(redacted),
        )
    finally:
        if harness is not None:
            harness.close()
        if base_dir is None:
            shutil.rmtree(base, ignore_errors=True)
