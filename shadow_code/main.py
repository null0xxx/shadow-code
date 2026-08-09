# shadow_code/main.py -- Full integrated REPL
#
# Orchestrates all modules:
#   - Rich UI (panels, markdown, spinners) via ui.py + streaming.py
#   - prompt_toolkit REPL (history, multiline, completion) via repl.py
#   - SQLite session persistence via db.py
#   - Destructive command warnings via safety.py
#   - 3-tier context management (result clearing, compaction, emergency truncate)
#   - Ollama streaming with tool_call buffer via display.py
#   - All 7 tools via tool registry
#
# Falls back to plain-text mode if rich/prompt_toolkit not installed.

import difflib
import os
import platform
import shlex
import signal
import subprocess
import sys
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import cast

from . import tools as tool_reg
from .config import (
    BASH_STRICT,
    CONTEXT_WINDOW,
    LEGACY_MARKDOWN_TOOLS,
    MAX_CONSECUTIVE_ERRORS,
    MAX_NATIVE_TOOL_TURNS,
    MAX_TOOL_TURNS,
    MODEL_NAME,
    MUTATION_STRICT,
)
from .context_compaction import (
    CompactionError,
    active_snapshot,
    append_context_snapshot,
    build_provider_messages,
    build_snapshot,
    context_diagnostics,
    group_events,
    select_closed_range,
    validate_snapshot,
)
from .conversation import COMPACTION_RATIO, Conversation
from .display import TAG_START, StreamDisplay
from .domain.approval import ActionPlan, ApprovalAuthority
from .domain.policy import PolicyFacts, WorkspaceAccessError
from .domain.tools import Capability, ToolError, ToolResult
from .engine import (
    AgentEngine,
    EngineRound,
    EngineState,
    ProviderRound,
    StreamCancelledError,
    StreamError,
)
from .events import (
    AssistantTextPayload,
    EventStore,
    EventStoreError,
    NewEvent,
    SessionEndedPayload,
    SessionStartedPayload,
    TurnCompletedPayload,
    UserMessagePayload,
    default_events_db_path,
    project_events,
)
from .ollama_client import OllamaClient, render_ollama_tool_schemas
from .parser import LegacyMarkdownToolCall, parse_legacy_markdown_tool_calls
from .policy.engine import PolicyEngine
from .policy.workspace import WorkspaceGuard
from .process import build_process_env, detect_sandbox
from .prompt_compiler import (
    CompiledPrompt,
    PromptCompileError,
    PromptManager,
    default_user_overlay_path,
    default_workspace_overlay_path,
    validate_prompt,
)
from .prompt_store import PromptStore, PromptStoreError, default_store_dir
from .safety import check_destructive
from .skills import get_skill, list_skills
from .tool_context import ToolContext
from .tools.catalog import (
    BASH_SPEC,
    EDIT_FILE_SPEC,
    READ_FILE_SPEC,
    WRITE_FILE_SPEC,
    WorkspaceContext,
)
from .tools.registry import ToolRegistry

# Optional imports -- graceful fallback
try:
    from rich.console import Console

    from .streaming import StreamCancelled, StreamController
    from .ui import HAS_RICH, UIRenderer

    _RICH = HAS_RICH
except ImportError:
    _RICH = False

try:
    from .repl import create_prompt_session, get_input

    _HAS_REPL = True
except ImportError:
    _HAS_REPL = False

try:
    from .db import Database

    _HAS_DB = True
except ImportError:
    _HAS_DB = False


def _get_legacy_markdown_tool_calls(
    response: str, *, enabled: bool = LEGACY_MARKDOWN_TOOLS
) -> list[LegacyMarkdownToolCall]:
    """Parse legacy Markdown tool calls only after explicit runtime opt-in."""
    if not enabled:
        return []
    _, calls = parse_legacy_markdown_tool_calls(response)
    return calls


def _legacy_markdown_protocol_error(
    response: str, *, enabled: bool = LEGACY_MARKDOWN_TOOLS
) -> ToolError | None:
    """Describe a rejected legacy envelope without parsing or executing it."""
    if enabled or TAG_START not in response:
        return None
    return ToolError(
        code="protocol_mismatch",
        message="Legacy Markdown tool calls are disabled; no action was executed.",
    )


def _granted_capabilities(bash_strict: bool, sandbox_label: str) -> frozenset[Capability]:
    """Granted capabilities; bash strict mode withholds shell execution.

    Bash strict mode denies shell execution entirely when no kernel
    sandboxing (bwrap/firejail) is available; the policy engine then rejects
    bash with CAPABILITY_NOT_GRANTED instead of running it unconfined.
    Mutation strict mode keeps FILESYSTEM_WRITE granted: approved changes
    are exported as reviewed patches instead of being applied, so the
    capability is still required on the admission path.
    """
    granted = {
        Capability.FILESYSTEM_READ,
        Capability.FILESYSTEM_WRITE,
        Capability.PROCESS_EXECUTE,
    }
    if bash_strict and sandbox_label == "unconfined":
        granted.discard(Capability.PROCESS_EXECUTE)
    return frozenset(granted)


def _request_approval(plan: ActionPlan) -> bool:
    """Render the action plan and ask for a one-shot interactive approval.

    Fail-closed: only an explicit "y" approves; empty input, EOF, interrupt,
    or anything else denies. (The TUI approval control is a later unit.)
    """
    print("Action requires approval:")
    print(f"  tool:       {plan.tool_name} v{plan.tool_version}")
    print(f"  capability: {plan.capability}")
    print(f"  arguments:  {plan.canonical_arguments_json}")
    print(f"  workspace:  device={plan.workspace_device} inode={plan.workspace_inode}")
    print(f"  plan:       sha256:{plan.digest()[:16]}...")
    print(f"  preview:    {plan.preview}")
    try:
        answer = input("Approve this exact action? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer == "y"


def _emit(store: EventStore | None, session_id: str, events: list[NewEvent]) -> None:
    """Append events; a store failure degrades to a warning, never a crash."""
    if store is None or not events:
        return
    try:
        if len(events) == 1:
            store.append(session_id, events[0])
        else:
            store.append_group(session_id, events)
    except (EventStoreError, OSError) as e:
        print(f"[events warning: {e}]")


def _resolve_pending_events(store: EventStore) -> bool:
    """Report unfinished calls from the most recent event session.

    Fail closed: pending work is NEVER re-executed. The user either
    acknowledges and abandons it (fresh session) or leaves. Returns True to
    continue startup, False to exit before any new session starts.
    """
    previous = store.latest_session_id()
    if previous is None:
        return True
    pending = store.pending_tool_calls(previous)
    if not pending:
        return True
    print(
        f"[events] previous session {previous[:12]} interrupted with "
        f"{len(pending)} unfinished tool call(s):"
    )
    for call in pending:
        plan = f"  plan={call.plan_digest[:12]}..." if call.plan_digest else ""
        print(f"  {call.call_id}  {call.name}{plan}")
    print("[events] nothing will be re-executed; continuing abandons the pending calls.")
    try:
        answer = input("Abandon pending calls and start a fresh session? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if answer.strip().lower() == "y":
        return True
    print("[events] leaving; the pending record stays in the event store.")
    return False


def _admit_native_calls(
    native_calls: list[dict],
    registry: ToolRegistry,
    policy_engine: PolicyEngine,
    execution_context: WorkspaceContext,
    approval_authority: ApprovalAuthority,
    *,
    event_store: EventStore | None = None,
    event_session_id: str = "",
) -> list[ToolResult]:
    """Compatibility wrapper: one admission pass through the bounded engine.

    Fail-closed at every step: invalid envelopes, unregistered tools, and
    policy denials produce typed error results and never reach a handler.
    Approval-required calls ask interactively for a one-shot, digest-bound
    token; denial or cancellation is final and the call is not retried.
    """
    engine = AgentEngine(
        registry,
        policy_engine,
        execution_context,
        approval_authority,
        consent=_request_approval,
        event_store=event_store,
        event_session_id=event_session_id,
        on_store_warning=lambda message: print(f"[events warning: {message}]"),
    )
    return engine.admit_calls(native_calls)


def _report_prompt_switch(compiled: CompiledPrompt, previous: str | None) -> None:
    """Print the snapshot attribution line and, on a switch, the audit line.

    The active snapshot digest is also recorded per turn in the event
    store (turn_completed, WU-06).
    """
    print(f"prompt snapshot: {compiled.digest[:12]}")
    if previous:
        print(f"prompt: active {previous[:12]} -> {compiled.digest[:12]}")


def _prompt_show(manager: PromptManager) -> None:
    lines = manager.active.compiled_text.splitlines()
    for line in lines[:200]:
        print(line)
    if len(lines) > 200:
        print(f"... [{len(lines) - 200} more lines truncated]")


def _prompt_diff(manager: PromptManager, prefix: str) -> None:
    active = manager.active
    if prefix:
        base = manager.store.load(prefix)
    else:
        snapshots = [s for s in manager.store.history() if s.digest != active.digest]
        if not snapshots:
            print("  No previous snapshot to diff against")
            return
        base = snapshots[0]
    diff = list(
        difflib.unified_diff(
            base.compiled_text.splitlines(),
            active.compiled_text.splitlines(),
            fromfile=f"snapshot {base.digest[:12]}",
            tofile=f"active {active.digest[:12]}",
            lineterm="",
        )
    )
    if not diff:
        print("  No differences")
        return
    for line in diff[:400]:
        print(line)
    if len(diff) > 400:
        print(f"... [{len(diff) - 400} more diff lines truncated]")


def _prompt_edit(manager: PromptManager) -> None:
    """Open the user overlay in $EDITOR, then recompile + activate."""
    editor = os.environ.get("EDITOR", "vi")
    path = manager.user_path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# shadow-code user prompt overlay\n", encoding="utf-8")
    argv = [*shlex.split(editor), str(path)]
    try:
        # User-chosen $EDITOR launched interactively on explicit request.
        subprocess.run(argv, check=False)  # nosec B603 B607
    except OSError as exc:
        print(f"  Cannot launch editor {editor!r}: {exc}")
        return
    compiled, previous = manager.reload()
    _report_prompt_switch(compiled, previous)


def _handle_prompt_command(user_input: str, manager: PromptManager) -> None:
    """Dispatch /prompt subcommands: inspect, validate, reload, roll back."""
    args = user_input[len("/prompt") :].strip().split()
    sub = args[0] if args else ""
    try:
        if sub == "show":
            _prompt_show(manager)
        elif sub == "sources":
            print(f"  active: {manager.active.digest[:12]} ({manager.active.created_utc})")
            for source in manager.active.sources:
                print(
                    f"  {source.layer:10} {source.origin}  "
                    f"sha256:{source.sha256[:12]}  {source.size} bytes"
                )
        elif sub == "history":
            snapshots = manager.store.history()
            if not snapshots:
                print("  No prompt snapshots")
            for snapshot in snapshots:
                marker = "*" if snapshot.digest == manager.active.digest else " "
                layers = ",".join(source.layer for source in snapshot.sources)
                print(f" {marker} {snapshot.digest[:12]}  {snapshot.created_utc}  {layers}")
        elif sub == "diff":
            _prompt_diff(manager, args[1] if len(args) > 1 else "")
        elif sub == "validate":
            issues = validate_prompt(manager.active, manager.registry)
            if issues:
                for issue in issues:
                    print(f"  issue: {issue}")
            else:
                print("  prompt OK: structure, digests, and tool docs verified")
        elif sub == "reload":
            compiled, previous = manager.reload()
            _report_prompt_switch(compiled, previous)
        elif sub == "edit":
            _prompt_edit(manager)
        elif sub == "rollback":
            if len(args) < 2:
                print("  Usage: /prompt rollback <digest-prefix>")
                return
            target, previous = manager.rollback(args[1])
            _report_prompt_switch(target, previous)
        else:
            print(
                "  Usage: /prompt show|sources|history|diff [digest]|validate|"
                "reload|edit|rollback <digest-prefix>"
            )
    except (PromptCompileError, PromptStoreError) as exc:
        # Fail visibly; the active snapshot stays untouched on any error.
        print(f"  prompt {sub or '?'} failed [{exc.code}]: {exc.message}")


def _summarize_with_model(client: OllamaClient, messages: list[dict], system_prompt: str) -> str:
    """Summarize messages through the existing LLM compaction path."""
    from .compaction import compact

    return compact(client, messages, system_prompt)


def _compact_from_events(
    store: EventStore,
    session_id: str,
    conv: Conversation,
    workspace_guard: WorkspaceGuard,
    summarize: Callable[[list[dict]], str],
) -> str:
    """Event-sourced /compact (WU-08): snapshot a closed range of complete
    causal groups, then rebuild the live conversation from the projection.

    Fail-closed: any summarizer failure, validation issue, or overlap is
    raised BEFORE the snapshot is appended and BEFORE the conversation is
    touched. The event log is never modified -- one context_snapshot event
    is appended; the original events stay queryable.
    """
    events = store.events_for(session_id)
    groups = group_events(events)
    active = active_snapshot(store, session_id)
    snapshot_event_ids = {event.event_id for event in events if event.type == "context_snapshot"}
    eligible = [
        group
        for group in groups
        if (active is None or group.seq_start > active.covered_seq_end)
        and not any(event_id in snapshot_event_ids for event_id in group.event_ids)
    ]
    budget = int(CONTEXT_WINDOW * COMPACTION_RATIO)
    selected = select_closed_range(eligible, budget)
    if not selected:
        raise CompactionError("nothing_to_compact", "no complete groups beyond the active snapshot")
    by_id = {event.event_id: event for event in events}
    covered_events = [by_id[event_id] for group in selected for event_id in group.event_ids]
    summary_text = summarize(project_events(covered_events))
    snapshot = build_snapshot(session_id, selected, summary_text)
    issues = validate_snapshot(snapshot, selected, workspace_guard)
    if issues:
        raise CompactionError("snapshot_invalid", "; ".join(issues))
    append_context_snapshot(store, snapshot)
    messages = build_provider_messages(store, session_id)
    tokens_before = conv.total_prompt_tokens
    conv.messages = messages
    tokens_after = max(1, sum(len(str(message.get("content", ""))) for message in messages) // 4)
    conv.update_tokens(tokens_after)
    return (
        f"[Compacted: {len(selected)} group(s) summarized, "
        f"~{tokens_before} -> ~{tokens_after} tokens; original events retained]"
    )


def _show_context_diagnostics(store: EventStore, session_id: str) -> None:
    """Print context_diagnostics for the /context command."""
    diag = context_diagnostics(store, session_id)
    kinds = diag["groups_by_kind"]
    print(
        f"  events: {diag['total_events']}  groups: {diag['groups_total']} "
        f"(messages {kinds['message']}, tool calls {kinds['tool_call']})"
    )
    print(
        f"  terminal: {diag['terminal_groups']}  pending: {diag['pending_groups']}  "
        f"~{diag['estimated_uncovered_tokens']} uncovered tokens"
    )
    snapshot = diag["active_snapshot"]
    if snapshot is None:
        print("  snapshot: none")
    else:
        print(
            f"  snapshot: seq {snapshot['covered_seq_start']}-"
            f"{snapshot['covered_seq_end']} ({snapshot['covered_group_count']} "
            f"groups, {snapshot['covered_group_percent']}% covered)"
        )
        print(
            f"  digests: source {snapshot['source_digest'][:12]}  "
            f"events {snapshot['covered_event_ids_digest'][:12]}"
        )
    for issue in diag["issues"]:
        print(f"  issue: {issue}")


def main():
    cwd = os.getcwd()
    ctx = ToolContext(cwd)

    # Admission: containment guard, policy facts, runtime registry.
    # bash executes UNCONFINED after an explicit one-shot approval bound to
    # the exact command, workspace, environment digest, and sandbox facts;
    # strict mode withholds the capability when no sandbox is available.
    # write_file/edit_file mutate only after an explicit one-shot approval
    # bound to the exact arguments and previewed diff; mutation strict mode
    # keeps the capability but exports approved changes as reviewed patches
    # instead of applying them.
    try:
        workspace_guard = WorkspaceGuard(cwd)
    except WorkspaceAccessError as e:
        print(f"Error: cannot establish workspace containment: {e}", file=sys.stderr)
        sys.exit(1)
    runtime_registry = ToolRegistry((READ_FILE_SPEC, WRITE_FILE_SPEC, EDIT_FILE_SPEC, BASH_SPEC))
    process_env = build_process_env()
    sandbox_label = detect_sandbox()
    capabilities = _granted_capabilities(BASH_STRICT, sandbox_label)
    policy_engine = PolicyEngine(PolicyFacts(capabilities, workspace_guard.identity))
    execution_context = WorkspaceContext(
        guard=workspace_guard,
        workspace_root=cwd,
        process_env=process_env,
        sandbox_label=sandbox_label,
        mutation_mode="export" if MUTATION_STRICT else "apply",
    )
    if Capability.PROCESS_EXECUTE not in capabilities:
        print("[bash disabled: strict mode and no sandbox (bwrap/firejail) available]")
    else:
        print("[bash runs UNCONFINED; every execution requires explicit approval]")
    if MUTATION_STRICT:
        print(
            "[file mutations run in strict mode: changes are exported as "
            "reviewed patches, never applied]"
        )
    approval_authority = ApprovalAuthority()
    tool_schemas = render_ollama_tool_schemas(runtime_registry)

    # Layered prompt: compiled from builtin base + optional user/workspace
    # overlays + generated tool docs (registry is the only tool truth).
    # Prompt contents never touch PolicyFacts -- text cannot grant capabilities.
    try:
        prompt_manager, previous_prompt = PromptManager.bootstrap(
            registry=runtime_registry,
            store=PromptStore(default_store_dir()),
            user_path=default_user_overlay_path(),
            workspace_path=default_workspace_overlay_path(cwd),
            legacy_markdown_tools=LEGACY_MARKDOWN_TOOLS,
        )
    except (PromptCompileError, PromptStoreError) as e:
        print(f"Error: cannot compile system prompt [{e.code}]: {e.message}", file=sys.stderr)
        sys.exit(1)
    system_prompt = prompt_manager.active.compiled_text
    _report_prompt_switch(prompt_manager.active, previous_prompt)

    # Register all tools
    from .tools.bash import BashTool

    tool_reg.register(BashTool(ctx))
    _register_optional_tools(ctx)

    # Ollama health check
    client = OllamaClient()
    ok, msg = client.health_check()
    if not ok:
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)

    # Setup UI
    if _RICH:
        console = Console()
        ui = UIRenderer()
        display = StreamDisplay()
        stream_ctrl = StreamController(client, ui, console, display)
        console.print(ui.render_welcome())
    else:
        console = None
        ui = None
        display = StreamDisplay()
        stream_ctrl = None
        print(f"shadow-code v0.1.0 | {MODEL_NAME}")
        print(f"CWD: {cwd}")

    # Session state for toolbar
    from .status_bar import SessionState

    state = SessionState(
        model_name=MODEL_NAME,
        tokens_total=CONTEXT_WINDOW,
        max_turns=MAX_TOOL_TURNS,
    )

    # Setup REPL (pass state for bottom toolbar)
    prompt_session = create_prompt_session(state) if _HAS_REPL else None

    # Setup DB
    db = None
    session_id = None
    if _HAS_DB:
        try:
            db = Database()
            session_id = db.create_session(MODEL_NAME)
        except Exception as e:
            print(f"[DB warning: {e}]")
            db = None

    # Event store (WU-06): append-only causal authority for resume and
    # audit. Any failure downgrades to a warning -- the store must never
    # break the CLI. A previous session's unfinished calls are reported and
    # abandoned only with explicit acknowledgment; nothing is re-executed.
    event_store = None
    event_session_id = ""
    try:
        event_store = EventStore(default_events_db_path())
        if not _resolve_pending_events(event_store):
            event_store.close()
            workspace_guard.close()
            if db:
                db.close()
            return
        event_session_id = uuid.uuid4().hex
        _emit(
            event_store,
            event_session_id,
            [
                NewEvent(
                    "session_started",
                    SessionStartedPayload(model=MODEL_NAME, cwd=cwd),
                )
            ],
        )
    except (EventStoreError, OSError) as e:
        print(f"[events warning: {e}]")
        event_store = None

    print("Commands: /help /clear /exit /tokens /save /load /list /info /prompt /events /context\n")

    conv = Conversation()
    interrupted = False
    first_message = True

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, on_sigint)

    def handle_round(engine_round: EngineRound) -> None:
        """Mirror one admitted round into the transcript and conversation."""
        conv.add_assistant_tool_call(list(engine_round.native_calls))
        for result in engine_round.results:
            if result.success:
                result_text = result.output or ""
                print(f"  [{result.tool_name}] ok")
            else:
                error = cast(ToolError, result.error)  # not success implies error
                result_text = f"[{error.code}] {error.message}"
                print(f"  [{result.tool_name}] {error.code}")
            conv.add_native_tool_result(result.tool_name, result_text)

    # Bounded engine (WU-07): drives the native tool rounds of each user
    # turn. All I/O seams are injected -- consent is the interactive
    # approval prompt, cancellation reads the SIGINT flag.
    engine = AgentEngine(
        runtime_registry,
        policy_engine,
        execution_context,
        approval_authority,
        consent=_request_approval,
        event_store=event_store,
        event_session_id=event_session_id,
        cancel_requested=lambda: interrupted,
        on_round=handle_round,
        on_store_warning=lambda message: print(f"[events warning: {message}]"),
    )

    def stream_round() -> ProviderRound:
        """One provider round: stream + display, return text and calls.

        Defined once; it reads the per-turn locals (system_prompt, turns,
        interrupted) live at call time.
        """
        if _RICH and stream_ctrl:
            try:
                resp, eval_tokens = stream_ctrl.stream_response(
                    conv.get_messages(), system_prompt, tools=tool_schemas
                )
            except StreamCancelled:
                raise StreamCancelledError from None
            except Exception as error:
                raise StreamError("provider_error", str(error)) from error
        else:
            display.reset()
            cancelled = False
            try:
                for chunk in client.chat_stream(
                    conv.get_messages(), system_prompt, tools=tool_schemas
                ):
                    if interrupted:
                        cancelled = True
                        break
                    display.feed(chunk)
            except KeyboardInterrupt:
                cancelled = True
            except Exception as error:
                raise StreamError("provider_error", str(error)) from error
            display.flush()
            print()
            if cancelled:
                raise StreamCancelledError
            resp = display.get_full_response()
        conv.update_tokens(client.last_prompt_tokens)
        state.tokens_used = conv.total_prompt_tokens
        state.turn = turns
        return ProviderRound(
            text=resp or "",
            native_calls=tuple(getattr(client, "last_tool_calls", [])),
        )

    while True:
        # Get input
        if _HAS_REPL:
            user_input = get_input(prompt_session, MODEL_NAME)
            if user_input is None:
                break
        else:
            try:
                user_input = input("shadow> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

        if not user_input:
            continue

        # === Slash commands ===
        if user_input == "/exit":
            break

        if user_input == "/clear":
            conv.clear()
            first_message = True
            if _RICH:
                console.clear()
                console.print(ui.render_welcome())
            print("[Cleared]")
            continue

        if user_input == "/tokens":
            used = conv.total_prompt_tokens
            if _RICH:
                console.print(ui.render_context_status(used, CONTEXT_WINDOW))
            else:
                pct = (used / CONTEXT_WINDOW * 100) if CONTEXT_WINDOW else 0
                print(f"Context: {used} tokens ({pct:.0f}% of {CONTEXT_WINDOW})")
            continue

        if user_input == "/info":
            print(f"  Model:    {MODEL_NAME}")
            print(f"  CWD:      {ctx.cwd}")
            print(f"  Messages: {len(conv.get_messages())}")
            print(f"  Tokens:   {conv.total_prompt_tokens}")
            print(f"  Tools:    {', '.join(tool_reg._REGISTRY.keys())}")
            print(f"  Skills:   {len(list_skills())}")
            if session_id:
                print(f"  Session:  #{session_id}")
            continue

        if user_input.startswith("/cd"):
            target = user_input[3:].strip()
            if not target:
                print(f"  CWD: {ctx.cwd}")
            else:
                import os as _os

                new = _os.path.normpath(_os.path.join(ctx.cwd, _os.path.expanduser(target)))
                if _os.path.isdir(new):
                    ctx.cwd = new
                    print(f"  CWD: {ctx.cwd}")
                else:
                    print(f"  Not a directory: {new}")
            continue

        if user_input == "/version":
            from . import __version__

            print(f"  shadow-code v{__version__}")
            print(f"  Model: {MODEL_NAME}")
            print(f"  Context: {CONTEXT_WINDOW // 1024}K")
            continue

        if user_input == "/history":
            msgs = conv.get_messages()
            if not msgs:
                print("  No messages yet")
            else:
                for i, m in enumerate(msgs[-10:], max(1, len(msgs) - 9)):
                    role = m["role"]
                    preview = m["content"][:80].replace("\n", " ")
                    print(f"  {i:3}. [{role:9}] {preview}...")
            continue

        if user_input == "/compact":
            if event_store is not None and event_session_id:
                # WU-08: snapshot complete causal groups via the event store;
                # the legacy message-level path remains for a degraded store.
                try:
                    line = _compact_from_events(
                        event_store,
                        event_session_id,
                        conv,
                        workspace_guard,
                        lambda messages, prompt=system_prompt: _summarize_with_model(
                            client, messages, prompt
                        ),
                    )
                except CompactionError as e:
                    if e.code == "nothing_to_compact":
                        print(f"  {e.message}")
                    else:
                        print(f"[Compaction failed ({e.code}): {e.message}]")
                except Exception as e:
                    print(f"[Compaction failed: {e}]")
                else:
                    print(line)
            elif conv.total_prompt_tokens > 0:
                print("[Compacting conversation...]")
                try:
                    from .compaction import compact

                    summary = compact(client, conv.get_messages(), system_prompt)
                    conv.apply_compaction_summary(summary)
                    print("[Compaction complete]")
                except Exception as e:
                    print(f"[Compaction failed: {e}]")
            else:
                print("  Nothing to compact")
            continue

        if user_input == "/context":
            if event_store is not None and event_session_id:
                _show_context_diagnostics(event_store, event_session_id)
            else:
                print("  [event store not available]")
            continue

        if user_input == "/help":
            cmds = [
                ("/help", "Show this help"),
                ("/clear", "Clear conversation"),
                ("/exit", "Exit shadow-code"),
                ("/tokens", "Show context usage"),
                ("/info", "Show session info"),
                ("/cd [path]", "Show or change working directory"),
                ("/compact", "Compact complete context groups into a snapshot"),
                ("/context", "Show context diagnostics"),
                ("/history", "Show last 10 messages"),
                ("/version", "Show version info"),
                ("/save [name]", "Save session"),
                ("/load [id]", "Load session"),
                ("/list", "List saved sessions"),
                ("/skills", "List available skills"),
                ("/prompt <sub>", "Inspect/reload/rollback the system prompt"),
                ("/events", "Verify event store integrity"),
            ]
            # Add skills
            for skill_name, skill_desc in list_skills():
                cmds.append((f"/{skill_name}", skill_desc))
            # Keyboard shortcuts
            keys = [
                ("", ""),  # separator
                ("Ctrl+C", "Stop current generation"),
                ("Ctrl+D", "Exit"),
                ("Ctrl+X", "Exit"),
                ("Ctrl+L", "Clear screen"),
                ("Ctrl+U", "Clear input line"),
                ("Alt+Enter", "New line (multiline input)"),
                ("Up/Down", "Command history"),
                ("Ctrl+R", "Search history"),
            ]
            cmds.extend(keys)
            if _RICH:
                console.print(ui.render_help(cmds))
            else:
                for cmd, desc in cmds:
                    print(f"  {cmd:20} {desc}")
            continue

        if user_input == "/skills":
            print("  Available skills:")
            for skill_name, skill_desc in list_skills():
                print(f"    /{skill_name:15} {skill_desc}")
            continue

        if user_input.startswith("/save"):
            if db:
                name = user_input[5:].strip() or f"Session #{session_id}"
                db.rename_session(session_id, name)
                print(f"  Session saved as '{name}'")
            else:
                print("  [DB not available]")
            continue

        if user_input.startswith("/load"):
            if db:
                arg = user_input[5:].strip()
                if arg:
                    try:
                        sid = int(arg)
                        s = db.get_session(sid)
                        if s:
                            conv.clear()
                            first_message = True
                            for m in s["messages"]:
                                if m["role"] == "user":
                                    conv.add_user(m["content"])
                                    first_message = False
                                elif m["role"] == "assistant":
                                    conv.add_assistant(m["content"])
                            session_id = sid
                            print(f"  Loaded session #{sid} ({len(s['messages'])} messages)")
                        else:
                            print(f"  Session #{sid} not found")
                    except ValueError:
                        print("  Usage: /load <id>")
                else:
                    print("  Usage: /load <id>")
            else:
                print("  [DB not available]")
            continue

        if user_input == "/list":
            if db:
                sessions = db.list_sessions()
                if sessions:
                    for s in sessions:
                        name = s.get("name", "") or f"Session #{s['id']}"
                        msgs = s.get("message_count", 0)
                        print(f"  #{s['id']:4}  {name:30} ({msgs} msgs)")
                else:
                    print("  No saved sessions")
            else:
                print("  [DB not available]")
            continue

        if user_input == "/events":
            if event_store is not None:
                issues = event_store.verify(event_session_id)
                count = len(event_store.events_for(event_session_id))
                if issues:
                    print(f"  event store: {len(issues)} issue(s) across {count} events")
                    for issue in issues:
                        print(f"  issue: {issue}")
                else:
                    print(f"  event store OK: {count} events, integrity verified")
            else:
                print("  [event store not available]")
            continue

        if user_input.startswith("/prompt"):
            _handle_prompt_command(user_input, prompt_manager)
            continue

        # Check if it's a skill command (e.g., /commit, /simplify, /review file.py)
        if user_input.startswith("/"):
            parts = user_input[1:].split(None, 1)
            skill_name = parts[0] if parts else ""
            skill_args = parts[1] if len(parts) > 1 else ""
            skill = get_skill(skill_name)
            if skill:
                desc, prompt_template = skill
                skill_msg = prompt_template
                if skill_args:
                    skill_msg += f"\n\nUser specified: {skill_args}"
                user_input = skill_msg  # falls through to message handling below
                # DO NOT continue -- let it fall through to send the skill prompt to the model
            else:
                print(f"  Unknown command: /{skill_name}. Type /help")
                continue

        # === Per-turn prompt source watch ===
        # An overlay edit affects only the next turn; on any failure the
        # previous active snapshot stays in place and a warning is printed.
        try:
            watched = prompt_manager.watch()
        except (PromptCompileError, PromptStoreError) as e:
            print(f"[prompt warning: keeping {prompt_manager.active.digest[:12]}: {e.message}]")
        else:
            if watched is not None:
                compiled, previous = watched
                system_prompt = compiled.compiled_text
                _report_prompt_switch(compiled, previous)

        # === Inject environment on first message ===
        if first_message:
            shell = os.environ.get("SHELL", "/bin/bash").rsplit("/", 1)[-1]
            env_prefix = (
                f"[Environment]\n"
                f"CWD: {ctx.cwd}\n"
                f"Platform: {platform.system()} {platform.release()}\n"
                f"Shell: {shell}\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
                f"[/Environment]\n\n"
            )
            # Auto-inject CLAUDE.md / SHADOW.md if present
            for ctx_file in ["CLAUDE.md", "SHADOW.md", ".shadow-code/context.md"]:
                ctx_path = os.path.join(ctx.cwd, ctx_file)
                if os.path.isfile(ctx_path):
                    try:
                        with open(ctx_path, encoding="utf-8") as f:
                            ctx_content = f.read()
                        if len(ctx_content) < 8000:
                            env_prefix += (
                                f"[Project context from {ctx_file}]\n"
                                f"{ctx_content}\n"
                                f"[/Project context]\n\n"
                            )
                    except OSError:
                        pass
                    break
            conv.add_user(env_prefix + user_input)
            user_content = env_prefix + user_input
            first_message = False
        else:
            conv.add_user(user_input)
            user_content = user_input

        _emit(
            event_store,
            event_session_id,
            [NewEvent("user_message", UserMessagePayload(content=user_content))],
        )

        # Save to DB
        if db:
            db.add_message(session_id, "user", user_input)

        # === Tool execution loop (bounded engine drives native rounds) ===
        turns = 0
        errors = 0

        while turns < MAX_TOOL_TURNS:
            interrupted = False
            outcome = engine.run_turn(stream_round)
            turns += outcome.steps

            if outcome.status is EngineState.CANCELLED:
                print("[Interrupted]")
                break
            if outcome.status is EngineState.FAILED:
                if _RICH:
                    console.print(ui.render_error(outcome.detail))
                else:
                    print(f"\n[Error: {outcome.detail}]")
                break
            if outcome.status is EngineState.BUDGET_EXHAUSTED:
                if outcome.reason == "budget_steps":
                    print(
                        f"[Budget exhausted (budget_steps): "
                        f"native tool limit ({MAX_NATIVE_TOOL_TURNS}) reached]"
                    )
                else:
                    print(f"[Budget exhausted ({outcome.reason})]")
                break

            # No native tool calls -- check for text response
            resp = outcome.text or ""
            if resp and resp.strip():
                conv.add_assistant(resp)
                _emit(
                    event_store,
                    event_session_id,
                    [NewEvent("assistant_text", AssistantTextPayload(content=resp))],
                )
                if db:
                    db.add_message(session_id, "assistant", resp)
                    db.update_session_tokens(session_id, conv.total_prompt_tokens)

                # Explicitly opt-in compatibility path for old Markdown tool calls.
                markdown_calls = _get_legacy_markdown_tool_calls(resp)
                if not markdown_calls:
                    protocol_error = _legacy_markdown_protocol_error(resp)
                    if protocol_error:
                        if _RICH:
                            console.print(ui.render_error(protocol_error.message))
                        else:
                            print(f"[{protocol_error.code}] {protocol_error.message}")
                    break  # Pure text response, done

                # Execute markdown tool calls (legacy path)
                results = []
                for tc in markdown_calls:
                    if tc.tool == "__invalid__":
                        r = tool_reg.ToolResult(False, tc.params.get("error", "Invalid"))
                    else:
                        desc = _build_tool_desc(tc.tool, tc.params)
                        if tc.tool == "bash":
                            warning = check_destructive(tc.params.get("command", ""))
                            if warning:
                                print(f"  {warning}")
                                try:
                                    confirm = input("  Proceed? (y/n): ").strip().lower()
                                except (EOFError, KeyboardInterrupt):
                                    confirm = "n"
                                if confirm != "y":
                                    r = tool_reg.ToolResult(False, "Command cancelled")
                                    results.append(tool_reg.format_result(tc.tool, r))
                                    continue
                        if _RICH:
                            console.print(ui.render_tool_call(tc.tool, desc))
                        r = tool_reg.dispatch(tc.tool, tc.params)
                        if _RICH:
                            console.print(ui.render_tool_result(tc.tool, r.output, r.success))
                    results.append(tool_reg.format_result(tc.tool, r))
                    errors = errors + 1 if not r.success else 0

                conv.add_tool_results("\n\n".join(results))
                turns += 1
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    break
                continue

            # Empty response
            break

        if turns >= MAX_TOOL_TURNS:
            print(f"[Tool limit ({MAX_TOOL_TURNS}) reached]")

        # Record the active prompt snapshot digest for this completed turn.
        _emit(
            event_store,
            event_session_id,
            [
                NewEvent(
                    "turn_completed",
                    TurnCompletedPayload(prompt_digest=prompt_manager.active.digest),
                )
            ],
        )

        # === Context status (always visible) ===
        conv.update_tokens(client.last_prompt_tokens)
        if conv.total_prompt_tokens > 0:
            _show_context_status(
                conv.total_prompt_tokens,
                CONTEXT_WINDOW,
                client.last_eval_tokens,
                console if _RICH else None,
                ui if _RICH else None,
            )

        # === 3-Tier Context Management ===
        if conv.needs_result_clearing():
            conv.clear_old_tool_results()

        if conv.needs_compaction():
            print("[Compacting conversation...]")
            try:
                from .compaction import compact

                summary = compact(client, conv.get_messages(), system_prompt)
                conv.apply_compaction_summary(summary)
                print("[Compaction complete]")
            except Exception as e:
                print(f"[Compaction failed: {e}]")

            if conv.needs_emergency_truncate():
                conv.emergency_truncate()
                print("[Emergency truncation applied]")

    # Cleanup
    _emit(
        event_store,
        event_session_id,
        [NewEvent("session_ended", SessionEndedPayload(reason="exit"))],
    )
    if event_store is not None:
        event_store.close()
    workspace_guard.close()
    if db:
        db.close()
    print("Goodbye!")


def _build_tool_desc(tool_name: str, params: dict) -> str:
    """Build a human-readable description of a tool call."""
    if tool_name == "bash":
        return str(params.get("command", ""))[:100]
    elif tool_name in ("read_file", "write_file", "edit_file"):
        desc = str(params.get("file_path", ""))
        if tool_name == "edit_file":
            old = params.get("old_string", "")[:40]
            new = params.get("new_string", "")[:40]
            desc += f'  "{old}" -> "{new}"'
        elif tool_name == "write_file":
            desc += f"  ({len(params.get('content', ''))} chars)"
        return desc
    elif tool_name == "grep":
        return f'"{params.get("pattern", "")}" in {params.get("path", ".")}'
    elif tool_name == "glob":
        return f"{params.get('pattern', '')} in {params.get('path', '.')}"
    elif tool_name == "multi_read":
        paths = params.get("paths", [])
        return f"{len(paths)} files"
    else:
        return str(params)[:80]


def _register_optional_tools(ctx):
    """Register Phase 2 tools if available."""
    optional = [
        ("shadow_code.tools.read_file", "ReadFileTool"),
        ("shadow_code.tools.edit_file", "EditFileTool"),
        ("shadow_code.tools.write_file", "WriteFileTool"),
        ("shadow_code.tools.glob_tool", "GlobTool"),
        ("shadow_code.tools.grep_tool", "GrepTool"),
        ("shadow_code.tools.list_dir", "ListDirTool"),
        ("shadow_code.tools.multi_read", "MultiReadTool"),
        ("shadow_code.tools.project_summary", "ProjectSummaryTool"),
        ("shadow_code.tools.file_backup", "FileBackupTool"),
        ("shadow_code.tools.file_backup", "FileRestoreTool"),
        ("shadow_code.tools.get_language_rules", "GetLanguageRulesTool"),
    ]
    for mod_path, cls_name in optional:
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            tool_reg.register(getattr(mod, cls_name)(ctx))
        except (ImportError, AttributeError):
            pass


def _show_context_status(used: int, total: int, last_eval: int, console=None, ui=None):
    """Show context usage after every turn. Works in both Rich and plain mode."""
    pct = (used / total * 100) if total else 0
    bar_width = 20
    filled = int(bar_width * pct / 100)
    bar = "=" * filled + "-" * (bar_width - filled)

    if console and ui:
        console.print(ui.render_context_status(used, total))
    else:
        # Soft ANSI colors (Claude Code inspired)
        if pct < 50:
            color = "\033[38;5;71m"  # soft green
        elif pct < 75:
            color = "\033[38;5;179m"  # soft amber
        else:
            color = "\033[38;5;167m"  # soft red
        dim = "\033[2m"
        reset = "\033[0m"
        print(f"  {color}[{bar}] {used // 1000}K/{total // 1000}K{reset} {dim}({pct:.0f}%){reset}")


if __name__ == "__main__":
    main()
