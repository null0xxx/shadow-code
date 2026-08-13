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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from . import mcp, ops
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
    TUI_ENABLED,
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
    McpServerPayload,
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
from .status_bar import SessionState
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
    from .db import Database, default_db_path

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


def _granted_capabilities(
    bash_strict: bool, sandbox_label: str, *, mcp_active: bool = False
) -> frozenset[Capability]:
    """Granted capabilities; bash strict mode withholds shell execution.

    Bash strict mode denies shell execution entirely when no kernel
    sandboxing (bwrap/firejail) is available; the policy engine then rejects
    bash with CAPABILITY_NOT_GRANTED instead of running it unconfined.
    Mutation strict mode keeps FILESYSTEM_WRITE granted: approved changes
    are exported as reviewed patches instead of being applied, so the
    capability is still required on the admission path. MCP_INVOKE is
    granted only when at least one configured MCP server discovered usable
    tools this session; every MCP call still requires one-shot approval.
    """
    granted = {
        Capability.FILESYSTEM_READ,
        Capability.FILESYSTEM_WRITE,
        Capability.PROCESS_EXECUTE,
    }
    if bash_strict and sandbox_label == "unconfined":
        granted.discard(Capability.PROCESS_EXECUTE)
    if mcp_active:
        granted.add(Capability.MCP_INVOKE)
    return frozenset(granted)


def _authority_facts(
    sandbox_label: str, *, mcp_active: bool = False
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Granted capability values plus every withheld capability and its reason."""
    capabilities = _granted_capabilities(BASH_STRICT, sandbox_label, mcp_active=mcp_active)
    granted = tuple(sorted(capability.value for capability in capabilities))
    withheld: list[tuple[str, str]] = []
    if Capability.PROCESS_EXECUTE not in capabilities:
        withheld.append(
            (
                Capability.PROCESS_EXECUTE.value,
                "SHADOW_BASH_STRICT=1, no sandbox (bwrap/firejail) available",
            )
        )
    withheld.append((Capability.NETWORK_ACCESS.value, "not supported by this build"))
    if Capability.MCP_INVOKE not in capabilities:
        withheld.append((Capability.MCP_INVOKE.value, "no MCP servers configured"))
    return granted, tuple(withheld)


def _doctor_facts(rt: "SessionRuntime") -> ops.DoctorFacts:
    """Snapshot the live runtime into the facts the doctor report consumes."""
    guard = rt.workspace_guard
    exec_ctx = rt.execution_context
    prompt_manager = rt.prompt_manager
    if guard is None or exec_ctx is None or prompt_manager is None:
        raise RuntimeError("doctor requires a fully bootstrapped runtime")
    mcp_manager = rt.mcp_manager
    mcp_active = mcp_manager is not None and bool(mcp_manager.ready_servers)
    granted, withheld = _authority_facts(exec_ctx.sandbox_label, mcp_active=mcp_active)
    ok, message = rt.client.health_check() if rt.client is not None else (False, "no client")
    return ops.DoctorFacts(
        workspace_root=exec_ctx.workspace_root,
        workspace_device=guard.identity.device,
        workspace_inode=guard.identity.inode,
        containment="openat2",
        granted=granted,
        withheld=withheld,
        sandbox_label=exec_ctx.sandbox_label,
        mutation_mode=exec_ctx.mutation_mode,
        model_name=MODEL_NAME,
        ollama_ok=ok,
        ollama_message=message,
        prompt_digest=prompt_manager.active.digest,
        prompt_layer_count=len(prompt_manager.active.sources),
        prompt_store_path=str(prompt_manager.store.root),
        events_db_path=str(rt.events_db_path) if rt.events_db_path else None,
        legacy_db_path=rt.db_path,
        event_store=rt.event_store,
        mcp_servers=mcp_manager.report_lines() if mcp_manager is not None else (),
    )


@dataclass
class SessionRuntime:
    """Mutable composition root shared by the line REPL and the TUI (WU-09).

    main() fills every field during bootstrap; the slash-command dispatch
    and the per-turn handler operate only through this object, so both
    frontends drive the identical session machinery. Output reaches the
    user exclusively through injected writers, never through a frontend
    assumption baked into the shared logic.
    """

    cwd: str = ""
    ctx: ToolContext | None = None
    workspace_guard: WorkspaceGuard | None = None
    registry: ToolRegistry | None = None
    policy_engine: PolicyEngine | None = None
    execution_context: WorkspaceContext | None = None
    approval_authority: ApprovalAuthority | None = None
    tool_schemas: list[dict] = field(default_factory=list)
    prompt_manager: PromptManager | None = None
    system_prompt: str = ""
    client: OllamaClient | None = None
    console: Any = None
    ui: Any = None
    display: Any = None
    stream_ctrl: Any = None
    state: SessionState | None = None
    prompt_session: Any = None
    db: Any = None
    db_path: Any = None  # str; the TUI opens its worker-thread Database here
    session_id: Any = None
    event_store: EventStore | None = None
    event_session_id: str = ""
    events_db_path: Any = None  # Path; the TUI opens its worker-thread store here
    # Deferred factory: tests patch main.Conversation before main() runs.
    conv: Conversation = field(default_factory=lambda: Conversation())
    engine: AgentEngine | None = None
    mcp_manager: mcp.McpManager | None = None
    permission_labels: tuple[str, ...] = ()
    first_message: bool = True
    interrupted: bool = False


def _want_tui() -> bool:
    """Opt-in persistent TUI: explicit flag, real TTYs, prompt_toolkit.

    Anything missing degrades to the line-oriented REPL, which doubles as
    the documented minimal diagnostic client (roadmap rollback boundary).
    """
    if not TUI_ENABLED:
        return False
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def _input_confirm(prompt: str) -> bool:
    """One-shot y/N confirmation via stdin; anything but "y" denies."""
    try:
        return input(prompt).strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def _mirror_round(
    engine_round: EngineRound, conv: Conversation, write: Callable[[str], None]
) -> None:
    """Mirror one admitted round into the transcript and conversation."""
    native_calls = cast(list[dict], list(engine_round.native_calls))
    conv.add_assistant_tool_call(native_calls)
    for result in engine_round.results:
        if result.success:
            result_text = result.output or ""
            write(f"  [{result.tool_name}] ok")
        else:
            error = cast(ToolError, result.error)  # not success implies error
            result_text = f"[{error.code}] {error.message}"
            write(f"  [{result.tool_name}] {error.code}")
        conv.add_native_tool_result(result.tool_name, result_text)


def _request_approval(
    plan: ActionPlan,
    console: "Console | None" = None,
    ui: "UIRenderer | None" = None,
) -> bool:
    """Render the action plan and ask for a one-shot interactive approval.

    Fail-closed: only an explicit "y" approves; empty input, EOF, interrupt,
    or anything else denies. (The TUI routes the same decision through its
    own approval bridge in tui.py.) With Rich available the plan renders as
    a colored approval panel; without Rich the plain prints below stay
    byte-identical to the historical output.
    """
    if _RICH:
        if console is None:
            console = Console()
        if ui is None:
            ui = UIRenderer()
        console.print(ui.render_approval_panel(plan))
    else:
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


def _emit(
    store: EventStore | None,
    session_id: str,
    events: list[NewEvent],
    write: Callable[[str], None] = print,
) -> None:
    """Append events; a store failure degrades to a warning, never a crash."""
    if store is None or not events:
        return
    try:
        if len(events) == 1:
            store.append(session_id, events[0])
        else:
            store.append_group(session_id, events)
    except (EventStoreError, OSError) as e:
        write(f"[events warning: {e}]")


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


def _report_prompt_switch(
    compiled: CompiledPrompt, previous: str | None, write: Callable[[str], None] = print
) -> None:
    """Print the snapshot attribution line and, on a switch, the audit line.

    The active snapshot digest is also recorded per turn in the event
    store (turn_completed, WU-06).
    """
    write(f"prompt snapshot: {compiled.digest[:12]}")
    if previous:
        write(f"prompt: active {previous[:12]} -> {compiled.digest[:12]}")


def _prompt_show(manager: PromptManager, write: Callable[[str], None] = print) -> None:
    lines = manager.active.compiled_text.splitlines()
    for line in lines[:200]:
        write(line)
    if len(lines) > 200:
        write(f"... [{len(lines) - 200} more lines truncated]")


def _prompt_diff(manager: PromptManager, prefix: str, write: Callable[[str], None]) -> None:
    active = manager.active
    if prefix:
        base = manager.store.load(prefix)
    else:
        snapshots = [s for s in manager.store.history() if s.digest != active.digest]
        if not snapshots:
            write("  No previous snapshot to diff against")
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
        write("  No differences")
        return
    for line in diff[:400]:
        write(line)
    if len(diff) > 400:
        write(f"... [{len(diff) - 400} more diff lines truncated]")


def _prompt_edit(manager: PromptManager, write: Callable[[str], None] = print) -> None:
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
        write(f"  Cannot launch editor {editor!r}: {exc}")
        return
    compiled, previous = manager.reload()
    _report_prompt_switch(compiled, previous, write)


def _handle_prompt_command(
    user_input: str, manager: PromptManager, write: Callable[[str], None] = print
) -> None:
    """Dispatch /prompt subcommands: inspect, validate, reload, roll back."""
    args = user_input[len("/prompt") :].strip().split()
    sub = args[0] if args else ""
    try:
        if sub == "show":
            _prompt_show(manager, write)
        elif sub == "sources":
            write(f"  active: {manager.active.digest[:12]} ({manager.active.created_utc})")
            for source in manager.active.sources:
                write(
                    f"  {source.layer:10} {source.origin}  "
                    f"sha256:{source.sha256[:12]}  {source.size} bytes"
                )
        elif sub == "history":
            snapshots = manager.store.history()
            if not snapshots:
                write("  No prompt snapshots")
            for snapshot in snapshots:
                marker = "*" if snapshot.digest == manager.active.digest else " "
                layers = ",".join(source.layer for source in snapshot.sources)
                write(f" {marker} {snapshot.digest[:12]}  {snapshot.created_utc}  {layers}")
        elif sub == "diff":
            _prompt_diff(manager, args[1] if len(args) > 1 else "", write)
        elif sub == "validate":
            issues = validate_prompt(manager.active, manager.registry)
            if issues:
                for issue in issues:
                    write(f"  issue: {issue}")
            else:
                write("  prompt OK: structure, digests, and tool docs verified")
        elif sub == "reload":
            compiled, previous = manager.reload()
            _report_prompt_switch(compiled, previous, write)
        elif sub == "edit":
            _prompt_edit(manager, write)
        elif sub == "rollback":
            if len(args) < 2:
                write("  Usage: /prompt rollback <digest-prefix>")
                return
            target, previous = manager.rollback(args[1])
            _report_prompt_switch(target, previous, write)
        else:
            write(
                "  Usage: /prompt show|sources|history|diff [digest]|validate|"
                "reload|edit|rollback <digest-prefix>"
            )
    except (PromptCompileError, PromptStoreError) as exc:
        # Fail visibly; the active snapshot stays untouched on any error.
        write(f"  prompt {sub or '?'} failed [{exc.code}]: {exc.message}")


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


def _show_context_diagnostics(
    store: EventStore, session_id: str, write: Callable[[str], None] = print
) -> None:
    """Print context_diagnostics for the /context command."""
    diag = context_diagnostics(store, session_id)
    kinds = diag["groups_by_kind"]
    write(
        f"  events: {diag['total_events']}  groups: {diag['groups_total']} "
        f"(messages {kinds['message']}, tool calls {kinds['tool_call']})"
    )
    write(
        f"  terminal: {diag['terminal_groups']}  pending: {diag['pending_groups']}  "
        f"~{diag['estimated_uncovered_tokens']} uncovered tokens"
    )
    snapshot = diag["active_snapshot"]
    if snapshot is None:
        write("  snapshot: none")
    else:
        write(
            f"  snapshot: seq {snapshot['covered_seq_start']}-"
            f"{snapshot['covered_seq_end']} ({snapshot['covered_group_count']} "
            f"groups, {snapshot['covered_group_percent']}% covered)"
        )
        write(
            f"  digests: source {snapshot['source_digest'][:12]}  "
            f"events {snapshot['covered_event_ids_digest'][:12]}"
        )
    for issue in diag["issues"]:
        write(f"  issue: {issue}")


def _restore_backup_dir(arg: str) -> str | None:
    """Explicit backup directory, or the newest one under the default root."""
    if arg:
        return os.path.expanduser(arg)
    root = ops.default_backup_root()
    backups = sorted(root.glob("shadow-code-backup-*")) if root.is_dir() else []
    return str(backups[-1]) if backups else None


def _handle_restore_command(
    user_input: str,
    rt: SessionRuntime,
    write: Callable[[str], None],
    confirm: Callable[[str], bool] | None,
) -> None:
    """Preview a database restore, then apply only on explicit confirmation.

    Same approval discipline as the tool pipeline: the dry-run plan is shown
    first and nothing is written until the user explicitly confirms. The
    apply pass re-verifies every backup file against its manifest digest
    before overwriting a single byte, so a tampered backup fails closed.
    """
    directory = _restore_backup_dir(user_input[len("/restore") :].strip())
    if directory is None:
        write("  No backups found; run /backup first or pass /restore <dir>")
        return
    sentinels = ops.collect_sentinels()
    sessions_path = rt.db_path
    events_path = str(rt.events_db_path) if rt.events_db_path else None
    try:
        plan = ops.restore_databases(
            directory, sessions_path=sessions_path, events_path=events_path
        )
    except ops.OpsError as exc:
        write(f"[restore failed ({exc.code}): {exc.message}]")
        return
    for line in ops.render_restore_plan(plan, sentinels).splitlines():
        write(line)
    if not any(action.would_change for action in plan.actions):
        write("  Nothing to restore; all databases already match the backup")
        return
    if confirm is None or not confirm("  Apply this restore? [y/N] "):
        write("  Restore NOT applied; no changes written")
        return
    try:
        applied = ops.restore_databases(
            directory, sessions_path=sessions_path, events_path=events_path, apply=True
        )
    except ops.OpsError as exc:
        write(f"[restore failed ({exc.code}): {exc.message}]")
        return
    for line in ops.render_restore_plan(applied, sentinels).splitlines():
        write(line)


# Slash dispatch outcomes; the frontend decides how to render each one.
_DISPATCH_EXIT = "exit"
_DISPATCH_CLEAR = "clear"
_DISPATCH_HANDLED = "handled"


def _dispatch_slash_command(
    user_input: str,
    rt: SessionRuntime,
    write: Callable[[str], None],
    confirm: Callable[[str], bool] | None = None,
) -> str | tuple[str, str]:
    """Handle a slash command; both frontends share this exact dispatch.

    Returns "exit" to leave the session, "clear" when the frontend should
    also clear its own screen, "handled" when the command was fully dealt
    with, or ("message", text) when the input falls through to the model
    (a skill command expands into its prompt template). ``confirm`` is the
    frontend's one-shot y/N seam used by /restore; without it, destructive
    confirmations fail closed.
    """
    if user_input == "/exit":
        return _DISPATCH_EXIT

    if user_input == "/clear":
        rt.conv.clear()
        rt.first_message = True
        if rt.console is not None and rt.ui is not None:
            rt.console.clear()
            rt.console.print(rt.ui.render_welcome())
        return _DISPATCH_CLEAR

    ctx = rt.ctx
    prompt_manager = rt.prompt_manager
    client = rt.client
    workspace_guard = rt.workspace_guard
    if ctx is None or prompt_manager is None or client is None or workspace_guard is None:
        raise RuntimeError("slash dispatch requires a fully bootstrapped runtime")
    conv = rt.conv
    console = rt.console
    ui = rt.ui
    db = rt.db
    event_store = rt.event_store

    if user_input == "/tokens":
        used = conv.total_prompt_tokens
        if console is not None and ui is not None:
            console.print(ui.render_context_status(used, CONTEXT_WINDOW))
        else:
            pct = (used / CONTEXT_WINDOW * 100) if CONTEXT_WINDOW else 0
            write(f"Context: {used} tokens ({pct:.0f}% of {CONTEXT_WINDOW})")
        return _DISPATCH_HANDLED

    if user_input == "/info":
        write(f"  Model:    {MODEL_NAME}")
        write(f"  CWD:      {ctx.cwd}")
        write(f"  Messages: {len(conv.get_messages())}")
        write(f"  Tokens:   {conv.total_prompt_tokens}")
        write(f"  Tools:    {', '.join(tool_reg._REGISTRY.keys())}")
        write(f"  Skills:   {len(list_skills())}")
        if rt.session_id:
            write(f"  Session:  #{rt.session_id}")
        return _DISPATCH_HANDLED

    if user_input.startswith("/cd"):
        target = user_input[3:].strip()
        if not target:
            write(f"  CWD: {ctx.cwd}")
        else:
            new = os.path.normpath(os.path.join(ctx.cwd, os.path.expanduser(target)))
            if os.path.isdir(new):
                ctx.cwd = new
                write(f"  CWD: {ctx.cwd}")
            else:
                write(f"  Not a directory: {new}")
        return _DISPATCH_HANDLED

    if user_input == "/version":
        from . import __version__

        write(f"  shadow-code v{__version__}")
        write(f"  Model: {MODEL_NAME}")
        write(f"  Context: {CONTEXT_WINDOW // 1024}K")
        return _DISPATCH_HANDLED

    if user_input == "/history":
        msgs = conv.get_messages()
        if not msgs:
            write("  No messages yet")
        else:
            for i, m in enumerate(msgs[-10:], max(1, len(msgs) - 9)):
                role = m["role"]
                preview = m["content"][:80].replace("\n", " ")
                write(f"  {i:3}. [{role:9}] {preview}...")
        return _DISPATCH_HANDLED

    if user_input == "/compact":
        if event_store is not None and rt.event_session_id:
            # WU-08: snapshot complete causal groups via the event store;
            # the legacy message-level path remains for a degraded store.
            def summarize(messages: list[dict]) -> str:
                return _summarize_with_model(client, messages, rt.system_prompt)

            try:
                line = _compact_from_events(
                    event_store,
                    rt.event_session_id,
                    conv,
                    workspace_guard,
                    summarize,
                )
            except CompactionError as e:
                if e.code == "nothing_to_compact":
                    write(f"  {e.message}")
                else:
                    write(f"[Compaction failed ({e.code}): {e.message}]")
            except Exception as e:
                write(f"[Compaction failed: {e}]")
            else:
                write(line)
        elif conv.total_prompt_tokens > 0:
            write("[Compacting conversation...]")
            try:
                from .compaction import compact

                summary = compact(client, conv.get_messages(), rt.system_prompt)
                conv.apply_compaction_summary(summary)
                write("[Compaction complete]")
            except Exception as e:
                write(f"[Compaction failed: {e}]")
        else:
            write("  Nothing to compact")
        return _DISPATCH_HANDLED

    if user_input == "/context":
        if event_store is not None and rt.event_session_id:
            _show_context_diagnostics(event_store, rt.event_session_id, write)
        else:
            write("  [event store not available]")
        return _DISPATCH_HANDLED

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
            ("/doctor", "Diagnose config, model, workspace, and stores"),
            ("/backup", "Back up the local databases with a manifest"),
            ("/restore [dir]", "Preview a database restore, then confirm to apply"),
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
        if console is not None and ui is not None:
            console.print(ui.render_help(cmds))
        else:
            for cmd, desc in cmds:
                write(f"  {cmd:20} {desc}")
        return _DISPATCH_HANDLED

    if user_input == "/skills":
        write("  Available skills:")
        for skill_name, skill_desc in list_skills():
            write(f"    /{skill_name:15} {skill_desc}")
        return _DISPATCH_HANDLED

    if user_input.startswith("/save"):
        if db:
            name = user_input[5:].strip() or f"Session #{rt.session_id}"
            db.rename_session(rt.session_id, name)
            write(f"  Session saved as '{name}'")
        else:
            write("  [DB not available]")
        return _DISPATCH_HANDLED

    if user_input.startswith("/load"):
        if db:
            arg = user_input[5:].strip()
            if arg:
                try:
                    sid = int(arg)
                    s = db.get_session(sid)
                    if s:
                        conv.clear()
                        rt.first_message = True
                        for m in s["messages"]:
                            if m["role"] == "user":
                                conv.add_user(m["content"])
                                rt.first_message = False
                            elif m["role"] == "assistant":
                                conv.add_assistant(m["content"])
                        rt.session_id = sid
                        write(f"  Loaded session #{sid} ({len(s['messages'])} messages)")
                    else:
                        write(f"  Session #{sid} not found")
                except ValueError:
                    write("  Usage: /load <id>")
            else:
                write("  Usage: /load <id>")
        else:
            write("  [DB not available]")
        return _DISPATCH_HANDLED

    if user_input == "/list":
        if db:
            sessions = db.list_sessions()
            if sessions:
                for s in sessions:
                    name = s.get("name", "") or f"Session #{s['id']}"
                    msgs = s.get("message_count", 0)
                    write(f"  #{s['id']:4}  {name:30} ({msgs} msgs)")
            else:
                write("  No saved sessions")
        else:
            write("  [DB not available]")
        return _DISPATCH_HANDLED

    if user_input == "/events":
        if event_store is not None:
            issues = event_store.verify(rt.event_session_id)
            count = len(event_store.events_for(rt.event_session_id))
            if issues:
                write(f"  event store: {len(issues)} issue(s) across {count} events")
                for issue in issues:
                    write(f"  issue: {issue}")
            else:
                write(f"  event store OK: {count} events, integrity verified")
        else:
            write("  [event store not available]")
        return _DISPATCH_HANDLED

    if user_input == "/doctor":
        # WU-12: redacted diagnostic report; a broken store or unreachable
        # Ollama shows up as report content, never as a session failure.
        try:
            report = ops.doctor(_doctor_facts(rt))
        except Exception as exc:
            write(f"[doctor failed: {exc}]")
            return _DISPATCH_HANDLED
        for line in ops.render_doctor(report).splitlines():
            write(line)
        return _DISPATCH_HANDLED

    if user_input == "/backup":
        sentinels = ops.collect_sentinels()
        try:
            receipt = ops.backup_databases(
                sessions_path=rt.db_path,
                events_path=str(rt.events_db_path) if rt.events_db_path else None,
                dest_root=ops.default_backup_root(),
                prompt_store_dir=prompt_manager.store.root,
                sentinels=sentinels,
            )
        except ops.OpsError as exc:
            write(f"[backup failed ({exc.code}): {exc.message}]")
            return _DISPATCH_HANDLED
        for line in ops.render_backup_receipt(receipt, sentinels).splitlines():
            write(line)
        return _DISPATCH_HANDLED

    if user_input.startswith("/restore"):
        _handle_restore_command(user_input, rt, write, confirm)
        return _DISPATCH_HANDLED

    if user_input.startswith("/prompt"):
        _handle_prompt_command(user_input, prompt_manager, write)
        return _DISPATCH_HANDLED

    # Skill commands (e.g., /commit, /review file.py) expand into prompts.
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
            return ("message", skill_msg)
        write(f"  Unknown command: /{skill_name}. Type /help")
        return _DISPATCH_HANDLED

    return ("message", user_input)


def _handle_user_message(
    rt: SessionRuntime,
    user_input: str,
    stream_round: Callable[[], ProviderRound],
    write: Callable[[str], None],
    confirm: Callable[[str], bool],
) -> None:
    """Drive one user message to a completed turn; identical per frontend.

    Only the injected seams differ: the stream display, the destructive-
    command confirmation, and the output writer.
    """
    prompt_manager = rt.prompt_manager
    ctx = rt.ctx
    client = rt.client
    engine = rt.engine
    if prompt_manager is None or ctx is None or client is None or engine is None:
        raise RuntimeError("user messages require a fully bootstrapped runtime")
    conv = rt.conv
    console = rt.console
    ui = rt.ui
    db = rt.db
    event_store = rt.event_store

    # === Per-turn prompt source watch ===
    # An overlay edit affects only the next turn; on any failure the
    # previous active snapshot stays in place and a warning is printed.
    try:
        watched = prompt_manager.watch()
    except (PromptCompileError, PromptStoreError) as e:
        write(f"[prompt warning: keeping {prompt_manager.active.digest[:12]}: {e.message}]")
    else:
        if watched is not None:
            compiled, previous = watched
            rt.system_prompt = compiled.compiled_text
            _report_prompt_switch(compiled, previous, write)

    # === Inject environment on first message ===
    if rt.first_message:
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
        rt.first_message = False
    else:
        conv.add_user(user_input)
        user_content = user_input

    _emit(
        event_store,
        rt.event_session_id,
        [NewEvent("user_message", UserMessagePayload(content=user_content))],
        write,
    )

    # Save to DB
    if db:
        db.add_message(rt.session_id, "user", user_input)

    # === Tool execution loop (bounded engine drives native rounds) ===
    turns = 0
    errors = 0

    while turns < MAX_TOOL_TURNS:
        rt.interrupted = False
        outcome = engine.run_turn(stream_round)
        turns += outcome.steps
        if rt.state is not None:
            rt.state.turn = turns

        if outcome.status is EngineState.CANCELLED:
            write("[Interrupted]")
            break
        if outcome.status is EngineState.FAILED:
            if console is not None and ui is not None:
                console.print(ui.render_error(outcome.detail))
            else:
                write(f"\n[Error: {outcome.detail}]")
            break
        if outcome.status is EngineState.BUDGET_EXHAUSTED:
            if outcome.reason == "budget_steps":
                write(
                    f"[Budget exhausted (budget_steps): "
                    f"native tool limit ({MAX_NATIVE_TOOL_TURNS}) reached]"
                )
            else:
                write(f"[Budget exhausted ({outcome.reason})]")
            break

        # No native tool calls -- check for text response
        resp = outcome.text or ""
        if resp and resp.strip():
            conv.add_assistant(resp)
            _emit(
                event_store,
                rt.event_session_id,
                [NewEvent("assistant_text", AssistantTextPayload(content=resp))],
                write,
            )
            if db:
                db.add_message(rt.session_id, "assistant", resp)
                db.update_session_tokens(rt.session_id, conv.total_prompt_tokens)

            # Explicitly opt-in compatibility path for old Markdown tool calls.
            markdown_calls = _get_legacy_markdown_tool_calls(resp)
            if not markdown_calls:
                protocol_error = _legacy_markdown_protocol_error(resp)
                if protocol_error:
                    if console is not None and ui is not None:
                        console.print(ui.render_error(protocol_error.message))
                    else:
                        write(f"[{protocol_error.code}] {protocol_error.message}")
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
                            write(f"  {warning}")
                            if not confirm("  Proceed? (y/n): "):
                                r = tool_reg.ToolResult(False, "Command cancelled")
                                results.append(tool_reg.format_result(tc.tool, r))
                                continue
                    if console is not None and ui is not None:
                        console.print(ui.render_tool_call(tc.tool, desc))
                    r = tool_reg.dispatch(tc.tool, tc.params)
                    if console is not None and ui is not None:
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
        write(f"[Tool limit ({MAX_TOOL_TURNS}) reached]")

    # Record the active prompt snapshot digest for this completed turn.
    _emit(
        event_store,
        rt.event_session_id,
        [
            NewEvent(
                "turn_completed",
                TurnCompletedPayload(prompt_digest=prompt_manager.active.digest),
            )
        ],
        write,
    )

    # === Context status (always visible) ===
    conv.update_tokens(client.last_prompt_tokens)
    if conv.total_prompt_tokens > 0:
        _show_context_status(
            conv.total_prompt_tokens,
            CONTEXT_WINDOW,
            client.last_eval_tokens,
            console,
            ui,
            write,
        )

    # === 3-Tier Context Management ===
    if conv.needs_result_clearing():
        conv.clear_old_tool_results()

    if conv.needs_compaction():
        write("[Compacting conversation...]")
        try:
            from .compaction import compact

            summary = compact(client, conv.get_messages(), rt.system_prompt)
            conv.apply_compaction_summary(summary)
            write("[Compaction complete]")
        except Exception as e:
            write(f"[Compaction failed: {e}]")

        if conv.needs_emergency_truncate():
            conv.emergency_truncate()
            write("[Emergency truncation applied]")


def _run_line_loop(rt: SessionRuntime) -> None:
    """Default line-oriented frontend: prompt input + direct terminal output.

    Also the documented minimal diagnostic client (roadmap rollback
    boundary): no TUI chrome, the same engine and event machinery.
    """
    client = rt.client
    conv = rt.conv
    if (
        client is None
        or rt.registry is None
        or rt.policy_engine is None
        or rt.execution_context is None
        or rt.approval_authority is None
    ):
        raise RuntimeError("line loop requires a fully bootstrapped runtime")

    def on_sigint(signum, frame):
        rt.interrupted = True

    signal.signal(signal.SIGINT, on_sigint)

    def handle_round(engine_round: EngineRound) -> None:
        """Mirror one admitted round into the transcript and conversation."""
        _mirror_round(engine_round, conv, print)

    # Bounded engine (WU-07): drives the native tool rounds of each user
    # turn. All I/O seams are injected -- consent is the interactive
    # approval prompt, cancellation reads the SIGINT flag. The lambda keeps
    # a late-bound reference so tests can patch _request_approval.
    rt.engine = AgentEngine(
        rt.registry,
        rt.policy_engine,
        rt.execution_context,
        rt.approval_authority,
        consent=lambda plan: _request_approval(plan, rt.console, rt.ui),
        event_store=rt.event_store,
        event_session_id=rt.event_session_id,
        cancel_requested=lambda: rt.interrupted,
        on_round=handle_round,
        on_store_warning=lambda message: print(f"[events warning: {message}]"),
    )

    def stream_round() -> ProviderRound:
        """One provider round: stream + display, return text and calls."""
        if rt.console is not None and rt.stream_ctrl is not None:
            try:
                resp, eval_tokens = rt.stream_ctrl.stream_response(
                    conv.get_messages(), rt.system_prompt, tools=rt.tool_schemas
                )
            except StreamCancelled:
                raise StreamCancelledError from None
            except Exception as error:
                raise StreamError("provider_error", str(error)) from error
        else:
            rt.display.reset()
            cancelled = False
            try:
                for chunk in client.chat_stream(
                    conv.get_messages(), rt.system_prompt, tools=rt.tool_schemas
                ):
                    if rt.interrupted:
                        cancelled = True
                        break
                    rt.display.feed(chunk)
            except KeyboardInterrupt:
                cancelled = True
            except Exception as error:
                raise StreamError("provider_error", str(error)) from error
            rt.display.flush()
            print()
            if cancelled:
                raise StreamCancelledError
            resp = rt.display.get_full_response()
        conv.update_tokens(client.last_prompt_tokens)
        if rt.state is not None:
            rt.state.tokens_used = conv.total_prompt_tokens
        return ProviderRound(
            text=resp or "",
            native_calls=tuple(getattr(client, "last_tool_calls", [])),
        )

    while True:
        if rt.prompt_session is not None:
            user_input = get_input(rt.prompt_session, MODEL_NAME)
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

        action = _dispatch_slash_command(user_input, rt, print, _input_confirm)
        if action == _DISPATCH_EXIT:
            break
        if action == _DISPATCH_CLEAR:
            print("[Cleared]")
            continue
        if isinstance(action, tuple):
            _handle_user_message(rt, action[1], stream_round, print, _input_confirm)


def main(tui_input: Any = None, tui_output: Any = None) -> None:
    cwd = os.getcwd()
    ctx = ToolContext(cwd)
    use_tui = _want_tui()
    boot_lines: list[str] = []
    write: Callable[[str], None] = boot_lines.append if use_tui else print

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
    builtin_specs = (READ_FILE_SPEC, WRITE_FILE_SPEC, EDIT_FILE_SPEC, BASH_SPEC)
    runtime_registry = ToolRegistry(builtin_specs)
    # MCP adapter (WU-13): discover configured stdio servers and register
    # their sanitized tools as namespaced specs in the SAME registry, so
    # policy, approval, events, and budgets apply identically. Discovery
    # never blocks startup: a failing server is reported unavailable.
    mcp_manager = mcp.McpManager.discover(
        mcp.default_mcp_config_path(),
        reserved_names=frozenset(spec.name for spec in builtin_specs),
    )
    if mcp_manager.specs:
        runtime_registry = ToolRegistry((*runtime_registry.specs, *mcp_manager.specs))
    mcp_active = bool(mcp_manager.ready_servers)
    for line in mcp_manager.report_lines():
        write(f"[mcp] {line}")
    process_env = build_process_env()
    sandbox_label = detect_sandbox()
    capabilities = _granted_capabilities(BASH_STRICT, sandbox_label, mcp_active=mcp_active)
    policy_engine = PolicyEngine(PolicyFacts(capabilities, workspace_guard.identity))
    execution_context = WorkspaceContext(
        guard=workspace_guard,
        workspace_root=cwd,
        process_env=process_env,
        sandbox_label=sandbox_label,
        mutation_mode="export" if MUTATION_STRICT else "apply",
    )
    if Capability.PROCESS_EXECUTE not in capabilities:
        write("[bash disabled: strict mode and no sandbox (bwrap/firejail) available]")
    else:
        write("[bash runs UNCONFINED; every execution requires explicit approval]")
    if MUTATION_STRICT:
        write(
            "[file mutations run in strict mode: changes are exported as "
            "reviewed patches, never applied]"
        )
    permission_labels: list[str] = []
    if Capability.PROCESS_EXECUTE not in capabilities:
        permission_labels.append("bash:off")
    elif sandbox_label == "unconfined":
        permission_labels.append("bash:UNCONFINED")
    else:
        permission_labels.append(f"bash:{sandbox_label}")
    if MUTATION_STRICT:
        permission_labels.append("mutation:export")
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
    _report_prompt_switch(prompt_manager.active, previous_prompt, write)

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

    # Setup UI. TUI mode owns the terminal itself: no Rich console is ever
    # constructed, so Rich performs zero direct active-terminal writes.
    if use_tui:
        console = None
        ui = None
        display = None
        stream_ctrl = None
        write(f"shadow-code v0.1.0 | {MODEL_NAME}")
        write(f"CWD: {cwd}")
    elif _RICH:
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
    state = SessionState(
        model_name=MODEL_NAME,
        tokens_total=CONTEXT_WINDOW,
        max_turns=MAX_TOOL_TURNS,
    )

    # Setup REPL (pass state for bottom toolbar); the TUI owns its own input.
    prompt_session = None if use_tui else (create_prompt_session(state) if _HAS_REPL else None)

    # Setup DB
    db = None
    db_path = None
    session_id = None
    if _HAS_DB:
        try:
            db_path = default_db_path()
            db = Database(db_path)
            session_id = db.create_session(MODEL_NAME)
        except Exception as e:
            write(f"[DB warning: {e}]")
            db = None
            db_path = None

    # Event store (WU-06): append-only causal authority for resume and
    # audit. Any failure downgrades to a warning -- the store must never
    # break the CLI. A previous session's unfinished calls are reported and
    # abandoned only with explicit acknowledgment; nothing is re-executed.
    event_store = None
    event_session_id = ""
    events_db_path = default_events_db_path()
    try:
        event_store = EventStore(events_db_path)
        if not _resolve_pending_events(event_store):
            event_store.close()
            workspace_guard.close()
            mcp_manager.close()
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
            write,
        )
        # Record the MCP discovery outcome for this session (WU-13).
        if mcp_manager.statuses:
            _emit(
                event_store,
                event_session_id,
                [
                    NewEvent(
                        "mcp_server",
                        McpServerPayload(
                            server=status.name, status=status.state, detail=status.detail
                        ),
                    )
                    for status in mcp_manager.statuses
                ],
                write,
            )
    except (EventStoreError, OSError) as e:
        write(f"[events warning: {e}]")
        event_store = None

    # Startup authority block (WU-12): name every active authority boundary
    # and every unavailable capability before the first prompt, so the owner
    # always sees exactly what this session may and may not do.
    granted, withheld = _authority_facts(sandbox_label, mcp_active=mcp_active)
    for line in ops.authority_summary(
        workspace_root=cwd,
        device=workspace_guard.identity.device,
        inode=workspace_guard.identity.inode,
        containment="openat2",
        granted=granted,
        withheld=withheld,
        sandbox_label=sandbox_label,
        mutation_mode=execution_context.mutation_mode,
        prompt_digest=prompt_manager.active.digest,
        sentinels=ops.collect_sentinels(),
    ):
        write(line)

    write(
        "Commands: /help /clear /exit /tokens /save /load /list /info /prompt "
        "/events /context /doctor /backup /restore\n"
    )

    rt = SessionRuntime(
        cwd=cwd,
        ctx=ctx,
        workspace_guard=workspace_guard,
        registry=runtime_registry,
        policy_engine=policy_engine,
        execution_context=execution_context,
        approval_authority=approval_authority,
        tool_schemas=tool_schemas,
        prompt_manager=prompt_manager,
        system_prompt=system_prompt,
        client=client,
        console=console,
        ui=ui,
        display=display,
        stream_ctrl=stream_ctrl,
        state=state,
        prompt_session=prompt_session,
        db=db,
        db_path=db_path,
        session_id=session_id,
        event_store=event_store,
        event_session_id=event_session_id,
        events_db_path=events_db_path,
        mcp_manager=mcp_manager,
        permission_labels=tuple(permission_labels),
    )

    if use_tui:
        from . import tui as tui_module

        try:
            tui_module.run_tui(rt, boot_lines, input=tui_input, output=tui_output)
        except Exception as error:
            if not sys.stdin.isatty():
                raise
            for line in boot_lines:
                print(line)
            print(f"[TUI unavailable ({error}); continuing in line mode]")
            _run_line_loop(rt)
    else:
        _run_line_loop(rt)

    # Cleanup (shared by both frontends). MCP cleanup comes first: every
    # owned client/process group is killed and recorded before the session
    # closes, so the event log shows exactly which servers were terminated.
    closed_servers = mcp_manager.close()
    if closed_servers:
        _emit(
            event_store,
            event_session_id,
            [
                NewEvent(
                    "mcp_server",
                    McpServerPayload(
                        server=name, status="closed", detail="process group terminated"
                    ),
                )
                for name in closed_servers
            ],
        )
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


def _show_context_status(
    used: int,
    total: int,
    last_eval: int,
    console=None,
    ui=None,
    write: Callable[[str], None] = print,
):
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
        write(f"  {color}[{bar}] {used // 1000}K/{total // 1000}K{reset} {dim}({pct:.0f}%){reset}")


if __name__ == "__main__":
    main()
