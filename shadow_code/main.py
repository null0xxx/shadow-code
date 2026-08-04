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
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
from datetime import datetime

from pydantic import ValidationError

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
from .conversation import Conversation
from .display import TAG_START, StreamDisplay
from .domain.approval import (
    ActionPlan,
    ApprovalAuthority,
    build_action_plan,
    render_action_preview,
)
from .domain.policy import PolicyDisposition, PolicyFacts, WorkspaceAccessError
from .domain.tools import Capability, ToolError, ToolResult, ValidatedToolCall
from .executor import execute_validated_call
from .mutation import MutationError, MutationPlan, build_edit_plan, build_write_plan
from .ollama_client import OllamaClient, render_ollama_tool_schemas
from .parser import LegacyMarkdownToolCall, parse_legacy_markdown_tool_calls
from .policy.engine import PolicyEngine
from .policy.workspace import WorkspaceGuard
from .process import build_process_env, classify_command, detect_sandbox, execution_facts
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
    EditFileArgs,
    WorkspaceContext,
    WriteFileArgs,
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


def _admit_native_calls(
    native_calls: list[dict],
    registry: ToolRegistry,
    policy_engine: PolicyEngine,
    execution_context: WorkspaceContext,
    approval_authority: ApprovalAuthority,
) -> list[ToolResult]:
    """Run collected native calls through validate -> policy -> execute.

    Fail-closed at every step: invalid envelopes, unregistered tools, and
    policy denials produce typed error results and never reach a handler.
    Approval-required calls ask interactively for a one-shot, digest-bound
    token; denial or cancellation is final and the call is not retried.
    """
    results = []
    for raw_call in native_calls:
        validated = registry.validate_call(raw_call)
        if isinstance(validated, ToolError):
            fallback_id = raw_call.get("call_id") if isinstance(raw_call, dict) else None
            fallback_name = raw_call.get("name") if isinstance(raw_call, dict) else None
            results.append(
                ToolResult(
                    call_id=fallback_id or "invalid-call",
                    tool_name=fallback_name or "unknown",
                    error=validated,
                )
            )
            continue

        decision = policy_engine.decide(validated)
        if decision.disposition is PolicyDisposition.ALLOW:
            results.append(execute_validated_call(validated, execution_context))
        elif decision.disposition is PolicyDisposition.DENY:
            results.append(
                ToolResult(
                    call_id=validated.call.call_id,
                    tool_name=validated.call.name,
                    error=ToolError(
                        code="policy_denied",
                        message=f"Policy denied execution ({decision.reason.value}).",
                    ),
                )
            )
        else:
            preview = render_action_preview(validated)
            facts = ""
            if validated.spec.capability is Capability.PROCESS_EXECUTE:
                facts = execution_facts(
                    execution_context.process_env,
                    execution_context.workspace_root,
                    execution_context.sandbox_label,
                )
                command = str(validated.arguments.get("command", ""))
                features = sorted(classify_command(command))
                sandbox_label = execution_context.sandbox_label
                if sandbox_label == "unconfined":
                    sandbox_line = "sandbox: unconfined (no sandbox helper available)"
                else:
                    sandbox_line = (
                        f"sandbox: unconfined ({sandbox_label} available but not applied)"
                    )
                preview += (
                    f"\n{sandbox_line}"
                    f" (no confinement; approval is the only control)"
                    f"\nfeatures: {', '.join(features) if features else 'none detected'}"
                )
            elif validated.spec.capability is Capability.FILESYSTEM_WRITE:
                mutation_plan = _mutation_plan_for_call(validated, execution_context)
                if isinstance(mutation_plan, ToolError):
                    # Plan-build failures (no_match, ambiguous_match, ...)
                    # surface as typed results, never as approval prompts.
                    results.append(
                        ToolResult(
                            call_id=validated.call.call_id,
                            tool_name=validated.call.name,
                            error=mutation_plan,
                        )
                    )
                    continue
                strict_note = (
                    " [strict: patch export]" if execution_context.mutation_mode == "export" else ""
                )
                preview += (
                    f"\nmutation: {mutation_plan.operation} "
                    f"{mutation_plan.relative_path}{strict_note}\n{mutation_plan.preview}"
                )
            plan = build_action_plan(
                validated,
                registry_digest=registry.digest,
                workspace=execution_context.guard.identity,
                preview=preview,
                execution_facts=facts,
            )
            if not _request_approval(plan):
                results.append(
                    ToolResult(
                        call_id=validated.call.call_id,
                        tool_name=validated.call.name,
                        error=ToolError(
                            code="approval_denied",
                            message=(
                                "User denied or cancelled the approval; the call was not executed."
                            ),
                        ),
                    )
                )
                continue
            token = approval_authority.issue(plan)
            results.append(
                execute_validated_call(
                    validated,
                    execution_context,
                    approval=token,
                    authority=approval_authority,
                    plan=plan,
                )
            )
    return results


def _report_prompt_switch(compiled: CompiledPrompt, previous: str | None) -> None:
    """Print the snapshot attribution line and, on a switch, the audit line.

    WU-04: the audit trail is this printed line plus the snapshot history;
    the durable event store arrives with WU-06.
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

    print("Commands: /help /clear /exit /tokens /save /load /list /info /prompt\n")

    conv = Conversation()
    interrupted = False
    first_message = True

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, on_sigint)

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
            if conv.total_prompt_tokens > 0:
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

        if user_input == "/help":
            cmds = [
                ("/help", "Show this help"),
                ("/clear", "Clear conversation"),
                ("/exit", "Exit shadow-code"),
                ("/tokens", "Show context usage"),
                ("/info", "Show session info"),
                ("/cd [path]", "Show or change working directory"),
                ("/compact", "Manually compact conversation"),
                ("/history", "Show last 10 messages"),
                ("/version", "Show version info"),
                ("/save [name]", "Save session"),
                ("/load [id]", "Load session"),
                ("/list", "List saved sessions"),
                ("/skills", "List available skills"),
                ("/prompt <sub>", "Inspect/reload/rollback the system prompt"),
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
            first_message = False
        else:
            conv.add_user(user_input)

        # Save to DB
        if db:
            db.add_message(session_id, "user", user_input)

        # === Tool execution loop (native tool calling) ===
        turns = 0
        errors = 0
        native_rounds = 0

        while turns < MAX_TOOL_TURNS:
            interrupted = False

            # Stream response
            if _RICH and stream_ctrl:
                try:
                    resp, eval_tokens = stream_ctrl.stream_response(
                        conv.get_messages(), system_prompt, tools=tool_schemas
                    )
                except StreamCancelled:
                    print("[Interrupted]")
                    break
                except Exception as e:
                    if _RICH:
                        console.print(ui.render_error(str(e)))
                    else:
                        print(f"[Error: {e}]")
                    break
            else:
                display.reset()
                try:
                    for chunk in client.chat_stream(
                        conv.get_messages(), system_prompt, tools=tool_schemas
                    ):
                        if interrupted:
                            break
                        display.feed(chunk)
                except KeyboardInterrupt:
                    interrupted = True
                except Exception as e:
                    print(f"\n[Error: {e}]")
                    break
                display.flush()
                print()
                resp = display.get_full_response()

                if interrupted:
                    print("[Interrupted]")
                    break

            conv.update_tokens(client.last_prompt_tokens)
            state.tokens_used = conv.total_prompt_tokens
            state.turn = turns

            # Native tool calls: admission pipeline (validate -> policy -> execute)
            native_calls = list(getattr(client, "last_tool_calls", []))
            if native_calls:
                conv.add_assistant_tool_call(native_calls)
                results = _admit_native_calls(
                    native_calls,
                    runtime_registry,
                    policy_engine,
                    execution_context,
                    approval_authority,
                )
                for result in results:
                    if result.success:
                        text = result.output or ""
                        print(f"  [{result.tool_name}] ok")
                    else:
                        text = f"[{result.error.code}] {result.error.message}"
                        print(f"  [{result.tool_name}] {result.error.code}")
                    conv.add_native_tool_result(result.tool_name, text)
                turns += 1
                native_rounds += 1
                if native_rounds >= MAX_NATIVE_TOOL_TURNS:
                    print(f"[Native tool limit ({MAX_NATIVE_TOOL_TURNS}) reached]")
                    break
                continue

            # No native tool calls -- check for text response
            if resp and resp.strip():
                conv.add_assistant(resp)
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
