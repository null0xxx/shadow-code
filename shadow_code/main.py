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

import os
import platform
import signal
import sys
from datetime import datetime

from . import tools as tool_reg
from .config import (
    CONTEXT_WINDOW,
    LEGACY_MARKDOWN_TOOLS,
    MAX_CONSECUTIVE_ERRORS,
    MAX_TOOL_TURNS,
    MODEL_NAME,
)
from .conversation import Conversation
from .display import StreamDisplay
from .ollama_client import OllamaClient
from .parser import LegacyMarkdownToolCall, parse_legacy_markdown_tool_calls
from .prompt import SYSTEM_PROMPT
from .safety import check_destructive
from .skills import get_skill, list_skills
from .tool_context import ToolContext

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


def main():
    cwd = os.getcwd()
    ctx = ToolContext(cwd)

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

    print("Commands: /help /clear /exit /tokens /save /load /list /info\n")

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

                    summary = compact(client, conv.get_messages(), SYSTEM_PROMPT)
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

        while turns < MAX_TOOL_TURNS:
            interrupted = False

            # Stream response
            if _RICH and stream_ctrl:
                try:
                    resp, eval_tokens = stream_ctrl.stream_response(
                        conv.get_messages(), SYSTEM_PROMPT
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
                    for chunk in client.chat_stream(conv.get_messages(), SYSTEM_PROMPT):
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

            # Check for native tool calls first
            native_calls = getattr(client, "last_tool_calls", [])

            if native_calls:
                # Native tool calling path (Gemma 4+)
                conv.add_assistant_tool_call(native_calls)

                if db:
                    db.add_message(session_id, "assistant", f"[tool calls: {len(native_calls)}]")

                for tc in native_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    params = func.get("arguments", {})
                    desc = _build_tool_desc(tool_name, params)

                    # Safety check
                    if tool_name == "bash":
                        warning = check_destructive(params.get("command", ""))
                        if warning:
                            if _RICH:
                                console.print(ui.render_error(warning))
                            else:
                                print(f"  {warning}")
                            try:
                                confirm = input("  Proceed? (y/n): ").strip().lower()
                            except (EOFError, KeyboardInterrupt):
                                confirm = "n"
                            if confirm != "y":
                                conv.add_native_tool_result(tool_name, "Command cancelled by user")
                                errors += 1
                                continue

                    if _RICH:
                        console.print(ui.render_tool_call(tool_name, desc))

                    r = tool_reg.dispatch(tool_name, params)

                    if _RICH:
                        if tool_name == "edit_file" and r.success:
                            console.print(
                                ui.render_diff(
                                    params.get("old_string", ""),
                                    params.get("new_string", ""),
                                    params.get("file_path", ""),
                                )
                            )
                        else:
                            console.print(
                                ui.render_tool_result(
                                    tool_name,
                                    r.output,
                                    r.success,
                                    params=params,
                                )
                            )
                    else:
                        status = "\u2713" if r.success else "\u2717"
                        print(f"  {status} [{tool_name}] {desc}")
                        for line in r.output.splitlines()[:20]:
                            print(f"    {line}")

                    conv.add_native_tool_result(tool_name, r.output)
                    errors = errors + 1 if not r.success else 0

                turns += 1
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"[{errors} consecutive errors]")
                    break
                continue  # Loop back for model's next response

            # No native tool calls -- check for text response
            if resp and resp.strip():
                conv.add_assistant(resp)
                if db:
                    db.add_message(session_id, "assistant", resp)
                    db.update_session_tokens(session_id, conv.total_prompt_tokens)

                # Explicitly opt-in compatibility path for old Markdown tool calls.
                markdown_calls = _get_legacy_markdown_tool_calls(resp)
                if not markdown_calls:
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

                summary = compact(client, conv.get_messages(), SYSTEM_PROMPT)
                conv.apply_compaction_summary(summary)
                print("[Compaction complete]")
            except Exception as e:
                print(f"[Compaction failed: {e}]")

            if conv.needs_emergency_truncate():
                conv.emergency_truncate()
                print("[Emergency truncation applied]")

    # Cleanup
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
