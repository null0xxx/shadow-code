# shadow_code/tui.py -- Persistent terminal shell (WU-09 + WU-10)
#
# One prompt_toolkit Application owns the whole session: a read-only
# transcript (user messages, streamed assistant text, tool lifecycle,
# slash-command output), a multiline input area with history and slash
# completion, and a footer with model/tokens/workspace/snapshot/permission
# state. The engine turn runs on a worker thread; every transcript
# mutation is posted back to the app thread, so input stays responsive
# and prompt_toolkit owns every redraw -- nothing tears, nothing writes
# to the active terminal behind its back.
#
# WU-10 adds the tool lifecycle layer: calls render grouped per engine
# round ("step n") with status tokens that update IN PLACE on transitions,
# long results collapse with an explicit marker and expand via Ctrl+E, and
# approvals run through a focused, bounded panel above the input area --
# one fresh single-focus control per side-effecting call, showing every
# action-plan fact and digest before the y/n/Esc decision. Transcript rows
# are immutable text EXCEPT the tool-group entries, which re-render from
# the live ToolLifecycleModel on every refresh (the simplest correct
# in-place update: the group object is shared, its rows are replaced).
#
# View-model first: TranscriptModel, ToolLifecycleModel (tui_tools.py),
# FooterModel, sanitize_terminal_text, and the renderers are pure and
# headless-testable. ANSI control sequences from model or tool output are
# neutralized before they can reach the terminal. NO_COLOR disables all
# styling; ASCII mode replaces box drawing while preserving every semantic
# label as text.
#
# This frontend is strictly opt-in (SHADOW_TUI=1 + TTYs); the line-oriented
# REPL in main.py stays the default and the minimal diagnostic client.

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import CallEvent, ProviderRound, StreamCancelledError, StreamError
from .events import EventStore
from .main import (
    _DISPATCH_CLEAR,
    _DISPATCH_EXIT,
    SessionRuntime,
    _dispatch_slash_command,
    _handle_user_message,
    _mirror_round,
)
from .repl import _SLASH_COMMANDS
from .tui_tools import (
    APPROVAL_PANEL_MAX_HEIGHT,
    THEME_TOKENS,
    ToolGroup,
    ToolLifecycleModel,
    approval_panel_fragments,
    render_markdown_lite,
    render_tool_group,
    render_tool_group_fragments,
    sanitize_terminal_text,
)

__all__ = [
    "FooterModel",
    "TranscriptEntry",
    "TranscriptModel",
    "TuiApp",
    "TuiTheme",
    "render_footer",
    "run_tui",
    "sanitize_terminal_text",
    "width_bucket",
]

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.styles import Style as PTStyle
    from prompt_toolkit.widgets import Frame, TextArea

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False


# -- transcript view model --------------------------------------------------

_KIND_USER = "user"
_KIND_ASSISTANT = "assistant"
_KIND_TOOL = "tool"
_KIND_TOOL_GROUP = "tool_group"
_KIND_SYSTEM = "system"


@dataclass(frozen=True)
class TranscriptEntry:
    """One logical transcript row; text may span multiple physical lines.

    Tool-group entries carry the live group instead of text: the group's
    frozen rows are replaced on status transitions and re-rendered on the
    next refresh, which is what makes in-place lifecycle updates work.
    """

    kind: str
    text: str
    group: ToolGroup | None = None


def _prefix(kind: str, theme: TuiTheme) -> str:
    if kind == _KIND_USER:
        return "> " if theme.ascii_mode else "› "
    if kind == _KIND_TOOL:
        return "  "
    return ""


class TranscriptModel:
    """Append-only conversation projection; the only transcript truth.

    Assistant text streams in as deltas that accumulate into the trailing
    assistant entry (the live response area); tool lifecycle renders as
    grouped entries projected from the ToolLifecycleModel; system lines are
    standalone entries. All text is sanitized on the way in.
    """

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []
        self.tools = ToolLifecycleModel()

    @property
    def entries(self) -> tuple[TranscriptEntry, ...]:
        return tuple(self._entries)

    def append_user(self, text: str) -> None:
        self._entries.append(TranscriptEntry(_KIND_USER, sanitize_terminal_text(text)))

    def append_assistant_delta(self, delta: str) -> None:
        delta = sanitize_terminal_text(delta)
        if not delta:
            return
        if self._entries and self._entries[-1].kind == _KIND_ASSISTANT:
            last = self._entries[-1]
            self._entries[-1] = TranscriptEntry(_KIND_ASSISTANT, last.text + delta)
        else:
            self._entries.append(TranscriptEntry(_KIND_ASSISTANT, delta))

    def append_tool_line(self, tool_name: str, status: str) -> None:
        line = f"[{sanitize_terminal_text(tool_name)}] {sanitize_terminal_text(status)}"
        self._entries.append(TranscriptEntry(_KIND_TOOL, line))

    def append_system_line(self, text: str) -> None:
        self._entries.append(TranscriptEntry(_KIND_SYSTEM, sanitize_terminal_text(text)))

    def clear(self) -> None:
        self._entries.clear()
        self.tools = ToolLifecycleModel()

    # -- tool lifecycle glue (engine CallEvent -> groups) ---------------

    def tool_round(self, step: int) -> None:
        group, is_new = self.tools.begin_round(step)
        if is_new:
            self._entries.append(TranscriptEntry(_KIND_TOOL_GROUP, "", group))

    def tool_proposed(self, call_id: str, tool_name: str, arguments_json: str) -> None:
        groups_before = len(self.tools.groups)
        self.tools.note_proposed(call_id, tool_name, arguments_json)
        if len(self.tools.groups) > groups_before:
            self._entries.append(TranscriptEntry(_KIND_TOOL_GROUP, "", self.tools.groups[-1]))

    def tool_result(self, result: Any) -> None:
        error_code = result.error.code if result.error else ""
        error_message = result.error.message if result.error else ""
        groups_before = len(self.tools.groups)
        self.tools.note_result(
            result.call_id, result.tool_name, result.output, error_code, error_message
        )
        if len(self.tools.groups) > groups_before:
            self._entries.append(TranscriptEntry(_KIND_TOOL_GROUP, "", self.tools.groups[-1]))

    # -- rendering ---------------------------------------------------------

    def _render_entry(self, entry: TranscriptEntry, theme: TuiTheme, width: int) -> list[str]:
        if entry.group is not None:
            return render_tool_group(entry.group, theme, width)
        prefix = _prefix(entry.kind, theme)
        lines = entry.text.split("\n")
        return [
            (prefix if index == 0 else " " * len(prefix)) + line for index, line in enumerate(lines)
        ]

    def render(self, theme: TuiTheme, width: int = 80) -> str:
        """Project the entries into plain display text; no Rich objects."""
        lines: list[str] = []
        for entry in self._entries:
            lines.extend(self._render_entry(entry, theme, width))
        return "\n".join(lines)

    def render_fragments(self, theme: TuiTheme, width: int = 80) -> list[tuple[str, str]]:
        """Styled projection for the color transcript.

        Tool groups get status-token and diff styling; assistant text gets
        markdown-lite styling. Without colors this is the plain render as a
        single unstyled fragment.
        """
        if not theme.colors:
            return [("", self.render(theme, width))]
        fragments: list[tuple[str, str]] = []
        for index, entry in enumerate(self._entries):
            if index > 0:
                fragments.append(("", "\n"))
            if entry.group is not None:
                fragments.extend(render_tool_group_fragments(entry.group, theme, width))
                continue
            if entry.kind == _KIND_ASSISTANT:
                fragments.extend(render_markdown_lite(entry.text, theme))
                continue
            fragments.append(("", "\n".join(self._render_entry(entry, theme, width))))
        return fragments


# -- footer view model ------------------------------------------------------


@dataclass(frozen=True)
class FooterModel:
    """Frozen footer snapshot: session state visible at a glance."""

    model_name: str
    tokens_used: int
    tokens_total: int
    workspace_root: str
    snapshot_digest: str = ""
    permission_labels: tuple[str, ...] = ()
    state: str = "idle"  # idle | busy | approval

    @property
    def context_pct(self) -> float:
        if self.tokens_total <= 0:
            return 0.0
        return self.tokens_used / self.tokens_total * 100


def width_bucket(width: int) -> str:
    """Layout buckets: the footer condenses as the terminal narrows."""
    if width >= 120:
        return "full"
    if width >= 80:
        return "standard"
    if width >= 40:
        return "compact"
    return "tiny"


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3) :]


def render_footer(model: FooterModel, width: int, *, ascii_mode: bool = False) -> tuple[str, ...]:
    """Pure layout: one or two footer lines per width bucket.

    Every semantic label (model, tokens, context %, workspace, snapshot,
    permissions, key hints) survives as plain text in every bucket; narrow
    buckets only drop decoration and secondary fields.
    """
    sep = "|" if ascii_mode else "│"
    pct = model.context_pct
    tokens = f"{model.tokens_used // 1000}K/{model.tokens_total // 1000}K"
    perms = " ".join(model.permission_labels)
    digest = f"snap {model.snapshot_digest}" if model.snapshot_digest else ""
    bucket = width_bucket(width)
    if bucket == "full":
        head = [model.model_name, tokens, f"ctx {pct:.0f}%", model.workspace_root]
        if digest:
            head.append(digest)
        if perms:
            head.append(perms)
        hints = "Enter: send | Alt+Enter: newline | Ctrl+C: cancel | Ctrl+D: exit | /help"
        if not ascii_mode:
            hints = hints.replace("|", sep)
        return (f" {sep} ".join(head), f" {model.state} {sep} {hints}")
    if bucket == "standard":
        head = [model.model_name, tokens, f"ctx {pct:.0f}%"]
        if perms:
            head.append(perms)
        head.append(model.state)
        tail = [_shorten(model.workspace_root, 40)]
        if digest:
            tail.append(digest)
        return (f" {sep} ".join(head), f" {sep} ".join(tail))
    if bucket == "compact":
        return (f" {sep} ".join([model.model_name, f"ctx {pct:.0f}%", model.state]),)
    return (f"{model.model_name} {pct:.0f}%",)


# -- theme ------------------------------------------------------------------


@dataclass(frozen=True)
class TuiTheme:
    """Color/ASCII decisions; NO_COLOR and SHADOW_ASCII opt out."""

    colors: bool = True
    ascii_mode: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TuiTheme:
        values = os.environ if env is None else env
        no_color = "NO_COLOR" in values
        ascii_flag = values.get("SHADOW_ASCII", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(colors=not no_color, ascii_mode=ascii_flag or no_color)


# -- the application ----------------------------------------------------------


class TuiApp:
    """One prompt_toolkit Application owning the persistent session.

    Layout: transcript window (read-only, wraps, pinned to the bottom),
    approval panel (bounded, scrollable, visible only while the engine
    awaits consent; single focus -- the input area is unreachable until the
    decision lands), input area (multiline, history, slash completion;
    Enter sends, Alt+Enter inserts a newline), footer window (state
    snapshot per width bucket). Engine turns run via asyncio.to_thread;
    approvals bridge back through a threading.Event.
    """

    def __init__(
        self,
        rt: SessionRuntime,
        theme: TuiTheme | None = None,
        *,
        boot_lines: Iterable[str] = (),
        input: Any = None,
        output: Any = None,
    ) -> None:
        if not _HAS_PROMPT_TOOLKIT:
            raise RuntimeError("prompt_toolkit is required for the TUI")
        self.rt = rt
        self.theme = theme or TuiTheme.from_env()
        self.model = TranscriptModel()
        for line in boot_lines:
            self.model.append_system_line(line)
        self._inputs: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._driver_task: asyncio.Task[None] | None = None
        self._busy = False
        self._approval_pending = False
        self._approval_answer = False
        self._approval_event = threading.Event()
        self._approval_plan: Any = None
        self._approval_freeform: tuple[str, ...] = ()
        self._approval_count = 0
        self._panel_scroll = 0
        self._panel_lines = 0
        self._transcript_lines = 1
        # All engine/dispatch work runs on ONE worker thread: the worker
        # event store (sqlite is thread-affine) is opened and closed on it.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shadow-tui")

        self._transcript_control = FormattedTextControl(
            self._transcript_content,
            focusable=False,
            get_cursor_position=lambda: Point(0, max(0, self._transcript_lines - 1)),
        )
        transcript_window = Window(content=self._transcript_control, wrap_lines=True)

        self._approval_control = FormattedTextControl(
            self._approval_content,
            focusable=True,
            get_cursor_position=lambda: Point(0, self._panel_scroll),
        )
        approval_window = Window(
            content=self._approval_control,
            height=Dimension(min=1, max=APPROVAL_PANEL_MAX_HEIGHT),
            always_hide_cursor=True,
            wrap_lines=True,
        )
        self._approval_window = approval_window
        approval_container = ConditionalContainer(
            Frame(approval_window, title="approval"),
            filter=Condition(lambda: self._approval_pending),
        )

        history_path = Path("~/.shadow-code/prompt_history").expanduser()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.input_area = TextArea(
            multiline=True,
            completer=WordCompleter(_SLASH_COMMANDS, sentence=True),
            history=FileHistory(str(history_path)),
            focus_on_click=True,
            wrap_lines=True,
            prompt="",
        )
        input_container: Any  # Frame is not a Container subclass, but renders as one
        if self.theme.ascii_mode:
            input_container = HSplit(
                [
                    Window(FormattedTextControl(self._ascii_input_label), height=1),
                    self.input_area,
                ]
            )
        else:
            input_container = Frame(self.input_area, title=self._title())
        footer_window = Window(
            content=FormattedTextControl(self._footer_text),
            height=2,
            style="class:footer",
        )
        layout = Layout(
            HSplit([transcript_window, approval_container, input_container, footer_window]),
            focused_element=self.input_area,
        )
        self.app: Application[None] = Application(
            layout=layout,
            key_bindings=self._build_key_bindings(),
            style=self._build_style(),
            full_screen=True,
            mouse_support=False,
            input=input,
            output=output,
        )

    # -- layout helpers -------------------------------------------------

    def _title(self) -> str:
        state = self.rt.state
        return state.model_name if state is not None else "shadow"

    def _term_width(self) -> int:
        try:
            return self.app.output.get_size().columns or 80
        except Exception:
            return 80

    def _ascii_input_label(self) -> str:
        label = f" {self._title()} "
        fill = max(0, self._term_width() - 5 - len(label))
        return f"+--{label}{'-' * fill}+"

    def _build_style(self) -> Any:
        if not self.theme.colors:
            return None  # NO_COLOR: no ANSI styles at all
        styles = {
            "frame.border": "#888888",
            "frame.label": "bold #d77757",
            "text-area": "#e0e0e0",
            "footer": "bg:#1a1a1a #888888",
        }
        styles.update(THEME_TOKENS)
        return PTStyle.from_dict(styles)

    def _transcript_content(self) -> Any:
        """Transcript text/fragments; also tracks the bottom line for pinning."""
        width = self._term_width()
        if not self.theme.colors:
            text = self.model.render(self.theme, width)
            self._transcript_lines = text.count("\n") + 1
            return text
        fragments = self.model.render_fragments(self.theme, width)
        plain = "".join(text for _style, text in fragments)
        self._transcript_lines = plain.count("\n") + 1
        return fragments

    def _approval_content(self) -> Any:
        """Approval panel content: every plan fact, or freeform prompt lines."""
        if self._approval_plan is not None:
            fragments = approval_panel_fragments(
                self._approval_plan, self.theme, self._term_width()
            )
            plain = "".join(text for _style, text in fragments)
            self._panel_lines = plain.count("\n") + 1
            return fragments
        text = "\n".join(self._approval_freeform)
        self._panel_lines = text.count("\n") + 1
        return text

    def _footer_model(self) -> FooterModel:
        rt = self.rt
        state = rt.state
        workspace = rt.ctx.cwd if rt.ctx is not None else rt.cwd
        digest = ""
        if rt.prompt_manager is not None:
            digest = rt.prompt_manager.active.digest[:12]
        if self._approval_pending:
            status = "approval"
        elif self._busy:
            status = "busy"
        else:
            status = "idle"
        return FooterModel(
            model_name=state.model_name if state is not None else "",
            tokens_used=state.tokens_used if state is not None else 0,
            tokens_total=state.tokens_total if state is not None else 0,
            workspace_root=workspace,
            snapshot_digest=digest,
            permission_labels=rt.permission_labels,
            state=status,
        )

    def _footer_text(self) -> str:
        lines = render_footer(
            self._footer_model(), self._term_width(), ascii_mode=self.theme.ascii_mode
        )
        return "\n".join(lines)

    # -- key bindings -----------------------------------------------------

    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        approval_active = Condition(lambda: self._approval_pending)

        @bindings.add("enter")
        def _submit(event: Any) -> None:
            self._on_enter()

        @bindings.add("escape", "enter")
        def _newline(event: Any) -> None:
            self.input_area.buffer.insert_text("\n")

        @bindings.add("c-c")
        def _cancel(event: Any) -> None:
            self._on_cancel()

        @bindings.add("c-d")
        def _exit(event: Any) -> None:
            self._on_exit()

        @bindings.add("c-u")
        def _clear(event: Any) -> None:
            self.input_area.text = ""

        @bindings.add("c-l")
        def _clear_screen(event: Any) -> None:
            event.app.renderer.clear()

        @bindings.add("c-e")
        def _expand(event: Any) -> None:
            # Expand/collapse the latest truncated tool result; a no-op
            # while the approval panel owns the focus.
            if self._approval_pending:
                return
            if self.model.tools.toggle_expand():
                self.app.invalidate()

        @bindings.add("y", filter=approval_active)
        def _approve(event: Any) -> None:
            self._resolve_approval(True)

        @bindings.add("n", filter=approval_active)
        def _deny(event: Any) -> None:
            self._resolve_approval(False)

        @bindings.add("escape", filter=approval_active)
        def _deny_escape(event: Any) -> None:
            self._resolve_approval(False)

        @bindings.add("up", filter=approval_active)
        def _panel_up(event: Any) -> None:
            self._panel_scroll = max(0, self._panel_scroll - 1)
            self.app.invalidate()

        @bindings.add("down", filter=approval_active)
        def _panel_down(event: Any) -> None:
            self._panel_scroll = min(max(0, self._panel_lines - 1), self._panel_scroll + 1)
            self.app.invalidate()

        return bindings

    def _resolve_approval(self, answer: bool) -> None:
        """Land the one-shot decision; the row shows the outcome."""
        if not self._approval_pending:
            return
        self._approval_answer = answer
        self._approval_pending = False
        self._approval_plan = None
        self._approval_freeform = ()
        self.app.layout.focus(self.input_area)
        self.app.invalidate()
        self._approval_event.set()

    def _on_enter(self) -> None:
        if self._approval_pending:
            return  # the approval panel owns the keys: y/n/Esc decide
        text = self.input_area.text.strip()
        if not text:
            return
        self.input_area.text = ""
        self._inputs.put_nowait(text)

    def _on_cancel(self) -> None:
        """Ctrl+C: cancel active work; never corrupt the terminal."""
        if self._approval_pending:
            self._resolve_approval(False)
        elif self._busy:
            self.rt.interrupted = True
            self.model.append_system_line("[Cancelled]")
            self._refresh()
        else:
            self.input_area.text = ""

    def _on_exit(self) -> None:
        """Ctrl+D: deny a pending approval, cancel work, or leave."""
        if self._approval_pending:
            self._resolve_approval(False)
        elif self._busy:
            self.rt.interrupted = True
        else:
            self._inputs.put_nowait(None)

    # -- thread-safe transcript posting -----------------------------------

    def post(self, apply: Callable[[], None]) -> None:
        """Run a transcript mutation on the app thread, then redraw."""
        loop = self._loop
        if loop is None or loop.is_closed():
            self._apply(apply)
            return
        loop.call_soon_threadsafe(self._apply, apply)

    def _apply(self, apply: Callable[[], None]) -> None:
        apply()
        self._refresh()

    def _refresh(self) -> None:
        self.app.invalidate()

    def _write(self, text: str) -> None:
        self.post_system(text)

    def post_system(self, text: str) -> None:
        self.post(lambda: self.model.append_system_line(text))

    def post_user(self, text: str) -> None:
        self.post(lambda: self.model.append_user(text))

    def post_delta(self, delta: str) -> None:
        self.post(lambda: self.model.append_assistant_delta(delta))

    # -- approval bridge --------------------------------------------------

    def _begin_approval(self, *, plan: Any = None, freeform: Iterable[str] = ()) -> None:
        """Arm the single-focus approval control; called on the app thread.

        Every call gets its OWN fresh control: no batch approval, no
        inherited consent. The input area is unreachable while pending.
        """
        self._approval_plan = plan
        self._approval_freeform = tuple(freeform)
        self._approval_answer = False
        self._approval_pending = True
        self._approval_count += 1
        self._panel_scroll = 0
        self.app.layout.focus(self._approval_control)
        self.app.invalidate()

    def _wait_approval(self) -> bool:
        """Block the engine worker until y/n/Esc lands; fail-closed."""
        self._approval_event.wait()
        return self._approval_answer

    def _ask_approval(self, prompt_text: str) -> bool:
        """Blocking consent bridge for freeform prompts (legacy confirms)."""
        self._approval_event.clear()
        self.post(lambda: self._begin_approval(freeform=prompt_text.splitlines()))
        return self._wait_approval()

    def _consent(self, plan: Any) -> bool:
        """Engine consent seam: show the full action plan in the panel.

        The exact preview approved or denied is recorded on the call's row
        first, so the transcript keeps the evidence after the panel closes.
        """

        def begin() -> None:
            self.model.tools.note_preview(plan.call_id, plan.preview)
            self._begin_approval(plan=plan)

        self._approval_event.clear()
        self.post(begin)
        return self._wait_approval()

    # -- engine bridge ------------------------------------------------------

    def _on_call_event(self, event: CallEvent) -> None:
        """Engine lifecycle seam: update group rows on the app thread."""

        def apply() -> None:
            if event.stage == "round":
                self.model.tool_round(event.step)
            elif event.stage == "proposed":
                self.model.tool_proposed(event.call_id, event.tool_name, event.arguments_json)
            elif event.stage == "awaiting_approval":
                self.model.tools.note_awaiting_approval(event.call_id)
            elif event.stage == "executing":
                self.model.tools.note_executing(event.call_id)
            elif event.stage == "result" and event.result is not None:
                self.model.tool_result(event.result)

        self.post(apply)

    def _build_engine(self) -> Any:
        rt = self.rt
        if (
            rt.registry is None
            or rt.policy_engine is None
            or rt.execution_context is None
            or rt.approval_authority is None
        ):
            raise RuntimeError("TUI engine requires a fully bootstrapped runtime")
        from .engine import AgentEngine

        def on_round(engine_round: Any) -> None:
            # Conversation mirroring only; tool lifecycle rows render live
            # from the CallEvent stream instead of plain mirror lines.
            _mirror_round(engine_round, rt.conv, lambda _line: None)

        return AgentEngine(
            rt.registry,
            rt.policy_engine,
            rt.execution_context,
            rt.approval_authority,
            consent=self._consent,
            event_store=rt.event_store,
            event_session_id=rt.event_session_id,
            cancel_requested=lambda: rt.interrupted,
            on_round=on_round,
            on_call_event=self._on_call_event,
            on_store_warning=lambda message: self._write(f"[events warning: {message}]"),
        )

    def _stream_round(self) -> ProviderRound:
        """Provider round seam: deltas post live into the transcript."""
        rt = self.rt
        if rt.client is None:
            raise RuntimeError("TUI stream requires a client")
        client = rt.client
        chunks: list[str] = []
        cancelled = False
        try:
            for chunk in client.chat_stream(
                rt.conv.get_messages(), rt.system_prompt, tools=rt.tool_schemas
            ):
                if rt.interrupted:
                    cancelled = True
                    break
                chunks.append(chunk)
                self.post_delta(chunk)
        except StreamCancelledError:
            raise
        except Exception as error:
            raise StreamError("provider_error", str(error)) from error
        if cancelled:
            raise StreamCancelledError
        rt.conv.update_tokens(client.last_prompt_tokens)
        if rt.state is not None:
            rt.state.tokens_used = rt.conv.total_prompt_tokens
        return ProviderRound(
            text="".join(chunks),
            native_calls=tuple(getattr(client, "last_tool_calls", [])),
        )

    def _run_turn(self, message: str) -> None:
        """One full user turn; runs on the worker thread."""
        _handle_user_message(self.rt, message, self._stream_round, self._write, self._ask_approval)

    # -- session driver -----------------------------------------------------

    async def _driver(self) -> None:
        """Async session loop: inputs in, turns out, UI never blocks.

        The main-thread event store stays untouched: the session appends
        through a second store opened on the single worker thread (same
        SQLite file; WAL handles the two sequential connections). The
        original store is restored before main() runs its cleanup.
        """
        loop = asyncio.get_running_loop()
        original_store = self.rt.event_store
        original_db = self.rt.db
        worker_store: EventStore | None = None
        worker_db: Any = None
        try:
            if original_store is not None and self.rt.events_db_path is not None:
                worker_store = await loop.run_in_executor(
                    self._executor, EventStore, self.rt.events_db_path
                )
                self.rt.event_store = worker_store
            if original_db is not None and self.rt.db_path is not None:
                # Same thread-affinity rule as the event store: the worker
                # opens its own Database on the same WAL file; the original
                # is restored before main() runs its cleanup.
                from .db import Database

                worker_db = await loop.run_in_executor(self._executor, Database, self.rt.db_path)
                self.rt.db = worker_db
            self.rt.engine = self._build_engine()
            while True:
                item = await self._inputs.get()
                if item is None:
                    self.app.exit()
                    return
                action = await loop.run_in_executor(
                    self._executor, _dispatch_slash_command, item, self.rt, self._write
                )
                if action == _DISPATCH_EXIT:
                    self.app.exit()
                    return
                if action == _DISPATCH_CLEAR:
                    self.model.clear()
                    self.post_system("[Cleared]")
                    continue
                if not isinstance(action, tuple):
                    continue
                self.post_user(item)
                self._set_busy(True)
                try:
                    await loop.run_in_executor(self._executor, self._run_turn, action[1])
                except Exception as error:
                    self.post_system(f"[Error: {error}]")
                finally:
                    self._set_busy(False)
        finally:
            if worker_db is not None:
                self.rt.db = original_db
                await loop.run_in_executor(self._executor, worker_db.close)
            if worker_store is not None:
                # Queued behind any in-flight turn on the single worker.
                self.rt.event_store = original_store
                await loop.run_in_executor(self._executor, worker_store.close)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.app.invalidate()

    def run(self) -> None:
        """Run the application; returns after a clean exit."""

        def pre_run() -> None:
            self._loop = asyncio.get_running_loop()
            self._driver_task = asyncio.ensure_future(self._driver())

        self._refresh()
        try:
            self.app.run(pre_run=pre_run)
        finally:
            # Unstick any engine work waiting on a consent answer or polling
            # the cancel flag; prompt_toolkit has already restored the
            # terminal, and the loop cancels the driver task itself.
            self.rt.interrupted = True
            self._approval_event.set()
            self._executor.shutdown(wait=True)


def run_tui(
    rt: SessionRuntime,
    boot_lines: Iterable[str] = (),
    *,
    theme: TuiTheme | None = None,
    input: Any = None,
    output: Any = None,
) -> TuiApp:
    """Run the persistent TUI session; returns the app after a clean exit."""
    if not _HAS_PROMPT_TOOLKIT:
        raise RuntimeError("prompt_toolkit is required for the TUI")
    app = TuiApp(rt, theme=theme, boot_lines=boot_lines, input=input, output=output)
    app.run()
    return app
