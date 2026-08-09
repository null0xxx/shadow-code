# shadow_code/tui.py -- Persistent terminal shell (WU-09)
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
# View-model first: TranscriptModel, FooterModel, sanitize_terminal_text,
# and render_footer are pure and headless-testable. ANSI control sequences
# from model or tool output are neutralized before they can reach the
# terminal. NO_COLOR disables all styling; ASCII mode replaces box drawing
# while preserving every semantic label as text.
#
# This frontend is strictly opt-in (SHADOW_TUI=1 + TTYs); the line-oriented
# REPL in main.py stays the default and the minimal diagnostic client.

from __future__ import annotations

import asyncio
import os
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import ProviderRound, StreamCancelledError, StreamError
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

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.styles import Style as PTStyle
    from prompt_toolkit.widgets import Frame, TextArea

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False


# -- terminal text sanitization -------------------------------------------

_OSC_RE = re.compile("\x1b\\][^\x07\x1b]*(?:\x07|\x1b\\\\)")
_CSI_RE = re.compile("\x1b\\[[0-?]*[ -/]*[@-~]")
# Remaining C0 controls (incl. stray ESC and CR) and C1 controls; \n and
# \t are legitimate transcript content and survive.
_CTRL_RE = re.compile("[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]")


def sanitize_terminal_text(text: str) -> str:
    """Neutralize terminal control sequences in model/tool output.

    OSC (title/set-clipboard) and CSI (colors, cursor moves) sequences are
    removed entirely; any remaining control character except newline and
    tab is stripped. The result can never inject terminal control into the
    transcript, regardless of what a provider or a tool produced.
    """
    text = _OSC_RE.sub("", text)
    text = _CSI_RE.sub("", text)
    return _CTRL_RE.sub("", text)


# -- transcript view model --------------------------------------------------

_KIND_USER = "user"
_KIND_ASSISTANT = "assistant"
_KIND_TOOL = "tool"
_KIND_SYSTEM = "system"


@dataclass(frozen=True)
class TranscriptEntry:
    """One logical transcript row; text may span multiple physical lines."""

    kind: str
    text: str


def _prefix(kind: str, theme: TuiTheme) -> str:
    if kind == _KIND_USER:
        return "> " if theme.ascii_mode else "› "
    if kind == _KIND_TOOL:
        return "  "
    return ""


class TranscriptModel:
    """Append-only conversation projection; the only transcript truth.

    Assistant text streams in as deltas that accumulate into the trailing
    assistant entry (the live response area); tool lifecycle and system
    lines are standalone entries. All text is sanitized on the way in.
    """

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []

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

    def render(self, theme: TuiTheme) -> str:
        """Project the entries into plain display text; no Rich objects."""
        lines: list[str] = []
        for entry in self._entries:
            prefix = _prefix(entry.kind, theme)
            for index, line in enumerate(entry.text.split("\n")):
                lines.append((prefix if index == 0 else " " * len(prefix)) + line)
        return "\n".join(lines)


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

    Layout: transcript window (read-only, wraps, auto-scrolls), input area
    (multiline, history, slash completion; Enter sends, Alt+Enter inserts a
    newline), footer window (state snapshot per width bucket). Engine turns
    run via asyncio.to_thread; approvals bridge back through the input area.
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
        # All engine/dispatch work runs on ONE worker thread: the worker
        # event store (sqlite is thread-affine) is opened and closed on it.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shadow-tui")

        self._transcript_buffer = Buffer(read_only=True)
        transcript_window = Window(
            content=BufferControl(buffer=self._transcript_buffer, focusable=False),
            wrap_lines=True,
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
            HSplit([transcript_window, input_container, footer_window]),
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
        return PTStyle.from_dict(
            {
                "frame.border": "#888888",
                "frame.label": "bold #d77757",
                "text-area": "#e0e0e0",
                "footer": "bg:#1a1a1a #888888",
            }
        )

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

        return bindings

    def _deny_approval(self) -> None:
        self._approval_answer = False
        self._approval_pending = False
        self.model.append_system_line("[Denied]")
        self._refresh()
        self._approval_event.set()

    def _on_enter(self) -> None:
        text = self.input_area.text.strip()
        if self._approval_pending:
            self._approval_answer = text.lower() in {"y", "yes"}
            self._approval_pending = False
            self.input_area.text = ""
            self.model.append_system_line("[Approved]" if self._approval_answer else "[Denied]")
            self._refresh()
            self._approval_event.set()
            return
        if not text:
            return
        self.input_area.text = ""
        self._inputs.put_nowait(text)

    def _on_cancel(self) -> None:
        """Ctrl+C: cancel active work; never corrupt the terminal."""
        if self._approval_pending:
            self._deny_approval()
        elif self._busy:
            self.rt.interrupted = True
            self.model.append_system_line("[Cancelled]")
            self._refresh()
        else:
            self.input_area.text = ""

    def _on_exit(self) -> None:
        """Ctrl+D: deny a pending approval, cancel work, or leave."""
        if self._approval_pending:
            self._deny_approval()
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
        text = self.model.render(self.theme)
        document = Document(text, cursor_position=len(text))
        self._transcript_buffer.set_document(document, bypass_readonly=True)
        self.app.invalidate()

    def _write(self, text: str) -> None:
        self.post_system(text)

    def post_system(self, text: str) -> None:
        self.post(lambda: self.model.append_system_line(text))

    def post_user(self, text: str) -> None:
        self.post(lambda: self.model.append_user(text))

    def post_delta(self, delta: str) -> None:
        self.post(lambda: self.model.append_assistant_delta(delta))

    def post_tool(self, tool_name: str, status: str) -> None:
        self.post(lambda: self.model.append_tool_line(tool_name, status))

    # -- approval bridge --------------------------------------------------

    def _ask_approval(self, prompt_text: str) -> bool:
        """Blocking y/N consent bridge; called from the engine worker.

        Fail-closed: only an explicit "y"/"yes" approves; Ctrl+C, Ctrl+D,
        empty input, or anything else denies. Single focus: the input area
        captures the answer, no second prompt is ever spawned.
        """
        self._approval_event.clear()

        def begin() -> None:
            self._approval_pending = True
            self._approval_answer = False
            self.model.append_system_line(prompt_text)
            self.model.append_system_line("Approve this exact action? [y/N]")

        self.post(begin)
        self._approval_event.wait()
        return self._approval_answer

    def _consent(self, plan: Any) -> bool:
        """Engine consent seam: render the action plan in the transcript."""
        lines = [
            "Action requires approval:",
            f"  tool:       {plan.tool_name} v{plan.tool_version}",
            f"  capability: {plan.capability}",
            f"  arguments:  {plan.canonical_arguments_json}",
            f"  workspace:  device={plan.workspace_device} inode={plan.workspace_inode}",
            f"  plan:       sha256:{plan.digest()[:16]}...",
            f"  preview:    {plan.preview}",
        ]
        return self._ask_approval("\n".join(lines))

    # -- engine bridge ------------------------------------------------------

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
            _mirror_round(engine_round, rt.conv, self._write)

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
        worker_store: EventStore | None = None
        try:
            if original_store is not None and self.rt.events_db_path is not None:
                worker_store = await loop.run_in_executor(
                    self._executor, EventStore, self.rt.events_db_path
                )
                self.rt.event_store = worker_store
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
