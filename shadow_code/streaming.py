# shadow_code/streaming.py -- Rich Live streaming controller
#
# Handles streaming display with Rich Live panels.
# Tool call blocks are hidden by StreamDisplay, only visible text shown.
# Includes real-time token estimation and shimmer thinking animation.

import io
import sys
import time

from .config import THINK_ENABLED
from .display import StreamDisplay


class StreamCancelled(Exception):
    """Raised when the user interrupts streaming with Ctrl+C."""

    pass


try:
    from rich.console import Console
    from rich.live import Live

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class StreamController:
    """Controls streaming LLM responses with Rich Live display."""

    def __init__(self, client, ui, console=None, display=None):
        self.client = client
        self.ui = ui
        self.display = display or StreamDisplay()
        if HAS_RICH:
            self.console = console or Console()
        else:
            self.console = console

    def stream_response(
        self, messages: list[dict], system: str, tools: list[dict] | None = None
    ) -> tuple[str, int]:
        """Stream response. Returns (full_response_with_tool_calls, eval_tokens)."""
        if HAS_RICH:
            return self._stream_rich(messages, system, tools)
        else:
            return self._stream_plain(messages, system, tools)

    def _stream_rich(
        self, messages: list[dict], system: str, tools: list[dict] | None = None
    ) -> tuple[str, int]:
        """Stream with Rich Live panel + real-time token estimation."""
        self.display.reset()
        accumulated_visible = ""
        char_count = 0
        thinking_parts: list[str] = []
        thinking_started: float | None = None

        try:
            with Live(
                self.ui.render_thinking(),
                console=self.console,
                refresh_per_second=10,
                transient=True,
            ) as live:
                if THINK_ENABLED:
                    # Thinking deltas arrive synchronously from chat_stream on
                    # this thread; render them dim above the answer, live.
                    def _capture_thinking(text: str) -> None:
                        nonlocal thinking_started
                        if thinking_started is None:
                            thinking_started = time.monotonic()
                        thinking_parts.append(text)
                        live.update(
                            self.ui.render_streaming_with_tokens(
                                accumulated_visible,
                                char_count // 4,
                                thinking="".join(thinking_parts),
                            )
                        )

                    self.client.thinking_handler = _capture_thinking
                try:
                    for chunk in self.client.chat_stream(messages, system, tools=tools):
                        visible_text = self._feed_and_capture(chunk)
                        char_count += len(chunk)
                        if visible_text:
                            accumulated_visible += visible_text
                            estimated_tokens = char_count // 4
                            live.update(
                                self.ui.render_streaming_with_tokens(
                                    accumulated_visible, estimated_tokens
                                )
                            )
                finally:
                    if THINK_ENABLED:
                        self.client.thinking_handler = None

                remaining = self._flush_and_capture()
                if remaining:
                    accumulated_visible += remaining

        except KeyboardInterrupt:
            remaining = self._flush_and_capture()
            if remaining:
                accumulated_visible += remaining
            raise StreamCancelled() from None

        # Print final response OUTSIDE Live (stays on screen permanently)
        if thinking_parts:
            # The thinking body stays ephemeral: only a collapsed one-line
            # summary persists above the final response block.
            elapsed = time.monotonic() - (thinking_started or time.monotonic())
            thinking_tokens = sum(len(part) for part in thinking_parts) // 4
            self.console.print(self.ui.render_thought_summary(elapsed, thinking_tokens))
        if accumulated_visible.strip():
            tokens = self.client.last_eval_tokens
            self.console.print(self.ui.render_response(accumulated_visible, tokens))

        # If no visible text but native tool calls exist, return empty string;
        # main.py runs the collected calls through the admission pipeline
        # (registry validation -> policy -> executor).
        return self.display.get_full_response(), self.client.last_eval_tokens

    def _stream_plain(
        self, messages: list[dict], system: str, tools: list[dict] | None = None
    ) -> tuple[str, int]:
        """Fallback: plain stdout streaming."""
        self.display.reset()
        thinking_seen = False

        def _mark_thinking(_text: str) -> None:
            # Plain path: a single static marker, never raw ANSI, body hidden.
            nonlocal thinking_seen
            if not thinking_seen:
                thinking_seen = True
                print("[thinking...]")

        if THINK_ENABLED:
            self.client.thinking_handler = _mark_thinking

        try:
            for chunk in self.client.chat_stream(messages, system, tools=tools):
                self.display.feed(chunk)
        except KeyboardInterrupt:
            self.display.flush()
            print()
            raise StreamCancelled() from None
        finally:
            if THINK_ENABLED:
                self.client.thinking_handler = None

        self.display.flush()
        print()

        return self.display.get_full_response(), self.client.last_eval_tokens

    def _feed_and_capture(self, chunk: str) -> str:
        """Feed chunk to StreamDisplay, capture what it would print."""
        capture = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = capture
            self.display.feed(chunk)
        finally:
            sys.stdout = old_stdout
        return capture.getvalue()

    def _flush_and_capture(self) -> str:
        """Flush StreamDisplay, capture remaining text."""
        capture = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = capture
            self.display.flush()
        finally:
            sys.stdout = old_stdout
        return capture.getvalue()
