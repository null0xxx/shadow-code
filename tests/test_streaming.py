"""Tests for streaming.py -- Rich Live streaming controller."""

import io
import sys
import unittest
from unittest.mock import MagicMock, patch

from shadow_code.streaming import StreamCancelled, StreamController


class TestStreamCancelled(unittest.TestCase):
    """Test StreamCancelled exception."""

    def test_is_exception(self):
        self.assertTrue(issubclass(StreamCancelled, Exception))

    def test_can_raise_and_catch(self):
        with self.assertRaises(StreamCancelled):
            raise StreamCancelled()


class TestStreamControllerPlain(unittest.TestCase):
    """Test StreamController in plain mode (no Rich)."""

    def _make_controller(self, chunks):
        """Create a StreamController with a mock client."""
        client = MagicMock()
        client.chat_stream.return_value = iter(chunks)
        client.last_eval_tokens = 42
        ui = MagicMock()
        ctrl = StreamController(client, ui, console=None, display=None)
        return ctrl

    def test_stream_plain_returns_response(self):
        ctrl = self._make_controller(["Hello ", "world"])
        resp, tokens = ctrl._stream_plain([], "system")
        self.assertEqual(resp, "Hello world")
        self.assertEqual(tokens, 42)

    def test_stream_plain_empty(self):
        ctrl = self._make_controller([])
        resp, tokens = ctrl._stream_plain([], "system")
        self.assertEqual(resp, "")

    def test_stream_response_uses_plain_when_no_rich(self):
        ctrl = self._make_controller(["test"])
        with patch("shadow_code.streaming.HAS_RICH", False):
            resp, _ = ctrl.stream_response([], "system")
        self.assertEqual(resp, "test")

    def test_stream_plain_sanitizes_stdout_but_returns_raw(self):
        raw = "Hello \x1b[31mworld\x1b]0;evil\x07!"
        ctrl = self._make_controller([raw])
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            resp, _ = ctrl._stream_plain([], "system")
        finally:
            sys.stdout = old
        # Returned response is byte-identical to the raw model output...
        self.assertEqual(resp, raw)
        # ...while the terminal saw sanitized text (plus the trailing print()).
        self.assertEqual(captured.getvalue(), "Hello world!\n")

    def test_stream_plain_flushes_and_raises_typed_cancellation(self):
        client = MagicMock()

        def interrupted():
            yield "visible"
            raise KeyboardInterrupt

        client.chat_stream.return_value = interrupted()
        ctrl = StreamController(client, MagicMock())

        with self.assertRaises(StreamCancelled):
            ctrl._stream_plain([], "system")

        self.assertEqual(ctrl.display.get_full_response(), "visible")


class TestStreamControllerCapture(unittest.TestCase):
    """Test _feed_and_capture and _flush_and_capture."""

    def test_feed_and_capture_plain_text(self):
        client = MagicMock()
        ui = MagicMock()
        ctrl = StreamController(client, ui)
        ctrl.display.reset()
        result = ctrl._feed_and_capture("Hello")
        self.assertIn("Hello", result)

    def test_feed_and_capture_tool_call_hidden(self):
        client = MagicMock()
        ui = MagicMock()
        ctrl = StreamController(client, ui)
        ctrl.display.reset()
        result = ctrl._feed_and_capture('```tool_call\n{"tool": "bash", "params": {}}\n```')
        self.assertNotIn('"tool"', result)

    def test_flush_and_capture(self):
        client = MagicMock()
        ui = MagicMock()
        ctrl = StreamController(client, ui)
        ctrl.display.reset()
        ctrl.display.buffer = "leftover"
        result = ctrl._flush_and_capture()
        self.assertIn("leftover", result)


class TestStreamControllerRich(unittest.TestCase):
    """Rich-mode behavior uses local fakes rather than a real terminal."""

    def test_rich_stream_updates_visible_text_and_prints_final_tokens(self):
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        client = MagicMock()
        client.chat_stream.return_value = iter(
            ["Hello ", '```tool_call\n{"tool":"bash"}\n```\n', "world"]
        )
        client.last_eval_tokens = 17
        ui = MagicMock()
        ui.render_thinking.return_value = "thinking"
        ui.render_streaming_with_tokens.side_effect = lambda text, tokens: (text, tokens)
        ui.render_response.side_effect = lambda text, tokens: (text, tokens)
        console = MagicMock()
        ctrl = StreamController(client, ui, console=console)

        with patch("shadow_code.streaming.Live", return_value=live):
            response, tokens = ctrl._stream_rich([{"role": "user"}], "system")

        self.assertIn("tool_call", response)
        self.assertEqual(tokens, 17)
        self.assertGreaterEqual(live.update.call_count, 2)
        ui.render_streaming_with_tokens.assert_any_call("Hello ", 1)
        ui.render_response.assert_called_once_with("Hello world", 17)
        console.print.assert_called_once_with(("Hello world", 17))

    def test_rich_stream_sanitizes_visible_text_but_returns_raw(self):
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        client = MagicMock()
        client.chat_stream.return_value = iter(["Hello \x1b[31", "mworld\x07"])
        client.last_eval_tokens = 5
        ui = MagicMock()
        ui.render_thinking.return_value = "thinking"
        ui.render_streaming_with_tokens.side_effect = lambda text, tokens: (text, tokens)
        ui.render_response.side_effect = lambda text, tokens: (text, tokens)
        console = MagicMock()
        ctrl = StreamController(client, ui, console=console)

        with patch("shadow_code.streaming.Live", return_value=live):
            response, tokens = ctrl._stream_rich([], "system")

        # Stored response keeps the raw bytes (incl. the split CSI sequence)...
        self.assertEqual(response, "Hello \x1b[31mworld\x07")
        # ...while the visible text fed to Live and the final print is clean.
        self.assertEqual(tokens, 5)
        ui.render_response.assert_called_once_with("Hello world", 5)

    def test_rich_stream_with_only_hidden_call_does_not_print_response(self):
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        client = MagicMock()
        client.chat_stream.return_value = iter(['```tool_call\n{"tool":"bash"}\n```'])
        client.last_eval_tokens = 3
        console = MagicMock()
        ctrl = StreamController(client, MagicMock(), console=console)

        with patch("shadow_code.streaming.Live", return_value=live):
            response, tokens = ctrl._stream_rich([], "system")

        self.assertIn("tool_call", response)
        self.assertEqual(tokens, 3)
        console.print.assert_not_called()

    def test_rich_stream_flushes_and_raises_typed_cancellation(self):
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        client = MagicMock()

        def interrupted():
            yield "visible"
            raise KeyboardInterrupt

        client.chat_stream.return_value = interrupted()
        ctrl = StreamController(client, MagicMock(), console=MagicMock())

        with (
            patch("shadow_code.streaming.Live", return_value=live),
            self.assertRaises(StreamCancelled),
        ):
            ctrl._stream_rich([], "system")

        self.assertEqual(ctrl.display.get_full_response(), "visible")


class TestStreamControllerThinking(unittest.TestCase):
    """SHADOW_THINK: display-only thinking channel in both stream paths."""

    class FakeThinkingClient:
        """Scripted client whose stream emits thinking via the handler."""

        def __init__(self):
            self.thinking_handler = None
            self.last_prompt_tokens = 5
            self.last_eval_tokens = 11

        def chat_stream(self, messages, system, model=None, tools=None):
            handler = self.thinking_handler
            if handler is not None:
                handler("let me think")
                handler(" about it")
            yield "real answer"

    def test_rich_thinking_renders_live_and_collapses_to_summary(self):
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        client = self.FakeThinkingClient()
        ui = MagicMock()
        ui.render_thinking.return_value = "spinner"
        ui.render_streaming_with_tokens.side_effect = lambda text, tokens, thinking="": (
            text,
            tokens,
            thinking,
        )
        ui.render_thought_summary.side_effect = lambda seconds, tokens: ("summary", seconds, tokens)
        ui.render_response.side_effect = lambda text, tokens: (text, tokens)
        console = MagicMock()
        ctrl = StreamController(client, ui, console=console)

        with (
            patch("shadow_code.streaming.THINK_ENABLED", True),
            patch("shadow_code.streaming.Live", return_value=live),
        ):
            response, tokens = ctrl._stream_rich([], "system")

        # Thinking never joins the response text or the display buffer.
        self.assertEqual(response, "real answer")
        self.assertEqual(tokens, 11)
        self.assertEqual(ctrl.display.get_full_response(), "real answer")
        # The Live view saw the thinking channel above the (empty) answer.
        ui.render_streaming_with_tokens.assert_any_call("", 0, thinking="let me think")
        ui.render_streaming_with_tokens.assert_any_call("", 0, thinking="let me think about it")
        # Final render: collapsed summary line first, then the response body.
        self.assertEqual(console.print.call_count, 2)
        self.assertEqual(console.print.call_args_list[0].args[0][0], "summary")
        ui.render_response.assert_called_once_with("real answer", 11)
        # The handler is uninstalled after the stream.
        self.assertIsNone(client.thinking_handler)

    def test_rich_off_by_default_never_touches_handler(self):
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        client = self.FakeThinkingClient()
        ui = MagicMock()
        ui.render_streaming_with_tokens.side_effect = lambda text, tokens: (text, tokens)
        ui.render_response.side_effect = lambda text, tokens: (text, tokens)
        ctrl = StreamController(client, ui, console=MagicMock())

        with patch("shadow_code.streaming.Live", return_value=live):
            response, _ = ctrl._stream_rich([], "system")

        self.assertEqual(response, "real answer")
        self.assertIsNone(client.thinking_handler)  # untouched: still the initial None
        # No summary line without thinking.
        ui.render_thought_summary.assert_not_called()

    def test_plain_thinking_prints_single_marker_and_hides_body(self):
        client = self.FakeThinkingClient()
        ctrl = StreamController(client, MagicMock())
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            with patch("shadow_code.streaming.THINK_ENABLED", True):
                resp, tokens = ctrl._stream_plain([], "system")
        finally:
            sys.stdout = old

        self.assertEqual(resp, "real answer")
        self.assertEqual(tokens, 11)
        self.assertEqual(captured.getvalue().count("[thinking...]"), 1)
        self.assertNotIn("let me think", captured.getvalue())
        self.assertNotIn("\x1b", captured.getvalue())
        self.assertIsNone(client.thinking_handler)

    def test_flag_on_vs_off_store_byte_identical_conversation(self):
        from shadow_code.conversation import Conversation

        def run(enabled: bool):
            client = self.FakeThinkingClient()
            ui = MagicMock()
            ui.render_streaming_with_tokens.side_effect = lambda text, tokens, thinking="": (
                text,
                tokens,
                thinking,
            )
            ui.render_response.side_effect = lambda text, tokens: (text, tokens)
            live = MagicMock()
            live.__enter__.return_value = live
            live.__exit__.return_value = False
            ctrl = StreamController(client, ui, console=MagicMock())
            with (
                patch("shadow_code.streaming.THINK_ENABLED", enabled),
                patch("shadow_code.streaming.Live", return_value=live),
            ):
                resp, _ = ctrl._stream_rich([], "system")
            conv = Conversation()
            conv.add_user("hi")
            conv.add_assistant(resp)
            return resp, conv.get_messages()

        resp_on, messages_on = run(True)
        resp_off, messages_off = run(False)
        self.assertEqual(resp_on, resp_off)
        self.assertEqual(messages_on, messages_off)
        self.assertNotIn("think", resp_on)


if __name__ == "__main__":
    unittest.main()
