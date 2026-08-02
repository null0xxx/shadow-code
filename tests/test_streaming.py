"""Tests for streaming.py -- Rich Live streaming controller."""

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


if __name__ == "__main__":
    unittest.main()
