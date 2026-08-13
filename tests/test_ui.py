"""Tests for ui.py -- Rich UI rendering (requires rich)."""

import io
import unittest

try:
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from shadow_code.ui import HAS_RICH as UI_HAS_RICH


def _render_to_str(renderable, *, force_terminal=False, width=100):
    """Render a Rich renderable to a plain string via an in-memory console."""
    from rich.console import Console

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=force_terminal, width=width)
    console.print(renderable)
    return buf.getvalue()


@unittest.skipUnless(HAS_RICH and UI_HAS_RICH, "rich not installed")
class TestUIRenderer(unittest.TestCase):
    """Tests for UIRenderer methods."""

    def setUp(self):
        from shadow_code.ui import UIRenderer

        self.ui = UIRenderer()

    def test_render_welcome(self):
        result = self.ui.render_welcome()
        self.assertIsNotNone(result)

    def test_render_thinking(self):
        result = self.ui.render_thinking()
        self.assertIsNotNone(result)

    def test_render_streaming(self):
        result = self.ui.render_streaming("Hello world")
        self.assertIsNotNone(result)

    def test_render_streaming_empty(self):
        result = self.ui.render_streaming("")
        self.assertIsNotNone(result)

    def test_render_response(self):
        from rich.console import Group
        from rich.markdown import Markdown

        result = self.ui.render_response("Some response", tokens=100)
        self.assertIsInstance(result, Group)
        renderables = list(result.renderables)
        self.assertIsInstance(renderables[0], Markdown)
        self.assertIsInstance(renderables[-1], Text)

    def test_render_response_no_tokens(self):
        from rich.console import Group
        from rich.markdown import Markdown

        result = self.ui.render_response("Response")
        self.assertIsInstance(result, Group)
        renderables = list(result.renderables)
        self.assertEqual(len(renderables), 1)
        self.assertIsInstance(renderables[0], Markdown)

    def test_render_response_fenced_code_block_survives(self):
        md = "Here is the code:\n\n```python\ndef hello():\n    return 1\n```\n"
        out = _render_to_str(self.ui.render_response(md))
        self.assertIn("def hello():", out)
        self.assertIn("return 1", out)

    def test_render_response_strips_control_sequences(self):
        payload = "safe\x1b]0;evil title\x07 \x1b[31mred\x1b[0m\x1b[2Jdone\x00\x9b"
        out = _render_to_str(self.ui.render_response(payload), force_terminal=True)
        self.assertNotIn("\x1b]", out)
        self.assertNotIn("\x1b[2J", out)
        self.assertNotIn("\x07", out)
        self.assertNotIn("\x00", out)
        self.assertNotIn("\x9b", out)
        self.assertNotIn("evil title", out)
        self.assertIn("safe", out)
        self.assertIn("red", out)
        self.assertIn("done", out)

    def test_render_response_no_color_env_emits_no_ansi(self):
        import os
        from unittest.mock import patch

        from rich.console import Console

        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            buf = io.StringIO()
            console = Console(file=buf, force_terminal=True, width=100)
            console.print(self.ui.render_response("**bold** plain", tokens=5))
        self.assertNotIn("\x1b", buf.getvalue())
        self.assertIn("bold", buf.getvalue())

    def test_render_response_non_tty_emits_no_ansi(self):
        out = _render_to_str(self.ui.render_response("**bold** plain", tokens=5))
        self.assertNotIn("\x1b", out)
        self.assertIn("bold", out)
        self.assertIn("5 tokens", out)

    def test_render_response_georgian_text_round_trips(self):
        text = "გამარჯობა, როგორ ხარ?"
        out = _render_to_str(self.ui.render_response(text))
        self.assertIn(text, out)

    def test_render_response_token_line_present(self):
        out = _render_to_str(self.ui.render_response("answer", tokens=1234))
        self.assertIn("1,234 tokens", out)

    def test_render_tool_call(self):
        result = self.ui.render_tool_call("bash", "ls -la")
        self.assertIsInstance(result, Text)

    def test_render_tool_result_success(self):
        result = self.ui.render_tool_result("bash", "file1\nfile2", True)
        self.assertIsNotNone(result)

    def test_render_tool_result_failure(self):
        result = self.ui.render_tool_result("bash", "command not found", False)
        self.assertIsNotNone(result)

    def test_render_tool_result_long_output(self):
        long_output = "x" * 5000
        result = self.ui.render_tool_result("bash", long_output, True)
        self.assertIsNotNone(result)

    def test_render_tool_result_syntax_highlight(self):
        code = "def hello():\n    print('hi')\n" * 5
        result = self.ui.render_tool_result("read_file", code, True)
        self.assertIsNotNone(result)

    def test_render_tool_result_short_text(self):
        result = self.ui.render_tool_result("glob", "short", True)
        self.assertIsNotNone(result)

    def test_render_error(self):
        result = self.ui.render_error("Something went wrong")
        self.assertIsInstance(result, Text)

    def test_render_context_status_low(self):
        result = self.ui.render_context_status(10000, 131072)
        self.assertIsInstance(result, Text)

    def test_render_context_status_medium(self):
        result = self.ui.render_context_status(80000, 131072)
        self.assertIsInstance(result, Text)

    def test_render_context_status_high(self):
        result = self.ui.render_context_status(120000, 131072)
        self.assertIsInstance(result, Text)

    def test_render_context_status_zero(self):
        result = self.ui.render_context_status(0, 0)
        self.assertIsInstance(result, Text)

    def test_render_help(self):
        commands = [("/help", "Show help"), ("/exit", "Exit")]
        result = self.ui.render_help(commands)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
