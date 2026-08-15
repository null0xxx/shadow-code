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
        from rich.console import Group

        result = self.ui.render_welcome()
        self.assertIsInstance(result, Group)

    def test_render_welcome_contains_banner_tips_and_tagline(self):
        out = _render_to_str(self.ui.render_welcome(100))
        self.assertIn("####", out)  # wordmark art
        self.assertIn("Tips for getting started:", out)
        self.assertIn("/help", out)

    def test_render_welcome_gradient_when_color_enabled(self):
        import os
        from unittest.mock import patch

        from rich.console import Console

        # The runner env may pin NO_COLOR/TERM=dumb; force a truecolor
        # terminal so the gradient assertion is runner-independent.
        with patch.dict(os.environ, {"TERM": "xterm-256color", "COLORTERM": "truecolor"}):
            os.environ.pop("NO_COLOR", None)
            buf = io.StringIO()
            console = Console(file=buf, force_terminal=True, width=100)
            self.assertFalse(console.no_color)
            console.print(self.ui.render_welcome(100))
        out = buf.getvalue()
        self.assertIn("\x1b[38;2;", out)  # truecolor gradient escapes
        self.assertIn("175;135;215", out)  # theme accent at the left edge

    def test_render_welcome_no_color_env_emits_no_ansi(self):
        import os
        from unittest.mock import patch

        from rich.console import Console

        # Pin TERM=dumb so the zero-ANSI assertion is runner-independent
        # (same gotcha as test_render_response_no_color_env_emits_no_ansi).
        with patch.dict(os.environ, {"NO_COLOR": "1", "TERM": "dumb"}):
            buf = io.StringIO()
            console = Console(file=buf, force_terminal=True, width=100)
            console.print(self.ui.render_welcome(100))
        out = buf.getvalue()
        self.assertNotIn("\x1b", out)
        self.assertIn("####", out)  # plain ASCII art survives
        self.assertIn("Tips for getting started:", out)

    def test_render_welcome_narrow_terminal_falls_back(self):
        out = _render_to_str(self.ui.render_welcome(20), width=40)
        self.assertIn("SHADOW CODE", out)
        self.assertNotIn("####", out)

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
        self.assertIsInstance(renderables[0], Text)  # ⏺ accent marker
        self.assertIsInstance(renderables[1], Markdown)
        self.assertIsInstance(renderables[-1], Text)

    def test_render_response_no_tokens(self):
        from rich.console import Group
        from rich.markdown import Markdown

        result = self.ui.render_response("Response")
        self.assertIsInstance(result, Group)
        renderables = list(result.renderables)
        self.assertEqual(len(renderables), 2)
        self.assertIsInstance(renderables[0], Text)
        self.assertIsInstance(renderables[1], Markdown)

    def test_render_response_marker_present(self):
        out = _render_to_str(self.ui.render_response("answer"))
        self.assertIn("⏺", out)
        self.assertIn("answer", out)

    def test_render_streaming_with_thinking_separates_and_sanitizes(self):
        from rich.console import Group

        result = self.ui.render_streaming_with_tokens(
            "answer", 3, thinking="plotting \x1b[31mevil\x07"
        )
        self.assertIsInstance(result, Group)
        out = _render_to_str(result, force_terminal=True)
        # The model's own control sequences never reach the terminal...
        self.assertNotIn("\x1b[31m", out)
        self.assertNotIn("\x07", out)
        # ...while the thinking text renders above the answer.
        self.assertLess(out.index("plotting evil"), out.index("answer"))

    def test_render_streaming_without_thinking_unchanged(self):
        from rich.text import Text

        result = self.ui.render_streaming_with_tokens("answer", 3)
        self.assertIsInstance(result, Text)

    def test_render_thought_summary_line(self):
        out = _render_to_str(self.ui.render_thought_summary(1.234, 517))
        self.assertIn("thought for 1.2s", out)
        self.assertIn("517", out)
        out_no_tokens = _render_to_str(self.ui.render_thought_summary(0.5))
        self.assertIn("thought for 0.5s", out_no_tokens)
        self.assertNotIn("tokens", out_no_tokens)

    def test_render_thought_summary_no_ansi_on_dumb_terminal(self):
        import os
        from unittest.mock import patch

        from rich.console import Console

        # Pin TERM=dumb so the zero-ANSI assertion is runner-independent
        # (same gotcha as test_render_response_no_color_env_emits_no_ansi).
        with patch.dict(os.environ, {"NO_COLOR": "1", "TERM": "dumb"}):
            buf = io.StringIO()
            console = Console(file=buf, force_terminal=True, width=100)
            console.print(self.ui.render_thought_summary(2.0, 12))
        self.assertNotIn("\x1b", buf.getvalue())
        self.assertIn("thought for 2.0s", buf.getvalue())

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

        # rich honors NO_COLOR by stripping color SGRs but keeps bold/dim
        # attributes; only a dumb terminal suppresses all escape output.
        # Pin TERM=dumb so the assertion is independent of the runner env
        # (CI runners have no TERM; dev shells vary).
        with patch.dict(os.environ, {"NO_COLOR": "1", "TERM": "dumb"}):
            buf = io.StringIO()
            console = Console(file=buf, force_terminal=True, width=100)
            self.assertTrue(console.no_color)
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

    def test_loose_markdown_numbered_run_becomes_list(self):
        out = _render_to_str(self.ui.render_response("1 first\n2 second\n3 third"))
        lines = [ln for ln in out.splitlines() if ln.strip()]
        # Recognized as a list: each item on its own line, no run-on merge.
        self.assertTrue(any("first" in ln for ln in lines))
        self.assertTrue(any("second" in ln for ln in lines))
        self.assertFalse(any("first" in ln and "second" in ln for ln in lines))

    def test_loose_markdown_bullet_marker(self):
        out = _render_to_str(self.ui.render_response("• alpha\n• beta"))
        self.assertIn("alpha", out)
        self.assertIn("beta", out)

    def test_loose_markdown_year_is_not_a_list(self):
        norm = self.ui._normalize_loose_markdown("2026 წლის გეგმა")
        self.assertEqual(norm, "2026 წლის გეგმა")

    def test_loose_markdown_non_run_numbers_untouched(self):
        norm = self.ui._normalize_loose_markdown("3 პუნქტი მხოლოდ")
        self.assertEqual(norm, "3 პუნქტი მხოლოდ")

    def test_loose_markdown_fence_content_untouched(self):
        raw = "```python\n1 not a list\n• safe\n```"
        self.assertEqual(self.ui._normalize_loose_markdown(raw), raw)

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
