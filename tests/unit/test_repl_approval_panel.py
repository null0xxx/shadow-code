"""PR 2 Part B: colored REPL approval panel.

Covers ui.render_unified_diff / ui.render_approval_panel and the rich/plain
split in main._request_approval. Plain-mode output must stay byte-identical
to the historical prints; plan digests must not change across rendering.
"""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from shadow_code.domain.approval import ActionPlan
from shadow_code.ui import HAS_RICH as UI_HAS_RICH

try:
    from rich.console import Console

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

_DIFF_PREVIEW = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n-old\n+new"
_BASH_PREVIEW = "$ echo hi | wc -c\nsandbox: none available\nfeatures: pipe redirect"


def _plan(preview: str = _DIFF_PREVIEW) -> ActionPlan:
    return ActionPlan(
        call_id="call-1",
        tool_name="edit_file",
        tool_version="1.0.0",
        capability="fs.write",
        canonical_arguments_json='{"file_path":"calc.py","new_string":"new","old_string":"old"}',
        workspace_device=64768,
        workspace_inode=1234567,
        registry_digest="0" * 64,
        preview=preview,
    )


def _render_panel(ui, plan, *, force_terminal=False, width=200) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=force_terminal, width=width)
    console.print(ui.render_approval_panel(plan))
    return buf.getvalue()


@unittest.skipUnless(HAS_RICH and UI_HAS_RICH, "rich not installed")
class TestUnifiedDiffClassifier(unittest.TestCase):
    """The Rich-native +/- /@@ classifier, tested directly."""

    def setUp(self):
        from shadow_code.ui import UIRenderer

        self.ui = UIRenderer()

    def test_add_del_hunk_lines_get_theme_styles(self):
        from shadow_code.theme import THEME

        text = self.ui.render_unified_diff(_DIFF_PREVIEW)
        spans = [(text.plain[s.start : s.end], str(s.style)) for s in text.spans]
        self.assertIn(("+new", THEME.diff_added), spans)
        self.assertIn(("-old", THEME.diff_removed), spans)
        self.assertIn(("@@ -1,2 +1,2 @@", THEME.info), spans)

    def test_file_headers_pass_through_unstyled(self):
        text = self.ui.render_unified_diff(_DIFF_PREVIEW)
        spans = [(text.plain[s.start : s.end], str(s.style)) for s in text.spans]
        styled_lines = {line for line, _ in spans}
        self.assertNotIn("--- a/calc.py", styled_lines)
        self.assertNotIn("+++ b/calc.py", styled_lines)
        self.assertEqual(text.plain, _DIFF_PREVIEW)

    def test_bash_preview_passes_through_unstyled_and_complete(self):
        text = self.ui.render_unified_diff(_BASH_PREVIEW)
        self.assertEqual(text.plain, _BASH_PREVIEW)
        self.assertEqual(list(text.spans), [])


@unittest.skipUnless(HAS_RICH and UI_HAS_RICH, "rich not installed")
class TestApprovalPanel(unittest.TestCase):
    """The Rich approval panel keeps every fact of the plain prints."""

    def setUp(self):
        from shadow_code.ui import UIRenderer

        self.ui = UIRenderer()

    def test_panel_contains_every_fact_line(self):
        plan = _plan()
        out = _render_panel(self.ui, plan)
        self.assertIn("Action requires approval:", out)
        self.assertIn("edit_file v1.0.0", out)
        self.assertIn("fs.write", out)
        self.assertIn(plan.canonical_arguments_json, out)
        self.assertIn("device=64768", out)
        self.assertIn("inode=1234567", out)
        self.assertIn(f"sha256:{plan.digest()[:16]}", out)
        for line in _DIFF_PREVIEW.split("\n"):
            self.assertIn(line, out)

    def test_panel_contains_bash_preview_complete(self):
        out = _render_panel(self.ui, _plan(preview=_BASH_PREVIEW))
        self.assertIn("sandbox: none available", out)
        self.assertIn("features: pipe redirect", out)

    def test_digest_and_preview_identical_across_rendering(self):
        plan = _plan()
        digest_before = plan.digest()
        preview_before = plan.preview
        _render_panel(self.ui, plan, force_terminal=True)
        self.assertEqual(plan.digest(), digest_before)
        self.assertEqual(plan.preview, preview_before)

    def test_ansi_injected_preview_is_neutralized(self):
        payload = "\x1b]0;pwned\x07\x1b[2J\x1b[31m+evil\x1b[0m\x00"
        plan = _plan(preview=payload)
        out = _render_panel(self.ui, plan, force_terminal=True)
        self.assertNotIn("\x1b]", out)
        self.assertNotIn("pwned", out)
        self.assertNotIn("\x1b[2J", out)
        self.assertNotIn("\x00", out)
        self.assertIn("+evil", out)
        # Stored preview stays raw (digest-bound); sanitization is render-only.
        self.assertEqual(plan.preview, payload)

    def test_no_color_dumb_terminal_emits_no_ansi(self):
        # rich honors NO_COLOR by stripping color SGRs but keeps bold/dim
        # attributes; only a dumb terminal suppresses all escape output.
        # Pin TERM=dumb so the assertion is independent of the runner env.
        with patch.dict(os.environ, {"NO_COLOR": "1", "TERM": "dumb"}):
            buf = io.StringIO()
            console = Console(file=buf, force_terminal=True, width=100)
            self.assertTrue(console.no_color)
            console.print(self.ui.render_approval_panel(_plan()))
        out = buf.getvalue()
        self.assertNotIn("\x1b", out)
        # The +/- prefixes still carry the diff semantics without color.
        self.assertIn("+new", out)
        self.assertIn("-old", out)


class TestRequestApprovalPlainMode(unittest.TestCase):
    """main._request_approval without Rich: byte-identical legacy output."""

    def test_plain_output_byte_identical_to_legacy_prints(self):
        from shadow_code import main

        plan = _plan()
        expected = (
            "Action requires approval:\n"
            f"  tool:       {plan.tool_name} v{plan.tool_version}\n"
            f"  capability: {plan.capability}\n"
            f"  arguments:  {plan.canonical_arguments_json}\n"
            f"  workspace:  device={plan.workspace_device} inode={plan.workspace_inode}\n"
            f"  plan:       sha256:{plan.digest()[:16]}...\n"
            f"  preview:    {plan.preview}\n"
        )
        buf = io.StringIO()
        with (
            patch.object(main, "_RICH", False),
            patch("builtins.input", return_value="n"),
            redirect_stdout(buf),
        ):
            self.assertFalse(main._request_approval(plan))
        self.assertEqual(buf.getvalue(), expected)


class TestRequestApprovalSemantics(unittest.TestCase):
    """Fail-closed consent semantics, unchanged by the panel renderer."""

    def _ask(self, answer):
        from shadow_code import main

        kwargs = {}
        if HAS_RICH and UI_HAS_RICH:
            from shadow_code.ui import UIRenderer

            kwargs = {"console": Console(file=io.StringIO()), "ui": UIRenderer()}
        with patch("builtins.input", answer):
            return main._request_approval(_plan(), **kwargs)

    def test_y_approves(self):
        from unittest.mock import Mock

        self.assertTrue(self._ask(Mock(return_value="y")))

    def test_empty_denies(self):
        from unittest.mock import Mock

        self.assertFalse(self._ask(Mock(return_value="")))

    def test_n_denies(self):
        from unittest.mock import Mock

        self.assertFalse(self._ask(Mock(return_value="n")))

    def test_eof_denies(self):
        from unittest.mock import Mock

        self.assertFalse(self._ask(Mock(side_effect=EOFError)))

    def test_keyboard_interrupt_denies(self):
        from unittest.mock import Mock

        self.assertFalse(self._ask(Mock(side_effect=KeyboardInterrupt)))


if __name__ == "__main__":
    unittest.main()
