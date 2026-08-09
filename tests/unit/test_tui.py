"""Headless tests for the persistent TUI (WU-09).

The view models (TranscriptModel, FooterModel, sanitize_terminal_text,
render_footer, width buckets, themes) are pure and tested directly. The
TuiApp is tested through its key-binding handlers and, for clean exit,
through a full headless run with prompt_toolkit's pipe input + dummy
output. No test touches a real terminal.
"""

import tempfile
import threading
import time
import unittest
from pathlib import Path

from shadow_code.domain.approval import ApprovalAuthority
from shadow_code.domain.policy import PolicyFacts
from shadow_code.main import SessionRuntime
from shadow_code.policy.engine import PolicyEngine
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.status_bar import SessionState
from shadow_code.tool_context import ToolContext
from shadow_code.tools.catalog import WorkspaceContext
from shadow_code.tools.registry import ToolRegistry
from shadow_code.tui import (
    FooterModel,
    TranscriptModel,
    TuiApp,
    TuiTheme,
    render_footer,
    sanitize_terminal_text,
    width_bucket,
)

_UNICODE = TuiTheme(colors=True, ascii_mode=False)
_ASCII = TuiTheme(colors=False, ascii_mode=True)


class TestSanitizeTerminalText(unittest.TestCase):
    def test_plain_text_unchanged(self):
        self.assertEqual(sanitize_terminal_text("hello world"), "hello world")

    def test_newlines_and_tabs_survive(self):
        self.assertEqual(sanitize_terminal_text("a\nb\tc"), "a\nb\tc")

    def test_csi_color_sequence_removed(self):
        self.assertEqual(sanitize_terminal_text("\x1b[31mred\x1b[0m"), "red")

    def test_csi_cursor_move_removed(self):
        self.assertEqual(sanitize_terminal_text("ab\x1b[2Ac"), "abc")

    def test_osc_with_bel_terminator_removed(self):
        self.assertEqual(sanitize_terminal_text("\x1b]0;evil title\x07done"), "done")

    def test_osc_with_st_terminator_removed(self):
        self.assertEqual(sanitize_terminal_text("\x1b]52;c;payload\x1b\\done"), "done")

    def test_lone_escape_and_carriage_return_removed(self):
        self.assertEqual(sanitize_terminal_text("a\x1bb\rc"), "abc")

    def test_c1_controls_removed(self):
        self.assertEqual(sanitize_terminal_text("a\x9bb"), "ab")

    def test_delete_and_bell_removed(self):
        self.assertEqual(sanitize_terminal_text("a\x07b\x7fc"), "abc")


class TestTranscriptModel(unittest.TestCase):
    def test_entries_keep_append_order(self):
        model = TranscriptModel()
        model.append_user("hi")
        model.append_assistant_delta("Hel")
        model.append_assistant_delta("lo")
        model.append_tool_line("read_file", "ok")
        model.append_system_line("[Budget exhausted]")
        kinds = [entry.kind for entry in model.entries]
        self.assertEqual(kinds, ["user", "assistant", "tool", "system"])
        self.assertEqual(model.entries[1].text, "Hello")

    def test_assistant_delta_after_tool_line_starts_new_entry(self):
        model = TranscriptModel()
        model.append_assistant_delta("one")
        model.append_tool_line("bash", "approval_denied")
        model.append_assistant_delta("two")
        self.assertEqual([e.text for e in model.entries], ["one", "[bash] approval_denied", "two"])

    def test_empty_delta_ignored(self):
        model = TranscriptModel()
        model.append_assistant_delta("")
        self.assertEqual(model.entries, ())

    def test_tool_line_format(self):
        model = TranscriptModel()
        model.append_tool_line("bash", "approval_denied")
        self.assertEqual(model.entries[0].text, "[bash] approval_denied")

    def test_user_text_is_sanitized(self):
        model = TranscriptModel()
        model.append_user("\x1b[31mhi\x1b[0m")
        self.assertEqual(model.entries[0].text, "hi")

    def test_clear(self):
        model = TranscriptModel()
        model.append_system_line("x")
        model.clear()
        self.assertEqual(model.entries, ())

    def test_render_unicode_user_prefix(self):
        model = TranscriptModel()
        model.append_user("hello")
        model.append_assistant_delta("world")
        self.assertEqual(model.render(_UNICODE), "› hello\nworld")

    def test_render_ascii_user_prefix(self):
        model = TranscriptModel()
        model.append_user("hello")
        self.assertEqual(model.render(_ASCII), "> hello")

    def test_render_multiline_indents_continuation(self):
        model = TranscriptModel()
        model.append_user("a\nb")
        self.assertEqual(model.render(_ASCII), "> a\n  b")


class TestFooterModel(unittest.TestCase):
    def _model(self, **overrides) -> FooterModel:
        base = {
            "model_name": "shadow-gemma:latest",
            "tokens_used": 10_000,
            "tokens_total": 131_072,
            "workspace_root": "/home/user/project",
            "snapshot_digest": "abcdef123456",
            "permission_labels": ("bash:UNCONFINED", "mutation:export"),
            "state": "idle",
        }
        base.update(overrides)
        return FooterModel(**base)

    def test_context_pct(self):
        self.assertAlmostEqual(self._model().context_pct, 10_000 / 131_072 * 100)

    def test_context_pct_zero_total(self):
        self.assertEqual(self._model(tokens_total=0).context_pct, 0.0)

    def test_full_bucket_shows_everything(self):
        lines = render_footer(self._model(), 120)
        self.assertEqual(len(lines), 2)
        joined = "\n".join(lines)
        for label in (
            "shadow-gemma:latest",
            "10K/131K",
            "ctx 8%",
            "/home/user/project",
            "snap abcdef123456",
            "bash:UNCONFINED",
            "mutation:export",
            "idle",
            "Enter: send",
            "Ctrl+C: cancel",
            "Ctrl+D: exit",
        ):
            self.assertIn(label, joined)

    def test_standard_bucket_condenses(self):
        lines = render_footer(self._model(), 80)
        self.assertEqual(len(lines), 2)
        self.assertIn("bash:UNCONFINED", lines[0])
        self.assertIn("idle", lines[0])
        self.assertIn("snap abcdef123456", lines[1])

    def test_compact_bucket_single_line_with_labels(self):
        lines = render_footer(self._model(), 40)
        self.assertEqual(len(lines), 1)
        self.assertIn("shadow-gemma:latest", lines[0])
        self.assertIn("ctx 8%", lines[0])
        self.assertIn("idle", lines[0])

    def test_tiny_bucket_minimal_single_line(self):
        lines = render_footer(self._model(), 39)
        self.assertEqual(len(lines), 1)
        self.assertIn("shadow-gemma:latest", lines[0])
        self.assertIn("8%", lines[0])

    def test_ascii_mode_uses_pipe_separator(self):
        lines = render_footer(self._model(), 120, ascii_mode=True)
        self.assertNotIn("│", lines[0])
        self.assertIn("|", lines[0])

    def test_widths_119_79_fall_to_next_bucket(self):
        self.assertEqual(len(render_footer(self._model(), 119)), 2)
        self.assertIn("bash:UNCONFINED", render_footer(self._model(), 119)[0])
        self.assertEqual(len(render_footer(self._model(), 79)), 1)
        self.assertIn("ctx 8%", render_footer(self._model(), 79)[0])


class TestWidthBucket(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(width_bucket(120), "full")
        self.assertEqual(width_bucket(119), "standard")
        self.assertEqual(width_bucket(80), "standard")
        self.assertEqual(width_bucket(79), "compact")
        self.assertEqual(width_bucket(40), "compact")
        self.assertEqual(width_bucket(39), "tiny")
        self.assertEqual(width_bucket(0), "tiny")


class TestTuiTheme(unittest.TestCase):
    def test_default_theme(self):
        theme = TuiTheme.from_env({})
        self.assertTrue(theme.colors)
        self.assertFalse(theme.ascii_mode)

    def test_no_color_disables_colors_and_forces_ascii(self):
        theme = TuiTheme.from_env({"NO_COLOR": "1"})
        self.assertFalse(theme.colors)
        self.assertTrue(theme.ascii_mode)

    def test_ascii_flag_keeps_colors(self):
        theme = TuiTheme.from_env({"SHADOW_ASCII": "1"})
        self.assertTrue(theme.colors)
        self.assertTrue(theme.ascii_mode)


class TuiAppCase(unittest.TestCase):
    """Headless TuiApp with a minimal but real engine-ready runtime."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        workspace = Path(self._tmp.name)
        self.guard = WorkspaceGuard(str(workspace))
        self.addCleanup(self.guard.close)
        self.rt = SessionRuntime(
            cwd=str(workspace),
            ctx=ToolContext(str(workspace)),
            workspace_guard=self.guard,
            registry=ToolRegistry(()),
            policy_engine=PolicyEngine(PolicyFacts(frozenset(), self.guard.identity)),
            execution_context=WorkspaceContext(
                guard=self.guard,
                workspace_root=str(workspace),
                process_env={},
                sandbox_label="unconfined",
            ),
            approval_authority=ApprovalAuthority(),
            state=SessionState(model_name="test-model", tokens_total=1000),
            permission_labels=("bash:UNCONFINED",),
        )

    def _app(self, **kwargs) -> TuiApp:
        from prompt_toolkit.output import DummyOutput

        kwargs.setdefault("theme", _ASCII)
        kwargs.setdefault("output", DummyOutput())
        return TuiApp(self.rt, **kwargs)


class TestTuiAppLayout(TuiAppCase):
    def test_boot_lines_land_in_transcript(self):
        app = self._app(boot_lines=["[bash runs UNCONFINED]"])
        self.assertIn("[bash runs UNCONFINED]", app.model.render(_ASCII))

    def test_input_area_has_focus(self):
        app = self._app()
        self.assertIs(app.app.layout.current_control, app.input_area.control)

    def test_input_content_survives_layout_queries(self):
        # Resize behavior: prompt_toolkit re-renders on width change; the
        # buffer and focus are untouched by footer/layout recomputation.
        app = self._app()
        app.input_area.text = "partial draft"
        footer_before = app._footer_text()
        footer_after = app._footer_text()
        self.assertEqual(app.input_area.text, "partial draft")
        self.assertIs(app.app.layout.current_control, app.input_area.control)
        self.assertEqual(footer_before, footer_after)

    def test_slash_completion_suggests_help(self):
        from prompt_toolkit.document import Document

        app = self._app()
        completer = app.input_area.completer
        completions = completer.get_completions(Document("/he"), None)
        self.assertIn("/help", [c.text for c in completions])

    def test_footer_model_labels(self):
        app = self._app()
        footer = app._footer_model()
        self.assertEqual(footer.model_name, "test-model")
        self.assertEqual(footer.workspace_root, self._tmp.name)
        self.assertEqual(footer.permission_labels, ("bash:UNCONFINED",))
        self.assertEqual(footer.state, "idle")
        app._busy = True
        self.assertEqual(app._footer_model().state, "busy")
        app._busy = False
        app._approval_pending = True
        self.assertEqual(app._footer_model().state, "approval")


class TestTuiAppKeys(TuiAppCase):
    def test_enter_queues_text_and_clears_buffer(self):
        app = self._app()
        app.input_area.text = "hello tui"
        app._on_enter()
        self.assertEqual(app.input_area.text, "")
        self.assertEqual(app._inputs.get_nowait(), "hello tui")

    def test_enter_empty_does_nothing(self):
        app = self._app()
        app.input_area.text = "   "
        app._on_enter()
        self.assertTrue(app._inputs.empty())

    def test_multiline_text_queued_intact(self):
        app = self._app()
        app.input_area.text = "line one\nline two"
        app._on_enter()
        self.assertEqual(app._inputs.get_nowait(), "line one\nline two")

    def test_ctrl_c_while_busy_sets_cancel_flag(self):
        app = self._app()
        app._busy = True
        app._on_cancel()
        self.assertTrue(self.rt.interrupted)
        self.assertIn("[Cancelled]", app.model.render(_ASCII))

    def test_ctrl_c_while_idle_clears_input(self):
        app = self._app()
        app.input_area.text = "draft"
        app._on_cancel()
        self.assertEqual(app.input_area.text, "")
        self.assertFalse(self.rt.interrupted)

    def test_ctrl_d_idle_requests_exit(self):
        app = self._app()
        app._on_exit()
        self.assertIsNone(app._inputs.get_nowait())

    def test_ctrl_d_busy_cancels_instead_of_exiting(self):
        app = self._app()
        app._busy = True
        app._on_exit()
        self.assertTrue(self.rt.interrupted)
        self.assertTrue(app._inputs.empty())


class TestApprovalBridge(TuiAppCase):
    def _answer_in_thread(self, app: TuiApp, keys: str) -> bool:
        result: list[bool] = []

        def ask() -> None:
            result.append(app._ask_approval("Action requires approval:\n  tool: bash"))

        worker = threading.Thread(target=ask)
        worker.start()
        # No event loop is running, so post() applies inline: the bridge
        # sets _approval_pending synchronously before it blocks.
        for _ in range(1000):
            if app._approval_pending:
                break
            time.sleep(0.001)
        self.assertTrue(app._approval_pending)
        app.input_area.text = keys
        app._on_enter()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        return result[0]

    def test_y_approves(self):
        app = self._app()
        self.assertTrue(self._answer_in_thread(app, "y"))
        self.assertIn("[Approved]", app.model.render(_ASCII))

    def test_n_denies(self):
        app = self._app()
        self.assertFalse(self._answer_in_thread(app, "n"))
        self.assertIn("[Denied]", app.model.render(_ASCII))

    def test_garbage_denies(self):
        app = self._app()
        self.assertFalse(self._answer_in_thread(app, "sure why not"))

    def test_empty_denies(self):
        app = self._app()
        self.assertFalse(self._answer_in_thread(app, ""))

    def test_ctrl_c_denies_pending_approval(self):
        app = self._app()
        result: list[bool] = []

        def ask() -> None:
            result.append(app._ask_approval("prompt"))

        worker = threading.Thread(target=ask)
        worker.start()
        for _ in range(1000):
            if app._approval_pending:
                break
            time.sleep(0.001)
        app._on_cancel()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [False])

    def test_approval_prompt_appears_in_transcript(self):
        app = self._app()
        self._answer_in_thread(app, "n")
        rendered = app.model.render(_ASCII)
        self.assertIn("Action requires approval:", rendered)
        self.assertIn("Approve this exact action? [y/N]", rendered)


class TestCleanExit(TuiAppCase):
    def test_pipe_input_exit_runs_and_restores(self):
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        app = None
        with create_pipe_input() as pipe:
            pipe.send_text("/exit\r")
            app = TuiApp(
                self.rt, theme=_ASCII, boot_lines=["boot"], input=pipe, output=DummyOutput()
            )
            app.run()  # returns only after a clean app.exit()
        self.assertIn("boot", app.model.render(_ASCII))
        # prompt_toolkit owns the terminal; run() returning means the
        # application tore down cleanly (raw mode restored).


if __name__ == "__main__":
    unittest.main()
