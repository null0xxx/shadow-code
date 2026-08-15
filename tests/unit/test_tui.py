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
    git_branch,
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

    def test_render_unicode_user_boxed(self):
        model = TranscriptModel()
        model.append_user("hello")
        model.append_assistant_delta("world")
        self.assertEqual(
            model.render(_UNICODE),
            "╭─────────╮\n│ > hello │\n╰─────────╯\n⏺ world",
        )

    def test_render_ascii_user_boxed(self):
        model = TranscriptModel()
        model.append_user("hello")
        self.assertEqual(model.render(_ASCII), "+---------+\n| > hello |\n+---------+")

    def test_render_multiline_user_boxed(self):
        model = TranscriptModel()
        model.append_user("a\nb")
        self.assertEqual(model.render(_ASCII), "+-----+\n| > a |\n|   b |\n+-----+")

    def test_user_box_fragments_style_only_the_border(self):
        model = TranscriptModel()
        model.append_user("hi")
        fragments = model.render_fragments(_UNICODE, 80)
        border_styles = [text for style, text in fragments if style == "class:user-box"]
        self.assertTrue(any("╭" in text for text in border_styles))
        self.assertTrue(any("╯" in text for text in border_styles))
        # The user's own words stay unstyled inside the box.
        self.assertIn(("", "> hi"), fragments)

    def test_user_box_no_color_is_plain(self):
        model = TranscriptModel()
        model.append_user("hi")
        self.assertEqual(
            model.render_fragments(_ASCII, 80),
            [("", "+------+\n| > hi |\n+------+")],
        )

    def test_assistant_marker_fragments(self):
        model = TranscriptModel()
        model.append_assistant_delta("answer")
        fragments = model.render_fragments(_UNICODE, 80)
        self.assertEqual(fragments[0], ("class:assistant-marker", "⏺ "))
        self.assertIn(("", "answer"), fragments)

    def test_assistant_multiline_continuation_aligned(self):
        model = TranscriptModel()
        model.append_assistant_delta("one\ntwo")
        fragments = model.render_fragments(_UNICODE, 80)
        plain = "".join(text for _style, text in fragments)
        self.assertEqual(plain, "⏺ one\n  two")


class TestTranscriptModelThinking(unittest.TestCase):
    """SHADOW_THINK: thinking rows are dim, sanitized, and collapsible."""

    def test_thinking_delta_accumulates_sanitized_into_one_row(self):
        model = TranscriptModel()
        model.append_thinking_delta("pon\x1b[31mder")
        model.append_thinking_delta("ing")
        self.assertEqual(len(model.entries), 1)
        self.assertEqual(model.entries[0].kind, "thinking")
        self.assertEqual(model.entries[0].text, "pondering")

    def test_thinking_row_is_separate_from_assistant_row(self):
        model = TranscriptModel()
        model.append_thinking_delta("reasoning")
        model.append_assistant_delta("answer")
        self.assertEqual([entry.kind for entry in model.entries], ["thinking", "assistant"])
        self.assertEqual(model.entries[1].text, "answer")

    def test_collapse_thinking_replaces_body_with_summary(self):
        model = TranscriptModel()
        model.append_thinking_delta("long hidden reasoning")
        model.collapse_thinking("* thought for 1.2s (~5 tokens)")
        self.assertEqual(len(model.entries), 1)
        self.assertEqual(model.entries[0].text, "* thought for 1.2s (~5 tokens)")
        self.assertNotIn("hidden", model.render(_ASCII))

    def test_collapse_thinking_without_row_is_noop(self):
        model = TranscriptModel()
        model.append_assistant_delta("answer")
        model.collapse_thinking("* thought for 0.1s")
        self.assertEqual([entry.kind for entry in model.entries], ["assistant"])

    def test_thinking_renders_dim_fragment_with_colors(self):
        model = TranscriptModel()
        model.append_thinking_delta("why")
        fragments = model.render_fragments(_UNICODE)
        self.assertIn(("class:thinking", "why"), fragments)

    def test_thinking_no_color_renders_plain(self):
        model = TranscriptModel()
        model.append_thinking_delta("why")
        self.assertEqual(model.render(_ASCII), "why")
        fragments = model.render_fragments(_ASCII)
        self.assertEqual(fragments, [("", "why")])


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

    def test_git_branch_shown_next_to_workspace(self):
        lines = render_footer(self._model(git_branch="feat/visual-identity"), 120)
        self.assertIn("/home/user/project (feat/visual-identity)", lines[0])
        standard = render_footer(self._model(git_branch="main"), 80)
        self.assertIn("(main)", standard[1])
        compact = render_footer(self._model(git_branch="main"), 40)
        self.assertNotIn("(main)", compact[0])  # narrow buckets drop the workspace

    def test_no_branch_by_default(self):
        joined = "\n".join(render_footer(self._model(), 120))
        self.assertNotIn("(", joined)

    def test_widths_119_79_fall_to_next_bucket(self):
        self.assertEqual(len(render_footer(self._model(), 119)), 2)
        self.assertIn("bash:UNCONFINED", render_footer(self._model(), 119)[0])
        self.assertEqual(len(render_footer(self._model(), 79)), 1)
        self.assertIn("ctx 8%", render_footer(self._model(), 79)[0])


class TestGitBranch(unittest.TestCase):
    def test_branch_from_head_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = Path(tmp) / ".git"
            git.mkdir()
            (git / "HEAD").write_text("ref: refs/heads/feat/x\n", encoding="utf-8")
            self.assertEqual(git_branch(tmp), "feat/x")

    def test_detached_head_shows_short_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = Path(tmp) / ".git"
            git.mkdir()
            (git / "HEAD").write_text("9bd80a1" + "0" * 33 + "\n", encoding="utf-8")
            self.assertEqual(git_branch(tmp), "9bd80a1")

    def test_worktree_gitdir_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_git = Path(tmp) / "real-git"
            real_git.mkdir()
            (real_git / "HEAD").write_text("ref: refs/heads/wt\n", encoding="utf-8")
            (Path(tmp) / ".git").write_text(f"gitdir: {real_git}\n", encoding="utf-8")
            self.assertEqual(git_branch(tmp), "wt")

    def test_no_repo_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(git_branch(tmp), "")
        self.assertEqual(git_branch("/nonexistent/path"), "")


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

    def test_boot_starts_with_banner_and_tips(self):
        app = self._app(boot_lines=["[authority line]"])
        rendered = app.model.render(_ASCII)
        self.assertIn("####", rendered)  # wordmark art
        self.assertIn("Tips for getting started:", rendered)
        self.assertIn("/help", rendered)
        # Order: banner -> tips -> authority boot lines.
        self.assertLess(rendered.index("####"), rendered.index("Tips for getting started:"))
        self.assertLess(
            rendered.index("Tips for getting started:"), rendered.index("[authority line]")
        )

    def test_boot_banner_gradient_fragments_when_colored(self):
        theme = TuiTheme(colors=True, ascii_mode=False)
        app = self._app(theme=theme)
        fragments = app.model.render_fragments(theme, 80)
        hex_styles = {style for style, _t in fragments if style.startswith("#")}
        self.assertGreater(len(hex_styles), 10)  # a real gradient, not one flat color
        self.assertIn(("class:tips-accent", "/help"), fragments)

    def test_boot_banner_no_color_is_plain(self):
        app = self._app()  # _ASCII theme: colors off
        fragments = app.model.render_fragments(_ASCII, 80)
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0][0], "")
        self.assertIn("####", fragments[0][1])

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


class TestApprovalWidget(TuiAppCase):
    """WU-10 single-focus approval control; the WU-09 bridge evolved here."""

    def _plan(self, call_id: str = "c1", **overrides):
        from shadow_code.domain.approval import ActionPlan

        base = {
            "call_id": call_id,
            "tool_name": "bash",
            "tool_version": "1.0.0",
            "capability": "process.execute",
            "canonical_arguments_json": '{"command":"echo hi"}',
            "workspace_device": 1,
            "workspace_inode": 2,
            "registry_digest": "cd" * 32,
            "preview": "bash v1.0.0 [process.execute]\nsandbox: unconfined\nfeatures: none",
            "execution_facts": "uid=1000",
        }
        base.update(overrides)
        return ActionPlan(**base)

    def _run_consent(self, app: TuiApp, plan, answer: bool | None) -> bool:
        """Run _consent on a worker; resolve (or Ctrl+C) from the test thread."""
        result: list[bool] = []
        worker = threading.Thread(target=lambda: result.append(app._consent(plan)))
        worker.start()
        # No event loop is running, so post() applies inline: the widget
        # arms synchronously before the bridge blocks.
        for _ in range(1000):
            if app._approval_pending:
                break
            time.sleep(0.001)
        self.assertTrue(app._approval_pending)
        # Single focus: the panel owns the keys, the input area does not.
        self.assertIs(app.app.layout.current_control, app._approval_control)
        self.assertEqual(app._footer_model().state, "approval")
        if answer is None:
            app._on_cancel()  # Ctrl+C == deny
        else:
            app._resolve_approval(answer)
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        return result[0]

    def test_y_approves_and_restores_focus(self):
        app = self._app()
        self.assertTrue(self._run_consent(app, self._plan(), True))
        self.assertFalse(app._approval_pending)
        self.assertIs(app.app.layout.current_control, app.input_area.control)
        self.assertEqual(app._footer_model().state, "idle")

    def test_n_denies(self):
        app = self._app()
        self.assertFalse(self._run_consent(app, self._plan(), False))

    def test_ctrl_c_denies(self):
        app = self._app()
        self.assertFalse(self._run_consent(app, self._plan(), None))

    def test_ctrl_d_denies_pending_approval(self):
        app = self._app()
        result: list[bool] = []
        worker = threading.Thread(target=lambda: result.append(app._consent(self._plan())))
        worker.start()
        for _ in range(1000):
            if app._approval_pending:
                break
            time.sleep(0.001)
        app._on_exit()
        worker.join(timeout=5)
        self.assertEqual(result, [False])

    def test_enter_is_noop_while_pending(self):
        app = self._app()
        result: list[bool] = []
        worker = threading.Thread(target=lambda: result.append(app._consent(self._plan())))
        worker.start()
        for _ in range(1000):
            if app._approval_pending:
                break
            time.sleep(0.001)
        app.input_area.text = "y"  # typed text must NOT leak past the panel
        app._on_enter()
        self.assertTrue(app._inputs.empty())
        self.assertTrue(app._approval_pending)
        app._resolve_approval(False)
        worker.join(timeout=5)
        self.assertEqual(result, [False])

    def test_panel_shows_every_plan_fact(self):
        app = self._app()
        plan = self._plan()
        result: list[bool] = []
        worker = threading.Thread(target=lambda: result.append(app._consent(plan)))
        worker.start()
        for _ in range(1000):
            if app._approval_pending:
                break
            time.sleep(0.001)
        content = app._approval_content()
        text = "".join(t for _s, t in content) if isinstance(content, list) else content
        for fact in (
            "bash v1.0.0",
            "process.execute",
            '{"command":"echo hi"}',
            "device=1",
            "inode=2",
            f"sha256:{'cd' * 8}",
            f"sha256:{plan.digest()[:16]}",
            "uid=1000",
            "sandbox: unconfined",
            "y approve",
            "n deny",
        ):
            self.assertIn(fact, text, fact)
        app._resolve_approval(False)
        worker.join(timeout=5)

    def test_sequential_approvals_are_fresh_controls(self):
        """Batch of two side-effecting calls: two consents, no inheritance."""
        app = self._app()
        first = self._run_consent(app, self._plan("c1"), True)
        second = self._run_consent(app, self._plan("c2"), False)
        self.assertTrue(first)
        self.assertFalse(second)  # the first answer was not reused
        self.assertEqual(app._approval_count, 2)

    def test_consent_records_preview_on_the_row(self):
        app = self._app()
        plan = self._plan()
        app.model.tool_round(1)
        app.model.tool_proposed("c1", "bash", '{"command":"echo hi"}')
        self.assertFalse(self._run_consent(app, plan, False))
        row = app.model.tools.groups[0].rows[0]
        self.assertIn("sandbox: unconfined", row.preview_text)

    def test_freeform_confirm_uses_same_control(self):
        app = self._app()
        result: list[bool] = []
        worker = threading.Thread(
            target=lambda: result.append(app._ask_approval("  Proceed? (y/n): "))
        )
        worker.start()
        for _ in range(1000):
            if app._approval_pending:
                break
            time.sleep(0.001)
        self.assertIn("Proceed?", app._approval_content())
        app._resolve_approval(True)
        worker.join(timeout=5)
        self.assertEqual(result, [True])

    def test_panel_scroll_keys_clamped(self):
        app = self._app()
        keys = {binding.keys for binding in app.app.key_bindings.bindings}
        self.assertIn(("up",), keys)
        self.assertIn(("down",), keys)
        self.assertIn(("y",), keys)
        self.assertIn(("n",), keys)


class TestTranscriptLifecycle(TuiAppCase):
    """Engine CallEvent stream -> grouped, in-place transcript rows."""

    def _event(self, stage, call_id="", tool_name="", arguments_json="", step=0, result=None):
        from shadow_code.engine import CallEvent

        return CallEvent(
            stage=stage,
            call_id=call_id,
            tool_name=tool_name,
            arguments_json=arguments_json,
            step=step,
            result=result,
        )

    def test_full_round_lifecycle_renders_grouped_rows(self):
        from shadow_code.domain.tools import ToolError, ToolResult

        app = self._app()
        app._on_call_event(self._event("round", step=1))
        app._on_call_event(self._event("proposed", "c1", "read_file", '{"path":"a.py"}', 1))
        app._on_call_event(self._event("proposed", "c2", "bash", '{"command":"ls"}', 1))
        app._on_call_event(self._event("executing", "c1"))
        app._on_call_event(
            self._event(
                "result",
                "c1",
                "read_file",
                result=ToolResult(call_id="c1", tool_name="read_file", output="contents"),
            )
        )
        app._on_call_event(self._event("awaiting_approval", "c2"))
        app._on_call_event(
            self._event(
                "result",
                "c2",
                "bash",
                result=ToolResult(
                    call_id="c2",
                    tool_name="bash",
                    error=ToolError(code="approval_denied", message="denied"),
                ),
            )
        )
        rendered = app.model.render(_ASCII, 80)
        self.assertIn("step 1", rendered)
        self.assertIn("read_file", rendered)
        self.assertIn("a.py", rendered)
        self.assertIn("ok", rendered)
        self.assertIn("$ ls", rendered)
        self.assertIn("denied — not retried", rendered)
        self.assertIn("hint:", rendered)

    def test_expand_toggle_via_model(self):
        from shadow_code.domain.tools import ToolResult

        app = self._app()
        app._on_call_event(self._event("round", step=1))
        app._on_call_event(self._event("proposed", "c1", "bash", '{"command":"ls"}', 1))
        output = "\n".join(f"out {i}" for i in range(30))
        app._on_call_event(
            self._event(
                "result",
                "c1",
                "bash",
                result=ToolResult(call_id="c1", tool_name="bash", output=output),
            )
        )
        collapsed = app.model.render(_ASCII, 80)
        self.assertIn("more lines (Ctrl+E: expand)", collapsed)
        self.assertNotIn("out 10", collapsed)
        app.model.tools.toggle_expand()
        expanded = app.model.render(_ASCII, 80)
        self.assertIn("out 10", expanded)

    def test_clear_resets_tools(self):
        app = self._app()
        app._on_call_event(self._event("round", step=1))
        app._on_call_event(self._event("proposed", "c1", "bash", "{}", 1))
        app.model.clear()
        self.assertEqual(app.model.entries, ())
        self.assertEqual(app.model.tools.groups, [])

    def test_assistant_markdown_fragments(self):
        app = self._app(theme=TuiTheme(colors=True, ascii_mode=False))
        app.model.append_assistant_delta("plain **bold** `code`")
        fragments = app.model.render_fragments(app.theme, 80)
        styles = {style for style, _t in fragments if style}
        self.assertIn("class:md-bold", styles)
        self.assertIn("class:md-code", styles)
        self.assertIn("**bold** `code`", app.model.render(app.theme, 80))

    def test_approval_window_height_bounded(self):
        from prompt_toolkit.layout.containers import ConditionalContainer, HSplit

        from shadow_code.tui_tools import APPROVAL_PANEL_MAX_HEIGHT

        app = self._app()
        root = app.app.layout.container
        self.assertIsInstance(root, HSplit)
        conditional = [c for c in root.children if isinstance(c, ConditionalContainer)]
        self.assertEqual(len(conditional), 1)  # panel hidden unless pending
        height = app._approval_window.height
        self.assertEqual(height.max, APPROVAL_PANEL_MAX_HEIGHT)


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
