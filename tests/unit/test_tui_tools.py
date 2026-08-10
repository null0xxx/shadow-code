"""Headless tests for the WU-10 tool lifecycle view model and renderers.

Covers: every lifecycle status render (unicode + ASCII), in-place status
transitions grouped per round, exported-vs-executed labels, denial and
stale-plan (approval_invalid) terminal rows, output collapse/expand,
preview bounding, failure guidance, huge paths/commands (wrap, never
truncated away), zero-width and Georgian text, malicious ANSI
neutralization, the approval panel's complete fact set, and the
markdown/diff/command fragment renderers. No test touches a terminal.
"""

import json
import unittest

from shadow_code.domain.approval import ActionPlan
from shadow_code.tui import TuiTheme
from shadow_code.tui_tools import (
    APPROVAL_PANEL_MAX_HEIGHT,
    ERROR_GUIDANCE,
    OUTPUT_MAX_LINES,
    PREVIEW_MAX_LINES,
    STATUS_AWAITING,
    STATUS_DENIED,
    STATUS_DONE,
    STATUS_EXECUTING,
    STATUS_EXPORTED,
    STATUS_FAILED,
    STATUS_PROPOSED,
    THEME_TOKENS,
    ToolCallRow,
    ToolGroup,
    ToolLifecycleModel,
    approval_panel_fragments,
    clip_display,
    collapse_output,
    display_width,
    render_approval_panel,
    render_command,
    render_diff,
    render_markdown_lite,
    render_tool_group,
    render_tool_group_fragments,
    summarize_call,
)

_UNICODE = TuiTheme(colors=True, ascii_mode=False)
_ASCII = TuiTheme(colors=False, ascii_mode=True)


def _plan(**overrides) -> ActionPlan:
    base = {
        "call_id": "c1",
        "tool_name": "bash",
        "tool_version": "1.0.0",
        "capability": "process.execute",
        "canonical_arguments_json": '{"command":"pytest -q"}',
        "workspace_device": 1234,
        "workspace_inode": 5678,
        "registry_digest": "ab" * 32,
        "preview": (
            "bash v1.0.0 [process.execute] arguments=...\n"
            "sandbox: unconfined\n"
            "features: none detected"
        ),
        "execution_facts": "uid=1000 cwd=/tmp",
    }
    base.update(overrides)
    return ActionPlan(**base)


class TestStatuses(unittest.TestCase):
    def test_all_statuses_have_token_label_and_ascii(self):
        from shadow_code.tui_tools import _STATUS_TOKENS

        for status in (
            STATUS_PROPOSED,
            STATUS_AWAITING,
            STATUS_EXECUTING,
            STATUS_DONE,
            STATUS_EXPORTED,
            STATUS_DENIED,
            STATUS_FAILED,
        ):
            glyph, ascii_glyph, label = _STATUS_TOKENS[status]
            self.assertTrue(glyph)
            self.assertTrue(ascii_glyph.isascii())
            self.assertTrue(label)

    def test_render_every_status_unicode_and_ascii(self):
        group = ToolGroup(step=1)
        for index, status in enumerate(
            (
                STATUS_PROPOSED,
                STATUS_AWAITING,
                STATUS_EXECUTING,
                STATUS_DONE,
                STATUS_EXPORTED,
                STATUS_DENIED,
                STATUS_FAILED,
            )
        ):
            group.rows.append(ToolCallRow(call_id=f"c{index}", tool_name="bash", status=status))
        unicode_lines = render_tool_group(group, _UNICODE, 80)
        ascii_lines = render_tool_group(group, _ASCII, 80)
        self.assertEqual(unicode_lines[0], "step 1")
        self.assertEqual(len(unicode_lines), 8)
        for label in (
            "proposed",
            "awaiting approval",
            "executing",
            "ok",
            "exported",
            "denied",
            "failed",
        ):
            self.assertTrue(any(label in line for line in unicode_lines), label)
            self.assertTrue(any(label in line for line in ascii_lines), label)
        # ASCII mode: no unicode glyphs anywhere.
        for line in ascii_lines:
            self.assertNotIn("✓", line)
            self.assertNotIn("✗", line)
            self.assertNotIn("○", line)


class TestLifecycleModel(unittest.TestCase):
    def test_rounds_group_calls_and_rows_update_in_place(self):
        model = ToolLifecycleModel()
        group, is_new = model.begin_round(1)
        self.assertTrue(is_new)
        self.assertIs(model.begin_round(1)[0], group)  # idempotent per round
        self.assertFalse(model.begin_round(1)[1])
        model.note_proposed("c1", "read_file", '{"path":"a.py"}')
        model.note_proposed("c2", "bash", '{"command":"ls"}')
        model.note_awaiting_approval("c2")
        model.note_executing("c2")
        self.assertEqual([r.status for r in group.rows], [STATUS_PROPOSED, STATUS_EXECUTING])
        model.note_result("c2", "bash", "total 0", "", "")
        self.assertEqual(group.rows[1].status, STATUS_DONE)
        self.assertEqual(group.rows[1].result_text, "total 0")
        second, is_new = model.begin_round(2)
        self.assertTrue(is_new)
        self.assertIsNot(second, group)

    def test_summary_lines_for_command_path_and_json(self):
        self.assertEqual(summarize_call('{"command":"pytest -q"}'), "$ pytest -q")
        self.assertEqual(summarize_call('{"file_path":"src/a.py"}'), "src/a.py")
        self.assertEqual(summarize_call('{"path":"src/b.py"}'), "src/b.py")
        multi = summarize_call('{"command":"echo one\\necho two"}')
        self.assertTrue(multi.startswith("$ echo one"))
        self.assertIn("…", multi)
        self.assertIn("x", summarize_call('{"x":1}'))
        self.assertTrue(summarize_call("not json"))

    def test_result_without_proposal_keeps_evidence(self):
        model = ToolLifecycleModel()
        model.note_result("late", "bash", None, "policy_denied", "denied")
        self.assertEqual(len(model.groups), 1)
        row = model.groups[0].rows[0]
        self.assertEqual(row.status, STATUS_DENIED)

    def test_duplicate_proposal_ignored(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "bash", '{"command":"ls"}')
        model.note_proposed("c1", "bash", '{"command":"ls"}')
        self.assertEqual(len(model.groups[0].rows), 1)


class TestResultDerivation(unittest.TestCase):
    def test_exported_vs_executed_labels(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "write_file", '{"path":"a.py"}')
        model.note_proposed("c2", "write_file", '{"path":"b.py"}')
        model.note_result(
            "c1", "write_file", "mutation: write a.py\nstatus: exported\npatch: p", "", ""
        )
        model.note_result("c2", "write_file", "mutation: write b.py\nstatus: executed", "", "")
        lines = render_tool_group(model.groups[0], _UNICODE, 80)
        joined = "\n".join(lines)
        self.assertIn("[exported]", joined)
        exported_row = model.groups[0].rows[0]
        done_row = model.groups[0].rows[1]
        self.assertEqual(exported_row.status, STATUS_EXPORTED)
        self.assertEqual(done_row.status, STATUS_DONE)
        self.assertIn("status: executed", done_row.result_text)

    def test_denial_is_final_and_labeled(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "bash", '{"command":"rm -rf x"}')
        model.note_preview("c1", "bash v1 [process.execute]")
        model.note_result("c1", "bash", None, "approval_denied", "User denied.")
        row = model.groups[0].rows[0]
        self.assertEqual(row.status, STATUS_DENIED)
        lines = render_tool_group(model.groups[0], _UNICODE, 80)
        joined = "\n".join(lines)
        self.assertIn("denied — not retried", joined)
        self.assertIn("hint: denied by user — not retried", joined)
        self.assertIn("bash v1 [process.execute]", joined)  # preview kept

    def test_approval_invalid_renders_stale_plan_label(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "edit_file", '{"path":"a.py"}')
        model.note_result("c1", "edit_file", None, "approval_invalid", "token spent")
        lines = render_tool_group(model.groups[0], _UNICODE, 80)
        joined = "\n".join(lines)
        self.assertIn("plan changed — approval rejected", joined)
        self.assertIn("hint:", joined)

    def test_policy_denied_label(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "bash", '{"command":"ls"}')
        model.note_result("c1", "bash", None, "policy_denied", "no")
        joined = "\n".join(render_tool_group(model.groups[0], _UNICODE, 80))
        self.assertIn("denied by policy", joined)


class TestFailureGuidance(unittest.TestCase):
    def test_guidance_lines_for_known_codes(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        codes = [
            "no_match",
            "ambiguous_match",
            "workspace_drift",
            "approval_denied",
            "budget_exhausted",
        ]
        for index, code in enumerate(codes):
            model.note_proposed(f"c{index}", "edit_file", '{"path":"a"}')
            model.note_result(f"c{index}", "edit_file", None, code, "boom")
        joined = "\n".join(render_tool_group(model.groups[0], _UNICODE, 80))
        self.assertIn("re-read the file; the exact text changed", joined)
        self.assertIn("include more surrounding context", joined)
        self.assertIn("file changed on disk; re-read and retry", joined)
        self.assertIn("denied by user — not retried", joined)
        self.assertIn("turn budget reached; continue in a new turn", joined)

    def test_unknown_code_renders_no_hint(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "bash", "{}")
        model.note_result("c1", "bash", None, "mystery_code", "boom")
        joined = "\n".join(render_tool_group(model.groups[0], _UNICODE, 80))
        self.assertNotIn("hint:", joined)
        self.assertIn("[mystery_code] boom", joined)

    def test_every_engine_code_has_guidance(self):
        for code in (
            "no_match",
            "ambiguous_match",
            "workspace_drift",
            "readback_mismatch",
            "approval_denied",
            "approval_invalid",
            "policy_denied",
            "budget_exhausted",
            "duplicate_call",
            "executor_error",
            "handler_unavailable",
            "invalid_arguments",
            "unknown_tool",
            "process_error",
            "correlation_mismatch",
        ):
            self.assertIn(code, ERROR_GUIDANCE, code)


class TestCollapseExpand(unittest.TestCase):
    def _long_result(self, lines: int) -> str:
        return "\n".join(f"line {index}" for index in range(lines))

    def test_short_output_not_collapsed(self):
        shown, collapsed = collapse_output("a\nb\nc")
        self.assertEqual(shown, ["a", "b", "c"])
        self.assertFalse(collapsed)

    def test_long_output_head_tail_with_marker(self):
        shown, collapsed = collapse_output(self._long_result(30))
        self.assertTrue(collapsed)
        self.assertEqual(shown[0], "line 0")
        self.assertEqual(shown[-1], "line 29")
        marker = [line for line in shown if "more lines" in line]
        self.assertEqual(len(marker), 1)
        self.assertIn("22 more lines", marker[0])
        self.assertIn("Ctrl+E: expand", marker[0])
        self.assertLessEqual(len(shown), OUTPUT_MAX_LINES)

    def test_expanded_shows_everything(self):
        shown, collapsed = collapse_output(self._long_result(30), expanded=True)
        self.assertEqual(len(shown), 30)
        self.assertFalse(collapsed)

    def test_model_marks_and_toggles_truncation(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "bash", "{}")
        model.note_result("c1", "bash", self._long_result(30), "", "")
        row = model.groups[0].rows[0]
        self.assertTrue(row.result_truncated)
        self.assertFalse(row.expanded)
        collapsed = "\n".join(render_tool_group(model.groups[0], _UNICODE, 80))
        self.assertIn("more lines", collapsed)
        self.assertNotIn("line 10", collapsed)
        self.assertTrue(model.toggle_expand())
        expanded = "\n".join(render_tool_group(model.groups[0], _UNICODE, 80))
        self.assertIn("line 10", expanded)
        self.assertNotIn("more lines", expanded)
        self.assertTrue(model.toggle_expand())  # toggles back

    def test_toggle_expand_without_truncated_rows_is_noop(self):
        model = ToolLifecycleModel()
        self.assertFalse(model.toggle_expand())

    def test_preview_bounded_with_marker(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "edit_file", "{}")
        model.note_preview("c1", "\n".join(f"preview {i}" for i in range(30)))
        model.note_result("c1", "edit_file", None, "approval_denied", "no")
        lines = render_tool_group(model.groups[0], _UNICODE, 80)
        preview_lines = [line for line in lines if "preview" in line]
        self.assertLessEqual(len(preview_lines), PREVIEW_MAX_LINES + 1)
        self.assertTrue(any("more preview lines" in line for line in lines))


class TestWidthAndUnicode(unittest.TestCase):
    def test_huge_path_never_truncated_at_normal_width(self):
        huge = "/" + "/".join(f"segment-{index}" for index in range(60))
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "read_file", json.dumps({"path": huge}))
        joined = "\n".join(render_tool_group(model.groups[0], _UNICODE, 80))
        self.assertIn(huge, joined)  # wraps in the window, never truncated away

    def test_huge_command_never_truncated_at_normal_width(self):
        huge = "x" * 500
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "bash", json.dumps({"command": huge}))
        joined = "\n".join(render_tool_group(model.groups[0], _UNICODE, 80))
        self.assertIn(huge, joined)

    def test_narrow_width_condenses_summary_but_keeps_labels(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "bash", '{"command":"' + "x" * 100 + '"}')
        model.note_result("c1", "bash", None, "approval_denied", "no")
        narrow = "\n".join(render_tool_group(model.groups[0], _ASCII, 30))
        self.assertIn("denied — not retried", narrow)
        self.assertNotIn("x" * 100, narrow)
        self.assertIn("…", narrow)

    def test_zero_width_and_georgian_text(self):
        georgian = "აბგდე"
        self.assertEqual(display_width(georgian), 5)
        self.assertGreaterEqual(display_width("a\u200db"), 0)  # ZWJ: no crash
        self.assertEqual(display_width("e\u0301"), 1)  # combining mark
        clipped = clip_display(georgian + "xyz" * 40, 8)
        self.assertLessEqual(display_width(clipped), 8)
        self.assertTrue(clipped.endswith("…"))
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "bash", json.dumps({"command": f"echo {georgian}\u200d"}))
        model.note_result("c1", "bash", f"output {georgian} \u200d done", "", "")
        joined = "\n".join(render_tool_group(model.groups[0], _UNICODE, 20))
        self.assertTrue(joined)  # renders without crashing at tiny width


class TestSanitizationInLifecycle(unittest.TestCase):
    def test_malicious_ansi_in_output_neutralized(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        evil = "\x1b[2J\x1b[H\x1b]0;owned\x07real output\x1b[31m"
        model.note_proposed("c1", "bash", '{"command":"echo hi"}')
        model.note_result("c1", "bash", evil, "", "")
        joined = "\n".join(render_tool_group(model.groups[0], _UNICODE, 80))
        self.assertNotIn("\x1b", joined)
        self.assertIn("real output", joined)

    def test_malicious_ansi_in_tool_name_and_preview(self):
        model = ToolLifecycleModel()
        model.begin_round(1)
        model.note_proposed("c1", "ba\x1b[31msh", "{}")
        model.note_preview("c1", "diff\x1b[2J\n+ok")
        joined = "\n".join(render_tool_group(model.groups[0], _UNICODE, 80))
        self.assertNotIn("\x1b", joined)
        self.assertIn("bash", joined)


class TestApprovalPanel(unittest.TestCase):
    def test_panel_shows_every_plan_fact_and_digest(self):
        plan = _plan()
        lines = render_approval_panel(plan, _ASCII, 80)
        joined = "\n".join(lines)
        self.assertIn("bash v1.0.0", joined)
        self.assertIn("process.execute", joined)
        self.assertIn('{"command":"pytest -q"}', joined)
        self.assertIn("device=1234", joined)
        self.assertIn("inode=5678", joined)
        self.assertIn(f"sha256:{'ab' * 8}", joined)
        self.assertIn(f"sha256:{plan.digest()[:16]}", joined)
        self.assertIn("uid=1000 cwd=/tmp", joined)  # execution facts

    def test_panel_fragments_include_preview_and_keys(self):
        plan = _plan()
        fragments = approval_panel_fragments(plan, _UNICODE, 80)
        text = "".join(t for _s, t in fragments)
        self.assertIn("preview:", text)
        self.assertIn("sandbox: unconfined", text)
        self.assertIn("y approve", text)
        self.assertIn("n deny", text)
        self.assertIn("Esc", text)

    def test_bash_preview_renders_command_header(self):
        plan = _plan()
        fragments = approval_panel_fragments(plan, _UNICODE, 80)
        text = "".join(t for _s, t in fragments)
        self.assertIn("$ pytest -q", text)
        self.assertIn("features: none detected", text)

    def test_mutation_preview_renders_through_diff_classifier(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new"
        plan = _plan(
            tool_name="edit_file",
            capability="filesystem.write",
            canonical_arguments_json='{"path":"f.py"}',
            preview=f"edit_file v1 [filesystem.write]\nmutation: edit f.py\n{diff}",
            execution_facts="",
        )
        fragments = approval_panel_fragments(plan, _UNICODE, 80)
        styles = {style for style, text in fragments if style}
        self.assertIn("class:diff-add", styles)
        self.assertIn("class:diff-del", styles)
        text = "".join(t for _s, t in fragments)
        self.assertIn("+new", text)
        self.assertIn("-old", text)

    def test_panel_height_bounded_constant(self):
        self.assertLessEqual(APPROVAL_PANEL_MAX_HEIGHT, 20)
        long_preview = "\n".join(f"line {i}" for i in range(200))
        plan = _plan(preview=long_preview)
        fragments = approval_panel_fragments(plan, _ASCII, 80)
        text = "".join(t for _s, t in fragments)
        # Overflow is never dropped from the projection; the WIDGET bounds
        # the visible region (APPROVAL_PANEL_MAX_HEIGHT) and scrolls.
        self.assertIn("line 199", text)

    def test_execution_facts_omitted_when_empty(self):
        plan = _plan(execution_facts="")
        joined = "\n".join(render_approval_panel(plan, _ASCII, 80))
        self.assertNotIn("execution:", joined)


class TestFragmentRenderers(unittest.TestCase):
    def test_render_diff_styles(self):
        fragments = render_diff("@@ -1 +1 @@\n-old\n+new\n--- a/x\n+++ b/x\n ctx", _UNICODE)
        styles = {text: style for style, text in fragments}
        self.assertEqual(styles["@@ -1 +1 @@\n"], "class:diff-hunk")
        self.assertEqual(styles["-old\n"], "class:diff-del")
        self.assertEqual(styles["+new\n"], "class:diff-add")
        self.assertEqual(styles["--- a/x\n"], "")  # file headers not colored
        self.assertEqual(styles["+++ b/x\n"], "")
        plain = render_diff("+a\n-b", _ASCII)
        self.assertEqual(plain, [("", "+a\n"), ("", "-b")])

    def test_render_command(self):
        fragments = render_command("pytest -q", ["sandbox: unconfined", "features: x"], _UNICODE)
        text = "".join(t for _s, t in fragments)
        self.assertIn("$ pytest -q", text)
        self.assertIn("sandbox: unconfined", text)
        self.assertIn("features: x", text)
        styles = {style for style, _t in fragments}
        self.assertIn("class:command", styles)

    def test_render_markdown_lite(self):
        fragments = render_markdown_lite("normal **bold** and `code`\n```\nfenced\n```", _UNICODE)
        text = "".join(t for _s, t in fragments)
        self.assertIn("bold", text)
        self.assertNotIn("**bold**", text)
        self.assertIn("code", text)
        self.assertIn("fenced", text)
        styles = {style for style, _t in fragments}
        self.assertIn("class:md-bold", styles)
        self.assertIn("class:md-code", styles)
        plain = render_markdown_lite("**bold** `code`", _ASCII)
        self.assertEqual(plain, [("", "**bold** `code`")])

    def test_group_fragments_match_plain_without_colors(self):
        group = ToolGroup(step=1)
        group.rows.append(ToolCallRow(call_id="c1", tool_name="bash", summary_line="$ ls"))
        plain = render_tool_group(group, _ASCII, 80)
        fragments = render_tool_group_fragments(group, _ASCII, 80)
        self.assertEqual(fragments, [("", "\n".join(plain))])

    def test_group_fragments_style_status_heads(self):
        group = ToolGroup(step=1)
        group.rows.append(
            ToolCallRow(call_id="c1", tool_name="bash", status=STATUS_DONE, summary_line="$ ls")
        )
        fragments = render_tool_group_fragments(group, _UNICODE, 80)
        head = [f for f in fragments if "bash" in f[1] and "[" in f[1]]
        self.assertEqual(head[0][0], "class:status-ok")

    def test_theme_tokens_cover_used_classes(self):
        used = {
            "status-ok",
            "status-pending",
            "status-failed",
            "status-info",
            "diff-add",
            "diff-del",
            "diff-hunk",
            "command",
            "md-bold",
            "md-code",
        }
        self.assertLessEqual(used, set(THEME_TOKENS))


if __name__ == "__main__":
    unittest.main()
