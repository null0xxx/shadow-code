# shadow_code/tui_tools.py -- Tool lifecycle view model and renderers (WU-10)
#
# Pure, headless-testable projection of the engine's per-call lifecycle:
# proposals land as rows grouped per engine round ("step n"), rows update IN
# PLACE on status transitions (the transcript re-renders the group region),
# and terminal outcomes keep their evidence -- the preview that was approved
# or denied, the bounded result output, and a one-line failure hint.
#
# Evidence always wins over decoration: every status carries a text label
# that survives NO_COLOR/ASCII, exported mutations are labeled distinctly
# from executed ones, long output collapses with an explicit marker and
# expands in place (never pushing the editor off-screen), and all model/tool
# text is sanitized before it can reach the terminal.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

from wcwidth import wcswidth, wcwidth

from .terminal_text import sanitize_terminal_text
from .theme import THEME

# -- theme tokens -------------------------------------------------------------

# Semantic style tokens consumed by TuiApp._build_style (prompt_toolkit
# class names). NO_COLOR maps every token to "" -- no ANSI at all -- and
# every semantic also has a text label, so nothing depends on color.
THEME_TOKENS: dict[str, str] = {
    "status-ok": "ansigreen",
    "status-pending": "ansiyellow",
    "status-failed": "ansired",
    "status-info": "ansicyan",
    "diff-add": "ansigreen",
    "diff-del": "ansired",
    "diff-hunk": "ansicyan",
    "command": "ansiyellow",
    "md-bold": "bold",
    "md-code": "ansimagenta",
    "approval-frame": "#d77757",
    "assistant-marker": f"bold {THEME.accent}",
    "tips-accent": f"bold {THEME.accent}",
    "user-box": THEME.text_muted,
}


# -- lifecycle statuses ---------------------------------------------------------

STATUS_PROPOSED = "proposed"
STATUS_AWAITING = "awaiting_approval"
STATUS_EXECUTING = "executing"
STATUS_DONE = "done"
STATUS_EXPORTED = "exported"
STATUS_DENIED = "denied"
STATUS_FAILED = "failed"

TERMINAL_STATUSES = frozenset({STATUS_DONE, STATUS_EXPORTED, STATUS_DENIED, STATUS_FAILED})

# glyph, ASCII fallback, text label: the label is the primary evidence and
# survives every theme; the glyph only adds scannability.
_STATUS_TOKENS: dict[str, tuple[str, str, str]] = {
    STATUS_PROPOSED: ("○", "o", "proposed"),
    STATUS_AWAITING: ("●", "?", "awaiting approval"),
    STATUS_EXECUTING: ("▶", ">", "executing"),
    STATUS_DONE: ("✓", "ok", "ok"),
    STATUS_EXPORTED: ("✓", "ok", "exported"),
    STATUS_DENIED: ("✗", "x", "denied"),
    STATUS_FAILED: ("✗", "x", "failed"),
}

# prompt_toolkit style class per status (THEME_TOKENS); NO_COLOR emits none.
_STATUS_STYLES: dict[str, str] = {
    STATUS_PROPOSED: "class:status-pending",
    STATUS_AWAITING: "class:status-pending",
    STATUS_EXECUTING: "class:status-info",
    STATUS_DONE: "class:status-ok",
    STATUS_EXPORTED: "class:status-ok",
    STATUS_DENIED: "class:status-failed",
    STATUS_FAILED: "class:status-failed",
}

_DENIAL_LABELS = {
    "approval_denied": "denied — not retried",
    "policy_denied": "denied by policy",
}

# Typed error code -> one-line recovery suggestion rendered under failed rows.
ERROR_GUIDANCE: dict[str, str] = {
    "no_match": "re-read the file; the exact text changed",
    "ambiguous_match": "include more surrounding context",
    "workspace_drift": "file changed on disk; re-read and retry",
    "readback_mismatch": "written content failed readback; retry the mutation",
    "approval_denied": "denied by user — not retried",
    "approval_invalid": "plan changed — approval rejected; re-approve the new plan",
    "policy_denied": "denied by policy — not retried",
    "budget_exhausted": "turn budget reached; continue in a new turn",
    "duplicate_call": "same call repeated too often; change the arguments",
    "executor_error": "tool crashed; check the arguments and retry",
    "handler_unavailable": "tool has no executable handler in this build",
    "invalid_arguments": "arguments failed validation; fix and retry",
    "invalid_context": "tool received the wrong execution context",
    "invalid_plan": "mutation plan went stale; re-read and re-approve",
    "unknown_tool": "tool is not in the registry",
    "read_error": "file could not be read; check the path",
    "process_error": "command could not be started; check it exists",
    "process_timeout": "command timed out; retry with a smaller scope",
    "containment_violation": "path escapes the workspace; stay inside the root",
    "root_changed": "workspace root changed; restart from a stable directory",
    "correlation_mismatch": "tool returned a mismatched result; report as a bug",
    "io_error": "I/O failure; check the filesystem and retry",
}

# -- bounded display constants --------------------------------------------------

OUTPUT_MAX_LINES = 12  # collapse tool output beyond this many lines
_OUTPUT_HEAD = 4
_OUTPUT_TAIL = 4
PREVIEW_MAX_LINES = 8  # preview lines kept in the transcript after resolution
APPROVAL_PANEL_MAX_HEIGHT = 14  # approval widget never grows past this
SUMMARY_MAX_WIDTH = 96  # display columns kept for a one-line call summary


def display_width(text: str) -> int:
    """Terminal cell width; control chars (already sanitized) count as zero."""
    width = wcswidth(text)
    return max(width, 0)


def clip_display(text: str, max_width: int) -> str:
    """Truncate to a display-width budget, marking the cut with an ellipsis.

    Zero-width and combining characters cost nothing; a lone surrogate or
    unprintable is skipped rather than crashing the render.
    """
    if max_width <= 1 or display_width(text) <= max_width:
        return text
    cells: list[str] = []
    used = 0
    for char in text:
        char_width = wcwidth(char)
        if char_width < 0:
            continue
        if used + char_width > max_width - 1:
            break
        cells.append(char)
        used += char_width
    return "".join(cells) + "…"


def summarize_call(arguments_json: str) -> str:
    """One-line summary of a proposed call: command, path, or clipped JSON.

    Commands and paths are primary evidence and are NEVER clipped -- the
    transcript window wraps them (narrow width buckets condense them at
    render time in _row_head). Only the generic JSON fallback is clipped,
    since the approval panel always shows the full canonical arguments.
    """
    try:
        arguments = json.loads(arguments_json)
    except (TypeError, ValueError):
        arguments = None
    if isinstance(arguments, dict):
        command = arguments.get("command")
        if isinstance(command, str) and command:
            first = command.splitlines()[0]
            summary = f"$ {first}"
            if len(command.splitlines()) > 1:
                summary += " …"
            return summary  # never clipped: the window wraps long commands
        path = arguments.get("file_path") or arguments.get("path")
        if isinstance(path, str) and path:
            return path  # never clipped: the window wraps long paths
    return clip_display(sanitize_terminal_text(arguments_json), SUMMARY_MAX_WIDTH)


# -- rows and groups ---------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallRow:
    """One tool call's lifecycle projection; replaced wholesale on transitions.

    ``expanded`` is view-only: it never changes the stored evidence, just how
    much of a long result the transcript shows at once.
    """

    call_id: str
    tool_name: str
    status: str = STATUS_PROPOSED
    summary_line: str = ""
    preview_text: str = ""
    result_text: str = ""
    error_code: str = ""
    result_truncated: bool = False
    expanded: bool = False


@dataclass
class ToolGroup:
    """The calls of one engine round; rows are frozen, replaced in place."""

    step: int
    rows: list[ToolCallRow] = field(default_factory=list)

    def find(self, call_id: str) -> int:
        for index, row in enumerate(self.rows):
            if row.call_id == call_id:
                return index
        return -1


class ToolLifecycleModel:
    """Groups per engine round plus a call_id index; the transcript truth.

    Rows update in place: a transition replaces the frozen row inside its
    group, and the transcript re-renders the group region on the next
    refresh. The model itself is thread-confined to the app thread (all
    mutations arrive through TuiApp.post).
    """

    def __init__(self) -> None:
        self.groups: list[ToolGroup] = []

    @property
    def current_group(self) -> ToolGroup | None:
        return self.groups[-1] if self.groups else None

    def begin_round(self, step: int) -> tuple[ToolGroup, bool]:
        """Start the group for a round; idempotent on repeated notifications."""
        group = self.current_group
        if group is not None and group.step == step:
            return group, False
        group = ToolGroup(step=step)
        self.groups.append(group)
        return group, True

    def note_proposed(self, call_id: str, tool_name: str, arguments_json: str) -> None:
        group = self.current_group
        if group is None:
            group, _ = self.begin_round(len(self.groups) + 1)
        if group.find(call_id) >= 0:
            return
        group.rows.append(
            ToolCallRow(
                call_id=sanitize_terminal_text(call_id),
                tool_name=sanitize_terminal_text(tool_name),
                summary_line=sanitize_terminal_text(summarize_call(arguments_json)),
            )
        )

    def _replace(self, call_id: str, **changes: Any) -> None:
        for group in reversed(self.groups):
            index = group.find(call_id)
            if index >= 0:
                group.rows[index] = replace(group.rows[index], **changes)
                return

    def note_awaiting_approval(self, call_id: str) -> None:
        self._replace(call_id, status=STATUS_AWAITING)

    def note_executing(self, call_id: str) -> None:
        self._replace(call_id, status=STATUS_EXECUTING)

    def note_preview(self, call_id: str, preview_text: str) -> None:
        """Record the exact preview shown at approval (kept after resolution)."""
        self._replace(call_id, preview_text=sanitize_terminal_text(preview_text))

    def note_result(
        self,
        call_id: str,
        tool_name: str,
        output: str | None,
        error_code: str,
        error_message: str,
    ) -> None:
        """Land the terminal result; the status is derived from the evidence."""
        if error_code:
            status = STATUS_DENIED if error_code in _DENIAL_LABELS else STATUS_FAILED
            result_text = f"[{error_code}] {error_message}"
        else:
            text = output or ""
            status = STATUS_EXPORTED if "status: exported" in text else STATUS_DONE
            result_text = text
        result_text = sanitize_terminal_text(result_text)
        changes: dict[str, Any] = {
            "status": status,
            "result_text": result_text,
            "error_code": sanitize_terminal_text(error_code),
            "result_truncated": len(result_text.splitlines()) > OUTPUT_MAX_LINES,
        }
        for group in reversed(self.groups):
            index = group.find(call_id)
            if index >= 0:
                group.rows[index] = replace(group.rows[index], **changes)
                return
        # Result without a matching proposal (e.g. UI attached late): keep the
        # evidence anyway as a terminal row in a fresh group.
        group, _ = self.begin_round(len(self.groups) + 1)
        group.rows.append(
            ToolCallRow(
                call_id=sanitize_terminal_text(call_id),
                tool_name=sanitize_terminal_text(tool_name),
                **changes,
            )
        )

    def toggle_expand(self) -> bool:
        """Toggle expansion on the most recent truncated row; False if none."""
        for group in reversed(self.groups):
            for index in range(len(group.rows) - 1, -1, -1):
                row = group.rows[index]
                if row.result_truncated:
                    group.rows[index] = replace(row, expanded=not row.expanded)
                    return True
        return False


# -- output collapsing ---------------------------------------------------------------


def collapse_output(
    text: str, max_lines: int = OUTPUT_MAX_LINES, *, expanded: bool = False
) -> tuple[list[str], bool]:
    """Head/tail collapse with an explicit marker; long single lines survive.

    Returns (lines, hidden_marker_shown). Lines are never truncated
    horizontally -- the transcript window wraps them instead.
    """
    lines = text.splitlines()
    if expanded or len(lines) <= max_lines:
        return lines, False
    head = lines[:_OUTPUT_HEAD]
    tail = lines[len(lines) - _OUTPUT_TAIL :]
    hidden = len(lines) - len(head) - len(tail)
    return [*head, f"… {hidden} more lines (Ctrl+E: expand)", *tail], True


# -- group rendering ---------------------------------------------------------------


def _status_label(row: ToolCallRow) -> str:
    if row.status == STATUS_DENIED:
        return _DENIAL_LABELS.get(row.error_code, "denied")
    if row.status == STATUS_FAILED and row.error_code == "approval_invalid":
        return "plan changed — approval rejected"
    return _STATUS_TOKENS[row.status][2]


def _row_head(row: ToolCallRow, theme: Any, width: int) -> str:
    glyph, ascii_glyph, _label = _STATUS_TOKENS[row.status]
    marker = ascii_glyph if theme.ascii_mode else glyph
    head = f"  {marker} {row.tool_name}"
    summary = row.summary_line
    if summary:
        # Narrow buckets condense the summary hint (the full arguments are
        # always in the approval panel); normal widths keep it whole and let
        # the window wrap long paths/commands.
        if width < 40:
            summary = clip_display(summary, max(width - 8, 8))
        head += f"  {summary}"
    return f"{head}  [{_status_label(row)}]"


def render_tool_group(group: ToolGroup, theme: Any, width: int) -> list[str]:
    """Project one round's rows into plain display lines (NO_COLOR safe)."""
    indent = "    " if width >= 40 else "  "
    lines = [f"step {group.step}"]
    for row in group.rows:
        lines.append(_row_head(row, theme, width))
        if row.preview_text:
            preview_lines = row.preview_text.splitlines()
            for line in preview_lines[:PREVIEW_MAX_LINES]:
                lines.append(indent + line)
            if len(preview_lines) > PREVIEW_MAX_LINES:
                remaining = len(preview_lines) - PREVIEW_MAX_LINES
                lines.append(f"{indent}… {remaining} more preview lines")
        if row.result_text:
            shown, _collapsed = collapse_output(row.result_text, expanded=row.expanded)
            for line in shown:
                lines.append(indent + line)
        if row.status in (STATUS_DENIED, STATUS_FAILED):
            hint = ERROR_GUIDANCE.get(row.error_code)
            if hint:
                lines.append(f"{indent}hint: {hint}")
    return lines


def render_tool_group_fragments(group: ToolGroup, theme: Any, width: int) -> list[tuple[str, str]]:
    """Styled variant of render_tool_group for the color transcript.

    Row heads carry their status token's style; preview/result bodies are
    classified as diff lines. With colors disabled every style is "" and the
    text is identical to the plain render.
    """
    plain = render_tool_group(group, theme, width)
    if not theme.colors:
        return [("", "\n".join(plain))]
    head_styles = {_row_head(row, theme, width): _STATUS_STYLES[row.status] for row in group.rows}
    fragments: list[tuple[str, str]] = []
    for index, line in enumerate(plain):
        style = head_styles.get(line, "")
        if not style:
            body = line.lstrip()
            if body.startswith("@@"):
                style = "class:diff-hunk"
            elif body.startswith("+") and not body.startswith("+++"):
                style = "class:diff-add"
            elif body.startswith("-") and not body.startswith("---"):
                style = "class:diff-del"
        suffix = "\n" if index < len(plain) - 1 else ""
        fragments.append((style, line + suffix))
    return fragments


# -- diff / command / markdown renderers -------------------------------------------


def render_diff(preview_text: str, theme: Any) -> list[tuple[str, str]]:
    """Classify unified-diff lines into styled fragments (+/-/hunk markers).

    The +/- prefixes are part of the text itself, so ASCII/NO_COLOR modes
    keep the full semantics without any styling.
    """
    fragments: list[tuple[str, str]] = []
    lines = preview_text.split("\n")
    for index, line in enumerate(lines):
        style = ""
        if theme.colors:
            if line.startswith("@@"):
                style = "class:diff-hunk"
            elif line.startswith("+") and not line.startswith("+++"):
                style = "class:diff-add"
            elif line.startswith("-") and not line.startswith("---"):
                style = "class:diff-del"
        suffix = "\n" if index < len(lines) - 1 else ""
        fragments.append((style, line + suffix))
    return fragments


def render_command(command: str, extra_lines: list[str], theme: Any) -> list[tuple[str, str]]:
    """Bash preview block: `$ command` header plus sandbox/feature lines."""
    fragments: list[tuple[str, str]] = []
    head_style = "class:command" if theme.colors else ""
    info_style = "class:status-info" if theme.colors else ""
    lines = command.split("\n")
    for index, line in enumerate(lines):
        prefix = "$ " if index == 0 else "  "
        fragments.append((head_style, prefix + line + "\n"))
    for line in extra_lines:
        fragments.append((info_style, line + "\n"))
    return fragments


_MD_INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


def render_markdown_lite(text: str, theme: Any) -> list[tuple[str, str]]:
    """Markdown-lite: fenced blocks, **bold**, and `code` spans -> fragments.

    Hand-rolled and intentionally small; the input is already sanitized.
    Without colors the text passes through unchanged (markers stay visible).
    """
    if not theme.colors:
        return [("", text)]
    fragments: list[tuple[str, str]] = []
    in_fence = False
    lines = text.split("\n")
    for line_index, line in enumerate(lines):
        suffix = "\n" if line_index < len(lines) - 1 else ""
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            fragments.append(("class:md-code", line + suffix))
            continue
        if in_fence:
            fragments.append(("class:md-code", line + suffix))
            continue
        parts = _MD_INLINE_RE.split(line)
        for part in parts:
            if part.startswith("**") and part.endswith("**") and len(part) > 4:
                fragments.append(("class:md-bold", part[2:-2]))
            elif part.startswith("`") and part.endswith("`") and len(part) > 2:
                fragments.append(("class:md-code", part[1:-1]))
            else:
                fragments.append(("", part))
        if suffix:
            fragments.append(("", suffix))
    return fragments


# -- approval panel -----------------------------------------------------------------


def render_approval_panel(plan: Any, theme: Any, width: int) -> list[str]:
    """Every approval-bound fact of the action plan, as plain display lines.

    The widget bounds the VISIBLE region (APPROVAL_PANEL_MAX_HEIGHT, arrow
    keys scroll); this projection always contains the complete evidence:
    tool+version, capability, full canonical arguments, workspace identity,
    registry digest, plan digest, execution facts, and the full preview.
    """
    lines = [
        "Action requires approval — review every fact:",
        f"  tool:       {plan.tool_name} v{plan.tool_version}",
        f"  capability: {plan.capability}",
        f"  arguments:  {plan.canonical_arguments_json}",
        f"  workspace:  device={plan.workspace_device} inode={plan.workspace_inode}",
        f"  registry:   sha256:{plan.registry_digest[:16]}…",
        f"  plan:       sha256:{plan.digest()[:16]}…",
    ]
    if plan.execution_facts:
        lines.append(f"  execution:  {plan.execution_facts}")
    return lines


def render_approval_preview_fragments(plan: Any, theme: Any) -> list[tuple[str, str]]:
    """Styled preview section of the approval panel (diff or command block).

    For process executions the command is reconstructed from the canonical
    arguments with the engine's sandbox/feature lines attached; everything
    else renders through the diff classifier. Falls back to the raw preview
    text if the command cannot be parsed -- evidence is never dropped.
    """
    preview = plan.preview.rstrip("\n")
    if not preview:
        return []
    if plan.capability == "process.execute":
        try:
            arguments = json.loads(plan.canonical_arguments_json)
            command = arguments.get("command")
        except (TypeError, ValueError):
            command = None
        if isinstance(command, str) and command:
            extras = [
                line for line in preview.splitlines() if line.startswith(("sandbox:", "features:"))
            ]
            fragments = render_command(sanitize_terminal_text(command), extras, theme)
            rendered = {line for _, text in fragments for line in text.rstrip("\n").split("\n")}
            remainder = [
                line
                for line in preview.splitlines()
                if line not in rendered and not line.startswith("bash v")
            ]
            for line in remainder:
                fragments.append(("", line + "\n"))
            return fragments
    return render_diff(preview, theme)


def approval_panel_fragments(plan: Any, theme: Any, width: int) -> list[tuple[str, str]]:
    """Full approval panel as fragments: fact lines + styled preview + hints."""
    fragments: list[tuple[str, str]] = []
    for line in render_approval_panel(plan, theme, width):
        fragments.append(("", line + "\n"))
    if plan.preview.rstrip("\n"):
        fragments.append(("", "  preview:\n"))
        for style, text in render_approval_preview_fragments(plan, theme):
            fragments.append((style, "    " + text if text != "\n" else text))
    keys = "keys: y approve · n deny (default) · Esc/Ctrl+C cancel · Up/Down scroll"
    if theme.ascii_mode:
        keys = keys.replace("·", "-")
    fragments.append(("class:status-info" if theme.colors else "", keys))
    return fragments
