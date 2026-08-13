# shadow_code/ui.py -- Claude Code inspired terminal UI
#
# Professional CLI rendering with Rich: semantic colors, unicode symbols,
# syntax highlighting, diff view, error panels, progress bars.
# Falls back gracefully when Rich is not installed.

import os

from .theme import ERROR_SUGGESTIONS, SYMBOLS, THEME

try:
    from rich.box import MINIMAL, ROUNDED, SIMPLE
    from rich.console import Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .config import MODEL_NAME
from .terminal_text import sanitize_terminal_text

_t = THEME
_s = SYMBOLS


class UIRenderer:
    """Claude Code-style professional UI rendering."""

    # --- Welcome ---

    def render_welcome(self) -> "Panel":
        content = Text()
        content.append(f"\n  {_s.diamond_filled} ", style=f"bold {_t.brand}")
        content.append("shadow", style=f"bold {_t.brand}")
        content.append("-code", style=f"{_t.text_dim}")
        content.append("  v0.1.0\n", style="dim")
        content.append(f"  {MODEL_NAME}\n", style="dim")
        content.append("  /help for commands\n", style="dim italic")
        return Panel(content, border_style=_t.text_muted, box=ROUNDED, padding=(0, 0))

    # --- Thinking / Streaming ---

    def render_thinking(self, model: str = "") -> "Panel":
        from rich.spinner import Spinner

        text = Text()
        text.append(f" {_s.thinking} ", style=f"bold {_t.thinking}")
        text.append("thinking", style=f"{_t.thinking}")
        text.append("...", style=f"dim {_t.thinking}")
        spinner = Spinner("dots", text=text)
        return Panel(spinner, border_style=_t.text_muted, box=MINIMAL, padding=(0, 1))

    def _dots(self) -> "Text":
        return Text(f"{_s.dot}{_s.dot}{_s.dot}", style=f"{_t.text_dim}")

    def _prefix_text(self, text: str) -> Text:
        """Add ⎿ prefix to each line (Claude Code style)."""
        result = Text()
        for i, line in enumerate(text.splitlines()):
            if i > 0:
                result.append("\n")
            result.append("  \u238f  ", style="dim")
            result.append(line)
        return result

    def render_streaming(self, text: str) -> "Text":
        if not text.strip():
            t = Text()
            t.append("  \u238f  ", style="dim")
            t.append(f"{_s.dot}{_s.dot}{_s.dot}", style="dim")
            return t
        return self._prefix_text(text)

    def render_streaming_with_tokens(self, text: str, estimated_tokens: int) -> "Text":
        """Render streaming with ⎿ prefix and token estimate."""
        result = self.render_streaming(text)
        if isinstance(result, Text):
            result.append(f"\n{'':>50}~{estimated_tokens:,} tokens", style="dim")
        return result

    # --- Response ---

    def render_response(self, text: str, tokens: int = 0) -> "Group":
        """Render the final assistant response as Markdown.

        The text is sanitized BEFORE Markdown construction so injected
        terminal control sequences can never reach the console (storage
        keeps the raw bytes; sanitization is render-time only). The token
        count is a plain Text sibling below the Markdown body.
        """
        body = Markdown(sanitize_terminal_text(text))
        if tokens:
            token_line = Text(f"{'':>50}{tokens:,} tokens", style="dim")
            return Group(body, token_line)
        return Group(body)

    # --- Tool Calls (Claude Code style: ⎿ prefix) ---

    def render_tool_call(self, tool: str, desc: str) -> "Text":
        text = Text()
        text.append("  \u238f  ", style="dim")  # ⎿
        text.append(f"{_s.tool} {tool}", style=f"bold {_t.tool}")
        text.append(f"  {desc}", style="dim")
        return text

    def render_tool_result(
        self, tool: str, output: str, success: bool, params: dict | None = None
    ) -> "Text":
        """Claude Code style: condensed ✓/✗ result with indented output."""
        color = _t.success if success else _t.error
        icon = _s.success if success else _s.error

        result = Text()

        # First line: icon + condensed output
        first_line = output.split("\n")[0][:120] if output else ""
        result.append(f"     {icon} ", style=f"bold {color}")
        result.append(first_line, style="dim")

        # For read_file/bash/grep: show more lines indented
        if tool in ("read_file", "bash", "grep", "multi_read") and output:
            lines = output.splitlines()
            show = min(len(lines), 20)
            if show > 1:
                for line in lines[1:show]:
                    result.append(f"\n       {line}", style="dim")
                if len(lines) > show:
                    remaining = len(lines) - show
                    dots = f"{_s.dot}{_s.dot}{_s.dot}"
                    result.append(
                        f"\n       {dots} [{remaining} more lines]",
                        style="dim",
                    )

        return result

    # --- Diff View ---

    def render_diff(self, old_string: str, new_string: str, file_path: str) -> "Panel":
        """Render a diff view for edit_file operations."""
        diff_text = Text()

        # File path header
        diff_text.append(f"  {file_path}\n", style=f"bold {_t.text_primary}")
        diff_text.append(f"  {_s.line_heavy * 40}\n", style=f"{_t.text_dim}")

        # Removed lines
        for line in old_string.splitlines():
            diff_text.append(f"  {_s.error} ", style=f"{_t.diff_removed}")
            diff_text.append(f"{line}\n", style=f"{_t.diff_removed}")

        # Added lines
        for line in new_string.splitlines():
            diff_text.append(f"  {_s.success} ", style=f"{_t.diff_added}")
            diff_text.append(f"{line}\n", style=f"{_t.diff_added}")

        return Panel(
            diff_text,
            border_style=_t.text_muted,
            box=SIMPLE,
            title=f"[{_t.info}]diff[/{_t.info}]",
            padding=(0, 1),
        )

    # --- Error Display ---

    def render_error(self, message: str) -> "Text":
        """Simple inline error (backward compatible)."""
        text = Text()
        text.append(f"  {_s.error} ", style=f"bold {_t.error}")
        text.append(message, style=f"{_t.error}")
        return text

    def render_error_panel(
        self, message: str, severity: str = "error", suggestion: str = ""
    ) -> "Panel":
        """Structured error panel with severity and suggestions."""
        colors = {
            "error": (_t.error, _s.error),
            "warning": (_t.warning, _s.warning),
            "info": (_t.info, _s.info),
        }
        color, icon = colors.get(severity, colors["error"])

        body = Text()
        body.append(f" {icon} ", style=f"bold {color}")
        body.append(message, style=color)

        # Auto-detect suggestion from known patterns
        if not suggestion:
            for pattern, sug in ERROR_SUGGESTIONS.items():
                if pattern.lower() in message.lower():
                    suggestion = sug
                    break

        if suggestion:
            body.append(f"\n\n  {_s.arrow} ", style=f"bold {_t.success}")
            body.append(suggestion, style=_t.text_secondary)

        return Panel(
            body,
            border_style=color,
            box=ROUNDED,
            title=f"[{color}]{severity.upper()}[/{color}]",
            padding=(0, 1),
        )

    # --- Context Status ---

    def render_context_status(self, used: int, total: int) -> "Text":
        pct = (used / total) * 100 if total else 0
        if pct < 50:
            color = _t.bar_low
        elif pct < 75:
            color = _t.bar_mid
        else:
            color = _t.bar_high

        bar_w = 20
        filled = int(bar_w * pct / 100)
        bar = _s.progress_fill * filled + _s.progress_empty * (bar_w - filled)

        text = Text()
        text.append(f"  [{bar}] ", style=f"{color}")
        text.append(f"{used // 1000}K/{total // 1000}K ", style=f"bold {color}")
        text.append(f"({pct:.0f}%)", style=f"{_t.text_dim}")
        return text

    # --- Help ---

    def render_help(self, commands: list[tuple[str, str]]) -> "Table":
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            show_edge=False,
        )
        table.add_column(style=f"{_t.tool}", no_wrap=True, width=20)
        table.add_column(style=f"{_t.text_dim}")
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        return table

    # --- File Path ---

    def render_file_path(self, path: str) -> "Text":
        """Render a file path with dim directory and bold filename."""
        text = Text()
        dir_part = os.path.dirname(path)
        file_part = os.path.basename(path)
        if dir_part:
            text.append(dir_part + "/", style=f"dim {_t.text_dim}")
        text.append(file_part, style=f"bold {_t.tool}")
        return text
