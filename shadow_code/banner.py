# shadow_code/banner.py -- Startup wordmark and tips block
#
# Pure, headless-testable projection of the visual identity: a full-block
# (U+2588) "SHADOW CODE" wordmark with a dim drop-shadow row, a left-to-right
# color gradient across the theme palette (accent -> info -> highlight), and
# the getting-started tips. ASCII mode (SHADOW_ASCII / NO_COLOR / non-UTF-8)
# falls back to the original #-art. No rendering library is imported here:
# the Rich REPL and the prompt_toolkit TUI each map these primitives onto
# their own style model.

import os

from .theme import THEME, _supports_unicode

# -- wordmark -----------------------------------------------------------------

FULL_BLOCK = "█"  # ink
SHADOW_CHAR = "░"  # drop-shadow row, styled text_muted at render time

# 5x5 block letters, one row per string, pure ASCII so every terminal mode
# (including SHADOW_ASCII) renders the same art.
_FONT: dict[str, tuple[str, ...]] = {
    "S": (" ####", "#    ", " ### ", "    #", "#### "),
    "H": ("#   #", "#   #", "#####", "#   #", "#   #"),
    "A": (" ### ", "#   #", "#####", "#   #", "#   #"),
    "D": ("#### ", "#   #", "#   #", "#   #", "#### "),
    "O": (" ### ", "#   #", "#   #", "#   #", " ### "),
    "W": ("#   #", "#   #", "# # #", "## ##", "#   #"),
    "C": (" ####", "#    ", "#    ", "#    ", " ####"),
    "E": ("#####", "#    ", "#### ", "#    ", "#####"),
}

_WORD = "SHADOW CODE"
_NARROW_TITLE = "SHADOW CODE"


def _render_word(word: str) -> tuple[str, ...]:
    """Assemble the wordmark row by row; the space between words is wider."""
    rows: list[str] = []
    for row_index in range(5):
        parts = []
        for char in word:
            if char == " ":
                parts.append(" ")
            else:
                parts.append(_FONT[char][row_index])
        rows.append(" ".join(parts).rstrip())
    return tuple(rows)


BANNER_ART: tuple[str, ...] = _render_word(_WORD)
BANNER_ASCII_WIDTH: int = max(len(line) for line in BANNER_ART)

# 6-row full-block letters, the primary wordmark. W runs wider (8 cols) so
# its middle stroke stays legible; everything else is 6 cols.
_BLOCK_FONT: dict[str, tuple[str, ...]] = {
    "S": (
        "██████",
        "██    ",
        "██████",
        "    ██",
        "    ██",
        "██████",
    ),
    "H": (
        "██  ██",
        "██  ██",
        "██████",
        "██  ██",
        "██  ██",
        "██  ██",
    ),
    "A": (
        " ████ ",
        "██  ██",
        "██  ██",
        "██████",
        "██  ██",
        "██  ██",
    ),
    "D": (
        "█████ ",
        "██  ██",
        "██  ██",
        "██  ██",
        "██  ██",
        "█████ ",
    ),
    "O": (
        " ████ ",
        "██  ██",
        "██  ██",
        "██  ██",
        "██  ██",
        " ████ ",
    ),
    "W": (
        "██    ██",
        "██    ██",
        "██ ██ ██",
        "████ ████",
        " ██  ██ ",
        " ██  ██ ",
    ),
    "C": (
        " █████",
        "██    ",
        "██    ",
        "██    ",
        "██    ",
        " █████",
    ),
    "E": (
        "██████",
        "██    ",
        "█████ ",
        "██    ",
        "██    ",
        "██████",
    ),
}


def _render_block_word(word: str) -> tuple[str, ...]:
    """Assemble the block wordmark; the space between words is 3 columns."""
    rows: list[str] = []
    for row_index in range(6):
        parts = []
        for char in word:
            if char == " ":
                parts.append("  ")
            else:
                parts.append(_BLOCK_FONT[char][row_index])
        rows.append(" ".join(parts).rstrip())
    return tuple(rows)


_BLOCK_CORE: tuple[str, ...] = _render_block_word(_WORD)

# Extruded look: one shadow row beneath the letters -- the bottom row's
# silhouette shifted one column right, drawn in light shade and styled dim.
_SHADOW_ROW = " " + _BLOCK_CORE[-1].replace(FULL_BLOCK, SHADOW_CHAR)

BLOCK_ART: tuple[str, ...] = (*_BLOCK_CORE, _SHADOW_ROW)
BANNER_WIDTH: int = max(len(line) for line in BLOCK_ART)

TAGLINE = "Local-first agentic coding with explicit authority boundaries"

TIPS: tuple[str, ...] = (
    "Ask questions, edit files, or run commands",
    "Be specific for the best results",
    "Type /help for more information",
)


def _ascii_default() -> bool:
    """ASCII fallback: explicit opt-out, NO_COLOR, or a non-UTF-8 terminal."""
    if os.environ.get("SHADOW_ASCII", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if "NO_COLOR" in os.environ:
        return True
    return not _supports_unicode()


def banner_lines(width: int | None = None, ascii_mode: bool | None = None) -> list[str]:
    """Wordmark lines; a terminal narrower than the art gets a plain title.

    Block art is the default; ASCII mode (or auto-detected NO_COLOR /
    SHADOW_ASCII / non-unicode terminals) keeps the original #-art.
    """
    if ascii_mode is None:
        ascii_mode = _ascii_default()
    art = BANNER_ART if ascii_mode else BLOCK_ART
    art_width = BANNER_ASCII_WIDTH if ascii_mode else BANNER_WIDTH
    if width is not None and width < art_width:
        return [_NARROW_TITLE]
    return list(art)


def tips_lines() -> list[str]:
    """The getting-started block as plain numbered lines."""
    lines = ["Tips for getting started:"]
    lines.extend(f"  {index}. {tip}" for index, tip in enumerate(TIPS, 1))
    return lines


# -- gradient -------------------------------------------------------------------

# Stops reuse the theme palette (no hardcoded identity colors): purple accent
# on the left fading through blue into cyan highlight on the right.
_GRADIENT_STOPS: tuple[str, ...] = (THEME.accent, THEME.info, THEME.highlight)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def gradient_color(t: float) -> str:
    """Hex color at horizontal position t (0.0 left .. 1.0 right)."""
    stops = [_hex_to_rgb(color) for color in _GRADIENT_STOPS]
    segments = len(stops) - 1
    t = min(max(t, 0.0), 1.0)
    segment = min(int(t * segments), segments - 1)
    local = t * segments - segment
    start, end = stops[segment], stops[segment + 1]
    rgb = [round(start[i] + (end[i] - start[i]) * local) for i in range(3)]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def column_color(index: int) -> str:
    """Gradient color for a character column of the full-width wordmark."""
    return gradient_color(index / max(1, BANNER_WIDTH - 1))


# -- prompt_toolkit fragments -----------------------------------------------------


def styled_banner_fragments(
    colors: bool, width: int | None = None, ascii_mode: bool | None = None
) -> list[tuple[str, str]]:
    """(style, text) fragments for the wordmark; gradient only with colors.

    Space characters carry no style, so a terminal background always shows
    through the letter gaps; the shadow row is styled text_muted; without
    colors the art passes through plain.
    """
    fragments: list[tuple[str, str]] = []
    for line_index, line in enumerate(banner_lines(width, ascii_mode)):
        if line_index:
            fragments.append(("", "\n"))
        for column, char in enumerate(line):
            if char == " " or not colors:
                fragments.append(("", char))
            elif char == SHADOW_CHAR:
                fragments.append((THEME.text_muted, char))
            else:
                fragments.append((column_color(column), char))
    return fragments


def styled_tips_fragments(colors: bool) -> list[tuple[str, str]]:
    """(style, text) fragments for the tips block; /help takes the accent."""
    fragments: list[tuple[str, str]] = []
    for line_index, line in enumerate(tips_lines()):
        if line_index:
            fragments.append(("", "\n"))
        if colors and "/help" in line:
            before, _, after = line.partition("/help")
            fragments.append(("", before))
            fragments.append(("class:tips-accent", "/help"))
            fragments.append(("", after))
        else:
            fragments.append(("", line))
    return fragments
