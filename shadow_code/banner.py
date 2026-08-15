# shadow_code/banner.py -- Startup wordmark and tips block
#
# Pure, headless-testable projection of the visual identity: a block-letter
# "SHADOW CODE" ASCII wordmark, a left-to-right color gradient across the
# theme palette (accent -> info -> highlight), and the getting-started tips.
# No rendering library is imported here: the Rich REPL and the prompt_toolkit
# TUI each map these primitives onto their own style model, so NO_COLOR and
# ASCII modes degrade to the same plain text everywhere.

from .theme import THEME

# -- wordmark -----------------------------------------------------------------

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
BANNER_WIDTH: int = max(len(line) for line in BANNER_ART)

TAGLINE = "Local-first agentic coding with explicit authority boundaries"

TIPS: tuple[str, ...] = (
    "Ask questions, edit files, or run commands",
    "Be specific for the best results",
    "Type /help for more information",
)


def banner_lines(width: int | None = None) -> list[str]:
    """Wordmark lines; a terminal narrower than the art gets a plain title."""
    if width is not None and width < BANNER_WIDTH:
        return [_NARROW_TITLE]
    return list(BANNER_ART)


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


def styled_banner_fragments(colors: bool, width: int | None = None) -> list[tuple[str, str]]:
    """(style, text) fragments for the wordmark; gradient only with colors.

    Space characters carry no style, so a terminal background always shows
    through the letter gaps; without colors the art passes through plain.
    """
    fragments: list[tuple[str, str]] = []
    for line_index, line in enumerate(banner_lines(width)):
        if line_index:
            fragments.append(("", "\n"))
        for column, char in enumerate(line):
            if char == " " or not colors:
                fragments.append(("", char))
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
