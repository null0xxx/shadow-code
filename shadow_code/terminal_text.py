# shadow_code/terminal_text.py -- terminal control-sequence sanitization
#
# Dependency-free (stdlib only) so every render path -- line REPL, TUI,
# streaming preview -- can neutralize model/tool output without pulling in
# wcwidth, prompt_toolkit, or rich. Sanitization happens at RENDER time
# only; stored event payloads stay byte-identical to raw model output.

from __future__ import annotations

import re

_OSC_RE = re.compile("\x1b\\][^\x07\x1b]*(?:\x07|\x1b\\\\)")
_CSI_RE = re.compile("\x1b\\[[0-?]*[ -/]*[@-~]")
# Remaining C0 controls (incl. stray ESC and CR) and C1 controls; \n and
# \t are legitimate transcript content and survive.
_CTRL_RE = re.compile("[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]")


def sanitize_terminal_text(text: str) -> str:
    """Neutralize terminal control sequences in model/tool output.

    OSC (title/set-clipboard) and CSI (colors, cursor moves) sequences are
    removed entirely; any remaining control character except newline and
    tab is stripped. The result can never inject terminal control into the
    transcript, regardless of what a provider or a tool produced.
    """
    text = _OSC_RE.sub("", text)
    text = _CSI_RE.sub("", text)
    return _CTRL_RE.sub("", text)
