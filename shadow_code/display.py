# shadow_code/display.py -- Streaming buffer for terminal output
#
# With native tool calling (Gemma 4+), tool calls come via API metadata,
# not in the text stream. This module still handles text buffering and
# full response collection. Legacy ```tool_call hiding is kept for
# backward compatibility with models that don't support native tools.

import json
import re
import sys

from .terminal_text import sanitize_terminal_text, split_trailing_partial_escape

# The opening marker: three backticks followed by "tool_call"
TAG_START = "```tool_call"
# The closing marker: three backticks on their own line (after tool_call block)
TAG_END = "```"

# Candidate fences for the fake tool-call notice: ```json or untagged blocks.
# ```tool_call fences are excluded on purpose -- the legacy protocol-error
# path (main._legacy_markdown_protocol_error) already reports those, and
# double-reporting is a bug.
_TEXT_TOOL_CALL_FENCE_RE = re.compile(
    r"^```(?:json)?[ \t]*\r?\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def _looks_like_tool_call(value: object) -> bool:
    """True for {"tool_call": ...} or {"tool": ..., "params"/"arguments": ...}."""
    if not isinstance(value, dict):
        return False
    if "tool_call" in value:
        return True
    if not isinstance(value.get("tool"), str):
        return False
    return isinstance(value.get("params"), dict) or isinstance(value.get("arguments"), dict)


def detect_text_tool_call_fence(text: str) -> int:
    """Count fenced blocks that look like tool calls emitted as plain text.

    Only ```json and untagged fences are inspected; the body must parse as a
    JSON dict with a "tool_call" key (or a {"tool", "params"/"arguments"}
    shape), or a list of such dicts -- each matching dict counts once.
    Malformed JSON is ignored. The result is purely observational: it feeds a
    display notice and never the parser, and the text is never mutated.
    """
    count = 0
    for match in _TEXT_TOOL_CALL_FENCE_RE.finditer(text):
        try:
            body = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(body, list):
            count += sum(1 for item in body if _looks_like_tool_call(item))
        elif _looks_like_tool_call(body):
            count += 1
    return count


class StreamDisplay:
    """Streaming display buffer that hides tool_call blocks from terminal output.

    Usage:
        display = StreamDisplay()
        display.reset()
        for chunk in stream:
            display.feed(chunk)
        display.flush()
        full_text = display.get_full_response()
    """

    def __init__(self):
        self.buffer = ""
        self.buffering = False  # True when inside a ```tool_call block
        self.full_response = ""  # Accumulates ALL chunks (including hidden ones)
        self._esc_held = ""  # Trailing partial escape sequence held back from the terminal

    def reset(self):
        """Reset state for a new streaming response."""
        self.buffer = ""
        self.buffering = False
        self.full_response = ""
        self._esc_held = ""

    def feed(self, chunk: str):
        """Process a streaming chunk. Shows text to user, hides tool_call blocks."""
        self.full_response += chunk

        if self.buffering:
            # We are inside a ```tool_call block -- buffer everything, don't show
            self.buffer += chunk
            # Look for the closing ``` that ends the tool_call block.
            # The closing ``` must appear after the opening line.
            # We skip the opening ```tool_call line itself when searching for the end.
            # Find first newline after TAG_START to skip past the opening line.
            first_newline = self.buffer.find("\n")
            if first_newline == -1:
                # Haven't even finished the opening line yet
                return
            rest = self.buffer[first_newline + 1 :]
            # Look for closing ``` -- it will be on its own line (possibly with whitespace)
            close_pos = self._find_closing_backticks(rest)
            if close_pos is not None:
                # Found the closing ```. Everything after closing ``` + newline is shown.
                after_close = rest[close_pos:]
                # The closing ``` may be followed by a newline and more text
                end_of_close = after_close.find("\n")
                if end_of_close != -1:
                    remaining = after_close[end_of_close + 1 :]
                    if remaining:
                        self._write_terminal(remaining)
                else:
                    # closing ``` is at the very end, no trailing text
                    pass
                self.buffering = False
                self.buffer = ""
            return

        # Not currently buffering -- look for ```tool_call in the combined text
        combined = self.buffer + chunk
        idx = combined.find(TAG_START)

        if idx != -1:
            # Found ```tool_call -- show everything before it, start buffering
            before = combined[:idx]
            if before:
                self._write_terminal(before)
            self.buffering = True
            self.buffer = combined[idx:]
            return

        # Check for partial match at the end of combined (TAG_START split across chunks)
        # We need to hold back characters that could be the start of ```tool_call
        safe, held = self._split_partial(combined)
        if safe:
            self._write_terminal(safe)
        self.buffer = held

    def _find_closing_backticks(self, text: str) -> int | None:
        """Find closing ``` in text that ends a tool_call block.

        The closing ``` must be at the start of a line (possibly with leading whitespace).
        Returns the position of the ``` in text, or None if not found.
        """
        pos = 0
        while pos < len(text):
            idx = text.find("```", pos)
            if idx == -1:
                return None
            # Check that this ``` is at the start of a line (or start of text)
            if idx == 0 or text[idx - 1] == "\n":
                # Make sure this isn't another opening like ```python (just closing ```)
                after = text[idx + 3 :]
                # Closing ``` is followed by nothing, whitespace, or newline
                if not after or after[0] in ("\n", "\r", " ", "\t"):
                    return idx
                # If followed by a letter, it's a new code block opening, not our close
                # Skip past it
                pos = idx + 3
            else:
                pos = idx + 3
        return None

    def _split_partial(self, text: str) -> tuple[str, str]:
        """Split text into safe-to-print and held-back portions.

        We hold back the end of text if it could be the beginning of TAG_START.
        For example, if text ends with "``" that could be the start of "```tool_call".
        """
        # Check progressively longer suffixes of text against prefixes of TAG_START
        max_check = min(len(TAG_START), len(text))
        for i in range(max_check, 0, -1):
            if TAG_START.startswith(text[-i:]):
                return text[:-i], text[-i:]
        return text, ""

    def _write_terminal(self, text: str, *, final: bool = False):
        """Write text to the terminal, sanitized at the write point.

        full_response keeps the raw model bytes; this is the single choke
        point where terminal-bound output is neutralized. A trailing partial
        escape sequence is held back (self._esc_held) until the next write so
        a sequence split across chunks is stripped whole instead of leaking
        its remains as visible garbage. With final=True (end of stream) the
        hold-back is resolved through the sanitizer instead, which matches
        whole-text sanitization of the same bytes.
        """
        if self._esc_held:
            text = self._esc_held + text
            self._esc_held = ""
        if not final:
            text, self._esc_held = split_trailing_partial_escape(text)
        if text:
            sys.stdout.write(sanitize_terminal_text(text))
            sys.stdout.flush()

    def flush(self):
        """Flush any remaining buffer to the terminal.

        Called at the end of streaming. If we're still buffering (model produced
        an unclosed ```tool_call block), we flush it so the parser can still
        extract it from full_response.
        """
        if self.buffer and not self.buffering:
            # Not inside a tool call block -- safe to print remaining buffer.
            # final=True resolves any held-back partial escape through the
            # sanitizer instead of holding it further.
            self._write_terminal(self.buffer, final=True)
        elif self._esc_held:
            # Swallowing an unclosed tool_call block; still resolve a dangling
            # held-back escape fragment through the sanitizer.
            self._write_terminal("", final=True)
        # If buffering is True, we swallow the incomplete tool call block
        # (the parser will handle it from full_response)
        self.buffer = ""

    def get_full_response(self) -> str:
        """Return the complete response text (including hidden tool_call blocks)."""
        return str(self.full_response)
