"""Tests for display.py -- streaming buffer that hides tool_call blocks."""

import io
import sys
import unittest

from shadow_code.display import TAG_END, TAG_START, StreamDisplay
from shadow_code.terminal_text import split_trailing_partial_escape


class TestStreamDisplayBasic(unittest.TestCase):
    """Basic StreamDisplay behavior."""

    def setUp(self):
        self.display = StreamDisplay()
        self.display.reset()

    def test_reset_clears_state(self):
        self.display.full_response = "leftover"
        self.display.buffer = "leftover"
        self.display.buffering = True
        self.display.reset()
        self.assertEqual(self.display.full_response, "")
        self.assertEqual(self.display.buffer, "")
        self.assertFalse(self.display.buffering)

    def test_plain_text_passes_through(self):
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            self.display.feed("Hello world")
            self.display.flush()
        finally:
            sys.stdout = old
        self.assertIn("Hello world", captured.getvalue())

    def test_full_response_accumulates(self):
        self.display.feed("Hello ")
        self.display.feed("world")
        self.assertEqual(self.display.get_full_response(), "Hello world")

    def test_tool_call_hidden_from_output(self):
        tool_block = '```tool_call\n{"tool": "bash", "params": {"command": "ls"}}\n```'
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            self.display.feed(tool_block)
            self.display.flush()
        finally:
            sys.stdout = old
        # tool_call block should NOT appear in printed output
        self.assertNotIn("tool_call", captured.getvalue())
        self.assertNotIn('"tool"', captured.getvalue())

    def test_tool_call_in_full_response(self):
        tool_block = '```tool_call\n{"tool": "bash", "params": {"command": "ls"}}\n```'
        self.display.feed(tool_block)
        # Full response should contain the tool call
        self.assertIn("tool_call", self.display.get_full_response())

    def test_text_before_tool_call_shown(self):
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            self.display.feed('Let me check.\n```tool_call\n{"tool": "bash", "params": {}}\n```')
            self.display.flush()
        finally:
            sys.stdout = old
        output = captured.getvalue()
        self.assertIn("Let me check.", output)
        self.assertNotIn('"tool"', output)

    def test_text_after_tool_call_in_full_response(self):
        self.display.feed('```tool_call\n{"tool": "bash", "params": {}}\n```\nDone!')
        self.display.flush()
        # The full response captures everything regardless of display
        full = self.display.get_full_response()
        self.assertIn("Done!", full)


class TestStreamDisplayChunked(unittest.TestCase):
    """Test with chunks split across feed() calls."""

    def setUp(self):
        self.display = StreamDisplay()
        self.display.reset()

    def test_chunked_tool_call(self):
        """Tool call split across multiple chunks should still be hidden."""
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            self.display.feed("```tool_")
            self.display.feed('call\n{"tool": "bash", ')
            self.display.feed('"params": {"command": "ls"}}\n')
            self.display.feed("```")
            self.display.flush()
        finally:
            sys.stdout = old
        self.assertNotIn('"tool"', captured.getvalue())

    def test_normal_code_block_not_hidden(self):
        """A ```python block should NOT be hidden."""
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            self.display.feed('```python\nprint("hello")\n```')
            self.display.flush()
        finally:
            sys.stdout = old
        output = captured.getvalue()
        self.assertIn("python", output)
        self.assertIn("hello", output)


class TestStreamDisplayEdgeCases(unittest.TestCase):
    """Edge cases for StreamDisplay."""

    def setUp(self):
        self.display = StreamDisplay()
        self.display.reset()

    def test_unclosed_tool_call_swallowed(self):
        """Unclosed tool_call block is swallowed on flush."""
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            self.display.feed('```tool_call\n{"tool": "bash", "params": {}}')
            # No closing ```
            self.display.flush()
        finally:
            sys.stdout = old
        self.assertNotIn("tool_call", captured.getvalue())

    def test_empty_feed(self):
        self.display.feed("")
        self.assertEqual(self.display.get_full_response(), "")

    def test_multiple_tool_calls_in_full_response(self):
        text = (
            "First.\n"
            '```tool_call\n{"tool": "bash", "params": {"command": "pwd"}}\n```\n'
            "Middle.\n"
            '```tool_call\n{"tool": "bash", "params": {"command": "ls"}}\n```\n'
            "Last."
        )
        self.display.feed(text)
        self.display.flush()
        full = self.display.get_full_response()
        # Full response contains everything
        self.assertIn("First.", full)
        self.assertIn("Middle.", full)
        self.assertIn("Last.", full)
        self.assertIn('"tool"', full)


class TestStreamDisplayConstants(unittest.TestCase):
    """Test constants are correct."""

    def test_tag_start(self):
        self.assertEqual(TAG_START, "```tool_call")

    def test_tag_end(self):
        self.assertEqual(TAG_END, "```")


class TestSplitPartial(unittest.TestCase):
    """Test the _split_partial helper."""

    def setUp(self):
        self.display = StreamDisplay()

    def test_no_partial(self):
        safe, held = self.display._split_partial("hello world")
        self.assertEqual(safe, "hello world")
        self.assertEqual(held, "")

    def test_partial_backtick(self):
        safe, held = self.display._split_partial("hello `")
        self.assertEqual(safe, "hello ")
        self.assertEqual(held, "`")

    def test_partial_double_backtick(self):
        safe, held = self.display._split_partial("hello ``")
        self.assertEqual(safe, "hello ")
        self.assertEqual(held, "``")

    def test_partial_triple_backtick(self):
        safe, held = self.display._split_partial("hello ```")
        self.assertEqual(safe, "hello ")
        self.assertEqual(held, "```")


class TestFindClosingBackticks(unittest.TestCase):
    """Test the _find_closing_backticks helper."""

    def setUp(self):
        self.display = StreamDisplay()

    def test_closing_at_start(self):
        result = self.display._find_closing_backticks("```\n")
        self.assertEqual(result, 0)

    def test_closing_after_newline(self):
        result = self.display._find_closing_backticks("some json\n```\n")
        self.assertIsNotNone(result)

    def test_no_closing(self):
        result = self.display._find_closing_backticks("just text")
        self.assertIsNone(result)

    def test_opening_not_closing(self):
        # ```python is an opening, not a closing
        result = self.display._find_closing_backticks("```python\n")
        self.assertIsNone(result)


class TestStreamDisplaySanitization(unittest.TestCase):
    """Terminal-bound writes are sanitized; full_response stays raw."""

    def setUp(self):
        self.display = StreamDisplay()
        self.display.reset()

    def _capture(self, chunks):
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            for chunk in chunks:
                self.display.feed(chunk)
            self.display.flush()
        finally:
            sys.stdout = old
        return captured.getvalue()

    def test_control_sequences_stripped_from_terminal_output(self):
        raw = "Hello \x1b[31mred\x1b[0m \x1b]0;evil title\x07world\x07\x0b!"
        output = self._capture([raw])
        self.assertEqual(output, "Hello red world!")
        # Stored response is byte-identical to the raw model output.
        self.assertEqual(self.display.get_full_response(), raw)

    def test_escape_sequence_split_across_chunks(self):
        cases = [
            (["hello\x1b[31", "mworld"], "helloworld"),  # CSI split
            (["a\x1b]0;ti", "tle\x07b"], "ab"),  # OSC split, BEL terminator
            (["a\x1b]0;x\x1b", "\\b"], "ab"),  # OSC split, ST terminator
            (["a\x1b", "[2Kb"], "ab"),  # bare ESC at chunk boundary
            (["a\x1b[1;", "2 Hb"], "ab"),  # CSI params + intermediate split
        ]
        for chunks, expected in cases:
            with self.subTest(chunks=chunks):
                self.display.reset()
                output = self._capture(chunks)
                self.assertEqual(output, expected)
                self.assertEqual(self.display.get_full_response(), "".join(chunks))

    def test_fence_hiding_with_sanitization(self):
        block = '```tool_call\n{"tool": "bash", "params": {"command": "ls \x1b[2K"}}\n```'
        output = self._capture(["ok\x1b[31m\n" + block, "\ndone\x07"])
        self.assertEqual(output, "ok\ndone")
        full = self.display.get_full_response()
        self.assertIn('"tool"', full)
        self.assertIn("\x1b[2K", full)

    def test_normal_text_unchanged_byte_for_byte(self):
        chunks = ["Hello, ", "wor", "ld!\nSecond ", "line `", "code`."]
        output = self._capture(chunks)
        self.assertEqual(output, "".join(chunks))

    def test_flush_writes_held_buffer_sanitized(self):
        # "`" is held back as a potential ```tool_call prefix until flush.
        output = self._capture(["abc`"])
        self.assertEqual(output, "abc`")

    def test_flush_resolves_dangling_escape(self):
        # Stream ends mid-sequence: the held fragment goes through the
        # sanitizer, matching whole-text sanitization of the same bytes.
        output = self._capture(["tail\x1b[31"])
        self.assertEqual(output, "tail[31")
        self.assertEqual(self.display.get_full_response(), "tail\x1b[31")

    def test_reset_clears_held_escape(self):
        self.display.feed("a\x1b[31")
        self.display.reset()
        output = self._capture(["b"])
        self.assertEqual(output, "b")


class TestSplitTrailingPartialEscape(unittest.TestCase):
    """Test the split_trailing_partial_escape helper."""

    def test_no_escape(self):
        self.assertEqual(split_trailing_partial_escape("hello"), ("hello", ""))

    def test_complete_sequence_not_held(self):
        self.assertEqual(split_trailing_partial_escape("a\x1b[31m"), ("a\x1b[31m", ""))

    def test_trailing_bare_esc_held(self):
        self.assertEqual(split_trailing_partial_escape("a\x1b"), ("a", "\x1b"))

    def test_trailing_partial_csi_held(self):
        self.assertEqual(split_trailing_partial_escape("a\x1b[1;2"), ("a", "\x1b[1;2"))

    def test_trailing_partial_osc_held(self):
        self.assertEqual(split_trailing_partial_escape("a\x1b]0;title"), ("a", "\x1b]0;title"))

    def test_trailing_osc_with_st_prefix_held(self):
        # The trailing ESC may be the first half of the ST terminator.
        self.assertEqual(split_trailing_partial_escape("a\x1b]0;t\x1b"), ("a", "\x1b]0;t\x1b"))


if __name__ == "__main__":
    unittest.main()
