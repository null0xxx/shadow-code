"""Tests for tool modules: read_file, write_file, edit_file, glob, grep, list_dir."""

import os
import tempfile
import unittest
from unittest.mock import patch

from shadow_code.tool_context import ToolContext


class TestGetLanguageRulesTool(unittest.TestCase):
    """Tests for GetLanguageRulesTool defensive execution."""

    def setUp(self):
        from shadow_code.tools.get_language_rules import GetLanguageRulesTool

        self.tool = GetLanguageRulesTool(ToolContext("/tmp"))

    @patch(
        "shadow_code.tools.get_language_rules.is_rules_root_available",
        return_value=True,
    )
    def test_execute_rejects_missing_rule_selector(self, _rules_available):
        result = self.tool.execute({})

        self.assertFalse(result.success)
        self.assertIn("Provide either 'extension' or 'name'", result.output)


class TestReadFileTool(unittest.TestCase):
    """Tests for ReadFileTool."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        from shadow_code.tools.read_file import ReadFileTool

        self.tool = ReadFileTool(self.ctx)

    def _write(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_validate_missing_path(self):
        self.assertIsNotNone(self.tool.validate({}))

    def test_validate_relative_path(self):
        self.assertIsNotNone(self.tool.validate({"file_path": "relative.py"}))

    def test_validate_valid(self):
        self.assertIsNone(self.tool.validate({"file_path": "/tmp/test.py"}))

    def test_read_file(self):
        path = self._write("hello.txt", "line1\nline2\nline3\n")
        result = self.tool.execute({"file_path": path})
        self.assertTrue(result.success)
        self.assertIn("line1", result.output)
        self.assertIn("line2", result.output)

    def test_read_with_line_numbers(self):
        path = self._write("numbered.txt", "aaa\nbbb\nccc\n")
        result = self.tool.execute({"file_path": path})
        self.assertTrue(result.success)
        self.assertIn("1\t", result.output)
        self.assertIn("2\t", result.output)

    def test_read_with_offset_and_limit(self):
        path = self._write("big.txt", "\n".join(f"line{i}" for i in range(100)))
        result = self.tool.execute({"file_path": path, "offset": 10, "limit": 5})
        self.assertTrue(result.success)
        self.assertIn("line9", result.output)
        self.assertNotIn("line1\t", result.output)

    def test_read_nonexistent(self):
        result = self.tool.execute({"file_path": "/tmp/_nonexistent_test_file_xyz"})
        self.assertFalse(result.success)
        self.assertIn("not found", result.output)

    def test_read_directory(self):
        result = self.tool.execute({"file_path": self.tmpdir})
        self.assertFalse(result.success)
        self.assertIn("directory", result.output)

    def test_read_blocked_path(self):
        result = self.tool.execute({"file_path": "/dev/zero"})
        self.assertFalse(result.success)
        self.assertIn("Blocked", result.output)

    def test_marks_file_read(self):
        path = self._write("mark.txt", "content")
        self.assertFalse(self.ctx.was_file_read(path))
        self.tool.execute({"file_path": path})
        self.assertTrue(self.ctx.was_file_read(path))

    def test_empty_file(self):
        path = self._write("empty.txt", "")
        result = self.tool.execute({"file_path": path})
        self.assertTrue(result.success)
        self.assertIn("empty", result.output.lower())


class TestWriteFileTool(unittest.TestCase):
    """Tests for WriteFileTool."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        from shadow_code.tools.write_file import WriteFileTool

        self.tool = WriteFileTool(self.ctx)

    def test_validate_missing_path(self):
        self.assertIsNotNone(self.tool.validate({"content": "hi"}))

    def test_validate_missing_content(self):
        self.assertIsNotNone(self.tool.validate({"file_path": "/tmp/test.py"}))

    def test_validate_relative_path(self):
        err = self.tool.validate({"file_path": "relative.py", "content": "hi"})
        self.assertIsNotNone(err)

    def test_validate_valid(self):
        err = self.tool.validate({"file_path": "/tmp/test.py", "content": "hi"})
        self.assertIsNone(err)

    def test_create_new_file(self):
        path = os.path.join(self.tmpdir, "new.txt")
        result = self.tool.execute({"file_path": path, "content": "hello"})
        self.assertTrue(result.success)
        self.assertIn("Created", result.output)
        with open(path) as f:
            self.assertEqual(f.read(), "hello")

    def test_overwrite_requires_read(self):
        path = os.path.join(self.tmpdir, "existing.txt")
        with open(path, "w") as f:
            f.write("old content")
        result = self.tool.execute({"file_path": path, "content": "new"})
        self.assertFalse(result.success)
        self.assertIn("not been read", result.output)

    def test_overwrite_after_read(self):
        path = os.path.join(self.tmpdir, "existing.txt")
        with open(path, "w") as f:
            f.write("old content")
        self.ctx.mark_file_read(path)
        result = self.tool.execute({"file_path": path, "content": "new"})
        self.assertTrue(result.success)
        self.assertIn("Updated", result.output)

    def test_creates_parent_directories(self):
        path = os.path.join(self.tmpdir, "sub", "dir", "file.txt")
        result = self.tool.execute({"file_path": path, "content": "nested"})
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(path))

    def test_write_directory_fails(self):
        result = self.tool.execute({"file_path": self.tmpdir, "content": "nope"})
        self.assertFalse(result.success)


class TestEditFileTool(unittest.TestCase):
    """Tests for EditFileTool."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        from shadow_code.tools.edit_file import EditFileTool

        self.tool = EditFileTool(self.ctx)

    def _write_and_read(self, name, content):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            f.write(content)
        self.ctx.mark_file_read(path)
        return path

    def test_validate_missing_params(self):
        self.assertIsNotNone(self.tool.validate({}))
        self.assertIsNotNone(self.tool.validate({"file_path": "/tmp/x"}))
        self.assertIsNotNone(self.tool.validate({"file_path": "/tmp/x", "old_string": "a"}))

    def test_validate_relative_path(self):
        err = self.tool.validate({"file_path": "rel.py", "old_string": "a", "new_string": "b"})
        self.assertIsNotNone(err)

    def test_requires_read_first(self):
        path = os.path.join(self.tmpdir, "unread.txt")
        with open(path, "w") as f:
            f.write("content")
        result = self.tool.execute(
            {"file_path": path, "old_string": "content", "new_string": "changed"}
        )
        self.assertFalse(result.success)
        self.assertIn("not been read", result.output)

    def test_simple_replace(self):
        path = self._write_and_read("code.py", "def foo():\n    pass\n")
        result = self.tool.execute({"file_path": path, "old_string": "foo", "new_string": "bar"})
        self.assertTrue(result.success)
        with open(path) as f:
            self.assertIn("bar", f.read())

    def test_identical_strings_rejected(self):
        path = self._write_and_read("same.py", "content")
        result = self.tool.execute(
            {"file_path": path, "old_string": "content", "new_string": "content"}
        )
        self.assertFalse(result.success)
        self.assertIn("identical", result.output)

    def test_not_found_in_file(self):
        path = self._write_and_read("miss.py", "hello world")
        result = self.tool.execute(
            {"file_path": path, "old_string": "xyz_not_here", "new_string": "new"}
        )
        self.assertFalse(result.success)
        self.assertIn("not found", result.output)

    def test_multiple_matches_rejected_without_replace_all(self):
        path = self._write_and_read("dup.py", "foo\nfoo\nfoo\n")
        result = self.tool.execute({"file_path": path, "old_string": "foo", "new_string": "bar"})
        self.assertFalse(result.success)
        self.assertIn("3 times", result.output)

    def test_replace_all(self):
        path = self._write_and_read("dup.py", "foo\nfoo\nfoo\n")
        result = self.tool.execute(
            {
                "file_path": path,
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": True,
            }
        )
        self.assertTrue(result.success)
        with open(path) as f:
            content = f.read()
        self.assertEqual(content.count("bar"), 3)
        self.assertEqual(content.count("foo"), 0)

    def test_nonexistent_file(self):
        fake = os.path.join(self.tmpdir, "nope.py")
        self.ctx.mark_file_read(fake)
        result = self.tool.execute({"file_path": fake, "old_string": "a", "new_string": "b"})
        self.assertFalse(result.success)


class TestGlobTool(unittest.TestCase):
    """Tests for GlobTool."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        from shadow_code.tools.glob_tool import GlobTool

        self.tool = GlobTool(self.ctx)
        # Create test files
        for name in ["a.py", "b.py", "c.txt", "d.js"]:
            with open(os.path.join(self.tmpdir, name), "w") as f:
                f.write("content")
        os.makedirs(os.path.join(self.tmpdir, "sub"))
        with open(os.path.join(self.tmpdir, "sub", "e.py"), "w") as f:
            f.write("nested")

    def test_validate_missing_pattern(self):
        self.assertIsNotNone(self.tool.validate({}))

    def test_validate_empty_pattern(self):
        self.assertIsNotNone(self.tool.validate({"pattern": "  "}))

    def test_match_py_files(self):
        result = self.tool.execute({"pattern": "*.py", "path": self.tmpdir})
        self.assertTrue(result.success)
        self.assertIn("a.py", result.output)
        self.assertIn("b.py", result.output)
        self.assertNotIn("c.txt", result.output)

    def test_globstar(self):
        result = self.tool.execute({"pattern": "**/*.py", "path": self.tmpdir})
        self.assertTrue(result.success)
        self.assertIn("a.py", result.output)
        self.assertIn("e.py", result.output)

    def test_no_matches(self):
        result = self.tool.execute({"pattern": "*.xyz", "path": self.tmpdir})
        self.assertTrue(result.success)
        self.assertIn("No files matched", result.output)

    def test_invalid_directory(self):
        result = self.tool.execute({"pattern": "*.py", "path": "/nonexistent_dir_xyz"})
        self.assertFalse(result.success)


class TestGrepTool(unittest.TestCase):
    """Tests for GrepTool (Python fallback path)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        from shadow_code.tools.grep_tool import GrepTool

        self.tool = GrepTool(self.ctx)
        with open(os.path.join(self.tmpdir, "haystack.py"), "w") as f:
            f.write("def needle_function():\n    pass\n\ndef other():\n    needle = True\n")

    def test_validate_missing_pattern(self):
        self.assertIsNotNone(self.tool.validate({}))

    def test_validate_invalid_regex(self):
        err = self.tool.validate({"pattern": "[invalid"})
        self.assertIsNotNone(err)
        self.assertIn("Invalid regex", err)

    def test_search_finds_matches(self):
        result = self.tool.execute({"pattern": "needle", "path": self.tmpdir})
        self.assertTrue(result.success)
        self.assertIn("needle", result.output)
        self.assertIn("haystack.py", result.output)

    def test_search_no_matches(self):
        result = self.tool.execute({"pattern": "xyzzy_not_here", "path": self.tmpdir})
        self.assertTrue(result.success)
        self.assertIn("No matches", result.output)

    def test_search_single_file(self):
        path = os.path.join(self.tmpdir, "haystack.py")
        result = self.tool.execute({"pattern": "needle", "path": path})
        self.assertTrue(result.success)
        self.assertIn("needle", result.output)

    def test_case_insensitive(self):
        result = self.tool.execute(
            {"pattern": "NEEDLE", "path": self.tmpdir, "case_insensitive": True}
        )
        self.assertTrue(result.success)
        self.assertIn("needle", result.output)

    def test_include_filter(self):
        with open(os.path.join(self.tmpdir, "other.txt"), "w") as f:
            f.write("needle in txt\n")
        result = self.tool.execute({"pattern": "needle", "path": self.tmpdir, "include": "*.py"})
        self.assertTrue(result.success)
        self.assertIn("haystack.py", result.output)
        self.assertNotIn("other.txt", result.output)

    def test_max_results(self):
        result = self.tool.execute({"pattern": "needle", "path": self.tmpdir, "max_results": 1})
        self.assertTrue(result.success)
        lines = [ln for ln in result.output.strip().split("\n") if ln.strip()]
        self.assertEqual(len(lines), 1)


class TestGrepPythonFallback(unittest.TestCase):
    """Tests for _python_grep (pure Python fallback, tier 3)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create test files
        with open(os.path.join(self.tmpdir, "a.py"), "w") as f:
            f.write("def hello():\n    print('world')\n")
        with open(os.path.join(self.tmpdir, "b.txt"), "w") as f:
            f.write("hello from txt\nno match here\n")
        # Binary file
        with open(os.path.join(self.tmpdir, "binary.bin"), "wb") as f:
            f.write(b"\x00\x01\x02binary data")

    def test_search_directory(self):
        from shadow_code.tools.grep_tool import _python_grep

        result = _python_grep("hello", self.tmpdir, None, False, 200)
        self.assertTrue(result.success)
        self.assertIn("hello", result.output)

    def test_search_single_file(self):
        from shadow_code.tools.grep_tool import _python_grep

        path = os.path.join(self.tmpdir, "a.py")
        result = _python_grep("hello", path, None, False, 200)
        self.assertTrue(result.success)
        self.assertIn("a.py:1:", result.output)

    def test_no_matches(self):
        from shadow_code.tools.grep_tool import _python_grep

        result = _python_grep("xyzzy_not_here", self.tmpdir, None, False, 200)
        self.assertTrue(result.success)
        self.assertIn("No matches", result.output)

    def test_case_insensitive(self):
        from shadow_code.tools.grep_tool import _python_grep

        result = _python_grep("HELLO", self.tmpdir, None, True, 200)
        self.assertTrue(result.success)
        self.assertIn("hello", result.output)

    def test_include_filter(self):
        from shadow_code.tools.grep_tool import _python_grep

        result = _python_grep("hello", self.tmpdir, "*.py", False, 200)
        self.assertTrue(result.success)
        self.assertIn("a.py", result.output)
        self.assertNotIn("b.txt", result.output)

    def test_max_results(self):
        from shadow_code.tools.grep_tool import _python_grep

        result = _python_grep("hello", self.tmpdir, None, False, 1)
        self.assertTrue(result.success)
        lines = [ln for ln in result.output.strip().split("\n") if ln.strip()]
        self.assertEqual(len(lines), 1)

    def test_invalid_regex(self):
        from shadow_code.tools.grep_tool import _python_grep

        result = _python_grep("[invalid", self.tmpdir, None, False, 200)
        self.assertFalse(result.success)
        self.assertIn("Invalid regex", result.output)

    def test_skips_binary_files(self):
        from shadow_code.tools.grep_tool import _python_grep

        result = _python_grep("binary", self.tmpdir, None, False, 200)
        # Should not find match in binary file
        self.assertNotIn("binary.bin", result.output)

    def test_search_file_no_match(self):
        from shadow_code.tools.grep_tool import _python_grep

        path = os.path.join(self.tmpdir, "a.py")
        result = _python_grep("xyzzy", path, None, False, 200)
        self.assertTrue(result.success)
        self.assertIn("No matches", result.output)


class TestListDirTool(unittest.TestCase):
    """Tests for ListDirTool."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        from shadow_code.tools.list_dir import ListDirTool

        self.tool = ListDirTool(self.ctx)
        # Create test structure
        os.makedirs(os.path.join(self.tmpdir, "subdir"))
        with open(os.path.join(self.tmpdir, "file.txt"), "w") as f:
            f.write("hello")

    def test_list_default_cwd(self):
        result = self.tool.execute({})
        self.assertTrue(result.success)
        self.assertIn("subdir/", result.output)
        self.assertIn("file.txt", result.output)

    def test_dirs_before_files(self):
        result = self.tool.execute({"path": self.tmpdir})
        lines = result.output.strip().split("\n")
        dir_idx = next(i for i, ln in enumerate(lines) if "subdir/" in ln)
        file_idx = next(i for i, ln in enumerate(lines) if "file.txt" in ln)
        self.assertLess(dir_idx, file_idx)

    def test_shows_file_size(self):
        result = self.tool.execute({"path": self.tmpdir})
        # file.txt is 5 bytes
        self.assertIn("5 B", result.output)

    def test_nonexistent_path(self):
        result = self.tool.execute({"path": "/nonexistent_xyz"})
        self.assertFalse(result.success)
        self.assertIn("not found", result.output.lower())

    def test_not_a_directory(self):
        path = os.path.join(self.tmpdir, "file.txt")
        result = self.tool.execute({"path": path})
        self.assertFalse(result.success)
        self.assertIn("Not a directory", result.output)

    def test_empty_directory(self):
        empty = os.path.join(self.tmpdir, "empty")
        os.makedirs(empty)
        result = self.tool.execute({"path": empty})
        self.assertTrue(result.success)
        self.assertIn("empty", result.output.lower())


if __name__ == "__main__":
    unittest.main()
