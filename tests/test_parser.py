"""Offline unit tests for parser.py.

Tests the explicitly opt-in legacy ```tool_call``` Markdown format.
"""

import unittest

from shadow_code.parser import parse_legacy_markdown_tool_calls


class TestParseToolCalls(unittest.TestCase):
    """Test the full parse_tool_calls function."""

    def test_single_tool_call(self):
        text = 'Let me check.\n```tool_call\n{"tool": "bash", "params": {"command": "ls -la"}}\n```'
        clean, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "bash")
        self.assertEqual(calls[0].params, {"command": "ls -la"})
        self.assertEqual(clean, "Let me check.")

    def test_multiple_tool_calls(self):
        text = (
            "Checking both:\n"
            '```tool_call\n{"tool": "bash", "params": {"command": "pwd"}}\n```\n'
            '```tool_call\n{"tool": "bash", "params": {"command": "ls"}}\n```'
        )
        clean, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].params["command"], "pwd")
        self.assertEqual(calls[1].params["command"], "ls")

    def test_no_tool_calls(self):
        text = "This is just a regular response with no tool calls."
        clean, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 0)
        self.assertEqual(clean, text)

    def test_invalid_json(self):
        text = "```tool_call\n{not valid json}\n```"
        clean, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "__invalid__")
        self.assertIn("Invalid JSON", calls[0].params["error"])

    def test_missing_tool_key(self):
        text = '```tool_call\n{"params": {"command": "ls"}}\n```'
        clean, calls = parse_legacy_markdown_tool_calls(text)
        # Missing "tool" key -- not a valid tool call, should be skipped
        self.assertEqual(len(calls), 0)

    def test_missing_params_key(self):
        text = '```tool_call\n{"tool": "bash"}\n```'
        clean, calls = parse_legacy_markdown_tool_calls(text)
        # Missing "params" key -- not a valid tool call, should be skipped
        self.assertEqual(len(calls), 0)

    def test_wrong_types(self):
        text = '```tool_call\n{"tool": 123, "params": {"command": "ls"}}\n```'
        clean, calls = parse_legacy_markdown_tool_calls(text)
        # tool is not a string -- should be skipped
        self.assertEqual(len(calls), 0)

    def test_params_not_dict(self):
        text = '```tool_call\n{"tool": "bash", "params": "not a dict"}\n```'
        clean, calls = parse_legacy_markdown_tool_calls(text)
        # params is not a dict -- should be skipped
        self.assertEqual(len(calls), 0)

    def test_text_before_and_after(self):
        text = (
            "I'll run this command for you.\n"
            '```tool_call\n{"tool": "bash", "params": {"command": "echo hello"}}\n```\n'
            "That should do it."
        )
        clean, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertIn("I'll run this command for you.", clean)
        self.assertIn("That should do it.", clean)

    def test_raw_field_contains_full_block(self):
        text = '```tool_call\n{"tool": "bash", "params": {"command": "ls"}}\n```'
        _, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].raw, text)

    def test_nested_backticks_in_command(self):
        # Command contains backticks but not the full marker pattern
        text = '```tool_call\n{"tool": "bash", "params": {"command": "echo `date`"}}\n```'
        _, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].params["command"], "echo `date`")

    def test_empty_params(self):
        text = '```tool_call\n{"tool": "list_dir", "params": {}}\n```'
        _, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "list_dir")
        self.assertEqual(calls[0].params, {})

    def test_regular_code_block_not_matched(self):
        # A normal ```python block should NOT be matched
        text = '```python\nprint("hello")\n```'
        _, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 0)

    def test_bash_fence_remains_assistant_text(self):
        text = "Here is an example:\n```bash\nrm -rf /tmp/example\n```"

        clean, calls = parse_legacy_markdown_tool_calls(text)

        self.assertEqual(calls, [])
        self.assertEqual(clean, text)

    def test_sh_fence_with_tool_shaped_json_is_not_a_tool_call(self):
        text = '```sh\n{"tool": "bash", "params": {"command": "pwd"}}\n```'

        clean, calls = parse_legacy_markdown_tool_calls(text)

        self.assertEqual(calls, [])
        self.assertEqual(clean, text)

    def test_tool_call_with_complex_params(self):
        text = (
            "```tool_call\n"
            '{"tool": "edit_file", "params": {"file_path": "/tmp/test.py", '
            '"old_string": "def foo():", "new_string": "def bar():"}}\n'
            "```"
        )
        _, calls = parse_legacy_markdown_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "edit_file")
        self.assertEqual(calls[0].params["old_string"], "def foo():")
        self.assertEqual(calls[0].params["new_string"], "def bar():")


class TestConversation(unittest.TestCase):
    """Basic tests for conversation management."""

    def test_add_messages(self):
        from shadow_code.conversation import Conversation

        conv = Conversation()
        conv.add_user("hello")
        conv.add_assistant("hi")
        msgs = conv.get_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")

    def test_add_tool_results(self):
        from shadow_code.conversation import Conversation

        conv = Conversation()
        conv.add_tool_results('<tool_result tool="bash" success="true">\nfoo\n</tool_result>')
        msgs = conv.get_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIn("[Tool results below", msgs[0]["content"])
        self.assertIn("<tool_result", msgs[0]["content"])

    def test_clear(self):
        from shadow_code.conversation import Conversation

        conv = Conversation()
        conv.add_user("hello")
        conv.add_assistant("hi")
        conv.total_prompt_tokens = 1000
        conv.clear()
        self.assertEqual(len(conv.get_messages()), 0)
        self.assertEqual(conv.total_prompt_tokens, 0)

    def test_result_clearing(self):
        from shadow_code.conversation import KEEP_RECENT_RESULTS, Conversation

        conv = Conversation()
        # Add more tool results than KEEP_RECENT_RESULTS
        for i in range(KEEP_RECENT_RESULTS + 3):
            conv.add_tool_results(
                f'<tool_result tool="bash" success="true">\nresult {i}\n</tool_result>'
            )
        conv.clear_old_tool_results()
        # First 3 should be cleared, last KEEP_RECENT_RESULTS kept
        cleared = [m for m in conv.get_messages() if "cleared to save context" in m["content"]]
        self.assertEqual(len(cleared), 3)

    def test_emergency_truncate(self):
        from shadow_code.conversation import KEEP_RECENT_MESSAGES, Conversation

        conv = Conversation()
        for i in range(KEEP_RECENT_MESSAGES + 10):
            conv.add_user(f"msg {i}")
        conv.emergency_truncate()
        # Should have KEEP_RECENT_MESSAGES + 1 (the truncation notice)
        self.assertEqual(len(conv.get_messages()), KEEP_RECENT_MESSAGES + 1)
        self.assertIn("truncated", conv.get_messages()[0]["content"])

    def test_emergency_truncate_small_history(self):
        from shadow_code.conversation import Conversation

        conv = Conversation()
        for i in range(5):
            conv.add_user(f"msg {i}")
        conv.emergency_truncate()
        # Should not truncate when under KEEP_RECENT_MESSAGES
        self.assertEqual(len(conv.get_messages()), 5)


class TestToolContext(unittest.TestCase):
    """Tests for ToolContext shared state."""

    def test_initial_cwd(self):
        from shadow_code.tool_context import ToolContext

        ctx = ToolContext("/tmp")
        self.assertEqual(ctx.cwd, "/tmp")

    def test_mark_and_check_file(self):
        from shadow_code.tool_context import ToolContext

        ctx = ToolContext("/tmp")
        ctx.mark_file_read("/tmp/test.py")
        self.assertTrue(ctx.was_file_read("/tmp/test.py"))
        self.assertFalse(ctx.was_file_read("/tmp/other.py"))

    def test_read_files_initially_empty(self):
        from shadow_code.tool_context import ToolContext

        ctx = ToolContext("/home")
        self.assertEqual(len(ctx.read_files), 0)


class TestToolRegistry(unittest.TestCase):
    """Tests for tool registry and dispatch."""

    def test_register_and_dispatch(self):
        from shadow_code.tools import _REGISTRY, dispatch, register
        from shadow_code.tools.base import BaseTool, ToolResult

        # Save and restore registry state
        saved = dict(_REGISTRY)
        try:

            class DummyTool(BaseTool):
                name = "dummy"

                def execute(self, params: dict) -> ToolResult:
                    return ToolResult(True, f"got: {params.get('x', '')}")

            register(DummyTool())
            result = dispatch("dummy", {"x": "hello"})
            self.assertTrue(result.success)
            self.assertEqual(result.output, "got: hello")
        finally:
            _REGISTRY.clear()
            _REGISTRY.update(saved)

    def test_dispatch_unknown_tool(self):
        from shadow_code.tools import dispatch

        result = dispatch("nonexistent_tool_xyz", {})
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.output)

    def test_format_result(self):
        from shadow_code.tools import format_result
        from shadow_code.tools.base import ToolResult

        formatted = format_result("bash", ToolResult(True, "hello world"))
        self.assertIn('<tool_result tool="bash" success="true">', formatted)
        self.assertIn("hello world", formatted)

    def test_format_result_failure(self):
        from shadow_code.tools import format_result
        from shadow_code.tools.base import ToolResult

        formatted = format_result("bash", ToolResult(False, "command not found"))
        self.assertIn('success="false"', formatted)

    def test_truncate(self):
        from shadow_code.config import TOOL_OUTPUT_MAX_CHARS
        from shadow_code.tools import _truncate

        short = "hello"
        self.assertEqual(_truncate(short), short)
        long = "x" * (TOOL_OUTPUT_MAX_CHARS + 1000)
        truncated = _truncate(long)
        self.assertIn("chars truncated", truncated)
        self.assertTrue(len(truncated) < len(long))


class TestBashTool(unittest.TestCase):
    """Tests for the bash tool."""

    def test_simple_command(self):
        from shadow_code.tool_context import ToolContext
        from shadow_code.tools.bash import BashTool

        ctx = ToolContext("/tmp")
        bash = BashTool(ctx)
        result = bash.execute({"command": "echo hello"})
        self.assertTrue(result.success)
        self.assertEqual(result.output.strip(), "hello")

    def test_command_failure(self):
        from shadow_code.tool_context import ToolContext
        from shadow_code.tools.bash import BashTool

        ctx = ToolContext("/tmp")
        bash = BashTool(ctx)
        result = bash.execute({"command": "false"})
        self.assertFalse(result.success)
        self.assertIn("exit code", result.output)

    def test_validate_missing_command(self):
        from shadow_code.tool_context import ToolContext
        from shadow_code.tools.bash import BashTool

        ctx = ToolContext("/tmp")
        bash = BashTool(ctx)
        err = bash.validate({})
        self.assertIsNotNone(err)
        self.assertIn("Missing", err)

    def test_validate_empty_command(self):
        from shadow_code.tool_context import ToolContext
        from shadow_code.tools.bash import BashTool

        ctx = ToolContext("/tmp")
        bash = BashTool(ctx)
        err = bash.validate({"command": "  "})
        self.assertIsNotNone(err)
        self.assertIn("empty", err)

    def test_validate_interactive_command(self):
        from shadow_code.tool_context import ToolContext
        from shadow_code.tools.bash import BashTool

        ctx = ToolContext("/tmp")
        bash = BashTool(ctx)
        err = bash.validate({"command": "vim test.py"})
        self.assertIsNotNone(err)
        self.assertIn("Interactive", err)

    def test_cwd_tracking(self):
        import tempfile

        from shadow_code.tool_context import ToolContext
        from shadow_code.tools.bash import BashTool

        ctx = ToolContext("/tmp")
        bash = BashTool(ctx)
        # cd to a known directory
        with tempfile.TemporaryDirectory() as tmpdir:
            bash.execute({"command": f"cd {tmpdir}"})
            self.assertEqual(ctx.cwd, tmpdir)

    def test_stderr_included(self):
        from shadow_code.tool_context import ToolContext
        from shadow_code.tools.bash import BashTool

        ctx = ToolContext("/tmp")
        bash = BashTool(ctx)
        result = bash.execute({"command": "echo err >&2"})
        self.assertIn("STDERR", result.output)

    def test_timeout(self):
        from shadow_code.tool_context import ToolContext
        from shadow_code.tools.bash import BashTool

        ctx = ToolContext("/tmp")
        bash = BashTool(ctx)
        result = bash.execute({"command": "sleep 10", "timeout": 1})
        self.assertFalse(result.success)
        self.assertIn("Timed out", result.output)

    def test_respects_cwd(self):
        import tempfile

        from shadow_code.tool_context import ToolContext
        from shadow_code.tools.bash import BashTool

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = ToolContext(tmpdir)
            bash = BashTool(ctx)
            result = bash.execute({"command": "pwd"})
            self.assertTrue(result.success)
            self.assertEqual(result.output.strip(), tmpdir)


if __name__ == "__main__":
    unittest.main()
