"""Integration tests for the shadow-code REPL loop with mock Ollama.

These tests exercise the full request flow:
  user input -> conversation -> mock LLM -> parser -> tool dispatch -> tool result -> loop

No real Ollama server is needed. OllamaClient.chat_stream and health_check are mocked.
"""

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

from shadow_code import tools as tool_reg
from shadow_code.conversation import Conversation
from shadow_code.display import StreamDisplay
from shadow_code.ollama_client import OllamaClient
from shadow_code.parser import parse_legacy_markdown_tool_calls
from shadow_code.prompt import SYSTEM_PROMPT
from shadow_code.safety import check_destructive
from shadow_code.tool_context import ToolContext
from shadow_code.tools.bash import BashTool


def _tc(tool, **params):
    """Build a ```tool_call``` block string."""
    import json

    j = json.dumps({"tool": tool, "params": params})
    return f"```tool_call\n{j}\n```"


def _make_client(responses):
    """Create a mock OllamaClient that yields pre-defined responses.

    Args:
        responses: list of strings. Each string is the full model response
                   for one chat_stream() call. Called in order.
    """
    client = MagicMock(spec=OllamaClient)
    client.last_prompt_tokens = 1000
    client.last_eval_tokens = 50

    call_count = [0]

    def _chat_stream(messages, system):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        resp = responses[idx]
        # Yield the response in chunks to simulate streaming
        for i in range(0, len(resp), 20):
            yield resp[i : i + 20]

    client.chat_stream.side_effect = _chat_stream
    client.health_check.return_value = (True, "OK")
    return client


def _run_repl_turn(client, conv, ctx, user_input, first_message=False):
    """Simulate one REPL turn: add user input, stream response, parse+dispatch tools.

    Returns (assistant_response, tool_results_list, turns_used).
    """
    from shadow_code.config import MAX_CONSECUTIVE_ERRORS, MAX_TOOL_TURNS

    # Inject environment on first message
    if first_message:
        import platform
        from datetime import datetime

        shell = os.environ.get("SHELL", "/bin/bash").rsplit("/", 1)[-1]
        env_prefix = (
            f"[Environment: CWD={ctx.cwd}, "
            f"Platform={platform.system()} {platform.release()}, "
            f"Shell={shell}, Date={datetime.now().strftime('%Y-%m-%d')}]\n\n"
        )
        conv.add_user(env_prefix + user_input)
    else:
        conv.add_user(user_input)

    display = StreamDisplay()
    turns = 0
    errors = 0
    all_tool_results = []
    final_response = ""

    while turns < MAX_TOOL_TURNS:
        display.reset()
        # Redirect stdout to suppress display output during tests
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            for chunk in client.chat_stream(conv.get_messages(), SYSTEM_PROMPT):
                display.feed(chunk)
            display.flush()
        finally:
            sys.stdout = old_stdout

        resp = display.get_full_response()
        if not resp or not resp.strip():
            break

        final_response = resp
        conv.add_assistant(resp)
        conv.update_tokens(client.last_prompt_tokens)

        # Parse tool calls
        _, calls = parse_legacy_markdown_tool_calls(resp)
        if not calls:
            break

        # Execute tools
        results = []
        for tc in calls:
            if tc.tool == "__invalid__":
                r = tool_reg.ToolResult(False, tc.params.get("error", "Invalid"))
            else:
                # Safety check for bash
                if tc.tool == "bash":
                    warning = check_destructive(tc.params.get("command", ""))
                    if warning:
                        r = tool_reg.ToolResult(False, "Command cancelled by user")
                        results.append(tool_reg.format_result(tc.tool, r))
                        all_tool_results.append(("bash", False, r.output))
                        continue

                r = tool_reg.dispatch(tc.tool, tc.params)

            results.append(tool_reg.format_result(tc.tool, r))
            all_tool_results.append((tc.tool, r.success, r.output))
            errors = errors + 1 if not r.success else 0

        conv.add_tool_results("\n\n".join(results))
        turns += 1

        if errors >= MAX_CONSECUTIVE_ERRORS:
            break

    return final_response, all_tool_results, turns


class TestIntegrationPlainResponse(unittest.TestCase):
    """Test: model responds with plain text (no tool calls)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        self.conv = Conversation()
        # Register tools
        saved = dict(tool_reg._REGISTRY)
        self.saved_registry = saved
        tool_reg.register(BashTool(self.ctx))

    def tearDown(self):
        tool_reg._REGISTRY.clear()
        tool_reg._REGISTRY.update(self.saved_registry)

    def test_plain_text_response(self):
        client = _make_client(["Hello! I'm Shadow, your coding assistant."])
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "hello", first_message=True
        )
        self.assertIn("Shadow", resp)
        self.assertEqual(len(tools), 0)
        self.assertEqual(turns, 0)

    def test_conversation_has_messages(self):
        client = _make_client(["Hi there!"])
        _run_repl_turn(client, self.conv, self.ctx, "hello", first_message=True)
        msgs = self.conv.get_messages()
        self.assertEqual(len(msgs), 2)  # user + assistant
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")

    def test_first_message_has_env_prefix(self):
        client = _make_client(["OK"])
        _run_repl_turn(client, self.conv, self.ctx, "test", first_message=True)
        msgs = self.conv.get_messages()
        self.assertIn("[Environment:", msgs[0]["content"])
        self.assertIn("CWD=", msgs[0]["content"])

    def test_second_message_no_env_prefix(self):
        client = _make_client(["OK", "Sure"])
        _run_repl_turn(client, self.conv, self.ctx, "first", first_message=True)
        _run_repl_turn(client, self.conv, self.ctx, "second", first_message=False)
        msgs = self.conv.get_messages()
        # 4 messages: user1, assistant1, user2, assistant2
        self.assertEqual(len(msgs), 4)
        self.assertNotIn("[Environment:", msgs[2]["content"])


class TestIntegrationToolExecution(unittest.TestCase):
    """Test: model calls tools, tools execute, results feed back."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        self.conv = Conversation()
        self.saved_registry = dict(tool_reg._REGISTRY)
        tool_reg.register(BashTool(self.ctx))
        from shadow_code.main import _register_optional_tools

        _register_optional_tools(self.ctx)

    def tearDown(self):
        tool_reg._REGISTRY.clear()
        tool_reg._REGISTRY.update(self.saved_registry)

    def test_bash_tool_execution(self):
        """Model calls bash tool, gets result, then responds."""
        responses = [
            "Let me check.\n" + _tc("bash", command="echo hello_world"),
            "The output is hello_world.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "run echo hello_world", first_message=True
        )
        # Tool was called
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0][0], "bash")
        self.assertTrue(tools[0][1])  # success
        self.assertIn("hello_world", tools[0][2])
        # 1 tool turn used
        self.assertEqual(turns, 1)

    def test_read_file_tool(self):
        """Model reads a file via read_file tool."""
        # Create a test file
        test_file = os.path.join(self.tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("line one\nline two\n")

        responses = [
            _tc("read_file", file_path=test_file),
            "The file contains two lines.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "read test.txt", first_message=True
        )
        self.assertEqual(tools[0][0], "read_file")
        self.assertTrue(tools[0][1])
        self.assertIn("line one", tools[0][2])
        self.assertIn("line two", tools[0][2])

    def test_write_file_tool(self):
        """Model creates a new file via write_file tool."""
        new_file = os.path.join(self.tmpdir, "created.txt")
        responses = [
            _tc("write_file", file_path=new_file, content="hello file"),
            "File created.",
        ]
        client = _make_client(responses)
        _run_repl_turn(client, self.conv, self.ctx, "create a file", first_message=True)
        self.assertTrue(os.path.exists(new_file))
        with open(new_file) as f:
            self.assertEqual(f.read(), "hello file")

    def test_multiple_tools_in_one_response(self):
        """Model calls two tools in a single response."""
        responses = [
            "Checking both:\n"
            + _tc("bash", command="echo aaa")
            + "\n"
            + _tc("bash", command="echo bbb"),
            "Both commands ran.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "run two commands", first_message=True
        )
        self.assertEqual(len(tools), 2)
        self.assertIn("aaa", tools[0][2])
        self.assertIn("bbb", tools[1][2])

    def test_tool_result_feeds_back_to_model(self):
        """Verify tool results are added to conversation for the model to see."""
        responses = [
            _tc("bash", command="echo feedback_test"),
            "I see the output.",
        ]
        client = _make_client(responses)
        _run_repl_turn(client, self.conv, self.ctx, "test feedback", first_message=True)
        msgs = self.conv.get_messages()
        # user, assistant (with tool call), tool_result, assistant (final)
        self.assertEqual(len(msgs), 4)
        tool_result_msg = msgs[2]
        self.assertEqual(tool_result_msg["role"], "user")
        self.assertIn("<tool_result", tool_result_msg["content"])
        self.assertIn("feedback_test", tool_result_msg["content"])


class TestIntegrationSafety(unittest.TestCase):
    """Test: destructive commands are blocked."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        self.conv = Conversation()
        self.saved_registry = dict(tool_reg._REGISTRY)
        tool_reg.register(BashTool(self.ctx))

    def tearDown(self):
        tool_reg._REGISTRY.clear()
        tool_reg._REGISTRY.update(self.saved_registry)

    def test_rm_rf_blocked(self):
        """rm -rf should be cancelled automatically in integration tests."""
        responses = [
            _tc("bash", command="rm -rf /"),
            "OK, I won't do that.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "delete everything", first_message=True
        )
        self.assertEqual(tools[0][0], "bash")
        self.assertFalse(tools[0][1])  # blocked
        self.assertIn("cancelled", tools[0][2])

    def test_git_force_push_blocked(self):
        responses = [
            _tc("bash", command="git push --force origin main"),
            "Cancelled.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "force push", first_message=True
        )
        self.assertFalse(tools[0][1])

    def test_safe_command_executes(self):
        responses = [
            _tc("bash", command="echo safe"),
            "Done.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "echo safe", first_message=True
        )
        self.assertTrue(tools[0][1])
        self.assertIn("safe", tools[0][2])


class TestIntegrationUnknownTool(unittest.TestCase):
    """Test: model calls a tool that doesn't exist."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        self.conv = Conversation()
        self.saved_registry = dict(tool_reg._REGISTRY)
        tool_reg.register(BashTool(self.ctx))

    def tearDown(self):
        tool_reg._REGISTRY.clear()
        tool_reg._REGISTRY.update(self.saved_registry)

    def test_unknown_tool_returns_error(self):
        responses = [
            _tc("nonexistent_xyz"),
            "Sorry, that tool doesn't exist.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "use fake tool", first_message=True
        )
        self.assertEqual(tools[0][0], "nonexistent_xyz")
        self.assertFalse(tools[0][1])
        self.assertIn("Unknown tool", tools[0][2])


class TestIntegrationInvalidJSON(unittest.TestCase):
    """Test: model emits a tool_call with invalid JSON."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        self.conv = Conversation()
        self.saved_registry = dict(tool_reg._REGISTRY)
        tool_reg.register(BashTool(self.ctx))

    def tearDown(self):
        tool_reg._REGISTRY.clear()
        tool_reg._REGISTRY.update(self.saved_registry)

    def test_invalid_json_handled(self):
        responses = [
            "```tool_call\n{this is not valid json}\n```",
            "Let me try again.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "test invalid", first_message=True
        )
        self.assertEqual(tools[0][0], "__invalid__")
        self.assertFalse(tools[0][1])


class TestIntegrationMultiTurn(unittest.TestCase):
    """Test: model uses multiple tool turns before final response."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        self.conv = Conversation()
        self.saved_registry = dict(tool_reg._REGISTRY)
        tool_reg.register(BashTool(self.ctx))
        from shadow_code.main import _register_optional_tools

        _register_optional_tools(self.ctx)

    def tearDown(self):
        tool_reg._REGISTRY.clear()
        tool_reg._REGISTRY.update(self.saved_registry)

    def test_read_then_edit(self):
        """Model reads a file, then edits it -- two tool turns."""
        test_file = os.path.join(self.tmpdir, "code.py")
        with open(test_file, "w") as f:
            f.write("def foo():\n    pass\n")

        responses = [
            _tc("read_file", file_path=test_file),
            _tc(
                "edit_file",
                file_path=test_file,
                old_string="foo",
                new_string="bar",
            ),
            "Renamed foo to bar.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "rename foo to bar", first_message=True
        )
        self.assertEqual(turns, 2)
        self.assertEqual(tools[0][0], "read_file")
        self.assertTrue(tools[0][1])
        self.assertEqual(tools[1][0], "edit_file")
        self.assertTrue(tools[1][1])
        # Verify file was actually changed
        with open(test_file) as f:
            self.assertIn("bar", f.read())

    def test_empty_response_breaks_loop(self):
        """Empty model response should break the tool loop."""
        client = _make_client([""])
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "test empty", first_message=True
        )
        self.assertEqual(resp, "")
        self.assertEqual(turns, 0)


class TestIntegrationSlashCommands(unittest.TestCase):
    """Test slash command parsing in isolation (no REPL loop needed)."""

    def test_skill_dispatch(self):
        """Verify /commit resolves to a skill prompt."""
        from shadow_code.skills import get_skill

        result = get_skill("commit")
        self.assertIsNotNone(result)
        desc, prompt = result
        self.assertIn("git commit", desc.lower())
        self.assertGreater(len(prompt), 50)

    def test_unknown_slash_command(self):
        from shadow_code.skills import get_skill

        result = get_skill("nonexistent_command_xyz")
        self.assertIsNone(result)

    def test_skill_with_args(self):
        """Skills receive user args appended to prompt."""
        from shadow_code.skills import get_skill

        desc, prompt = get_skill("review")
        # The skill system appends args in main.py:
        # skill_msg += f"\n\nUser specified: {skill_args}"
        combined = prompt + "\n\nUser specified: myfile.py"
        self.assertIn("myfile.py", combined)


class TestIntegrationContextManagement(unittest.TestCase):
    """Test context management thresholds."""

    def test_result_clearing_threshold(self):
        from shadow_code.config import CONTEXT_WINDOW
        from shadow_code.conversation import RESULT_CLEARING_RATIO

        conv = Conversation()
        threshold = int(CONTEXT_WINDOW * RESULT_CLEARING_RATIO)
        conv.total_prompt_tokens = threshold + 1
        self.assertTrue(conv.needs_result_clearing())

    def test_compaction_threshold(self):
        from shadow_code.config import CONTEXT_WINDOW
        from shadow_code.conversation import COMPACTION_RATIO

        conv = Conversation()
        conv.total_prompt_tokens = int(CONTEXT_WINDOW * COMPACTION_RATIO) + 1
        self.assertTrue(conv.needs_compaction())

    def test_emergency_threshold(self):
        from shadow_code.config import CONTEXT_WINDOW
        from shadow_code.conversation import EMERGENCY_TRUNCATE_RATIO

        conv = Conversation()
        conv.total_prompt_tokens = int(CONTEXT_WINDOW * EMERGENCY_TRUNCATE_RATIO) + 1
        self.assertTrue(conv.needs_emergency_truncate())

    def test_below_all_thresholds(self):
        conv = Conversation()
        conv.total_prompt_tokens = 100
        self.assertFalse(conv.needs_result_clearing())
        self.assertFalse(conv.needs_compaction())
        self.assertFalse(conv.needs_emergency_truncate())


class TestIntegrationEditRequiresRead(unittest.TestCase):
    """Test that edit_file enforces read-before-write in the full flow."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = ToolContext(self.tmpdir)
        self.conv = Conversation()
        self.saved_registry = dict(tool_reg._REGISTRY)
        tool_reg.register(BashTool(self.ctx))
        from shadow_code.main import _register_optional_tools

        _register_optional_tools(self.ctx)

    def tearDown(self):
        tool_reg._REGISTRY.clear()
        tool_reg._REGISTRY.update(self.saved_registry)

    def test_edit_without_read_fails(self):
        """Model tries to edit a file it hasn't read -- should fail."""
        test_file = os.path.join(self.tmpdir, "unread.py")
        with open(test_file, "w") as f:
            f.write("original content")

        responses = [
            _tc(
                "edit_file",
                file_path=test_file,
                old_string="original",
                new_string="modified",
            ),
            "The edit failed because I didn't read first.",
        ]
        client = _make_client(responses)
        resp, tools, turns = _run_repl_turn(
            client, self.conv, self.ctx, "edit the file", first_message=True
        )
        self.assertFalse(tools[0][1])
        self.assertIn("not been read", tools[0][2])
        # File should be unchanged
        with open(test_file) as f:
            self.assertEqual(f.read(), "original content")


if __name__ == "__main__":
    unittest.main()
