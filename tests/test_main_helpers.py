"""Tests for main.py helper functions and slash command parsing.

We test the helper functions and importable parts without running the full REPL.
"""

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestShowContextStatus(unittest.TestCase):
    """Tests for _show_context_status."""

    def test_plain_mode_low(self):
        from shadow_code.main import _show_context_status

        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            _show_context_status(10000, 131072, 500)
        finally:
            sys.stdout = old
        output = captured.getvalue()
        self.assertIn("10K", output)
        self.assertIn("131K", output)

    def test_plain_mode_medium(self):
        from shadow_code.main import _show_context_status

        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            _show_context_status(80000, 131072, 500)
        finally:
            sys.stdout = old
        output = captured.getvalue()
        self.assertIn("80K", output)

    def test_plain_mode_high(self):
        from shadow_code.main import _show_context_status

        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            _show_context_status(120000, 131072, 500)
        finally:
            sys.stdout = old
        output = captured.getvalue()
        self.assertIn("120K", output)


class TestRegisterOptionalTools(unittest.TestCase):
    """Tests for _register_optional_tools."""

    def test_registers_tools(self):
        from shadow_code.main import _register_optional_tools
        from shadow_code.tool_context import ToolContext
        from shadow_code.tools import _REGISTRY

        ctx = ToolContext("/tmp")
        saved = dict(_REGISTRY)
        try:
            _register_optional_tools(ctx)
            # Should register the 6 optional tools
            expected = {"read_file", "write_file", "edit_file", "glob", "grep", "list_dir"}
            registered = set(_REGISTRY.keys())
            self.assertTrue(
                expected.issubset(registered),
                f"Missing: {expected - registered}",
            )
        finally:
            _REGISTRY.clear()
            _REGISTRY.update(saved)


class TestMainImports(unittest.TestCase):
    """Test that main module is importable and has expected attributes."""

    def test_main_function_exists(self):
        from shadow_code.main import main

        self.assertTrue(callable(main))

    def test_register_optional_tools_exists(self):
        from shadow_code.main import _register_optional_tools

        self.assertTrue(callable(_register_optional_tools))

    def test_show_context_status_exists(self):
        from shadow_code.main import _show_context_status

        self.assertTrue(callable(_show_context_status))


class TestLegacyMarkdownToolBoundary(unittest.TestCase):
    """Legacy Markdown tool calls must be explicitly enabled at runtime."""

    def test_disabled_boundary_does_not_invoke_legacy_parser(self):
        from shadow_code.main import _get_legacy_markdown_tool_calls

        with patch("shadow_code.main.parse_legacy_markdown_tool_calls") as parser:
            calls = _get_legacy_markdown_tool_calls("```tool_call\n{}\n```", enabled=False)

        self.assertEqual(calls, [])
        parser.assert_not_called()

    def test_enabled_boundary_invokes_legacy_parser(self):
        from shadow_code.main import _get_legacy_markdown_tool_calls

        expected = [MagicMock()]
        with patch(
            "shadow_code.main.parse_legacy_markdown_tool_calls",
            return_value=("", expected),
        ) as parser:
            calls = _get_legacy_markdown_tool_calls("response", enabled=True)

        self.assertEqual(calls, expected)
        parser.assert_called_once_with("response")

    def test_environment_switch_defaults_to_disabled(self):
        from shadow_code.config import _env_flag

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_env_flag("SHADOW_LEGACY_MARKDOWN_TOOLS"))

    def test_environment_switch_accepts_explicit_true_value(self):
        from shadow_code.config import _env_flag

        with patch.dict(os.environ, {"SHADOW_LEGACY_MARKDOWN_TOOLS": "true"}, clear=True):
            self.assertTrue(_env_flag("SHADOW_LEGACY_MARKDOWN_TOOLS"))

    def test_disabled_tool_only_fence_returns_typed_protocol_mismatch(self):
        from shadow_code.domain.tools import ToolError
        from shadow_code.main import _legacy_markdown_protocol_error

        response = '```tool_call\n{"tool":"read_file","params":{}}\n```'
        error = _legacy_markdown_protocol_error(response, enabled=False)

        self.assertIsInstance(error, ToolError)
        self.assertEqual(error.code, "protocol_mismatch")
        self.assertIn("no action was executed", error.message.lower())
        self.assertIsNone(_legacy_markdown_protocol_error(response, enabled=True))


class TestNativeToolAdmission(unittest.TestCase):
    """Read-only native calls execute through the admission pipeline."""

    def test_main_executes_read_only_native_calls_through_pipeline(self):
        import importlib
        import tempfile

        from shadow_code.conversation import Conversation
        from shadow_code.policy.workspace import WorkspaceGuard

        main_module = importlib.import_module("shadow_code.main")
        native_call = {
            "call_id": "call-0",
            "name": "read_file",
            "arguments": {"file_path": "hello.txt"},
        }

        with tempfile.TemporaryDirectory() as workspace:
            with open(os.path.join(workspace, "hello.txt"), "w", encoding="utf-8") as handle:
                handle.write("admitted content\n")

            client = MagicMock()
            client.health_check.return_value = (True, "OK")
            client.last_prompt_tokens = 0
            client.last_eval_tokens = 0
            responses = iter((([], [native_call]), (["finished"], [])))
            payloads = []
            prompts = []

            def chat_stream(messages, system, model=None, tools=None):
                prompts.append(system)
                payloads.append(tools)
                chunks, client.last_tool_calls = next(responses)
                return iter(chunks)

            client.chat_stream.side_effect = chat_stream
            conversation = Conversation()

            with (
                patch.object(main_module, "_RICH", False),
                patch.object(main_module, "_HAS_REPL", False),
                patch.object(main_module, "_HAS_DB", False),
                patch.object(main_module, "OllamaClient", return_value=client),
                patch.object(main_module, "Conversation", return_value=conversation),
                patch.object(
                    main_module,
                    "WorkspaceGuard",
                    side_effect=lambda root, **kw: WorkspaceGuard(workspace),
                ),
                patch.object(main_module, "_register_optional_tools"),
                patch.object(main_module.tool_reg, "register"),
                patch.object(
                    main_module.tool_reg,
                    "dispatch",
                    return_value=main_module.tool_reg.ToolResult(True, "unexpected"),
                ) as dispatch,
                patch.object(main_module.signal, "signal"),
                patch("builtins.input", side_effect=["read hello", "/exit"]),
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
                patch.object(
                    conversation,
                    "add_assistant_tool_call",
                    wraps=conversation.add_assistant_tool_call,
                ) as add_call,
                patch.object(
                    conversation,
                    "add_native_tool_result",
                    wraps=conversation.add_native_tool_result,
                ) as add_result,
            ):
                main_module.main()

            dispatch.assert_not_called()
            add_call.assert_called_once_with([native_call])
            add_result.assert_called_once()
            tool_messages = [m for m in conversation.get_messages() if m["role"] == "tool"]
            self.assertEqual(len(tool_messages), 1)
            self.assertEqual(tool_messages[0]["name"], "read_file")
            self.assertIn("admitted content", tool_messages[0]["content"])

            # Every streamed turn advertises the registry schemas (sorted by name).
            self.assertEqual(len(payloads), 2)
            for tools in payloads:
                self.assertIsNotNone(tools)
                self.assertEqual(
                    [t["function"]["name"] for t in tools], ["bash", "read_file"]
                )

            # System prompt uses the native section plus generated tool docs.
            self.assertNotIn("No executable tool protocol is active", prompts[0])
            self.assertIn("## read_file", prompts[0])
            self.assertIn("## bash", prompts[0])
            self.assertNotIn("native_tools_unavailable", stdout.getvalue())


class TestAdmitNativeCalls(unittest.TestCase):
    """The admission pipeline fails closed without executing handlers."""

    def test_bash_unknown_and_malformed_calls_never_execute(self):
        import tempfile

        from shadow_code.domain.approval import ApprovalAuthority
        from shadow_code.domain.tools import Capability
        from shadow_code.main import _admit_native_calls
        from shadow_code.policy.workspace import WorkspaceGuard
        from shadow_code.tools.catalog import READ_FILE_SPEC, WorkspaceContext

        calls = [
            {"call_id": "c1", "name": "bash", "arguments": {"command": "id"}},
            {"call_id": "c2", "name": "nope", "arguments": {}},
            {"call_id": "c3", "name": "", "arguments": {}},
            "not-a-dict",
        ]
        with tempfile.TemporaryDirectory() as workspace, WorkspaceGuard(workspace) as guard:
            from shadow_code.domain.policy import PolicyFacts
            from shadow_code.policy.engine import PolicyEngine
            from shadow_code.tools.registry import ToolRegistry

            registry = ToolRegistry((READ_FILE_SPEC,))
            engine = PolicyEngine(PolicyFacts({Capability.FILESYSTEM_READ}, guard.identity))
            results = _admit_native_calls(
                calls, registry, engine, WorkspaceContext(guard), ApprovalAuthority()
            )

        self.assertEqual(len(results), 4)
        self.assertEqual(
            [r.error.code for r in results],
            ["unknown_tool", "unknown_tool", "invalid_tool_call", "invalid_tool_call"],
        )
        for result in results:
            self.assertFalse(result.success)

    def _bash_admission(self):
        import tempfile

        from shadow_code.domain.policy import PolicyFacts
        from shadow_code.domain.tools import Capability
        from shadow_code.policy.engine import PolicyEngine
        from shadow_code.policy.workspace import WorkspaceGuard
        from shadow_code.tools.catalog import BASH_SPEC, READ_FILE_SPEC, WorkspaceContext
        from shadow_code.tools.registry import ToolRegistry

        workspace = tempfile.TemporaryDirectory()
        guard = WorkspaceGuard(workspace.name)
        registry = ToolRegistry((BASH_SPEC, READ_FILE_SPEC))
        capabilities = {Capability.FILESYSTEM_READ, Capability.PROCESS_EXECUTE}
        engine = PolicyEngine(PolicyFacts(capabilities, guard.identity))
        calls = [{"call_id": "b1", "name": "bash", "arguments": {"command": "id"}}]
        return workspace, guard, calls, registry, engine, WorkspaceContext(guard)

    def test_approved_side_effecting_call_reaches_executor_without_handler(self):
        from shadow_code.domain.approval import ApprovalAuthority
        from shadow_code.main import _admit_native_calls

        workspace, guard, calls, registry, engine, context = self._bash_admission()
        with workspace, guard, patch("builtins.input", return_value="y"):
            results = _admit_native_calls(calls, registry, engine, context, ApprovalAuthority())

        # Approved: the one-shot token is consumed by the executor; bash has
        # no handler yet, so the typed handler_unavailable result is returned.
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertEqual(results[0].error.code, "handler_unavailable")

    def test_denied_side_effecting_call_is_final_and_not_retried(self):
        from shadow_code.domain.approval import ApprovalAuthority
        from shadow_code.main import _admit_native_calls

        workspace, guard, calls, registry, engine, context = self._bash_admission()
        with workspace, guard, patch("builtins.input", return_value="n") as prompt:
            results = _admit_native_calls(calls, registry, engine, context, ApprovalAuthority())

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertEqual(results[0].error.code, "approval_denied")
        prompt.assert_called_once()

    def test_cancelled_side_effecting_call_is_a_typed_denial(self):
        from shadow_code.domain.approval import ApprovalAuthority
        from shadow_code.main import _admit_native_calls

        workspace, guard, calls, registry, engine, context = self._bash_admission()
        with workspace, guard, patch("builtins.input", side_effect=EOFError):
            results = _admit_native_calls(calls, registry, engine, context, ApprovalAuthority())

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertEqual(results[0].error.code, "approval_denied")

    def test_ungranted_capability_is_a_typed_policy_denial(self):
        import tempfile

        from shadow_code.domain.approval import ApprovalAuthority
        from shadow_code.domain.policy import PolicyFacts
        from shadow_code.domain.tools import Capability
        from shadow_code.main import _admit_native_calls
        from shadow_code.policy.engine import PolicyEngine
        from shadow_code.policy.workspace import WorkspaceGuard
        from shadow_code.tools.catalog import BASH_SPEC, READ_FILE_SPEC, WorkspaceContext
        from shadow_code.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as workspace, WorkspaceGuard(workspace) as guard:
            registry = ToolRegistry((BASH_SPEC, READ_FILE_SPEC))
            engine = PolicyEngine(PolicyFacts({Capability.FILESYSTEM_READ}, guard.identity))
            calls = [{"call_id": "b2", "name": "bash", "arguments": {"command": "id"}}]
            results = _admit_native_calls(
                calls, registry, engine, WorkspaceContext(guard), ApprovalAuthority()
            )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertEqual(results[0].error.code, "policy_denied")


class TestOllamaClient(unittest.TestCase):
    """Basic tests for OllamaClient (no actual server required)."""

    def test_init(self):
        from shadow_code.ollama_client import OllamaClient

        client = OllamaClient()
        self.assertEqual(client.last_prompt_tokens, 0)
        self.assertEqual(client.last_eval_tokens, 0)

    def test_health_check_connection_error(self):
        from shadow_code.ollama_client import OllamaClient

        client = OllamaClient()
        with patch("shadow_code.ollama_client.OLLAMA_BASE_URL", "http://localhost:99999"):
            ok, msg = client.health_check()
            self.assertFalse(ok)
            self.assertIsInstance(msg, str)
            self.assertGreater(len(msg), 0)

    def test_health_check_model_found(self):
        from shadow_code.ollama_client import OllamaClient

        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "test-model:latest"}]}
        mock_resp.raise_for_status = MagicMock()
        with (
            patch("shadow_code.ollama_client.MODEL_NAME", "test-model:latest"),
            patch("shadow_code.ollama_client.requests.get", return_value=mock_resp),
        ):
            ok, msg = client.health_check()
            self.assertTrue(ok)
            self.assertEqual(msg, "OK")

    def test_health_check_model_not_found(self):
        from shadow_code.ollama_client import OllamaClient

        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "other-model"}]}
        mock_resp.raise_for_status = MagicMock()
        with (
            patch("shadow_code.ollama_client.MODEL_NAME", "missing-model:latest"),
            patch("shadow_code.ollama_client.requests.get", return_value=mock_resp),
        ):
            ok, msg = client.health_check()
            self.assertFalse(ok)
            self.assertIn("not found", msg)

    def test_health_check_base_name_match(self):
        from shadow_code.ollama_client import OllamaClient

        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "shadow-gemma:v2"}]}
        mock_resp.raise_for_status = MagicMock()
        with (
            patch("shadow_code.ollama_client.MODEL_NAME", "shadow-gemma:latest"),
            patch("shadow_code.ollama_client.requests.get", return_value=mock_resp),
        ):
            ok, msg = client.health_check()
            self.assertTrue(ok)

    def test_chat_stream_mock(self):
        import json

        from shadow_code.ollama_client import OllamaClient

        client = OllamaClient()
        mock_resp = MagicMock()
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}).encode(),
            json.dumps({"message": {"content": " world"}, "done": False}).encode(),
            json.dumps(
                {
                    "message": {"content": ""},
                    "done": True,
                    "prompt_eval_count": 100,
                    "eval_count": 50,
                }
            ).encode(),
        ]
        mock_resp.iter_lines.return_value = lines
        mock_resp.raise_for_status = MagicMock()
        with patch("shadow_code.ollama_client.requests.post", return_value=mock_resp):
            chunks = list(client.chat_stream([], "system"))
            self.assertEqual(chunks, ["Hello", " world"])
            self.assertEqual(client.last_prompt_tokens, 100)
            self.assertEqual(client.last_eval_tokens, 50)

    def test_chat_stream_advertises_tools_and_collects_normalized_calls(self):
        import json

        from shadow_code.ollama_client import OllamaClient

        native_call = {"function": {"name": "read_file", "arguments": {"file_path": "a.txt"}}}
        schemas = [{"type": "function", "function": {"name": "read_file"}}]
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"tool_calls": [native_call]}, "done": True}).encode()
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("shadow_code.ollama_client.requests.post", return_value=mock_resp) as post:
            list(client.chat_stream([], "system", tools=schemas))

        self.assertEqual(post.call_args.kwargs["json"]["tools"], schemas)
        self.assertEqual(
            client.last_tool_calls,
            [{"call_id": "call-0", "name": "read_file", "arguments": {"file_path": "a.txt"}}],
        )
        self.assertFalse(hasattr(client, "last_rejected_native_calls"))

    def test_chat_stream_omits_tools_key_without_schemas(self):
        import json

        from shadow_code.ollama_client import OllamaClient

        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"content": "hi"}, "done": True}).encode()
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("shadow_code.ollama_client.requests.post", return_value=mock_resp) as post:
            self.assertEqual(list(client.chat_stream([], "system")), ["hi"])

        self.assertNotIn("tools", post.call_args.kwargs["json"])
        self.assertEqual(client.last_tool_calls, [])

    def test_chat_stream_normalizes_malformed_calls_without_executing(self):
        import json

        from shadow_code.ollama_client import OllamaClient

        raw_calls = [
            {"id": "provider-1", "function": {"name": "read_file", "arguments": '{"a": 1}'}},
            {"function": {"name": 42, "arguments": "not-json"}},
            "garbage",
        ]
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"tool_calls": raw_calls}, "done": True}).encode()
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("shadow_code.ollama_client.requests.post", return_value=mock_resp):
            list(client.chat_stream([], "system"))

        self.assertEqual(
            client.last_tool_calls,
            [
                {"call_id": "provider-1", "name": "read_file", "arguments": {"a": 1}},
                {"call_id": "call-1", "name": "", "arguments": "not-json"},
                {"call_id": "call-2", "name": "", "arguments": {}},
            ],
        )


class TestReplCreateSession(unittest.TestCase):
    """Test create_prompt_session."""

    def test_returns_none_without_prompt_toolkit(self):
        from shadow_code.repl import create_prompt_session

        with patch("shadow_code.repl._HAS_PROMPT_TOOLKIT", False):
            result = create_prompt_session()
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
