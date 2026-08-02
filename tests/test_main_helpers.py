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


class TestNativeToolBoundary(unittest.TestCase):
    """Native calls remain non-executable until admission wiring exists."""

    def test_main_rejects_requested_and_unsolicited_native_calls(self):
        import importlib

        from shadow_code.conversation import Conversation

        main_module = importlib.import_module("shadow_code.main")
        native_call = {"function": {"name": "bash", "arguments": {"command": "id"}}}

        for requested in (False, True):
            client = MagicMock()
            client.health_check.return_value = (True, "OK")
            client.last_prompt_tokens = 0
            client.last_eval_tokens = 0
            responses = iter((([], [native_call]), (["finished"], [])))
            prompts = []

            def chat_stream(messages, system, *, _client=client, _state=(prompts, responses)):
                _state[0].append(system)
                chunks, _client.last_tool_calls = next(_state[1])
                return iter(chunks)

            client.chat_stream.side_effect = chat_stream
            conversation = Conversation()
            with (
                self.subTest(requested=requested),
                patch.object(main_module, "NATIVE_TOOLS", requested, create=True),
                patch.object(main_module, "_RICH", False),
                patch.object(main_module, "_HAS_REPL", False),
                patch.object(main_module, "_HAS_DB", False),
                patch.object(main_module, "OllamaClient", return_value=client),
                patch.object(main_module, "Conversation", return_value=conversation),
                patch.object(main_module, "_register_optional_tools"),
                patch.object(main_module.tool_reg, "register"),
                patch.object(
                    main_module.tool_reg,
                    "dispatch",
                    return_value=main_module.tool_reg.ToolResult(True, "unexpected"),
                ) as dispatch,
                patch.object(main_module.signal, "signal"),
                patch("builtins.input", side_effect=["do something", "/exit"]),
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
            add_call.assert_not_called()
            add_result.assert_not_called()
            self.assertIn("No executable tool protocol is active", prompts[0])
            self.assertIn("native_tools_unavailable", stdout.getvalue())


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

    def test_chat_stream_never_advertises_native_schemas_before_admission_wiring(self):
        import json

        from shadow_code.ollama_client import OllamaClient

        native_call = {"function": {"name": "bash", "arguments": {"command": "id"}}}
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"tool_calls": [native_call]}, "done": True}).encode()
        ]
        mock_resp.raise_for_status = MagicMock()
        with (
            patch("shadow_code.ollama_client.NATIVE_TOOLS", True, create=True),
            patch("shadow_code.ollama_client.requests.post", return_value=mock_resp) as post,
        ):
            list(client.chat_stream([], "system"))

        self.assertNotIn("tools", post.call_args.kwargs["json"])
        self.assertEqual(client.last_rejected_native_calls, [native_call])
        self.assertEqual(getattr(client, "last_tool_calls", []), [])


class TestReplCreateSession(unittest.TestCase):
    """Test create_prompt_session."""

    def test_returns_none_without_prompt_toolkit(self):
        from shadow_code.repl import create_prompt_session

        with patch("shadow_code.repl._HAS_PROMPT_TOOLKIT", False):
            result = create_prompt_session()
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
