"""PR 3 Part C: fake tool-call fence detector + post-turn notice.

The detector (display.detect_text_tool_call_fence) is pure and observational:
it counts ```json/untagged fences whose body parses to a tool-call-shaped
JSON value, it never feeds the parser, and it never mutates the response.
The hook in main._handle_user_message prints ONE display-only notice per
turn; the stored AssistantTextPayload stays byte-identical to model output.
"""

import tempfile
import unittest
from unittest.mock import MagicMock

from shadow_code.display import detect_text_tool_call_fence

_FAKE_DICT = 'Here you go:\n```json\n{"tool_call": "write_file", "path": "a.txt"}\n```\nDone.'
_FAKE_LIST = (
    "```json\n"
    '[{"tool_call": "write_file", "path": "a.txt"},\n'
    ' {"tool_call": "read_file", "path": "b.txt"}]\n'
    "```"
)
_FAKE_TOOL_SHAPE = '```\n{"tool": "read_file", "params": {"file_path": "a.txt"}}\n```'
_FAKE_ARGUMENTS_SHAPE = '```json\n{"tool": "read_file", "arguments": {"file_path": "a.txt"}}\n```'
_LEGACY_FENCE = '```tool_call\n{"tool": "read_file", "params": {"file_path": "a.txt"}}\n```'
_PLAIN_JSON = 'Example payload:\n```json\n{"name": "x", "value": [1, 2]}\n```'
_MALFORMED_JSON = "```json\n{not valid json\n```"


def _run_turn(response, *, calls_executed=0):
    """Drive one _handle_user_message turn; return (writes, assistant payloads)."""
    from shadow_code.conversation import Conversation
    from shadow_code.engine import EngineResult, EngineState
    from shadow_code.main import SessionRuntime, _handle_user_message
    from shadow_code.tool_context import ToolContext

    workspace = tempfile.TemporaryDirectory()
    engine = MagicMock()
    engine.run_turn.return_value = EngineResult(
        status=EngineState.COMPLETED,
        steps=1,
        calls_executed=calls_executed,
        results=(),
        reason="",
        text=response,
    )
    prompt_manager = MagicMock()
    prompt_manager.watch.return_value = None
    prompt_manager.active.digest = "0" * 64
    client = MagicMock()
    client.last_prompt_tokens = 0
    client.last_eval_tokens = 0
    store = MagicMock()
    rt = SessionRuntime(
        cwd=workspace.name,
        ctx=ToolContext(workspace.name),
        prompt_manager=prompt_manager,
        client=client,
        engine=engine,
        conv=Conversation(),
        event_store=store,
    )
    writes = []
    _handle_user_message(rt, "hi", lambda: None, writes.append, lambda _: False)
    payloads = [
        call.args[1].payload.content
        for call in store.append.call_args_list
        if call.args[1].type == "assistant_text"
    ]
    workspace.cleanup()
    return writes, payloads


class TestDetectTextToolCallFence(unittest.TestCase):
    """The pure detector: narrow shape matching, no parser involvement."""

    def test_json_fence_with_tool_call_dict_counts_one(self):
        self.assertEqual(detect_text_tool_call_fence(_FAKE_DICT), 1)

    def test_json_fence_with_list_counts_each_entry(self):
        self.assertEqual(detect_text_tool_call_fence(_FAKE_LIST), 2)

    def test_tool_params_shape_counts(self):
        self.assertEqual(detect_text_tool_call_fence(_FAKE_TOOL_SHAPE), 1)

    def test_tool_arguments_shape_counts(self):
        self.assertEqual(detect_text_tool_call_fence(_FAKE_ARGUMENTS_SHAPE), 1)

    def test_legacy_tool_call_fence_is_not_counted(self):
        # Owned by _legacy_markdown_protocol_error; flagging it here would
        # double-report.
        self.assertEqual(detect_text_tool_call_fence(_LEGACY_FENCE), 0)

    def test_plain_json_example_is_not_counted(self):
        self.assertEqual(detect_text_tool_call_fence(_PLAIN_JSON), 0)

    def test_malformed_json_is_ignored_without_crashing(self):
        self.assertEqual(detect_text_tool_call_fence(_MALFORMED_JSON), 0)

    def test_no_fences_is_zero(self):
        self.assertEqual(detect_text_tool_call_fence("plain prose, no fences"), 0)


class TestTextToolCallNoticeHook(unittest.TestCase):
    """The post-turn hook in _handle_user_message (serves REPL and TUI)."""

    def test_notice_fires_once_with_count(self):
        writes, _ = _run_turn(_FAKE_DICT)
        notices = [w for w in writes if w.startswith("[notice:")]
        self.assertEqual(len(notices), 1)
        self.assertIn("1 tool call(s)", notices[0])
        self.assertIn("nothing was executed", notices[0])

    def test_notice_reflects_list_count(self):
        writes, _ = _run_turn(_FAKE_LIST)
        notices = [w for w in writes if w.startswith("[notice:")]
        self.assertEqual(len(notices), 1)
        self.assertIn("2 tool call(s)", notices[0])

    def test_legacy_fence_yields_protocol_error_not_notice(self):
        writes, _ = _run_turn(_LEGACY_FENCE)
        self.assertFalse(any(w.startswith("[notice:") for w in writes))
        self.assertTrue(any("protocol_mismatch" in w for w in writes))

    def test_native_tool_call_turn_yields_no_notice(self):
        writes, _ = _run_turn(None, calls_executed=1)
        self.assertFalse(any(w.startswith("[notice:") for w in writes))

    def test_plain_json_example_yields_no_notice(self):
        writes, _ = _run_turn(_PLAIN_JSON)
        self.assertFalse(any(w.startswith("[notice:") for w in writes))

    def test_malformed_json_yields_no_notice(self):
        writes, _ = _run_turn(_MALFORMED_JSON)
        self.assertFalse(any(w.startswith("[notice:") for w in writes))

    def test_stored_assistant_text_is_byte_identical_when_notice_fires(self):
        writes, payloads = _run_turn(_FAKE_DICT)
        self.assertTrue(any(w.startswith("[notice:") for w in writes))
        self.assertEqual(payloads, [_FAKE_DICT])


class TestTuiTranscriptNotice(unittest.TestCase):
    """The TUI routes write() into TranscriptModel.append_system_line."""

    def test_notice_renders_as_system_line(self):
        from shadow_code.tui import TranscriptModel

        writes, _ = _run_turn(_FAKE_DICT)
        notice = next(w for w in writes if w.startswith("[notice:"))
        model = TranscriptModel()
        model.append_system_line(notice)
        self.assertEqual(len(model.entries), 1)
        self.assertEqual(model.entries[0].kind, "system")
        self.assertEqual(model.entries[0].text, notice)


if __name__ == "__main__":
    unittest.main()
