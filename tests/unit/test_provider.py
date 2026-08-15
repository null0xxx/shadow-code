"""WU-05 provider contract tests: typed stream, assembly, cancellation, errors.

Async protocol is driven with asyncio.run (no pytest-asyncio dependency).
"""

import asyncio
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from shadow_code.provider import (
    OllamaProvider,
    ProviderError,
    ProviderStreamError,
    StreamAssembler,
    TextDelta,
    ThinkingDelta,
    ToolCallArgumentsDelta,
    ToolCallComplete,
    ToolCallStarted,
    TurnDone,
    UsageUpdate,
    collect_turn,
    iter_events_sync,
    thaw_arguments,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "provider"


def fixture_lines(name: str) -> list[bytes]:
    return (FIXTURES / name).read_bytes().splitlines()


class FakeResponse:
    """Minimal requests.Response stand-in backed by recorded lines."""

    def __init__(self, lines: list[bytes], status_error: Exception | None = None):
        self._lines = lines
        self._status_error = status_error
        self.closed = False

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def iter_lines(self):
        yield from self._lines

    def close(self) -> None:
        self.closed = True


def run_provider(lines=None, *, response=None, post_side_effect=None):
    """Drain OllamaProvider.stream with a patched requests.post."""
    provider = OllamaProvider("http://fixture.invalid", {"temperature": 0})
    if response is None:
        response = FakeResponse(lines or [])
    with patch("shadow_code.provider.requests.post") as post:
        if post_side_effect is not None:
            post.side_effect = post_side_effect
        else:
            post.return_value = response

        async def drain():
            return [event async for event in provider.stream([], "system")]

        events = asyncio.run(drain())
    return events, post


def events_of(events, kind):
    return [event for event in events if isinstance(event, kind)]


class TestProviderFixtures(unittest.TestCase):
    def test_text_only_fixture(self):
        events, _ = run_provider(fixture_lines("text_only.ndjson"))
        self.assertEqual(
            [type(event) for event in events],
            [TextDelta, TextDelta, UsageUpdate, TurnDone],
        )
        self.assertEqual(events[0].text, "Hello")
        self.assertEqual(events[1].text, " world")
        self.assertEqual((events[2].prompt_tokens, events[2].eval_tokens), (11, 7))
        self.assertEqual(events[3].stop_reason, "stop")

    def test_one_call_fixture(self):
        events, _ = run_provider(fixture_lines("one_call.ndjson"))
        completes = events_of(events, ToolCallComplete)
        self.assertEqual(len(completes), 1)
        call = completes[0]
        self.assertEqual(call.call_id, "provider-abc")
        self.assertEqual(call.name, "read_file")
        self.assertEqual(dict(call.arguments), {"file_path": "a.txt"})
        self.assertEqual(events_of(events, ToolCallArgumentsDelta), [])

    def test_multiple_calls_preserve_order_and_identity(self):
        events, _ = run_provider(fixture_lines("multiple_calls.ndjson"))
        completes = events_of(events, ToolCallComplete)
        self.assertEqual(
            [(c.index, c.call_id, c.name) for c in completes],
            [
                (0, "provider-1", "read_file"),
                (1, "provider-2", "read_file"),
                (2, "call-2", "list_dir"),
            ],
        )
        self.assertEqual(dict(completes[2].arguments), {"path": "."})

    def test_mixed_text_and_calls_preserved(self):
        events, _ = run_provider(fixture_lines("mixed_text_and_calls.ndjson"))
        self.assertEqual(
            [event.text for event in events_of(events, TextDelta)],
            ["Let me read that file. ", "Done."],
        )
        completes = events_of(events, ToolCallComplete)
        self.assertEqual(len(completes), 1)
        self.assertEqual(completes[0].name, "read_file")
        # Text order around the call is preserved in the event stream.
        self.assertLess(
            events.index(events_of(events, TextDelta)[0]),
            events.index(events_of(events, ToolCallStarted)[0]),
        )
        self.assertLess(
            events.index(events_of(events, ToolCallStarted)[0]),
            events.index(events_of(events, TextDelta)[1]),
        )

    def test_fragmented_arguments_emit_exactly_one_complete_call(self):
        events, _ = run_provider(fixture_lines("fragmented_arguments.ndjson"))
        deltas = events_of(events, ToolCallArgumentsDelta)
        self.assertEqual(
            [d.fragment for d in deltas], ['{"file_path": "x.txt", "con', 'tent": "hi"}']
        )
        completes = events_of(events, ToolCallComplete)
        self.assertEqual(len(completes), 1)
        self.assertEqual(
            dict(completes[0].arguments),
            {"file_path": "x.txt", "content": "hi"},
        )
        # No call completes before every fragment arrived.
        self.assertGreater(events.index(completes[0]), events.index(deltas[-1]))

    def test_missing_ids_generate_stable_ids(self):
        events, _ = run_provider(fixture_lines("missing_ids.ndjson"))
        completes = events_of(events, ToolCallComplete)
        self.assertEqual([c.call_id for c in completes], ["call-0", "call-1"])
        self.assertEqual([c.index for c in completes], [0, 1])

    def test_malformed_json_line_is_a_typed_terminal_error(self):
        events, _ = run_provider(fixture_lines("malformed_json_line.ndjson"))
        self.assertEqual(events_of(events, TextDelta)[0].text, "partial")
        errors = events_of(events, ProviderError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].code, "malformed_payload")
        self.assertIs(events[-1], errors[0])  # terminal: nothing after it

    def test_unknown_fields_are_ignored(self):
        events, _ = run_provider(fixture_lines("unknown_fields.ndjson"))
        self.assertEqual([e.text for e in events_of(events, TextDelta)], ["ok"])
        completes = events_of(events, ToolCallComplete)
        self.assertEqual(completes[0].name, "glob")
        self.assertEqual(dict(completes[0].arguments), {"pattern": "*.py"})
        self.assertEqual(events_of(events, ProviderError), [])

    def test_provider_dicts_never_cross_the_boundary(self):
        events, _ = run_provider(fixture_lines("one_call.ndjson"))
        call = events_of(events, ToolCallComplete)[0]
        self.assertFalse(isinstance(call.arguments, dict))
        self.assertEqual(thaw_arguments(call.arguments), {"file_path": "a.txt"})


class TestProviderTransportErrors(unittest.TestCase):
    def test_http_error_status(self):
        response = FakeResponse([], status_error=requests.HTTPError("500 boom"))
        events, _ = run_provider(response=response)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].code, "http_error")
        self.assertIn("500 boom", events[0].message)
        self.assertTrue(response.closed)

    def test_connect_timeout(self):
        events, _ = run_provider(post_side_effect=requests.ConnectTimeout("connect timed out"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].code, "timeout")

    def test_read_timeout(self):
        events, _ = run_provider(post_side_effect=requests.ReadTimeout("read timed out"))
        self.assertEqual(events[0].code, "timeout")

    def test_mid_stream_disconnect_escapes_nothing_partial(self):
        class DisconnectResponse(FakeResponse):
            def iter_lines(self):
                yield json.dumps(
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "write_file", "arguments": '{"par'}}
                            ]
                        },
                        "done": False,
                    }
                ).encode()
                raise requests.exceptions.ChunkedEncodingError("connection reset")

        events, _ = run_provider(response=DisconnectResponse([]))
        # The partial call never completes; the only terminal fact is the error.
        self.assertEqual(events_of(events, ToolCallComplete), [])
        self.assertEqual(len(events_of(events, ToolCallStarted)), 1)
        self.assertEqual(events[-1].code, "disconnect")

    def test_clean_eof_without_done_is_a_disconnect(self):
        lines = [json.dumps({"message": {"content": "hi"}, "done": False}).encode()]
        events, _ = run_provider(lines)
        self.assertEqual(events_of(events, TextDelta)[0].text, "hi")
        self.assertEqual(events[-1].code, "disconnect")

    def test_cancellation_closes_response_and_stops_thread(self):
        gate = threading.Event()
        state = {"closed": False, "iter_finished": False}

        class BlockingResponse:
            def raise_for_status(self):
                pass

            def iter_lines(self):
                yield b'{"message": {"content": "hi"}, "done": false}'
                gate.wait(10)
                state["iter_finished"] = True

            def close(self):
                state["closed"] = True
                gate.set()

        with patch("shadow_code.provider.requests.post", return_value=BlockingResponse()):
            provider = OllamaProvider("http://fixture.invalid")

            async def run():
                stream = provider.stream([], "system")
                first = await anext(stream)
                self.assertEqual(first, TextDelta("hi"))
                await stream.aclose()

            asyncio.run(run())

        self.assertTrue(state["closed"])
        self.assertTrue(state["iter_finished"])

    def test_sync_bridge_close_cancels_the_stream(self):
        response = FakeResponse(
            [json.dumps({"message": {"content": "hi"}, "done": False}).encode()]
        )
        with patch("shadow_code.provider.requests.post", return_value=response):
            provider = OllamaProvider("http://fixture.invalid")
            stream = iter_events_sync(provider.stream([], "system"))
            self.assertEqual(next(stream), TextDelta("hi"))
            stream.close()
        self.assertTrue(response.closed)


class TestStreamAssembler(unittest.TestCase):
    def test_partial_arguments_never_leave_before_flush(self):
        assembler = StreamAssembler()
        first = assembler.feed(
            {
                "message": {
                    "tool_calls": [
                        {"index": 0, "function": {"name": "write_file", "arguments": '{"a"'}}
                    ]
                },
                "done": False,
            }
        )
        self.assertEqual([type(e) for e in first], [ToolCallStarted, ToolCallArgumentsDelta])
        second = assembler.feed(
            {
                "message": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]},
                "done": False,
            }
        )
        self.assertEqual([type(e) for e in second], [ToolCallArgumentsDelta])
        done = assembler.feed({"message": {"content": ""}, "done": True, "done_reason": "stop"})
        completes = events_of(done, ToolCallComplete)
        self.assertEqual(len(completes), 1)
        self.assertEqual(dict(completes[0].arguments), {"a": 1})

    def test_unparseable_final_arguments_are_carried_raw(self):
        assembler = StreamAssembler()
        assembler.feed(
            {
                "message": {"tool_calls": [{"function": {"name": "x", "arguments": "not-json"}}]},
                "done": False,
            }
        )
        completes = events_of(assembler.finish(), ToolCallComplete)
        self.assertEqual(completes[0].arguments, "not-json")

    def test_announced_slot_without_payload_is_a_typed_error(self):
        assembler = StreamAssembler()
        assembler.feed({"message": {"tool_calls": [{"index": 2}]}, "done": False})
        errors = events_of(assembler.finish(), ProviderError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].code, "incomplete_tool_call")

    def test_finish_is_idempotent(self):
        assembler = StreamAssembler()
        assembler.feed(
            {
                "message": {"tool_calls": [{"function": {"name": "x", "arguments": {"a": 1}}}]},
                "done": True,
            }
        )
        self.assertEqual(assembler.finish(), [])


class TestCollectTurn(unittest.TestCase):
    def test_aggregates_text_calls_usage_and_stop_reason(self):
        events, _ = run_provider(fixture_lines("mixed_text_and_calls.ndjson"))

        async def replay():
            for event in events:
                yield event

        turn = asyncio.run(collect_turn(replay()))
        self.assertEqual(turn.text, "Let me read that file. Done.")
        self.assertEqual(len(turn.calls), 1)
        self.assertEqual(turn.calls[0].name, "read_file")
        self.assertEqual((turn.prompt_tokens, turn.eval_tokens), (40, 21))
        self.assertEqual(turn.stop_reason, "stop")

    def test_raises_typed_error_on_provider_error(self):
        async def failing():
            yield TextDelta("partial")
            yield ProviderError(code="timeout", message="boom")

        with self.assertRaises(ProviderStreamError) as ctx:
            asyncio.run(collect_turn(failing()))
        self.assertEqual(ctx.exception.code, "timeout")


class TestChatStreamAdapterParity(unittest.TestCase):
    def test_chat_stream_matches_fixture_events(self):
        from shadow_code.ollama_client import OllamaClient

        lines = fixture_lines("multiple_calls.ndjson")
        with patch("shadow_code.provider.requests.post", return_value=FakeResponse(lines)):
            client = OllamaClient()
            chunks = list(client.chat_stream([], "system"))

        self.assertEqual(chunks, [])
        self.assertEqual(
            client.last_tool_calls,
            [
                {
                    "call_id": "provider-1",
                    "name": "read_file",
                    "arguments": {"file_path": "one.txt"},
                },
                {
                    "call_id": "provider-2",
                    "name": "read_file",
                    "arguments": {"file_path": "two.txt"},
                },
                {"call_id": "call-2", "name": "list_dir", "arguments": {"path": "."}},
            ],
        )
        # Plain JSON-compatible dicts cross into the legacy surface.
        for call in client.last_tool_calls:
            json.dumps(call)
        self.assertEqual(client.last_prompt_tokens, 30)
        self.assertEqual(client.last_eval_tokens, 15)

    def test_chat_stream_text_parity(self):
        from shadow_code.ollama_client import OllamaClient

        lines = fixture_lines("text_only.ndjson")
        with patch("shadow_code.provider.requests.post", return_value=FakeResponse(lines)):
            client = OllamaClient()
            chunks = list(client.chat_stream([], "system"))

        self.assertEqual(chunks, ["Hello", " world"])
        self.assertEqual(client.last_tool_calls, [])
        self.assertEqual(client.last_eval_tokens, 7)

    def test_chat_stream_raises_typed_error_on_malformed_payload(self):
        from shadow_code.ollama_client import OllamaClient

        lines = fixture_lines("malformed_json_line.ndjson")
        with patch("shadow_code.provider.requests.post", return_value=FakeResponse(lines)):
            client = OllamaClient()
            stream = client.chat_stream([], "system")
            self.assertEqual(next(stream), "partial")
            with self.assertRaises(ProviderStreamError) as ctx:
                list(stream)
        self.assertEqual(ctx.exception.code, "malformed_payload")


class TestThinkingChannel(unittest.TestCase):
    """SHADOW_THINK: ThinkingDelta is a display-only side channel."""

    def test_thinking_field_parses_into_thinking_delta(self):
        assembler = StreamAssembler()
        events = assembler.feed(
            {"message": {"thinking": "pondering ", "content": "answer"}, "done": False}
        )
        self.assertEqual(events, [ThinkingDelta("pondering "), TextDelta("answer")])

    def test_chunk_without_thinking_field_emits_no_thinking_events(self):
        assembler = StreamAssembler()
        events = assembler.feed({"message": {"content": "hi"}, "done": True})
        self.assertEqual(events_of(events, ThinkingDelta), [])
        self.assertEqual(events_of(events, TextDelta), [TextDelta("hi")])

    def test_thinking_never_enters_tool_arguments(self):
        assembler = StreamAssembler()
        events = assembler.feed(
            {
                "message": {
                    "thinking": "I should read a.txt",
                    "tool_calls": [
                        {"function": {"name": "read_file", "arguments": {"file_path": "a.txt"}}}
                    ],
                },
                "done": True,
            }
        )
        completes = events_of(events, ToolCallComplete)
        self.assertEqual(len(completes), 1)
        self.assertEqual(dict(completes[0].arguments), {"file_path": "a.txt"})
        self.assertEqual(events_of(events, ThinkingDelta), [ThinkingDelta("I should read a.txt")])

    def test_collect_turn_excludes_thinking_from_text_and_tokens(self):
        async def replay():
            yield ThinkingDelta("hidden reasoning")
            yield TextDelta("visible")
            yield UsageUpdate(prompt_tokens=10, eval_tokens=7)
            yield TurnDone("stop")

        turn = asyncio.run(collect_turn(replay()))
        self.assertEqual(turn.text, "visible")
        self.assertEqual((turn.prompt_tokens, turn.eval_tokens), (10, 7))

    def test_think_true_sent_only_when_enabled(self):
        def payload_for(think: bool):
            provider = OllamaProvider("http://fixture.invalid", {}, think=think)
            with patch("shadow_code.provider.requests.post", return_value=FakeResponse([])) as post:

                async def drain():
                    return [event async for event in provider.stream([], "system")]

                asyncio.run(drain())
            return post.call_args.kwargs["json"]

        self.assertEqual(payload_for(True)["think"], True)
        self.assertNotIn("think", payload_for(False))

    def test_thinking_stream_keeps_text_and_calls_intact(self):
        lines = [
            json.dumps({"message": {"thinking": "hmm"}, "done": False}).encode(),
            json.dumps(
                {
                    "message": {
                        "thinking": "...",
                        "tool_calls": [
                            {"function": {"name": "glob", "arguments": {"pattern": "*.py"}}}
                        ],
                    },
                    "done": False,
                }
            ).encode(),
            json.dumps(
                {
                    "message": {"content": "done"},
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 9,
                }
            ).encode(),
        ]
        events, _ = run_provider(lines)
        self.assertEqual(
            events_of(events, ThinkingDelta), [ThinkingDelta("hmm"), ThinkingDelta("...")]
        )
        self.assertEqual(events_of(events, TextDelta), [TextDelta("done")])
        completes = events_of(events, ToolCallComplete)
        self.assertEqual(len(completes), 1)
        self.assertEqual(dict(completes[0].arguments), {"pattern": "*.py"})


if __name__ == "__main__":
    unittest.main()
