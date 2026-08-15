# shadow_code/provider.py -- Provider-neutral streaming contract (WU-05)
#
# One typed stream for text, usage, stop reasons, and native tool calls,
# independent of UI and engine. Raw provider dictionaries never leave this
# module: Ollama NDJSON chunks enter StreamAssembler and only frozen typed
# events come out. Partial tool-call arguments are accumulated per call and
# parsed only at completion, so a fragmented call can never execute early.

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast

import requests

# ---------------------------------------------------------------------------
# Typed stream events (the contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    """Display-only reasoning text (Ollama ``think: true``).

    A separate channel: it never feeds tool-call assembly, never joins the
    stored assistant text, and is never persisted or replayed. UI layers may
    render it dimmed; every other consumer ignores it.
    """

    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    index: int
    call_id: str
    name: str


@dataclass(frozen=True)
class ToolCallArgumentsDelta:
    index: int
    fragment: str


@dataclass(frozen=True)
class ToolCallComplete:
    """A finished tool call. arguments is an immutable mapping; a final
    argument string that does not parse is carried raw so downstream
    validation fails closed instead of executing a guess."""

    index: int
    call_id: str
    name: str
    arguments: Mapping[str, Any] | str


@dataclass(frozen=True)
class UsageUpdate:
    prompt_tokens: int
    eval_tokens: int


@dataclass(frozen=True)
class TurnDone:
    stop_reason: str


@dataclass(frozen=True)
class ProviderError:
    """Terminal stream failure with a stable code."""

    code: str  # http_error | timeout | disconnect | malformed_payload | incomplete_tool_call
    message: str


ProviderEvent: TypeAlias = (
    TextDelta
    | ThinkingDelta
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | ToolCallComplete
    | UsageUpdate
    | TurnDone
    | ProviderError
)


class ProviderStreamError(Exception):
    """Raised by collectors when a ProviderError event terminates a stream."""

    def __init__(self, error: ProviderError):
        super().__init__(f"[{error.code}] {error.message}")
        self.error = error
        self.code = error.code


@dataclass(frozen=True)
class ProviderTurn:
    """Aggregate of one completed provider turn."""

    text: str
    calls: tuple[ToolCallComplete, ...]
    prompt_tokens: int
    eval_tokens: int
    stop_reason: str


# ---------------------------------------------------------------------------
# Argument freezing: parsed provider JSON crosses the boundary immutable
# ---------------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def thaw_arguments(value: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-convert frozen event arguments back to plain JSON-compatible data."""
    return cast(dict[str, Any], _thaw(value))


# ---------------------------------------------------------------------------
# StreamAssembler: parsed chunk dicts in, typed events out (pure, testable)
# ---------------------------------------------------------------------------


@dataclass
class _CallBuffer:
    index: int
    call_id: str
    name: str
    explicit_index: bool
    fragments: list[str] = field(default_factory=list)
    final: Mapping[str, Any] | str | None = None
    flushed: bool = False


def _call_identity(raw: Mapping[str, Any], index: int) -> tuple[str, str, Any]:
    """Extract (call_id, name, arguments) from a provider tool-call envelope.

    Malformed envelopes normalize to empty name/arguments; downstream
    registry validation fails closed on them. Never raises.
    """
    function = raw.get("function")
    function = function if isinstance(function, Mapping) else {}
    call_id = raw.get("id") or raw.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call-{index}"
    name = function.get("name", "")
    if not isinstance(name, str):
        name = ""
    return call_id, name, function.get("arguments", {})


class StreamAssembler:
    """Assemble typed events from parsed Ollama NDJSON chunk dicts.

    String arguments accumulate per call index and calls complete only when
    the turn flushes (done chunk or finish()); partial arguments never leave
    the assembler, and completions always preserve call order and identity.
    """

    def __init__(self) -> None:
        self._buffers: dict[int, _CallBuffer] = {}
        self._next_index = 0

    def feed(self, chunk: Mapping[str, Any]) -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        message = chunk.get("message")
        message = message if isinstance(message, Mapping) else {}

        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            # Reasoning channel: emitted alongside content, assembled nowhere.
            events.append(ThinkingDelta(thinking))

        content = message.get("content")
        if isinstance(content, str) and content:
            events.append(TextDelta(content))

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for raw in tool_calls:
                events.extend(self._feed_call(raw))

        if chunk.get("done") is True:
            events.extend(self._flush_pending())
            prompt_tokens = chunk.get("prompt_eval_count")
            eval_tokens = chunk.get("eval_count")
            events.append(
                UsageUpdate(
                    prompt_tokens if isinstance(prompt_tokens, int) else 0,
                    eval_tokens if isinstance(eval_tokens, int) else 0,
                )
            )
            reason = chunk.get("done_reason")
            events.append(TurnDone(reason if isinstance(reason, str) else ""))
        return events

    def finish(self) -> list[ProviderEvent]:
        """Flush pending calls at end of stream. Nothing partial escapes."""
        return self._flush_pending()

    def _flush_pending(self) -> list[ProviderEvent]:
        # Calls complete in index order so the typed stream always preserves
        # call order and identity, regardless of chunk interleaving.
        events: list[ProviderEvent] = []
        for index in sorted(self._buffers):
            buffer = self._buffers[index]
            if buffer.flushed:
                continue
            buffer.flushed = True
            if (
                buffer.explicit_index
                and not buffer.name
                and not buffer.fragments
                and not buffer.final
            ):
                events.append(
                    ProviderError(
                        code="incomplete_tool_call",
                        message=f"tool call slot {index} never received a payload",
                    )
                )
                continue
            if buffer.final is not None:
                events.append(ToolCallComplete(index, buffer.call_id, buffer.name, buffer.final))
                continue
            raw = "".join(buffer.fragments)
            events.append(self._complete(buffer, raw))
        return events

    def _feed_call(self, raw: object) -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        raw_map = raw if isinstance(raw, Mapping) else {}
        index_value = raw_map.get("index")
        if isinstance(index_value, bool) or not isinstance(index_value, int):
            explicit = False
            index = self._next_index
        else:
            explicit = True
            index = index_value
        self._next_index = max(self._next_index, index + 1)

        buffer = self._buffers.get(index)
        if buffer is None:
            call_id, name, _ = _call_identity(raw_map, index)
            buffer = _CallBuffer(index, call_id, name, explicit_index=explicit)
            self._buffers[index] = buffer
            events.append(ToolCallStarted(index, buffer.call_id, buffer.name))

        arguments = _call_identity(raw_map, index)[2]
        if isinstance(arguments, str):
            buffer.fragments.append(arguments)
            events.append(ToolCallArgumentsDelta(index, arguments))
        elif isinstance(arguments, Mapping):
            # An empty mapping with pending fragments is a no-op; the flush
            # merges the accumulated fragments into the final arguments.
            if (arguments or not buffer.fragments) and buffer.final is None:
                buffer.final = _freeze(arguments)
        elif arguments:
            # Non-string, non-mapping scalar: keep it unexecutable by
            # carrying the JSON encoding as a raw string (fails closed).
            if buffer.final is None:
                buffer.final = json.dumps(arguments)
        # Missing/empty arguments complete as {} at flush time.
        return events

    @staticmethod
    def _complete(buffer: _CallBuffer, raw: str) -> ToolCallComplete:
        arguments: Mapping[str, Any] | str
        if not raw:
            arguments = MappingProxyType({})
        else:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
            arguments = _freeze(parsed) if isinstance(parsed, Mapping) else raw
        return ToolCallComplete(buffer.index, buffer.call_id, buffer.name, arguments)


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class ProviderProtocol(Protocol):
    """Async streaming contract every provider implements."""

    def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ProviderEvent]: ...


class OllamaProvider:
    """Ollama /api/chat provider over requests.

    The blocking NDJSON read loop runs in a worker thread that feeds an
    asyncio.Queue. Closing the async generator (or task cancellation) closes
    the HTTP response and stops the thread.
    """

    def __init__(
        self,
        base_url: str,
        options: Mapping[str, Any] | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
        think: bool = False,
    ):
        self._base_url = base_url.rstrip("/")
        self._options = dict(options or {})
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._think = think

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + list(messages),
            "stream": True,
            "options": self._options,
        }
        if self._think:
            # Ollama takes `think` as a top-level chat field (not a model
            # option); the model answers with a separate message.thinking
            # channel that StreamAssembler surfaces as ThinkingDelta.
            payload["think"] = True
        if tools is not None:
            payload["tools"] = tools

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ProviderEvent | None] = asyncio.Queue()
        cancel = threading.Event()
        holder: dict[str, Any] = {}

        def emit(item: ProviderEvent | None) -> None:
            with suppress(RuntimeError):  # loop already closed on shutdown
                loop.call_soon_threadsafe(queue.put_nowait, item)

        worker = threading.Thread(
            target=self._read_loop,
            args=(payload, emit, cancel, holder),
            name="ollama-provider",
            daemon=True,
        )
        worker.start()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item
                if isinstance(item, ProviderError):
                    return
        finally:
            cancel.set()
            response = holder.get("response")
            if response is not None:
                response.close()
            worker.join(timeout=5.0)

    def _read_loop(
        self,
        payload: dict[str, Any],
        emit: Any,
        cancel: threading.Event,
        holder: dict[str, Any],
    ) -> None:
        assembler = StreamAssembler()
        response = None
        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=(self._connect_timeout, self._read_timeout),
            )
            holder["response"] = response
            response.raise_for_status()
            saw_done = False
            for line in response.iter_lines():
                if cancel.is_set():
                    return
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    emit(
                        ProviderError(
                            code="malformed_payload",
                            message=f"unparseable NDJSON line: {line[:80]!r}",
                        )
                    )
                    return
                if not isinstance(chunk, dict):
                    emit(
                        ProviderError(
                            code="malformed_payload",
                            message=f"NDJSON line is not an object: {line[:80]!r}",
                        )
                    )
                    return
                for event in assembler.feed(chunk):
                    emit(event)
                if chunk.get("done") is True:
                    saw_done = True
            for event in assembler.finish():
                emit(event)
            if not saw_done and not cancel.is_set():
                emit(
                    ProviderError(
                        code="disconnect",
                        message="stream ended before a done chunk",
                    )
                )
        except requests.Timeout as e:
            emit(ProviderError(code="timeout", message=str(e)))
        except requests.HTTPError as e:
            emit(ProviderError(code="http_error", message=str(e)))
        except requests.RequestException as e:
            if not cancel.is_set():
                emit(ProviderError(code="disconnect", message=str(e)))
        except Exception as e:  # closing the response mid-read lands here
            if not cancel.is_set():
                emit(ProviderError(code="disconnect", message=str(e)))
        finally:
            if response is not None:
                response.close()
            emit(None)


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


async def collect_turn(events: AsyncIterator[ProviderEvent]) -> ProviderTurn:
    """Drain an event stream into a ProviderTurn; raises on ProviderError."""
    text_parts: list[str] = []
    calls: list[ToolCallComplete] = []
    prompt_tokens = 0
    eval_tokens = 0
    stop_reason = ""
    async for event in events:
        if isinstance(event, TextDelta):
            text_parts.append(event.text)
        elif isinstance(event, ToolCallComplete):
            calls.append(event)
        elif isinstance(event, UsageUpdate):
            prompt_tokens = event.prompt_tokens
            eval_tokens = event.eval_tokens
        elif isinstance(event, TurnDone):
            stop_reason = event.stop_reason
        elif isinstance(event, ProviderError):
            raise ProviderStreamError(event)
    return ProviderTurn(
        text="".join(text_parts),
        calls=tuple(calls),
        prompt_tokens=prompt_tokens,
        eval_tokens=eval_tokens,
        stop_reason=stop_reason,
    )


def iter_events_sync(events: AsyncIterator[ProviderEvent]) -> Iterator[ProviderEvent]:
    """Drive an async provider stream from sync code.

    Closing this generator acloses the underlying stream, which cancels the
    provider: its HTTP response is closed and the worker thread stops.
    """
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                event = loop.run_until_complete(events.__anext__())
            except StopAsyncIteration:
                return
            yield event
    finally:
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                loop.run_until_complete(aclose())
        loop.close()
