"""Parse the opt-in legacy Markdown tool-call format.

Native provider tool calls do not pass through this module. Only explicitly
labelled ``tool_call`` fences are executable legacy messages; every other
Markdown fence is assistant prose.
"""

import json
import re
from dataclasses import dataclass


@dataclass
class LegacyMarkdownToolCall:
    tool: str
    params: dict
    raw: str


CANONICAL_FENCE_RE = re.compile(r"```tool_call\s*\n(.*?)\n```", re.DOTALL)
# Unclosed fence at end of stream — happens when the model stops generating
# right after emitting the JSON (anti-hallucination rule side effect on 7B
# models). Only match when fence is at the very end and contains valid tool JSON.
UNCLOSED_FENCE_RE = re.compile(r"```tool_call\s*\n(\{[^`]*?\})\s*\Z", re.DOTALL)


def _try_parse_tool_json(json_str: str) -> dict | None:
    """Return dict if json_str parses to a valid {tool, params} object, else None."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if (
        isinstance(data, dict)
        and isinstance(data.get("tool"), str)
        and isinstance(data.get("params"), dict)
    ):
        return data
    return None


def parse_legacy_markdown_tool_calls(text: str) -> tuple[str, list[LegacyMarkdownToolCall]]:
    """Extract explicitly labelled legacy tool calls from model response.

    Returns (clean_text, calls) where clean_text has tool call blocks removed
    and calls is a list of parsed LegacyMarkdownToolCall objects.

    Plain Markdown code fences, including ``bash`` and ``sh``, are never tool
    calls. This parser is retained only for explicit compatibility with old
    ``tool_call`` responses and must be gated at the production boundary.

    Invalid JSON inside a canonical ```tool_call``` produces a LegacyMarkdownToolCall with
    tool="__invalid__" so the caller can report the error back to the model.
    """
    calls: list[LegacyMarkdownToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    # Closed canonical fences are strict: invalid JSON is reported to the model.
    for match in CANONICAL_FENCE_RE.finditer(text):
        raw_block = match.group(0)
        json_str = match.group(1)
        data = _try_parse_tool_json(json_str)
        if data is not None:
            calls.append(
                LegacyMarkdownToolCall(tool=data["tool"], params=data["params"], raw=raw_block)
            )
        else:
            # Distinguish "invalid JSON" from "valid JSON but wrong shape"
            try:
                json.loads(json_str)
                # Valid JSON, wrong shape — skip silently (model may show example)
                continue
            except json.JSONDecodeError:
                calls.append(
                    LegacyMarkdownToolCall(
                        tool="__invalid__",
                        params={"error": f"Invalid JSON: {json_str[:200]}"},
                        raw=raw_block,
                    )
                )
        consumed_spans.append(match.span())

    # Unclosed canonical fence at end-of-stream (model stopped generating after
    # emitting JSON — common with 7B models under strict anti-halluc rules).
    if not calls:  # only fall back if nothing matched above
        m = UNCLOSED_FENCE_RE.search(text)
        if m and not any(s <= m.start() < e for s, e in consumed_spans):
            data = _try_parse_tool_json(m.group(1))
            if data is not None:
                calls.append(
                    LegacyMarkdownToolCall(tool=data["tool"], params=data["params"], raw=m.group(0))
                )
                consumed_spans.append(m.span())

    # Strip consumed spans from text
    if consumed_spans:
        consumed_spans.sort()
        parts: list[str] = []
        last = 0
        for start, end in consumed_spans:
            parts.append(text[last:start])
            last = end
        parts.append(text[last:])
        clean = "".join(parts).strip()
    else:
        clean = text.strip()
    return clean, calls
