# shadow_code/ollama_client.py -- Ollama API client with native tool support
#
# Supports both streaming text and native tool calling (Gemma 4+).
# Tracks prompt_eval_count and eval_count for context management.
# Streaming itself lives in provider.py; this client is a thin sync adapter
# that preserves the historical chat_stream/last_tool_calls surface.

import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any

import requests

from .config import MODEL_NAME, MODEL_OPTIONS, OLLAMA_BASE_URL, THINK_ENABLED
from .provider import (
    OllamaProvider,
    ProviderError,
    ProviderStreamError,
    TextDelta,
    ThinkingDelta,
    ToolCallComplete,
    UsageUpdate,
    iter_events_sync,
    thaw_arguments,
)
from .tools.projections import flat_tool_schema
from .tools.registry import ToolRegistry


def render_ollama_tool_schemas(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Project a registry into deterministic Ollama function envelopes."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": flat_tool_schema(spec),
            },
        }
        for spec in registry.specs
    ]


# Tool schemas for Ollama native tool calling API
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file with line numbers. Use absolute paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to file"},
                    "offset": {"type": "integer", "description": "Starting line (1-based)"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file. Must read_file first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path"},
                    "old_string": {"type": "string", "description": "Exact text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file. Must read_file first for existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path"},
                    "content": {"type": "string", "description": "File content to write"},
                    "append": {"type": "boolean", "description": "Append mode for large files"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern like **/*.py"},
                    "path": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with regex",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "File or directory to search"},
                    "include": {"type": "string", "description": "File filter like *.py"},
                    "case_insensitive": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory contents with sizes",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_read",
            "description": "Read up to 10 files in one call for project orientation",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of absolute file paths",
                    },
                    "limit": {"type": "integer", "description": "Lines per file"},
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_summary",
            "description": "Detect project language, framework, and structure",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Project root directory"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_backup",
            "description": "Backup a file before risky edits",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to backup"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_restore",
            "description": "Restore a file from backup",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to restore"},
                },
                "required": ["file_path"],
            },
        },
    },
]


def _normalize_tool_call(raw: object, fallback_id: str) -> dict[str, Any]:
    """Compatibility shim: normalize one provider tool-call envelope.

    The streaming path normalizes inside StreamAssembler (provider.py);
    this helper keeps the historical single-call shape for importers.
    Malformed payloads are kept as structured data (never executed); the
    registry's validate_call fails closed on them downstream.
    """
    if not isinstance(raw, dict):
        return {"call_id": fallback_id, "name": "", "arguments": {}}
    function = raw.get("function")
    function = function if isinstance(function, dict) else {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        with suppress(json.JSONDecodeError):
            arguments = json.loads(arguments)
        # an unparseable string stays raw; envelope validation fails closed
    call_id = raw.get("id") or raw.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        call_id = fallback_id
    name = function.get("name", "")
    return {
        "call_id": call_id,
        "name": name if isinstance(name, str) else "",
        "arguments": arguments,
    }


class OllamaClient:
    """Client for Ollama /api/chat with native tool calling support."""

    def __init__(self):
        self.last_prompt_tokens: int = 0
        self.last_eval_tokens: int = 0
        self.last_tool_calls: list[dict] = []
        # Display-only sink for ThinkingDelta events (SHADOW_THINK). The
        # streaming frontends install a handler for the duration of one
        # stream; None means thinking text is dropped silently.
        self.thinking_handler: Callable[[str], None] | None = None
        self._provider = OllamaProvider(OLLAMA_BASE_URL, MODEL_OPTIONS, think=THINK_ENABLED)

    def health_check(self) -> tuple[bool, str]:
        """Verify Ollama is running and model is available."""
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if MODEL_NAME not in models:
                base_name = MODEL_NAME.split(":")[0]
                found = [m for m in models if m.startswith(base_name)]
                if not found:
                    return False, f"Model '{MODEL_NAME}' not found. Available: {models}"
            return True, "OK"
        except requests.ConnectionError:
            return False, f"Cannot connect to Ollama at {OLLAMA_BASE_URL}"
        except requests.RequestException as e:
            return False, f"Ollama error: {e}"

    def chat_stream(
        self,
        messages: list[dict],
        system: str,
        model: str | None = None,
        tools: list[dict] | None = None,
    ):
        """Stream text and collect native tool calls for the admission pipeline.

        Thin sync adapter over provider.py: the typed event stream is the
        single streaming implementation; this method only projects it onto
        the historical text-chunk/last_tool_calls surface.

        Yields:
            str: Text content chunks.
        """
        self.last_tool_calls = []

        events = self._provider.stream(messages, system, model or MODEL_NAME, tools)
        for event in iter_events_sync(events):
            if isinstance(event, TextDelta):
                yield event.text
            elif isinstance(event, ThinkingDelta):
                # Display-only: routed to the installed handler, never yielded
                # as text and never recorded on the client.
                if self.thinking_handler is not None:
                    self.thinking_handler(event.text)
            elif isinstance(event, ToolCallComplete):
                arguments = event.arguments
                self.last_tool_calls.append(
                    {
                        "call_id": event.call_id,
                        "name": event.name,
                        "arguments": thaw_arguments(arguments)
                        if isinstance(arguments, Mapping)
                        else arguments,
                    }
                )
            elif isinstance(event, UsageUpdate):
                self.last_prompt_tokens = event.prompt_tokens
                self.last_eval_tokens = event.eval_tokens
            elif isinstance(event, ProviderError):
                raise ProviderStreamError(event)

    def format_tool_result_message(self, tool_name: str, output: str) -> dict:
        """Format a tool result as an Ollama tool message."""
        return {"role": "tool", "content": output, "name": tool_name}
