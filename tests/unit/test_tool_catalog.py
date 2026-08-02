from pathlib import Path

import pytest

from shadow_code.domain.tools import (
    Capability,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolError,
    ValidatedToolCall,
)
from shadow_code.tool_context import ToolContext
from shadow_code.tools.catalog import (
    BASH_SPEC,
    DEFAULT_TOOL_REGISTRY,
    READ_FILE_OUTPUT_LIMIT,
    READ_FILE_SPEC,
    BashArgs,
    ReadFileArgs,
    _bounded_output,
    _read_file_handler,
)


def test_default_catalog_has_exact_order_and_truthful_metadata() -> None:
    assert DEFAULT_TOOL_REGISTRY.names == ("bash", "read_file")
    assert BASH_SPEC.handler is None
    assert BASH_SPEC.capability is Capability.PROCESS_EXECUTE
    assert BASH_SPEC.risk is RiskLevel.HIGH
    assert BASH_SPEC.side_effects is SideEffect.UNKNOWN
    assert READ_FILE_SPEC.capability is Capability.FILESYSTEM_READ
    assert READ_FILE_SPEC.risk is RiskLevel.LOW
    assert READ_FILE_SPEC.side_effects is SideEffect.NONE
    assert READ_FILE_SPEC.max_output_chars == 30_000


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"file_path": "relative.txt"},
        {"file_path": "/tmp/file", "limit": "1"},
        {"file_path": "/tmp/file", "limit": True},
        {"file_path": "/tmp/file", "offset": 0},
        {"file_path": "/tmp/file", "offset": -1},
        {"file_path": "/tmp/file", "limit": 0},
        {"file_path": "/tmp/file", "limit": -1},
        {"file_path": "/tmp/file", "limit": 2_001},
        {"file_path": "/tmp/file", "unknown": 1},
        "not-an-object",
    ],
)
def test_read_file_arguments_are_strict_and_never_reach_handler(arguments: object) -> None:
    result = DEFAULT_TOOL_REGISTRY.validate_call(
        {"call_id": "invalid-1", "name": "read_file", "arguments": arguments}
    )

    assert isinstance(result, ToolError)
    assert result.code in {"invalid_tool_call", "invalid_arguments"}


def test_read_file_handler_returns_correlated_bounded_result(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("x" * 31_000, encoding="utf-8")
    call = ToolCall(call_id="read-1", name="read_file", arguments={"file_path": str(path)})
    handler = READ_FILE_SPEC.handler

    assert handler is not None
    result = handler(call, ReadFileArgs(file_path=str(path)), ToolContext(str(tmp_path)))

    assert result.success is True
    assert result.call_id == "read-1"
    assert result.tool_name == "read_file"
    assert result.output is not None
    assert len(result.output) <= READ_FILE_SPEC.max_output_chars
    assert "1\t" in result.output


def test_read_file_handler_returns_typed_correlated_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    call = ToolCall(call_id="read-2", name="read_file", arguments={"file_path": str(missing)})
    handler = READ_FILE_SPEC.handler

    assert handler is not None
    result = handler(call, ReadFileArgs(file_path=str(missing)), ToolContext(str(tmp_path)))

    assert result.success is False
    assert result.call_id == "read-2"
    assert result.tool_name == "read_file"
    assert result.error is not None
    assert result.error.code == "read_error"
    assert "not found" in result.error.message.lower()


def test_read_file_handler_normalizes_text_read_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unreadable.txt"
    path.write_text("content", encoding="utf-8")
    call = ToolCall(call_id="read-io", name="read_file", arguments={"file_path": str(path)})
    original_open = open
    binary_probe_seen = False

    def fail_text_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        nonlocal binary_probe_seen
        if mode == "rb":
            binary_probe_seen = True
        if kwargs.get("encoding") == "utf-8":
            assert binary_probe_seen
            raise OSError("deterministic text read failure")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_text_open)
    handler = READ_FILE_SPEC.handler
    assert handler is not None

    result = handler(call, ReadFileArgs(file_path=str(path)), ToolContext(str(tmp_path)))

    assert result.error is not None
    assert result.call_id == "read-io"
    assert result.error.code == "read_error"
    assert "deterministic text read failure" in result.error.message


def test_bash_is_validatable_but_declaration_only() -> None:
    result = DEFAULT_TOOL_REGISTRY.validate_call(
        {"call_id": "bash-1", "name": "bash", "arguments": {"command": "printf ok"}}
    )

    assert isinstance(result, ValidatedToolCall)
    assert result.spec is BASH_SPEC
    assert result.spec.handler is None
    assert result.model_dump()["arguments"] == BashArgs(command="printf ok").model_dump()


@pytest.mark.parametrize("command", ["", " ", "\t\n"])
def test_bash_rejects_blank_command_declarations(command: str) -> None:
    result = DEFAULT_TOOL_REGISTRY.validate_call(
        {"call_id": "bash-blank", "name": "bash", "arguments": {"command": command}}
    )

    assert isinstance(result, ToolError)
    assert result.code == "invalid_arguments"
    assert result.issues[0].path == ("arguments", "command")


def test_bounded_output_preserves_both_ends_within_exact_limit() -> None:
    short = "short output"
    long = "a" * (READ_FILE_OUTPUT_LIMIT + 500)

    assert _bounded_output(short, READ_FILE_OUTPUT_LIMIT) == short
    bounded = _bounded_output(long, READ_FILE_OUTPUT_LIMIT)
    assert len(bounded) == READ_FILE_OUTPUT_LIMIT
    assert bounded.startswith("a") and bounded.endswith("a")
    assert "500 chars truncated" in bounded


def test_read_file_handler_rejects_wrong_arguments_and_context(tmp_path: Path) -> None:
    call = ToolCall(call_id="read-contract", name="read_file", arguments={"file_path": "/tmp/x"})

    wrong_arguments = _read_file_handler(call, BashArgs(command="printf ok"), object())
    wrong_context = _read_file_handler(call, ReadFileArgs(file_path=str(tmp_path / "x")), object())

    assert wrong_arguments.error is not None
    assert wrong_arguments.error.code == "invalid_arguments"
    assert wrong_context.error is not None
    assert wrong_context.error.code == "invalid_context"
