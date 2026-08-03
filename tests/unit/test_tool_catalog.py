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
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.tools.catalog import (
    BASH_SPEC,
    DEFAULT_TOOL_REGISTRY,
    READ_FILE_OUTPUT_LIMIT,
    READ_FILE_SPEC,
    BashArgs,
    ReadFileArgs,
    WorkspaceContext,
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
        {"file_path": "/absolute.txt"},
        {"file_path": ".."},
        {"file_path": "dir/../file.txt"},
        {"file_path": "dir//file.txt"},
        {"file_path": "./file.txt"},
        {"file_path": ""},
        {"file_path": "file\x00x.txt"},
        {"file_path": "file.txt", "limit": "1"},
        {"file_path": "file.txt", "limit": True},
        {"file_path": "file.txt", "offset": 0},
        {"file_path": "file.txt", "offset": -1},
        {"file_path": "file.txt", "limit": 0},
        {"file_path": "file.txt", "limit": -1},
        {"file_path": "file.txt", "limit": 2_001},
        {"file_path": "file.txt", "unknown": 1},
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
    call = ToolCall(call_id="read-1", name="read_file", arguments={"file_path": "large.txt"})
    handler = READ_FILE_SPEC.handler

    assert handler is not None
    with WorkspaceGuard(tmp_path) as guard:
        result = handler(call, ReadFileArgs(file_path="large.txt"), WorkspaceContext(guard))

    assert result.success is True
    assert result.call_id == "read-1"
    assert result.tool_name == "read_file"
    assert result.output is not None
    assert len(result.output) <= READ_FILE_SPEC.max_output_chars
    assert "1\t" in result.output
    assert "chars truncated" in result.output


def test_read_file_handler_reads_relative_path_through_guard(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hello.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    call = ToolCall(
        call_id="read-rel", name="read_file", arguments={"file_path": "nested/hello.txt"}
    )
    handler = READ_FILE_SPEC.handler

    assert handler is not None
    with WorkspaceGuard(tmp_path) as guard:
        result = handler(
            call,
            ReadFileArgs(file_path="nested/hello.txt", offset=2, limit=1),
            WorkspaceContext(guard),
        )

    assert result.success is True
    assert result.output == "2\tbeta"


def test_read_file_handler_rejects_symlinks_via_containment(tmp_path: Path) -> None:
    (tmp_path / "real.txt").write_text("real", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")
    call = ToolCall(call_id="read-link", name="read_file", arguments={"file_path": "link.txt"})
    handler = READ_FILE_SPEC.handler

    assert handler is not None
    with WorkspaceGuard(tmp_path) as guard:
        result = handler(call, ReadFileArgs(file_path="link.txt"), WorkspaceContext(guard))

    assert result.success is False
    assert result.call_id == "read-link"
    assert result.error is not None
    assert result.error.code == "containment_violation"


def test_read_file_handler_returns_typed_correlated_error(tmp_path: Path) -> None:
    call = ToolCall(call_id="read-2", name="read_file", arguments={"file_path": "missing.txt"})
    handler = READ_FILE_SPEC.handler

    assert handler is not None
    with WorkspaceGuard(tmp_path) as guard:
        result = handler(call, ReadFileArgs(file_path="missing.txt"), WorkspaceContext(guard))

    assert result.success is False
    assert result.call_id == "read-2"
    assert result.tool_name == "read_file"
    assert result.error is not None
    assert result.error.code == "io_error"


def test_read_file_handler_normalizes_read_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "unreadable.txt").write_text("content", encoding="utf-8")
    call = ToolCall(call_id="read-io", name="read_file", arguments={"file_path": "unreadable.txt"})

    def fail_read(descriptor: int, count: int) -> bytes:
        raise OSError("deterministic read failure")

    monkeypatch.setattr("shadow_code.tools.catalog.os.read", fail_read)
    handler = READ_FILE_SPEC.handler
    assert handler is not None

    with WorkspaceGuard(tmp_path) as guard:
        result = handler(call, ReadFileArgs(file_path="unreadable.txt"), WorkspaceContext(guard))

    assert result.error is not None
    assert result.call_id == "read-io"
    assert result.error.code == "read_error"
    assert "deterministic read failure" in result.error.message


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
    call = ToolCall(call_id="read-contract", name="read_file", arguments={"file_path": "x.txt"})

    wrong_arguments = _read_file_handler(call, BashArgs(command="printf ok"), object())
    wrong_context = _read_file_handler(call, ReadFileArgs(file_path="x.txt"), object())

    assert wrong_arguments.error is not None
    assert wrong_arguments.error.code == "invalid_arguments"
    assert wrong_context.error is not None
    assert wrong_context.error.code == "invalid_context"
