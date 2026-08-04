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
from shadow_code.process import build_process_env
from shadow_code.tools.catalog import (
    BASH_OUTPUT_LIMIT,
    BASH_SPEC,
    DEFAULT_TOOL_REGISTRY,
    EDIT_FILE_SPEC,
    READ_FILE_OUTPUT_LIMIT,
    READ_FILE_SPEC,
    WRITE_FILE_SPEC,
    BashArgs,
    EditFileArgs,
    ReadFileArgs,
    WorkspaceContext,
    WriteFileArgs,
    _bash_handler,
    _bounded_output,
    _edit_file_handler,
    _read_file_handler,
    _write_file_handler,
)


def test_default_catalog_has_exact_order_and_truthful_metadata() -> None:
    assert DEFAULT_TOOL_REGISTRY.names == ("bash", "edit_file", "read_file", "write_file")
    assert BASH_SPEC.handler is _bash_handler
    assert "UNCONFINED" in BASH_SPEC.description
    assert "no sandbox" in BASH_SPEC.description
    assert BASH_SPEC.capability is Capability.PROCESS_EXECUTE
    assert BASH_SPEC.risk is RiskLevel.HIGH
    assert BASH_SPEC.side_effects is SideEffect.UNKNOWN
    assert READ_FILE_SPEC.capability is Capability.FILESYSTEM_READ
    assert READ_FILE_SPEC.risk is RiskLevel.LOW
    assert READ_FILE_SPEC.side_effects is SideEffect.NONE
    assert READ_FILE_SPEC.max_output_chars == 30_000
    mutation_specs = (
        (WRITE_FILE_SPEC, _write_file_handler),
        (EDIT_FILE_SPEC, _edit_file_handler),
    )
    for spec, handler in mutation_specs:
        assert spec.handler is handler
        assert spec.capability is Capability.FILESYSTEM_WRITE
        assert spec.risk is RiskLevel.HIGH
        assert spec.side_effects is SideEffect.MUTATING
        assert spec.renderer_hint == "diff"
        assert spec.idempotency is False
        assert spec.parallel_safety is False


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


def test_bash_is_validatable_and_executable() -> None:
    result = DEFAULT_TOOL_REGISTRY.validate_call(
        {"call_id": "bash-1", "name": "bash", "arguments": {"command": "printf ok"}}
    )

    assert isinstance(result, ValidatedToolCall)
    assert result.spec is BASH_SPEC
    assert result.spec.handler is _bash_handler
    assert result.model_dump()["arguments"] == BashArgs(command="printf ok").model_dump()


def _bash_context(
    guard: WorkspaceGuard, root: Path, env: dict[str, str] | None = None
) -> WorkspaceContext:
    return WorkspaceContext(
        guard=guard,
        workspace_root=str(root),
        process_env={} if env is None else env,
        sandbox_label="unconfined",
    )


def test_bash_handler_runs_command_in_workspace(tmp_path: Path) -> None:
    call = ToolCall(call_id="bash-run", name="bash", arguments={"command": "echo ok"})
    with WorkspaceGuard(tmp_path) as guard:
        result = _bash_handler(call, BashArgs(command="echo ok"), _bash_context(guard, tmp_path))

    assert result.success is True
    assert result.call_id == "bash-run"
    assert result.tool_name == "bash"
    assert result.output is not None
    assert result.output.startswith("$ echo ok")
    assert "\nok\n" in result.output
    assert "exit code: 0" in result.output


def test_bash_handler_reports_non_zero_exit_and_stderr(tmp_path: Path) -> None:
    command = "echo oops >&2; exit 7"
    call = ToolCall(call_id="bash-fail", name="bash", arguments={"command": command})
    with WorkspaceGuard(tmp_path) as guard:
        result = _bash_handler(call, BashArgs(command=command), _bash_context(guard, tmp_path))

    # A non-zero exit is still a successful tool result: the model needs the output.
    assert result.success is True
    assert result.output is not None
    assert "[stderr]\noops" in result.output
    assert "exit code: 7" in result.output


def test_bash_handler_environment_drops_parent_secrets(tmp_path: Path) -> None:
    sentinel = "SHADOW_TEST_SECRET_TOKEN"
    source = {"PATH": "/usr/bin:/bin", sentinel: "sentinel-value"}
    call = ToolCall(call_id="bash-env", name="bash", arguments={"command": "env"})
    with WorkspaceGuard(tmp_path) as guard:
        context = _bash_context(guard, tmp_path, build_process_env(source))
        result = _bash_handler(call, BashArgs(command="env"), context)

    assert result.success is True
    assert result.output is not None
    assert sentinel not in result.output
    assert "sentinel-value" not in result.output


def test_bash_handler_marks_timeout(tmp_path: Path) -> None:
    command = "sleep 60"
    call = ToolCall(
        call_id="bash-timeout",
        name="bash",
        arguments={"command": command, "timeout": 1},
    )
    with WorkspaceGuard(tmp_path) as guard:
        context = _bash_context(guard, tmp_path)
        result = _bash_handler(call, BashArgs(command=command, timeout=1), context)

    assert result.success is True
    assert result.output is not None
    assert "timed out after 1s" in result.output
    assert "exit code: killed" in result.output


def test_bash_handler_truncates_and_records_removed_bytes(tmp_path: Path) -> None:
    command = "head -c 100000 /dev/zero | tr '\\0' a"
    call = ToolCall(call_id="bash-huge", name="bash", arguments={"command": command})
    env = build_process_env({"PATH": "/usr/bin:/bin"})
    with WorkspaceGuard(tmp_path) as guard:
        result = _bash_handler(call, BashArgs(command=command), _bash_context(guard, tmp_path, env))

    assert result.success is True
    assert result.output is not None
    assert "bytes removed" in result.output
    assert len(result.output) <= BASH_OUTPUT_LIMIT * 4


def test_bash_handler_rejects_wrong_arguments_and_context(tmp_path: Path) -> None:
    call = ToolCall(call_id="bash-contract", name="bash", arguments={"command": "echo ok"})

    wrong_arguments = _bash_handler(call, ReadFileArgs(file_path="x.txt"), object())
    wrong_context = _bash_handler(call, BashArgs(command="echo ok"), object())

    assert wrong_arguments.error is not None
    assert wrong_arguments.error.code == "invalid_arguments"
    assert wrong_context.error is not None
    assert wrong_context.error.code == "invalid_context"

    # A context without a workspace root cannot execute either.
    with WorkspaceGuard(tmp_path) as guard:
        rootless = _bash_handler(call, BashArgs(command="echo ok"), WorkspaceContext(guard))
    assert rootless.error is not None
    assert rootless.error.code == "invalid_context"


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


def test_write_file_args_are_strict() -> None:
    for arguments in (
        {},
        {"file_path": "/abs.txt", "content": "x"},
        {"file_path": "a.txt"},
        {"file_path": "a.txt", "content": 1},
        {"file_path": "a.txt", "content": "x" * 200_001},
        {"file_path": "a/../b.txt", "content": "x"},
        {"file_path": "a.txt", "content": "x", "extra": 1},
    ):
        result = DEFAULT_TOOL_REGISTRY.validate_call(
            {"call_id": "w-invalid", "name": "write_file", "arguments": arguments}
        )
        assert isinstance(result, ToolError)

    for arguments in (
        {"file_path": "a.txt", "old_text": "", "new_text": "b"},
        {"file_path": "a.txt", "old_text": "a"},
    ):
        result = DEFAULT_TOOL_REGISTRY.validate_call(
            {"call_id": "e-invalid", "name": "edit_file", "arguments": arguments}
        )
        assert isinstance(result, ToolError)


def test_write_file_handler_writes_through_guard(tmp_path: Path) -> None:
    call = ToolCall(
        call_id="write-1",
        name="write_file",
        arguments={"file_path": "out.txt", "content": "written\n"},
    )
    with WorkspaceGuard(tmp_path) as guard:
        result = _write_file_handler(
            call,
            WriteFileArgs(file_path="out.txt", content="written\n"),
            WorkspaceContext(guard),
        )

    assert result.success is True
    assert result.call_id == "write-1"
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "written\n"
    assert result.output is not None
    assert "mutation: write out.txt" in result.output
    assert "before: missing" in result.output
    assert "sha256=" in result.output
    assert "bytes written: 8" in result.output
    assert "not a security boundary" in result.output


def test_edit_file_handler_replaces_exact_text(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("value = 1\n", encoding="utf-8")
    call = ToolCall(
        call_id="edit-1",
        name="edit_file",
        arguments={"file_path": "code.py", "old_text": "1", "new_text": "2"},
    )
    with WorkspaceGuard(tmp_path) as guard:
        result = _edit_file_handler(
            call,
            EditFileArgs(file_path="code.py", old_text="1", new_text="2"),
            WorkspaceContext(guard),
        )

    assert result.success is True
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "value = 2\n"
    assert result.output is not None
    assert "mutation: edit code.py" in result.output
    assert "before: device=" in result.output


def test_edit_file_handler_surfaces_typed_plan_errors(tmp_path: Path) -> None:
    (tmp_path / "dup.txt").write_text("foo foo\n", encoding="utf-8")
    ambiguous_call = ToolCall(
        call_id="edit-amb",
        name="edit_file",
        arguments={"file_path": "dup.txt", "old_text": "foo", "new_text": "bar"},
    )
    no_match_call = ToolCall(
        call_id="edit-none",
        name="edit_file",
        arguments={"file_path": "missing.txt", "old_text": "x", "new_text": "y"},
    )
    with WorkspaceGuard(tmp_path) as guard:
        ambiguous = _edit_file_handler(
            ambiguous_call,
            EditFileArgs(file_path="dup.txt", old_text="foo", new_text="bar"),
            WorkspaceContext(guard),
        )
        no_match = _edit_file_handler(
            no_match_call,
            EditFileArgs(file_path="missing.txt", old_text="x", new_text="y"),
            WorkspaceContext(guard),
        )

    assert ambiguous.error is not None
    assert ambiguous.error.code == "ambiguous_match"
    assert no_match.error is not None
    assert no_match.error.code == "no_match"
    assert (tmp_path / "dup.txt").read_text(encoding="utf-8") == "foo foo\n"
    assert not (tmp_path / "missing.txt").exists()


def test_mutation_handlers_reject_wrong_arguments_and_context(tmp_path: Path) -> None:
    write_call = ToolCall(
        call_id="w-contract",
        name="write_file",
        arguments={"file_path": "x.txt", "content": "x"},
    )
    edit_call = ToolCall(
        call_id="e-contract",
        name="edit_file",
        arguments={"file_path": "x.txt", "old_text": "a", "new_text": "b"},
    )

    wrong_write_args = _write_file_handler(write_call, BashArgs(command="id"), object())
    wrong_write_ctx = _write_file_handler(
        write_call, WriteFileArgs(file_path="x.txt", content="x"), object()
    )
    wrong_edit_args = _edit_file_handler(edit_call, BashArgs(command="id"), object())
    wrong_edit_ctx = _edit_file_handler(
        edit_call,
        EditFileArgs(file_path="x.txt", old_text="a", new_text="b"),
        object(),
    )

    for result in (wrong_write_args, wrong_edit_args):
        assert result.error is not None
        assert result.error.code == "invalid_arguments"
    for result in (wrong_write_ctx, wrong_edit_ctx):
        assert result.error is not None
        assert result.error.code == "invalid_context"
