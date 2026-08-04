"""Typed built-in tool declarations with read and process-execution handlers."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shadow_code.config import BASH_DEFAULT_TIMEOUT, BASH_MAX_TIMEOUT, MAX_LINES_TO_READ
from shadow_code.domain.policy import WorkspaceAccessError
from shadow_code.domain.tools import (
    Capability,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.process import run_process

from .registry import ToolRegistry

READ_FILE_OUTPUT_LIMIT = 30_000
BASH_OUTPUT_LIMIT = 15_000
_READ_CHUNK_BYTES = 8192


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    """Execution context carrying the workspace containment guard.

    The process-execution fields default to inert values so read-only
    callers keep working; the bash handler requires a real workspace root.
    """

    guard: WorkspaceGuard
    workspace_root: str = ""
    process_env: Mapping[str, str] = field(default_factory=dict)
    sandbox_label: str = "unconfined"

    def __post_init__(self) -> None:
        # Deep-freeze the mapping so the context stays immutable.
        object.__setattr__(self, "process_env", MappingProxyType(dict(self.process_env)))


class CatalogArgs(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReadFileArgs(CatalogArgs):
    file_path: str = Field(description="Workspace-relative path to the file.")
    offset: int = Field(default=1, ge=1, description="One-based starting line.")
    limit: int = Field(
        default=MAX_LINES_TO_READ,
        ge=1,
        le=MAX_LINES_TO_READ,
        description="Maximum number of lines to return.",
    )

    @field_validator("file_path")
    @classmethod
    def _require_normalized_relative_path(cls, value: str) -> str:
        if not value or "\x00" in value or value.startswith("/"):
            raise ValueError("file_path must be a workspace-relative path")
        if any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("file_path must be normalized (no empty, '.' or '..' segments)")
        return value


class BashArgs(CatalogArgs):
    command: str = Field(min_length=1, description="Shell command to declare.")
    timeout: int = Field(
        default=BASH_DEFAULT_TIMEOUT,
        ge=1,
        le=BASH_MAX_TIMEOUT,
        description="Timeout in seconds.",
    )

    @field_validator("command")
    @classmethod
    def _require_command_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command cannot be blank")
        return value


def _bounded_output(output: str, limit: int) -> str:
    if len(output) <= limit:
        return output
    marker = f"\n\n[...{len(output) - limit} chars truncated...]\n\n"
    retained = limit - len(marker)
    head = retained // 2
    return output[:head] + marker + output[-(retained - head) :]


def _read_bounded_text(descriptor: int, max_chars: int) -> str:
    byte_budget = max_chars * 4 + 1  # UTF-8 worst case; +1 byte detects overflow
    data = bytearray()
    while len(data) < byte_budget:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, byte_budget - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    return data.decode("utf-8", errors="replace")


def _read_file_handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
    if not isinstance(arguments, ReadFileArgs):
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="invalid_arguments", message="Expected ReadFileArgs."),
        )
    if not isinstance(context, WorkspaceContext):
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="invalid_context", message="Expected WorkspaceContext."),
        )

    try:
        with context.guard.open_read(arguments.file_path) as descriptor:
            text = _read_bounded_text(descriptor, READ_FILE_OUTPUT_LIMIT)
    except WorkspaceAccessError as error:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code=error.reason.value, message=str(error)),
        )
    except OSError as error:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="read_error", message=f"{type(error).__name__}: {error}"),
        )

    lines = text.splitlines()
    selected = lines[arguments.offset - 1 : arguments.offset - 1 + arguments.limit]
    if not selected:
        output = "(empty file)" if not lines else "(no lines in requested range)"
    else:
        output = "\n".join(
            f"{number}\t{line}" for number, line in enumerate(selected, start=arguments.offset)
        )
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.name,
        output=_bounded_output(output, READ_FILE_OUTPUT_LIMIT),
    )


READ_FILE_SPEC = ToolSpec(
    name="read_file",
    version="1",
    description=(
        "Read a workspace-relative file path with optional one-based offset and line limit."
    ),
    args_model=ReadFileArgs,
    handler=_read_file_handler,
    capability=Capability.FILESYSTEM_READ,
    risk=RiskLevel.LOW,
    side_effects=SideEffect.NONE,
    timeout_seconds=30,
    max_output_chars=READ_FILE_OUTPUT_LIMIT,
    idempotency=True,
    parallel_safety=True,
    renderer_hint="text",
)


def _bash_handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
    if not isinstance(arguments, BashArgs):
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="invalid_arguments", message="Expected BashArgs."),
        )
    if not isinstance(context, WorkspaceContext) or not context.workspace_root:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(
                code="invalid_context",
                message="Expected WorkspaceContext with a workspace root.",
            ),
        )

    timeout = min(arguments.timeout, BASH_MAX_TIMEOUT)
    try:
        outcome = run_process(
            arguments.command,
            cwd=context.workspace_root,
            env=context.process_env,
            timeout_seconds=timeout,
            max_output_chars=BASH_OUTPUT_LIMIT,
        )
    except OSError as error:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="process_error", message=f"{type(error).__name__}: {error}"),
        )

    parts = [f"$ {arguments.command}"]
    stdout = _bounded_output(outcome.stdout, BASH_OUTPUT_LIMIT)
    if outcome.stdout_removed_bytes:
        stdout += f"\n[...stdout truncated: {outcome.stdout_removed_bytes} bytes removed...]"
    if stdout:
        parts.append(stdout)
    stderr = _bounded_output(outcome.stderr, BASH_OUTPUT_LIMIT)
    if outcome.stderr_removed_bytes:
        stderr += f"\n[...stderr truncated: {outcome.stderr_removed_bytes} bytes removed...]"
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if outcome.timed_out:
        parts.append(f"[timed out after {timeout}s; process group terminated]")
    exit_code = outcome.exit_code if outcome.exit_code is not None else "killed"
    parts.append(f"exit code: {exit_code}")
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.name,
        output="\n".join(parts),
    )


BASH_SPEC = ToolSpec(
    name="bash",
    version="1",
    description=(
        "Run an UNCONFINED shell command in the workspace after explicit "
        "one-shot approval; no sandbox is applied."
    ),
    args_model=BashArgs,
    handler=_bash_handler,
    capability=Capability.PROCESS_EXECUTE,
    risk=RiskLevel.HIGH,
    side_effects=SideEffect.UNKNOWN,
    timeout_seconds=BASH_MAX_TIMEOUT,
    max_output_chars=BASH_OUTPUT_LIMIT,
    idempotency=False,
    parallel_safety=False,
    renderer_hint="command",
)

DEFAULT_TOOL_REGISTRY = ToolRegistry((BASH_SPEC, READ_FILE_SPEC))
