"""Typed built-in tool declarations without production-loop wiring."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shadow_code.config import BASH_DEFAULT_TIMEOUT, BASH_MAX_TIMEOUT, MAX_LINES_TO_READ
from shadow_code.domain.tools import (
    Capability,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)
from shadow_code.tool_context import ToolContext

from .read_file import ReadFileTool
from .registry import ToolRegistry

READ_FILE_OUTPUT_LIMIT = 30_000
BASH_OUTPUT_LIMIT = 15_000


class CatalogArgs(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReadFileArgs(CatalogArgs):
    file_path: str
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=MAX_LINES_TO_READ, ge=1, le=MAX_LINES_TO_READ)

    @field_validator("file_path")
    @classmethod
    def _require_absolute_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("file_path must be an absolute path")
        return value


class BashArgs(CatalogArgs):
    command: str = Field(min_length=1)
    timeout: int = Field(default=BASH_DEFAULT_TIMEOUT, ge=1, le=BASH_MAX_TIMEOUT)

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


def _read_file_handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
    if not isinstance(arguments, ReadFileArgs):
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="invalid_arguments", message="Expected ReadFileArgs."),
        )
    if not isinstance(context, ToolContext):
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="invalid_context", message="Expected ToolContext."),
        )

    try:
        legacy_result = ReadFileTool(context).execute(arguments.model_dump())
    except (OSError, UnicodeError) as error:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="read_error", message=f"{type(error).__name__}: {error}"),
        )
    if not legacy_result.success:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="read_error", message=legacy_result.output),
        )
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.name,
        output=_bounded_output(legacy_result.output, READ_FILE_OUTPUT_LIMIT),
    )


READ_FILE_SPEC = ToolSpec(
    name="read_file",
    version="1",
    description="Read an absolute file path with optional one-based offset and line limit.",
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

BASH_SPEC = ToolSpec(
    name="bash",
    version="1",
    description="Declare an unconfined shell request; execution is intentionally unavailable.",
    args_model=BashArgs,
    handler=None,
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
