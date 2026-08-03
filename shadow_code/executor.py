"""Single executor entry point for validated, policy-allowed tool calls.

Admission order is registry validation, then policy, then this executor.
The executor never raises: containment and unexpected handler failures are
converted into typed ToolError results.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter

from shadow_code.domain.policy import WorkspaceAccessError
from shadow_code.domain.tools import (
    FrozenArguments,
    ToolError,
    ToolResult,
    ValidatedToolCall,
)
from shadow_code.tools.catalog import _bounded_output

_ARGUMENTS_ADAPTER: TypeAdapter[Mapping[str, Any]] = TypeAdapter(FrozenArguments)


def execute_validated_call(validated: ValidatedToolCall, context: object) -> ToolResult:
    """Dispatch a validated call to its spec handler; never raises."""
    call = validated.call
    spec = validated.spec
    handler = spec.handler
    if handler is None:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(
                code="handler_unavailable",
                message=f"Tool '{call.name}' has no executable handler.",
            ),
        )

    try:
        raw_arguments = _ARGUMENTS_ADAPTER.dump_python(validated.arguments, mode="json")
        arguments = spec.args_model.model_validate(raw_arguments, strict=True)
        result = handler(call, arguments, context)
    except WorkspaceAccessError as error:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code=error.reason.value, message=str(error)),
        )
    except Exception as error:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(code="executor_error", message=f"{type(error).__name__}: {error}"),
        )

    if result.call_id != call.call_id or result.tool_name != call.name:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.name,
            error=ToolError(
                code="correlation_mismatch",
                message="Handler result does not match the validated call.",
            ),
        )
    if result.output is not None and len(result.output) > spec.max_output_chars:
        result = ToolResult(
            call_id=result.call_id,
            tool_name=result.tool_name,
            output=_bounded_output(result.output, spec.max_output_chars),
        )
    return result
