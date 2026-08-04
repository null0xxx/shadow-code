"""Single executor entry point for validated, policy-allowed tool calls.

Admission order is registry validation, then policy, then this executor.
The executor never raises: containment and unexpected handler failures are
converted into typed ToolError results.

Approval-gated calls additionally carry an ActionPlan plus a one-shot
ApprovalToken issued by an ApprovalAuthority; the executor consumes the
token before any dispatch and fails closed with `approval_invalid` when the
plan does not match the validated call or the token is not honored.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import TypeAdapter

from shadow_code.domain.approval import ActionPlan, ApprovalAuthority, ApprovalToken
from shadow_code.domain.policy import WorkspaceAccessError
from shadow_code.domain.tools import (
    FrozenArguments,
    ToolError,
    ToolResult,
    ValidatedToolCall,
)
from shadow_code.tools.catalog import _bounded_output

_ARGUMENTS_ADAPTER: TypeAdapter[Mapping[str, Any]] = TypeAdapter(FrozenArguments)


def _plan_matches_call(plan: ActionPlan, validated: ValidatedToolCall) -> bool:
    """Check that every digest-relevant plan fact matches the validated call."""
    return (
        plan.call_id == validated.call.call_id
        and plan.tool_name == validated.call.name
        and plan.tool_version == validated.spec.version
        and plan.capability == validated.spec.capability.value
        and plan.canonical_arguments_json == validated.canonical_arguments_json()
    )


def _approval_invalid_result(call: ValidatedToolCall) -> ToolResult:
    return ToolResult(
        call_id=call.call.call_id,
        tool_name=call.call.name,
        error=ToolError(
            code="approval_invalid",
            message="Approval token is missing, spent, or bound to a different action plan.",
        ),
    )


def execute_validated_call(
    validated: ValidatedToolCall,
    context: object,
    approval: ApprovalToken | None = None,
    authority: ApprovalAuthority | None = None,
    plan: ActionPlan | None = None,
) -> ToolResult:
    """Dispatch a validated call to its spec handler; never raises.

    Without approval arguments this is the plain ALLOW path. When any of
    approval/authority/plan is provided, all three are required: the plan
    must match the validated call and the authority must consume the token
    before the handler is even considered.
    """
    approval_bound = approval is not None or authority is not None or plan is not None
    if approval_bound and (
        approval is None
        or authority is None
        or plan is None
        or not _plan_matches_call(plan, validated)
        or not authority.consume(approval, plan)
    ):
        return _approval_invalid_result(validated)

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
