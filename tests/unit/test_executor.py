from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from shadow_code.domain.approval import (
    ActionPlan,
    ApprovalAuthority,
    build_action_plan,
)
from shadow_code.domain.policy import (
    WorkspaceAccessError,
    WorkspaceFailure,
    WorkspaceIdentity,
)
from shadow_code.domain.tools import (
    Capability,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolHandler,
    ToolResult,
    ToolSpec,
    ValidatedToolCall,
)
from shadow_code.executor import execute_validated_call
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.tools.catalog import READ_FILE_SPEC, WorkspaceContext


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str = "default"


def _echo_handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
    assert isinstance(arguments, EchoArgs)
    return ToolResult(call_id=call.call_id, tool_name=call.name, output=arguments.text)


def _spec(handler: object = _echo_handler, max_output_chars: int = 100) -> ToolSpec:
    return ToolSpec(
        name="echo",
        version="1",
        description="Echo probe.",
        args_model=EchoArgs,
        handler=handler,  # type: ignore[arg-type]
        capability=Capability.FILESYSTEM_READ,
        risk=RiskLevel.LOW,
        side_effects=SideEffect.NONE,
        timeout_seconds=1,
        max_output_chars=max_output_chars,
        idempotency=True,
        parallel_safety=True,
        renderer_hint="text",
    )


def _validated(spec: ToolSpec, text: str = "hello", call_id: str = "call-1") -> ValidatedToolCall:
    call = ToolCall(call_id=call_id, name=spec.name, arguments={"text": text})
    return ValidatedToolCall(call=call, spec=spec)


def test_execute_dispatches_to_handler_with_typed_arguments() -> None:
    result = execute_validated_call(_validated(_spec(), text="hi", call_id="d-1"), object())

    assert result.success is True
    assert result.call_id == "d-1"
    assert result.tool_name == "echo"
    assert result.output == "hi"


def test_execute_fails_closed_without_handler() -> None:
    result = execute_validated_call(_validated(_spec(handler=None)), object())

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "handler_unavailable"


def test_execute_bounds_output_to_spec_limit() -> None:
    validated = _validated(_spec(max_output_chars=50), text="y" * 500)

    result = execute_validated_call(validated, object())

    assert result.success is True
    assert result.output is not None
    assert len(result.output) == 50
    assert "450 chars truncated" in result.output


def test_execute_maps_workspace_access_error_to_typed_error() -> None:
    def raising_handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
        raise WorkspaceAccessError(WorkspaceFailure.CONTAINMENT_VIOLATION, "escape attempt")

    result = execute_validated_call(_validated(_spec(handler=raising_handler)), object())

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "containment_violation"
    assert "escape attempt" in result.error.message


def test_execute_maps_unexpected_exception_to_typed_error() -> None:
    def exploding_handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
        raise RuntimeError("boom")

    validated = _validated(_spec(handler=exploding_handler), call_id="x-1")

    result = execute_validated_call(validated, object())

    assert result.success is False
    assert result.call_id == "x-1"
    assert result.error is not None
    assert result.error.code == "executor_error"
    assert "RuntimeError" in result.error.message


def test_execute_rejects_uncorrelated_handler_result() -> None:
    def lying_handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
        return ToolResult(call_id="other", tool_name=call.name, output="forged")

    validated = _validated(_spec(handler=lying_handler), call_id="c-1")

    result = execute_validated_call(validated, object())

    assert result.success is False
    assert result.call_id == "c-1"
    assert result.error is not None
    assert result.error.code == "correlation_mismatch"


def test_execute_runs_catalog_read_file_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("first\nsecond\n", encoding="utf-8")
    call = ToolCall(call_id="e2e-1", name="read_file", arguments={"file_path": "note.txt"})
    validated = ValidatedToolCall(call=call, spec=READ_FILE_SPEC)

    with WorkspaceGuard(tmp_path) as guard:
        result = execute_validated_call(validated, WorkspaceContext(guard))

    assert result.success is True
    assert result.call_id == "e2e-1"
    assert result.output is not None
    assert "1\tfirst" in result.output
    assert "2\tsecond" in result.output


@pytest.mark.parametrize("context", [None, object(), "guard"])
def test_execute_catalog_read_file_with_wrong_context_is_typed_error(context: object) -> None:
    call = ToolCall(call_id="ctx-1", name="read_file", arguments={"file_path": "note.txt"})
    validated = ValidatedToolCall(call=call, spec=READ_FILE_SPEC)

    result = execute_validated_call(validated, context)

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "invalid_context"


# -- Approval-bound execution -----------------------------------------------


def _plan_for(validated: ValidatedToolCall, preview: str = "preview") -> ActionPlan:
    return build_action_plan(
        validated,
        registry_digest="registry",
        workspace=WorkspaceIdentity(device=1, inode=2),
        preview=preview,
    )


def _counting_handler(calls: list[str]) -> ToolHandler:
    def handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
        calls.append(call.call_id)
        return ToolResult(call_id=call.call_id, tool_name=call.name, output="done")

    return handler


def test_execute_with_valid_approval_dispatches_once() -> None:
    calls: list[str] = []
    authority = ApprovalAuthority()
    validated = _validated(_spec(handler=_counting_handler(calls)), call_id="a-1")
    plan = _plan_for(validated)
    token = authority.issue(plan)

    result = execute_validated_call(
        validated, object(), approval=token, authority=authority, plan=plan
    )

    assert result.success is True
    assert calls == ["a-1"]
    # The token is one-shot: a second execution with it fails closed.
    replayed = execute_validated_call(
        validated, object(), approval=token, authority=authority, plan=plan
    )
    assert replayed.success is False
    assert replayed.error is not None
    assert replayed.error.code == "approval_invalid"
    assert calls == ["a-1"]


def test_execute_with_mismatched_plan_is_approval_invalid_and_never_dispatches() -> None:
    calls: list[str] = []
    authority = ApprovalAuthority()
    validated = _validated(_spec(handler=_counting_handler(calls)), text="hi")
    token = authority.issue(_plan_for(_validated(_spec(), text="hi")))
    wrong_plan = _plan_for(validated, preview="different preview")

    result = execute_validated_call(
        validated, object(), approval=token, authority=authority, plan=wrong_plan
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "approval_invalid"
    assert calls == []


def test_execute_rejects_plan_whose_facts_diverge_from_the_call() -> None:
    calls: list[str] = []
    authority = ApprovalAuthority()
    validated = _validated(_spec(handler=_counting_handler(calls)), text="hi", call_id="a-2")
    other = _validated(_spec(handler=_counting_handler(calls)), text="bye", call_id="a-3")
    plan = _plan_for(other)  # call_id and arguments do not match `validated`
    token = authority.issue(plan)

    result = execute_validated_call(
        validated, object(), approval=token, authority=authority, plan=plan
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "approval_invalid"
    assert calls == []


@pytest.mark.parametrize(
    "approval, authority, plan",
    [
        ("token", None, None),
        (None, "authority", None),
        (None, None, "plan"),
        ("token", "authority", None),
    ],
)
def test_execute_with_partial_approval_binding_is_approval_invalid(
    approval: object, authority: object, plan: object
) -> None:
    calls: list[str] = []
    validated = _validated(_spec(handler=_counting_handler(calls)))
    real_authority = ApprovalAuthority()
    real_plan = _plan_for(validated)
    real_token = real_authority.issue(real_plan)
    bound_approval = real_token if approval else None
    bound_authority = real_authority if authority else None
    bound_plan = real_plan if plan else None

    result = execute_validated_call(
        validated,
        object(),
        approval=bound_approval,
        authority=bound_authority,
        plan=bound_plan,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "approval_invalid"
    assert calls == []


def test_execute_without_approval_arguments_is_the_unchanged_allow_path() -> None:
    calls: list[str] = []
    validated = _validated(_spec(handler=_counting_handler(calls)), text="plain")

    result = execute_validated_call(validated, object())

    assert result.success is True
    assert result.output == "done"
    assert calls == ["call-1"]
