from types import MappingProxyType

import pytest
from pydantic import BaseModel, ValidationError

from shadow_code.domain.tools import (
    Capability,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
    ValidatedToolCall,
    ValidationIssue,
)


class CountArgs(BaseModel):
    count: int
    options: dict[str, list[int]] | None = None


def _handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
    raise AssertionError("domain declarations must not execute handlers")


def test_domain_models_and_spec_are_frozen_and_strict() -> None:
    issue = ValidationIssue(path=("arguments", "count"), code="invalid_type", message="bad")
    error = ToolError(code="invalid_arguments", message="Invalid arguments.", issues=(issue,))
    result = ToolResult(call_id="call-1", tool_name="counter", output=None, error=error)
    call = ToolCall(call_id="call-1", name="counter", arguments={"count": 1})
    spec = ToolSpec(
        name="counter",
        version="1",
        description="Count exactly.",
        args_model=CountArgs,
        handler=_handler,
        capability=Capability.FILESYSTEM_READ,
        risk=RiskLevel.LOW,
        side_effects=SideEffect.NONE,
        timeout_seconds=1,
        max_output_chars=100,
        idempotency=True,
        parallel_safety=True,
        renderer_hint="text",
    )

    for model, field, value in (
        (issue, "code", "changed"),
        (error, "message", "changed"),
        (result, "output", "changed"),
        (call, "name", "changed"),
        (spec, "description", "changed"),
    ):
        with pytest.raises(ValidationError):
            setattr(model, field, value)

    with pytest.raises(ValidationError):
        ToolCall.model_validate(
            {"call_id": "call-1", "name": "counter", "arguments": {}, "surprise": True},
            strict=True,
        )
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"call_id": 1, "name": "counter", "arguments": {}}, strict=True)


def test_capability_and_safety_metadata_are_stable_strings() -> None:
    assert Capability.PROCESS_EXECUTE.value == "process.execute"
    assert Capability.NETWORK_ACCESS.value == "network.access"
    assert RiskLevel.HIGH.value == "high"
    assert SideEffect.UNKNOWN.value == "unknown"


def test_declaration_only_tool_spec_is_valid_and_frozen() -> None:
    spec = ToolSpec(
        name="bash",
        version="1",
        description="Declare shell metadata without execution authority.",
        args_model=CountArgs,
        capability=Capability.PROCESS_EXECUTE,
        risk=RiskLevel.HIGH,
        side_effects=SideEffect.UNKNOWN,
        timeout_seconds=1,
        max_output_chars=100,
        idempotency=False,
        parallel_safety=False,
        renderer_hint="command",
    )

    assert spec.handler is None
    with pytest.raises(ValidationError):
        spec.handler = _handler


def test_tool_result_requires_exactly_one_output_or_error() -> None:
    error = ToolError(code="failed", message="Execution failed.")

    success = ToolResult(call_id="call-1", tool_name="counter", output="")
    failure = ToolResult(call_id="call-2", tool_name="counter", error=error)

    assert success.success is True
    assert failure.success is False
    with pytest.raises(ValidationError):
        ToolResult(call_id="call-3", tool_name="counter")
    with pytest.raises(ValidationError):
        ToolResult(call_id="call-4", tool_name="counter", output="value", error=error)


def test_call_and_validated_arguments_are_deeply_immutable() -> None:
    source = {"count": 1, "options": {"values": [1, 2]}}
    call = ToolCall(call_id="call-1", name="counter", arguments=source)
    spec = ToolSpec(
        name="counter",
        version="1",
        description="Count.",
        args_model=CountArgs,
        capability=Capability.FILESYSTEM_READ,
        risk=RiskLevel.LOW,
        side_effects=SideEffect.NONE,
        timeout_seconds=1,
        max_output_chars=100,
        idempotency=True,
        parallel_safety=True,
        renderer_hint="text",
    )
    validated = ValidatedToolCall(call=call, spec=spec)
    source["options"]["values"].append(3)

    assert call.model_dump()["arguments"] == {"count": 1, "options": {"values": [1, 2]}}
    assert validated.model_dump()["arguments"] == call.model_dump()["arguments"]
    with pytest.raises(TypeError):
        call.arguments["options"]["values"][0] = 9
    with pytest.raises(TypeError):
        validated.arguments["options"]["new"] = []

    with pytest.raises(ValidationError):
        ValidatedToolCall(call=call.model_copy(update={"name": "other"}), spec=spec)
    with pytest.raises(ValidationError):
        ValidatedToolCall.model_validate(
            MappingProxyType(
                {
                    "call": call.model_copy(update={"name": "other"}),
                    "spec": spec,
                    "arguments": validated.arguments,
                }
            ),
            strict=True,
        )
    for supplied in (
        ToolError(code="wrong", message="Wrong model."),
        CountArgs(count=2, options={"values": [1, 2]}),
    ):
        with pytest.raises(ValidationError):
            ValidatedToolCall(call=call, spec=spec, arguments=supplied)
    for arguments in (
        {"count": "1", "options": {"values": [1, 2]}},
        {"count": 1, "options": {"values": [1, 2]}, "extra": True},
    ):
        with pytest.raises(ValidationError):
            ValidatedToolCall(call=call.model_copy(update={"arguments": arguments}), spec=spec)
    with pytest.raises(ValidationError):
        ToolCall(call_id="call-2", name="counter", arguments={"bad": {1}})
