from collections.abc import Callable

import pytest
from pydantic import BaseModel

from shadow_code.domain.tools import (
    Capability,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
    ValidatedToolCall,
)
from shadow_code.tools.registry import ToolRegistry


class CountArgs(BaseModel):
    count: int
    label: str


def _spec(
    name: str,
    handler: Callable[[ToolCall, BaseModel, object], ToolResult] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        version="1",
        description=f"Validate {name}.",
        args_model=CountArgs,
        handler=handler,
        capability=Capability.FILESYSTEM_READ,
        risk=RiskLevel.LOW,
        side_effects=SideEffect.NONE,
        timeout_seconds=2,
        max_output_chars=100,
        idempotency=True,
        parallel_safety=True,
        renderer_hint="text",
    )


def test_registry_is_sorted_immutable_and_rejects_duplicates() -> None:
    handler = lambda call, arguments, context: ToolResult(  # noqa: E731
        call_id="unused", tool_name="unused", output="unused"
    )
    registry = ToolRegistry([_spec("zeta", handler), _spec("alpha", handler)])

    assert registry.names == ("alpha", "zeta")
    assert tuple(spec.name for spec in registry.specs) == registry.names
    assert ToolRegistry([_spec("declared")]).names == ("declared",)
    with pytest.raises(AttributeError):
        registry.names = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="Duplicate tool"):
        ToolRegistry([_spec("same", handler), _spec("same", handler)])


@pytest.mark.parametrize(
    ("arguments", "expected_paths"),
    [
        ({"count": 1}, {("arguments", "label")}),
        ({"count": "1", "label": "x"}, {("arguments", "count")}),
        ({"count": True, "label": "x"}, {("arguments", "count")}),
        ({"count": 1, "label": "x", "extra": "secret-value"}, {("arguments", "extra")}),
        ("not-an-object", {("arguments",)}),
    ],
)
def test_validation_rejects_invalid_arguments_without_invoking_handler(
    arguments: object, expected_paths: set[tuple[str | int, ...]]
) -> None:
    invocations = 0

    def handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
        nonlocal invocations
        invocations += 1
        return ToolResult(call_id="call-1", tool_name="counter", output="unexpected")

    result = ToolRegistry([_spec("counter", handler)]).validate_call(
        {"call_id": "call-1", "name": "counter", "arguments": arguments}
    )

    assert isinstance(result, ToolError)
    assert result.code in {"invalid_tool_call", "invalid_arguments"}
    assert {issue.path for issue in result.issues} == expected_paths
    assert invocations == 0
    serialized = result.model_dump()
    assert "input" not in str(serialized)
    assert "url" not in str(serialized)
    assert "secret-value" not in str(serialized)


def test_validation_rejects_unknown_tools_and_envelope_fields() -> None:
    registry = ToolRegistry([])

    unknown = registry.validate_call({"call_id": "call-1", "name": "missing", "arguments": {}})
    malformed = registry.validate_call(
        {"call_id": "call-1", "name": "missing", "arguments": {}, "extra": True}
    )

    assert isinstance(unknown, ToolError)
    assert unknown.code == "unknown_tool"
    assert isinstance(malformed, ToolError)
    assert malformed.code == "invalid_tool_call"
    assert malformed.issues[0].path == ("extra",)


def test_validation_normalizes_nested_non_string_mapping_keys() -> None:
    result = ToolRegistry([_spec("counter")]).validate_call(
        {
            "call_id": "call-1",
            "name": "counter",
            "arguments": {"count": 1, "label": "x", "nested": {1: "x", "a": "y"}},
        }
    )

    assert isinstance(result, ToolError)
    assert result.code == "invalid_tool_call"
    assert {issue.path for issue in result.issues} == {("arguments",)}


def test_valid_call_returns_exact_model_without_executing_handler() -> None:
    invocations = 0

    def handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
        nonlocal invocations
        invocations += 1
        return ToolResult(call_id="call-1", tool_name="counter", output="unexpected")

    result = ToolRegistry([_spec("counter", handler)]).validate_call(
        {"call_id": "call-1", "name": "counter", "arguments": {"count": 2, "label": "x"}}
    )

    assert isinstance(result, ValidatedToolCall)
    assert result.model_dump()["arguments"] == {"count": 2, "label": "x"}
    assert result.spec.name == "counter"
    assert invocations == 0


def test_registry_digest_is_order_stable_and_excludes_handler_identity() -> None:
    first_handler = lambda call, arguments, context: ToolResult(  # noqa: E731
        call_id="unused", tool_name="unused", output="one"
    )
    second_handler = lambda call, arguments, context: ToolResult(  # noqa: E731
        call_id="unused", tool_name="unused", output="two"
    )

    first = ToolRegistry([_spec("zeta", first_handler), _spec("alpha", first_handler)])
    reordered = ToolRegistry([_spec("alpha", second_handler), _spec("zeta", second_handler)])
    changed = ToolRegistry([_spec("alpha", second_handler)])

    assert first.digest == reordered.digest
    assert first.digest != changed.digest
    assert len(first.digest) == 64
