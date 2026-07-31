"""Immutable registry and strict tool-call validation."""

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from shadow_code.domain.tools import (
    ToolCall,
    ToolError,
    ToolSpec,
    ValidatedToolCall,
    ValidationIssue,
)

_ERROR_MESSAGES = {
    "missing": ("missing", "Field is required."),
    "extra_forbidden": ("unknown_field", "Unknown field."),
    "string_type": ("invalid_type", "Expected a string."),
    "int_type": ("invalid_type", "Expected an integer."),
    "float_type": ("invalid_type", "Expected a number."),
    "bool_type": ("invalid_type", "Expected a boolean."),
    "dict_type": ("invalid_type", "Expected an object."),
    "model_type": ("invalid_type", "Expected an object."),
}


def _normalized_issues(
    error: ValidationError, *, prefix: tuple[str | int, ...] = ()
) -> tuple[ValidationIssue, ...]:
    issues = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        code, message = _ERROR_MESSAGES.get(detail["type"], ("invalid_value", "Value is invalid."))
        issues.append(
            ValidationIssue(path=prefix + tuple(detail["loc"]), code=code, message=message)
        )
    return tuple(issues)


def _spec_metadata(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "version": spec.version,
        "description": spec.description,
        "arguments_schema": spec.args_model.model_json_schema(),
        "capability": spec.capability.value,
        "risk": spec.risk.value,
        "side_effects": spec.side_effects.value,
        "timeout_seconds": spec.timeout_seconds,
        "max_output_chars": spec.max_output_chars,
        "idempotency": spec.idempotency,
        "parallel_safety": spec.parallel_safety,
        "renderer_hint": spec.renderer_hint,
    }


@dataclass(frozen=True, slots=True, init=False)
class ToolRegistry:
    specs: tuple[ToolSpec, ...]
    digest: str
    _by_name: MappingProxyType[str, ToolSpec] = field(repr=False, compare=False)

    def __init__(self, specs: list[ToolSpec] | tuple[ToolSpec, ...]) -> None:
        ordered = tuple(sorted(specs, key=lambda spec: spec.name))
        names = tuple(spec.name for spec in ordered)
        if len(names) != len(set(names)):
            duplicate = next(name for index, name in enumerate(names) if name in names[:index])
            raise ValueError(f"Duplicate tool: {duplicate}")

        metadata = [_spec_metadata(spec) for spec in ordered]
        encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "specs", ordered)
        object.__setattr__(
            self, "_by_name", MappingProxyType({spec.name: spec for spec in ordered})
        )
        object.__setattr__(self, "digest", hashlib.sha256(encoded.encode()).hexdigest())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def validate_call(self, value: object) -> ValidatedToolCall | ToolError:
        try:
            call = ToolCall.model_validate(value, strict=True)
        except ValidationError as error:
            return ToolError(
                code="invalid_tool_call",
                message="Tool call envelope is invalid.",
                issues=_normalized_issues(error),
            )

        spec = self._by_name.get(call.name)
        if spec is None:
            return ToolError(code="unknown_tool", message="Tool is not registered.")

        try:
            return ValidatedToolCall.model_validate({"call": call, "spec": spec}, strict=True)
        except ValidationError as error:
            return ToolError(
                code="invalid_arguments",
                message="Tool arguments are invalid.",
                issues=_normalized_issues(error, prefix=("arguments",)),
            )
