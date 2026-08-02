"""Provider-neutral validation for flat ToolSpec JSON Schemas."""

import json
from copy import deepcopy
from typing import Any

from shadow_code.domain.tools import ToolSpec

_TOP_LEVEL_KEYS = {"additionalProperties", "description", "properties", "required", "title", "type"}
_PROPERTY_KEYS = {
    "default",
    "description",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maxLength",
    "maximum",
    "minLength",
    "minimum",
    "multipleOf",
    "pattern",
    "title",
    "type",
}
SUPPORTED_CONSTRAINT_FACTS = tuple(
    "minimum maximum exclusiveMaximum exclusiveMinimum multipleOf minLength "  # noqa: SIM905
    "maxLength pattern".split()
)
_SCALAR_TYPES = {"boolean", "integer", "number", "string"}


class UnsupportedToolSchemaError(ValueError):
    """Raised when a ToolSpec schema cannot be projected without rewriting."""


def _matches_type(kind: str, value: object) -> bool:
    expected = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,)}
    return type(value) in expected[kind]


def _validate_scalar_field(spec: ToolSpec, name: str, field: dict[str, Any]) -> None:
    kind = field["type"]
    try:
        json.dumps(field, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise UnsupportedToolSchemaError(f"{spec.name}.{name}: non-JSON value") from error
    if any(key in field and not isinstance(field[key], str) for key in ("title", "description")):
        raise UnsupportedToolSchemaError(f"{spec.name}.{name}: invalid text metadata")
    if "default" in field and not _matches_type(kind, field["default"]):
        raise UnsupportedToolSchemaError(f"{spec.name}.{name}: invalid default")
    for key in SUPPORTED_CONSTRAINT_FACTS[:5]:
        if key in field and (
            kind not in {"integer", "number"}
            or type(field[key]) not in (int, float)
            or (key == "multipleOf" and field[key] <= 0)
        ):
            raise UnsupportedToolSchemaError(f"{spec.name}.{name}: invalid {key}")
    for key in SUPPORTED_CONSTRAINT_FACTS[5:7]:
        if key in field and (kind != "string" or type(field[key]) is not int or field[key] < 0):
            raise UnsupportedToolSchemaError(f"{spec.name}.{name}: invalid {key}")
    pattern = SUPPORTED_CONSTRAINT_FACTS[-1]
    if pattern in field and (kind != "string" or not isinstance(field[pattern], str)):
        raise UnsupportedToolSchemaError(f"{spec.name}.{name}: invalid {pattern}")


def flat_tool_schema(spec: ToolSpec) -> dict[str, Any]:
    """Return an exact copied flat schema or fail closed on unsupported constructs."""
    schema = spec.args_model.model_json_schema()
    if set(schema) - _TOP_LEVEL_KEYS:
        raise UnsupportedToolSchemaError(f"{spec.name}: unsupported top-level schema keys")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise UnsupportedToolSchemaError(f"{spec.name}: schema must be a closed object")

    properties = schema.get("properties")
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise UnsupportedToolSchemaError(f"{spec.name}: invalid properties or required fields")
    if not all(isinstance(name, str) and name in properties for name in required):
        raise UnsupportedToolSchemaError(f"{spec.name}: invalid required field")

    for name, field in properties.items():
        if not isinstance(name, str) or not isinstance(field, dict):
            raise UnsupportedToolSchemaError(f"{spec.name}: invalid property")
        if set(field) - _PROPERTY_KEYS or field.get("type") not in _SCALAR_TYPES:
            raise UnsupportedToolSchemaError(f"{spec.name}.{name}: unsupported property schema")
        _validate_scalar_field(spec, name, field)
    return deepcopy(schema)
