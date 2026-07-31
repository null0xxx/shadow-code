import pytest
from pydantic import BaseModel, ConfigDict, Field

from shadow_code.domain.tools import ToolSpec, ValidatedToolCall
from shadow_code.ollama_client import render_ollama_tool_schemas
from shadow_code.prompt import render_tool_documentation
from shadow_code.tools.catalog import BASH_SPEC, DEFAULT_TOOL_REGISTRY, READ_FILE_SPEC
from shadow_code.tools.projections import UnsupportedToolSchemaError
from shadow_code.tools.registry import ToolRegistry


def test_default_registry_provider_and_prompt_schema_parity() -> None:
    envelopes = render_ollama_tool_schemas(DEFAULT_TOOL_REGISTRY)
    documentation = render_tool_documentation(DEFAULT_TOOL_REGISTRY)

    assert [item["function"]["name"] for item in envelopes] == [
        "bash",
        "read_file",
    ]
    assert documentation.index("## bash") < documentation.index("## read_file")
    for spec, envelope in zip(DEFAULT_TOOL_REGISTRY.specs, envelopes, strict=True):
        function = envelope["function"]
        schema = spec.args_model.model_json_schema()
        assert envelope["type"] == "function"
        assert function["name"] == spec.name
        assert function["description"] == spec.description
        assert function["parameters"] == schema
        assert schema["additionalProperties"] is False
        assert f"## {spec.name}" in documentation
        assert f"Description: {spec.description}" in documentation
        assert f"Version: {spec.version}" in documentation
        for name, field in schema["properties"].items():
            assert f"- {name}: {field['type']}" in documentation
            requirement = "required" if name in schema.get("required", []) else "optional"
            assert f"- {name}: {field['type']}; {requirement}" in documentation
            if "default" in field:
                assert f"default={field['default']}" in documentation
            assert field["description"] in documentation
            for bound in ("minimum", "maximum"):
                if bound in field:
                    assert f"{bound}={field[bound]}" in documentation


def test_canonical_examples_validate_without_execution() -> None:
    examples = {
        "bash": {"command": "printf ok"},
        "read_file": {"file_path": "/tmp/example", "offset": 1, "limit": 10},
    }

    for call_id, (name, arguments) in enumerate(examples.items()):
        result = DEFAULT_TOOL_REGISTRY.validate_call(
            {"call_id": f"example-{call_id}", "name": name, "arguments": arguments}
        )
        assert isinstance(result, ValidatedToolCall)


def test_projection_output_is_deterministic() -> None:
    reordered = ToolRegistry(tuple(reversed(DEFAULT_TOOL_REGISTRY.specs)))

    assert render_ollama_tool_schemas(reordered) == render_ollama_tool_schemas(
        DEFAULT_TOOL_REGISTRY
    )
    assert render_tool_documentation(reordered) == render_tool_documentation(DEFAULT_TOOL_REGISTRY)


def test_changed_spec_metadata_updates_projections_and_digest() -> None:
    changed = BASH_SPEC.model_copy(
        update={"description": "Changed declaration description.", "version": "2"}
    )
    registry = ToolRegistry((changed, READ_FILE_SPEC))
    envelope = render_ollama_tool_schemas(registry)[0]["function"]
    documentation = render_tool_documentation(registry)

    assert envelope["description"] == changed.description
    assert changed.description in documentation
    assert "Version: 2" in documentation
    assert registry.digest != DEFAULT_TOOL_REGISTRY.digest


def test_argument_metadata_is_shared_by_schema_prompt_and_digest() -> None:
    class ChangedBashArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
        command: str = Field(
            description="Changed command description.", min_length=2, pattern=r"^.+$"
        )
        ratio: float = Field(default=1.0, multiple_of=0.5)

    changed = BASH_SPEC.model_copy(update={"args_model": ChangedBashArgs})
    registry = ToolRegistry((changed,))
    schema = render_ollama_tool_schemas(registry)[0]["function"]["parameters"]
    documentation = render_tool_documentation(registry)

    assert schema["properties"]["command"]["description"] == "Changed command description."
    assert "Changed command description." in documentation
    assert schema["properties"]["command"]["pattern"] == "^.+$"
    assert "pattern=^.+$" in documentation
    assert schema["properties"]["ratio"]["multipleOf"] == 0.5
    assert "multipleOf=0.5" in documentation
    assert registry.digest != ToolRegistry((BASH_SPEC,)).digest


class NestedValue(BaseModel):
    value: str


class NestedArgs(BaseModel):
    nested: NestedValue


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class UnionArgs(StrictArgs):
    value: str | int


class MalformedMinimumArgs(StrictArgs):
    value: int = Field(json_schema_extra={"minimum": "zero"})


class IncompatibleConstraintArgs(StrictArgs):
    value: str = Field(json_schema_extra={"minimum": 1})


class IncompatibleDefaultArgs(StrictArgs):
    value: int = "1"  # type: ignore[assignment]


class NonFiniteDefaultArgs(StrictArgs):
    value: float = float("nan")


class FormatArgs(StrictArgs):
    value: str = Field(json_schema_extra={"format": "uuid"})


@pytest.mark.parametrize(
    "args_model",
    [
        NestedArgs,
        UnionArgs,
        MalformedMinimumArgs,
        IncompatibleConstraintArgs,
        IncompatibleDefaultArgs,
        NonFiniteDefaultArgs,
        FormatArgs,
    ],
)
def test_unsupported_complex_schemas_fail_closed(args_model: type[BaseModel]) -> None:
    complex_spec: ToolSpec = BASH_SPEC.model_copy(
        update={"name": "complex", "args_model": args_model}
    )
    registry = ToolRegistry((complex_spec,))

    with pytest.raises(UnsupportedToolSchemaError):
        render_ollama_tool_schemas(registry)
    with pytest.raises(UnsupportedToolSchemaError):
        render_tool_documentation(registry)
