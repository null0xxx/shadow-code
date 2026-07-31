"""Strict tool contracts shared by providers, policy, and executors."""

from collections.abc import Callable, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, PlainSerializer, model_validator
from typing_extensions import Self


class Capability(str, Enum):
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    PROCESS_EXECUTE = "process.execute"
    NETWORK_ACCESS = "network.access"
    MCP_INVOKE = "mcp.invoke"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SideEffect(str, Enum):
    NONE = "none"
    MUTATING = "mutating"
    UNKNOWN = "unknown"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("tool arguments must contain only JSON-compatible values")


def _thaw(value: Any) -> Any:
    return (
        {key: _thaw(item) for key, item in value.items()}
        if isinstance(value, Mapping)
        else [_thaw(item) for item in value]
        if isinstance(value, tuple)
        else value
    )


FrozenArguments = Annotated[
    Mapping[str, Any],
    AfterValidator(_freeze),
    PlainSerializer(_thaw, return_type=dict[str, Any]),
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ValidationIssue(FrozenModel):
    path: tuple[str | int, ...]
    code: str
    message: str


class ToolError(FrozenModel):
    code: str
    message: str
    issues: tuple[ValidationIssue, ...] = ()


class ToolCall(FrozenModel):
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: FrozenArguments


class ToolResult(FrozenModel):
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    output: str | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def _require_one_payload(self) -> Self:
        if (self.output is None) == (self.error is None):
            raise ValueError("exactly one of output or error is required")
        return self

    @property
    def success(self) -> bool:
        return self.error is None


ToolHandler = Callable[[ToolCall, BaseModel, object], ToolResult]


class ToolSpec(FrozenModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.-]*$")
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    args_model: type[BaseModel]
    handler: ToolHandler | None = None
    capability: Capability
    risk: RiskLevel
    side_effects: SideEffect
    timeout_seconds: float = Field(gt=0)
    max_output_chars: int = Field(gt=0)
    idempotency: bool
    parallel_safety: bool
    renderer_hint: str = Field(min_length=1)


class ValidatedToolCall(FrozenModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    call: ToolCall
    spec: ToolSpec
    arguments: FrozenArguments

    @model_validator(mode="before")
    @classmethod
    def _bind_arguments(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        value = dict(value)
        call, spec = value.get("call"), value.get("spec")
        if not isinstance(call, ToolCall) or not isinstance(spec, ToolSpec):
            return value
        if call.name != spec.name:
            raise ValueError("call and spec names must match")
        validated = spec.args_model.model_validate(
            _thaw(call.arguments), strict=True, extra="forbid"
        )
        canonical = validated.model_dump(mode="json")
        if "arguments" in value:
            supplied = value["arguments"]
            if (
                type(supplied) is not spec.args_model
                or supplied.model_dump(mode="json") != canonical
            ):
                raise ValueError("arguments do not match the call and spec")
        return {**value, "arguments": canonical}
