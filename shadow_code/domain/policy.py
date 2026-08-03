"""Immutable domain facts and typed failures for execution policy."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from shadow_code.domain.tools import Capability


class WorkspaceFailure(str, Enum):
    INVALID_PATH = "invalid_path"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    UNSUPPORTED_CONTAINMENT = "unsupported_containment"
    CONTAINMENT_VIOLATION = "containment_violation"
    ROOT_CHANGED = "root_changed"
    CLOSED = "closed"
    IO_ERROR = "io_error"


@dataclass(frozen=True, slots=True)
class WorkspaceIdentity:
    device: int
    inode: int


class PolicyDisposition(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class PolicyReason(str, Enum):
    INVALID_CALL = "invalid_call"
    CAPABILITY_NOT_GRANTED = "capability_not_granted"
    WORKSPACE_UNAVAILABLE = "workspace_unavailable"
    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"
    SIDE_EFFECT_UNKNOWN = "side_effect_unknown"
    SENSITIVE_CAPABILITY = "sensitive_capability"
    UNSUPPORTED_POLICY_CASE = "unsupported_policy_case"


@dataclass(frozen=True, slots=True, init=False)
class PolicyFacts:
    granted_capabilities: frozenset[Capability]
    workspace_identity: WorkspaceIdentity | None

    def __init__(
        self,
        granted_capabilities: Iterable[Capability],
        workspace_identity: WorkspaceIdentity | None,
    ) -> None:
        validated_capabilities = tuple(granted_capabilities)
        if any(not isinstance(capability, Capability) for capability in validated_capabilities):
            raise TypeError("granted_capabilities must contain only Capability values")
        object.__setattr__(self, "granted_capabilities", frozenset(validated_capabilities))
        object.__setattr__(self, "workspace_identity", workspace_identity)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    call_id: str | None
    tool_name: str | None
    disposition: PolicyDisposition
    reason: PolicyReason


class WorkspaceAccessError(RuntimeError):
    """A fail-closed workspace access failure with a stable reason."""

    def __init__(self, reason: WorkspaceFailure, message: str) -> None:
        self.reason = reason
        super().__init__(message)
