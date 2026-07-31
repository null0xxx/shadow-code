"""Immutable domain facts and typed failures for execution policy."""

from dataclasses import dataclass
from enum import Enum


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


class WorkspaceAccessError(RuntimeError):
    """A fail-closed workspace access failure with a stable reason."""

    def __init__(self, reason: WorkspaceFailure, message: str) -> None:
        self.reason = reason
        super().__init__(message)
