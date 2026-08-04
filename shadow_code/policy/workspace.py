"""Linux descriptor-relative workspace containment."""

import ctypes
import errno
import fcntl
import os
import platform
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Protocol

from shadow_code.domain.policy import (
    WorkspaceAccessError,
    WorkspaceFailure,
    WorkspaceIdentity,
)

_OPENAT2_SYSCALLS = {
    "x86_64": 437,
    "amd64": 437,
    "aarch64": 437,
    "arm64": 437,
}
_RESOLVE_NO_XDEV = 0x01
_RESOLVE_NO_MAGICLINKS = 0x02
_RESOLVE_NO_SYMLINKS = 0x04
_RESOLVE_BENEATH = 0x08
_RESOLVE_FLAGS = _RESOLVE_BENEATH | _RESOLVE_NO_SYMLINKS | _RESOLVE_NO_MAGICLINKS | _RESOLVE_NO_XDEV
_OS_FLAGS = ("O_PATH", "O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK")
_FLAGS_OK = all(hasattr(os, name) for name in _OS_FLAGS)
_ROOT_FLAGS = sum(getattr(os, name, 0) for name in _OS_FLAGS[:3])
_READ_FLAGS = os.O_RDONLY | sum(getattr(os, name, 0) for name in _OS_FLAGS[2:])
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
_MUTATION_LOCK_NAME = ".shadow-code.lock"


class _StatLike(Protocol):
    st_dev: int
    st_ino: int


OpenAt2 = Callable[[int, str, int], int]
Fstat = Callable[[int], os.stat_result | _StatLike]


class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("mode", ctypes.c_uint64),
        ("resolve", ctypes.c_uint64),
    ]


class _LinuxOpenAt2:
    """Minimal audited openat2 syscall adapter; unknown platforms fail closed."""

    def __init__(self, architecture: str | None = None) -> None:
        machine = (architecture or platform.machine()).lower()
        if platform.system() != "Linux" or machine not in _OPENAT2_SYSCALLS or not _FLAGS_OK:
            raise WorkspaceAccessError(
                WorkspaceFailure.UNSUPPORTED_PLATFORM,
                f"openat2 is unsupported on {platform.system()} {machine}",
            )
        self._number = _OPENAT2_SYSCALLS[machine]
        self._syscall = ctypes.CDLL(None, use_errno=True).syscall
        self._syscall.restype = ctypes.c_long

    def __call__(self, root_fd: int, path: str, flags: int) -> int:
        how = _OpenHow(flags=flags, mode=0, resolve=_RESOLVE_FLAGS)
        ctypes.set_errno(0)
        result = self._syscall(
            self._number,
            root_fd,
            ctypes.c_char_p(os.fsencode(path)),
            ctypes.byref(how),
            ctypes.sizeof(how),
        )
        if result < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), path)
        return int(result)


def _normalized_relative_path(path: str) -> str:
    if not isinstance(path, str) or not path or "\x00" in path or path.startswith("/"):
        raise WorkspaceAccessError(WorkspaceFailure.INVALID_PATH, "path must be relative")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise WorkspaceAccessError(WorkspaceFailure.INVALID_PATH, "path must be normalized")
    return "/".join(parts)


def _normalized_relative_dir(path: str) -> str:
    """Normalize a relative directory path; "" and "." denote the root."""
    if path in {"", "."}:
        return "."
    return _normalized_relative_path(path)


def _normalize_open_error(error: OSError) -> WorkspaceAccessError:
    if error.errno in {errno.ENOSYS, errno.EINVAL, errno.E2BIG}:
        reason = WorkspaceFailure.UNSUPPORTED_CONTAINMENT
    elif error.errno in {errno.ELOOP, errno.EXDEV, errno.EAGAIN}:
        reason = WorkspaceFailure.CONTAINMENT_VIOLATION
    else:
        reason = WorkspaceFailure.IO_ERROR
    return WorkspaceAccessError(reason, f"workspace open failed: {error.strerror}")


class WorkspaceGuard:
    """Own a pinned workspace descriptor and open files beneath it safely."""

    def __init__(
        self,
        root: str | Path,
        *,
        architecture: str | None = None,
        openat2: OpenAt2 | None = None,
        fstat: Fstat = os.fstat,
        close: Callable[[int], None] | None = None,
    ) -> None:
        self._openat2 = openat2 or _LinuxOpenAt2(architecture)
        self._fstat = fstat
        self._close = os.close if close is None else close
        try:
            self._root_fd: int | None = os.open(os.fspath(root), _ROOT_FLAGS)
            root_stat = self._fstat(self._root_fd)
        except OSError as error:
            root_fd = getattr(self, "_root_fd", None)
            if root_fd is not None:
                with suppress(OSError):
                    self._close(root_fd)
            raise WorkspaceAccessError(
                WorkspaceFailure.IO_ERROR, f"cannot pin workspace: {error.strerror}"
            ) from error
        self._identity = WorkspaceIdentity(device=root_stat.st_dev, inode=root_stat.st_ino)

    @property
    def identity(self) -> WorkspaceIdentity:
        return self._identity

    @property
    def closed(self) -> bool:
        return self._root_fd is None

    def _verified_root_fd(self) -> int:
        if self._root_fd is None:
            raise WorkspaceAccessError(WorkspaceFailure.CLOSED, "workspace guard is closed")
        try:
            current = self._fstat(self._root_fd)
        except OSError as error:
            raise WorkspaceAccessError(
                WorkspaceFailure.ROOT_CHANGED, "workspace identity cannot be verified"
            ) from error
        if (current.st_dev, current.st_ino) != (self._identity.device, self._identity.inode):
            raise WorkspaceAccessError(
                WorkspaceFailure.ROOT_CHANGED, "pinned workspace identity changed"
            )
        return self._root_fd

    @contextmanager
    def open_read(self, path: str) -> Iterator[int]:
        relative_path = _normalized_relative_path(path)
        root_fd = self._verified_root_fd()
        try:
            descriptor = self._openat2(root_fd, relative_path, _READ_FLAGS)
        except OSError as error:
            raise _normalize_open_error(error) from error
        primary: BaseException | None = None
        try:
            try:
                mode = os.fstat(descriptor).st_mode
            except OSError as error:
                raise WorkspaceAccessError(WorkspaceFailure.IO_ERROR, "fstat failed") from error
            if not stat.S_ISREG(mode):
                raise WorkspaceAccessError(WorkspaceFailure.IO_ERROR, "path is not a regular file")
            yield descriptor
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                self._close(descriptor)
            except OSError as error:
                if primary is None:
                    raise WorkspaceAccessError(WorkspaceFailure.IO_ERROR, "close failed") from error

    @contextmanager
    def open_dir(self, path: str) -> Iterator[int]:
        """Resolve a normalized relative directory beneath the pinned root.

        "" or "." selects the workspace root itself. The yielded descriptor
        is a real directory fd suitable for ``dir_fd=`` syscalls; it is
        resolved with the same no-symlink, beneath-root discipline as
        :meth:`open_read` and fails closed on non-directories or
        containment violations.
        """
        relative_dir = _normalized_relative_dir(path)
        root_fd = self._verified_root_fd()
        try:
            descriptor = self._openat2(root_fd, relative_dir, _DIR_FLAGS)
        except OSError as error:
            raise _normalize_open_error(error) from error
        primary: BaseException | None = None
        try:
            try:
                mode = os.fstat(descriptor).st_mode
            except OSError as error:
                raise WorkspaceAccessError(WorkspaceFailure.IO_ERROR, "fstat failed") from error
            if not stat.S_ISDIR(mode):
                raise WorkspaceAccessError(WorkspaceFailure.IO_ERROR, "path is not a directory")
            yield descriptor
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                self._close(descriptor)
            except OSError as error:
                if primary is None:
                    raise WorkspaceAccessError(WorkspaceFailure.IO_ERROR, "close failed") from error

    @contextmanager
    def mutation_lock(self) -> Iterator[None]:
        """Hold the cooperative per-workspace mutation lock.

        COOPERATIVE ONLY: this advisory flock is honored solely by Shadow
        Code writers that voluntarily take it, serializing their mutation
        commits. It is NOT a security boundary and offers no protection
        against uncooperative or hostile processes.

        The lock file is opened descriptor-relative to the pinned root with
        O_NOFOLLOW; the flock is released and descriptors closed on exit.
        """
        with self.open_dir(".") as dir_fd:
            try:
                lock_fd = os.open(
                    _MUTATION_LOCK_NAME,
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=dir_fd,
                )
            except OSError as error:
                raise WorkspaceAccessError(
                    WorkspaceFailure.IO_ERROR,
                    f"cannot open mutation lock: {error.strerror}",
                ) from error
            try:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except OSError as error:
                    raise WorkspaceAccessError(
                        WorkspaceFailure.IO_ERROR,
                        f"cannot take mutation lock: {error.strerror}",
                    ) from error
                yield
            finally:
                with suppress(OSError):
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                self._close(lock_fd)

    def close(self) -> None:
        root_fd = self._root_fd
        if root_fd is None:
            return
        self._root_fd = None
        try:
            self._close(root_fd)
        except OSError as error:
            raise WorkspaceAccessError(WorkspaceFailure.IO_ERROR, "root close failed") from error

    def __enter__(self) -> "WorkspaceGuard":
        self._verified_root_fd()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.close()
        except WorkspaceAccessError:
            if exc is None:
                raise
