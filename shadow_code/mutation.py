"""Planned, approval-bound workspace file mutations.

Threat model: a stable workspace with cooperating writers. The cooperative
workspace lock serializes Shadow Code writers that voluntarily take it; it
is NOT a security boundary and nothing here claims protection against
uncooperative or hostile same-UID processes.

A mutation is planned purely (snapshot, new-content digest, bounded preview)
so the approval binds the exact arguments and preview; it is then applied
atomically: temp file in the target directory, fsync, mode preservation,
an immediate pre-commit re-validation against the planned snapshot, a
descriptor-relative rename, a directory fsync, and a post-write readback.
Any drift in identity, content, mode, or presence before the commit aborts
with the original intact and the temp removed.

In strict mode the mutation is never executed: the planned change is
rendered as a full unified diff and exported as an isolated patch under
``<root>/.shadow-code-exports/`` with status ``exported``. The export is a
reviewed-patch fallback, not confinement; the workspace target is never
touched on that path.
"""

import difflib
import hashlib
import os
import secrets
import stat
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from shadow_code.domain.policy import WorkspaceAccessError
from shadow_code.policy.workspace import WorkspaceGuard

_READ_CHUNK_BYTES = 8192
_PREVIEW_MAX_LINES = 40
_NEW_FILE_MODE = 0o644
_TEMP_PREFIX = ".shadow-tmp-"
_EXPORT_DIR_MODE = 0o755
_EXPORT_FILE_MODE = 0o644


class MutationError(RuntimeError):
    """A fail-closed mutation failure with a stable typed code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Observed presence, identity, mode, size, and content digest."""

    exists: bool
    device: int | None
    inode: int | None
    mode: int | None
    size: int | None
    sha256: str | None


class WritePlanArgs(Protocol):
    """Structural arguments for a full-file write plan."""

    file_path: str
    content: str


class EditPlanArgs(Protocol):
    """Structural arguments for an exact-text edit plan."""

    file_path: str
    old_text: str
    new_text: str


@dataclass(frozen=True, slots=True)
class MutationPlan:
    """Pure description of an approved mutation; carries no descriptors."""

    operation: str  # "write" | "edit"
    relative_path: str
    before: FileSnapshot
    new_sha256: str
    new_size: int
    preview: str


@dataclass(frozen=True, slots=True)
class MutationReceipt:
    """Recorded evidence of a successfully applied mutation."""

    before: FileSnapshot
    after: FileSnapshot
    bytes_written: int


def _read_all_bytes(descriptor: int) -> bytes:
    chunks = []
    while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
        chunks.append(chunk)
    return b"".join(chunks)


def _snapshot_from(file_stat: os.stat_result, data: bytes) -> FileSnapshot:
    return FileSnapshot(
        exists=True,
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        mode=stat.S_IMODE(file_stat.st_mode),
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _exists_beneath(guard: WorkspaceGuard, relative_path: str) -> bool:
    """Check presence via the parent directory without following symlinks."""
    parent, _, basename = relative_path.rpartition("/")
    with guard.open_dir(parent or ".") as dir_fd:
        try:
            os.stat(basename, dir_fd=dir_fd, follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError):
            return False
    return True


def snapshot_file(guard: WorkspaceGuard, relative_path: str) -> FileSnapshot:
    """Snapshot the target; exists=False when it is absent beneath the root.

    Containment violations and I/O failures other than absence propagate as
    WorkspaceAccessError so callers fail closed instead of planning against
    an unreadable target.
    """
    try:
        with guard.open_read(relative_path) as descriptor:
            file_stat = os.fstat(descriptor)
            data = _read_all_bytes(descriptor)
    except WorkspaceAccessError:
        if not _exists_beneath(guard, relative_path):
            return FileSnapshot(
                exists=False,
                device=None,
                inode=None,
                mode=None,
                size=None,
                sha256=None,
            )
        raise
    return _snapshot_from(file_stat, data)


def _render_preview(
    relative_path: str,
    before: FileSnapshot,
    new_bytes: bytes,
    old_bytes: bytes | None,
) -> str:
    """Bounded unified diff for overwrites/edits; size line for creations."""
    if not before.exists or old_bytes is None:
        return f"+{len(new_bytes)} bytes (new file)"
    old_lines = old_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
    new_lines = new_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
    )
    if len(diff) > _PREVIEW_MAX_LINES:
        retained = diff[:_PREVIEW_MAX_LINES]
        retained.append(f"[...diff truncated: {len(diff) - _PREVIEW_MAX_LINES} more lines...]\n")
        diff = retained
    return "".join(diff).rstrip("\n") or "(no content change)"


def _build_plan(
    operation: str,
    relative_path: str,
    before: FileSnapshot,
    new_bytes: bytes,
    old_bytes: bytes | None,
) -> MutationPlan:
    return MutationPlan(
        operation=operation,
        relative_path=relative_path,
        before=before,
        new_sha256=hashlib.sha256(new_bytes).hexdigest(),
        new_size=len(new_bytes),
        preview=_render_preview(relative_path, before, new_bytes, old_bytes),
    )


def build_write_plan(guard: WorkspaceGuard, args: WritePlanArgs) -> MutationPlan:
    """Pure plan for a full-file write; performs no mutations."""
    before = snapshot_file(guard, args.file_path)
    old_bytes: bytes | None = None
    if before.exists:
        with guard.open_read(args.file_path) as descriptor:
            old_bytes = _read_all_bytes(descriptor)
    return _build_plan("write", args.file_path, before, args.content.encode("utf-8"), old_bytes)


def build_edit_plan(guard: WorkspaceGuard, args: EditPlanArgs) -> MutationPlan:
    """Pure plan for an exact-text edit; performs no mutations.

    Fail closed: the exact text must match EXACTLY ONCE. Zero matches raise
    `no_match`; more than one match raises `ambiguous_match` and nothing is
    planned or written.
    """
    try:
        with guard.open_read(args.file_path) as descriptor:
            file_stat = os.fstat(descriptor)
            old_bytes = _read_all_bytes(descriptor)
    except WorkspaceAccessError:
        if not _exists_beneath(guard, args.file_path):
            raise MutationError(
                "no_match", f"edit target '{args.file_path}' does not exist"
            ) from None
        raise
    before = _snapshot_from(file_stat, old_bytes)
    needle = args.old_text.encode("utf-8")
    matches = old_bytes.count(needle)
    if matches == 0:
        raise MutationError("no_match", f"edit text does not appear in '{args.file_path}'")
    if matches > 1:
        raise MutationError(
            "ambiguous_match",
            f"edit text appears {matches} times in '{args.file_path}'; "
            "exact-text edits require exactly one match",
        )
    new_bytes = old_bytes.replace(needle, args.new_text.encode("utf-8"), 1)
    return _build_plan("edit", args.file_path, before, new_bytes, old_bytes)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def _require_no_drift(before: FileSnapshot, current: FileSnapshot, path: str) -> None:
    if before != current:
        raise MutationError(
            "workspace_drift",
            f"'{path}' changed after the mutation was planned; aborted with the original intact",
        )


def apply_mutation(
    guard: WorkspaceGuard, plan: MutationPlan, new_content: bytes
) -> MutationReceipt:
    """Apply an approved plan atomically under the cooperative lock.

    The temp file is created, written, fsynced, and mode-adjusted inside the
    target directory via descriptor-relative, no-follow syscalls. An
    immediate pre-commit re-snapshot must equal the planned snapshot exactly;
    any drift aborts with the original intact and the temp removed. The
    rename and a directory fsync complete the commit, and a post-write
    readback through the guard verifies the landed digest.
    """
    if len(new_content) != plan.new_size or (
        hashlib.sha256(new_content).hexdigest() != plan.new_sha256
    ):
        raise MutationError("invalid_plan", "content does not match the mutation plan")

    parent, _, basename = plan.relative_path.rpartition("/")
    with guard.mutation_lock(), guard.open_dir(parent or ".") as dir_fd:
        temp_name = f"{_TEMP_PREFIX}{secrets.token_hex(8)}"
        descriptor: int | None = None
        committed = False
        try:
            try:
                descriptor = os.open(
                    temp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=dir_fd,
                )
                _write_all(descriptor, new_content)
                os.fsync(descriptor)
                mode = plan.before.mode if plan.before.mode is not None else _NEW_FILE_MODE
                os.fchmod(descriptor, mode)
                os.close(descriptor)
                descriptor = None
                # Immediate pre-commit re-validation: the preview-time plan
                # was snapshotted at approval, so any drift in identity,
                # content, mode, or presence since then aborts here.
                current = snapshot_file(guard, plan.relative_path)
                _require_no_drift(plan.before, current, plan.relative_path)
                os.rename(temp_name, basename, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                committed = True
                os.fsync(dir_fd)
            except (MutationError, WorkspaceAccessError):
                raise
            except OSError as error:
                raise MutationError(
                    "io_error", f"mutation failed: {error.strerror or error}"
                ) from error
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            if not committed:
                with suppress(OSError):
                    os.unlink(temp_name, dir_fd=dir_fd)

        # Post-write readback through the guard proves the landed content.
        after = snapshot_file(guard, plan.relative_path)
        if not after.exists or after.sha256 != plan.new_sha256:
            raise MutationError(
                "readback_mismatch",
                f"post-write readback of '{plan.relative_path}' does not match the approved digest",
            )
        return MutationReceipt(before=plan.before, after=after, bytes_written=len(new_content))


def render_patch(plan: MutationPlan, old_bytes: bytes | None, new_bytes: bytes) -> str:
    """Full, deterministic unified diff for the strict-mode patch export.

    Unlike the bounded approval preview, the exported patch is complete: it
    is the reviewable artifact the user applies manually. New files diff
    from ``/dev/null``; there are no timestamps, so identical plans render
    identical patches.
    """
    if len(new_bytes) != plan.new_size or (
        hashlib.sha256(new_bytes).hexdigest() != plan.new_sha256
    ):
        raise MutationError("invalid_plan", "content does not match the mutation plan")
    fromfile = "/dev/null" if old_bytes is None else f"a/{plan.relative_path}"
    old_lines = (
        []
        if old_bytes is None
        else old_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
    )
    new_lines = new_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=fromfile,
        tofile=f"b/{plan.relative_path}",
    )
    return "".join(diff)


def export_patch(
    guard: WorkspaceGuard,
    plan: MutationPlan,
    patch_text: str,
    exports_dir_name: str = ".shadow-code-exports",
) -> str:
    """Write a reviewed patch beneath the workspace root and return its path.

    This is the strict-mode fallback for an approved mutation: the change is
    recorded as an isolated patch under ``<root>/<exports_dir_name>/`` with
    status ``exported`` and the workspace target is NEVER touched. The export
    is a reviewed-patch fallback, not confinement.

    The exports directory is created descriptor-relative to the pinned root;
    a symlink or non-directory at that path fails closed through the guard.
    Names are deterministic (UTC stamp, operation, basename) with a short
    random token appended only on collision. The file is written, fsynced,
    and the directory fsynced; the workspace-relative patch path is returned.
    """
    _, _, basename = plan.relative_path.rpartition("/")
    with guard.mutation_lock():
        with guard.open_dir(".") as root_fd:
            try:
                os.mkdir(exports_dir_name, _EXPORT_DIR_MODE, dir_fd=root_fd)
            except FileExistsError:
                pass
            except OSError as error:
                raise MutationError(
                    "io_error", f"cannot create exports directory: {error.strerror or error}"
                ) from error
        with guard.open_dir(exports_dir_name) as exports_fd:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_base = basename.replace("/", "_")
            name = f"{stamp}-{plan.operation}-{safe_base}.patch"
            descriptor: int | None = None
            try:
                for _ in range(2):
                    try:
                        descriptor = os.open(
                            name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                            _EXPORT_FILE_MODE,
                            dir_fd=exports_fd,
                        )
                        break
                    except FileExistsError:
                        name = f"{stamp}-{plan.operation}-{safe_base}-{secrets.token_hex(4)}.patch"
                if descriptor is None:
                    raise MutationError("io_error", "cannot allocate a unique patch export name")
                _write_all(descriptor, patch_text.encode("utf-8"))
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
                os.fsync(exports_fd)
            except MutationError:
                raise
            except OSError as error:
                raise MutationError(
                    "io_error", f"patch export failed: {error.strerror or error}"
                ) from error
            finally:
                if descriptor is not None:
                    with suppress(OSError):
                        os.close(descriptor)
    return f"{exports_dir_name}/{name}"
