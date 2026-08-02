import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from shadow_code.domain.policy import WorkspaceFailure
from shadow_code.policy.workspace import WorkspaceAccessError, WorkspaceGuard


def test_guard_pins_root_identity_and_reads_nested_file_by_descriptor(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hello.txt").write_text("hello", encoding="utf-8")

    with WorkspaceGuard(tmp_path) as guard:
        expected = os.stat(tmp_path)
        assert guard.identity.device == expected.st_dev
        assert guard.identity.inode == expected.st_ino
        with pytest.raises(AttributeError):
            guard.identity = guard.identity
        with guard.open_read("nested/hello.txt") as descriptor:
            assert os.read(descriptor, 32) == b"hello"

    assert guard.closed is True


@pytest.mark.parametrize(
    "path",
    ["", ".", "..", "/etc/passwd", "./file", "dir/../file", "dir//file", "file\x00x"],
)
def test_guard_rejects_non_normalized_relative_paths(tmp_path: Path, path: str) -> None:
    with (
        WorkspaceGuard(tmp_path) as guard,
        pytest.raises(WorkspaceAccessError) as raised,
        guard.open_read(path),
    ):
        pass

    assert raised.value.reason is WorkspaceFailure.INVALID_PATH


@pytest.mark.parametrize("target", ["inside", "outside"])
def test_guard_rejects_symlinks_inside_and_outside_workspace(tmp_path: Path, target: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "inside.txt"
    outside = tmp_path / "outside.txt"
    inside.write_text("inside", encoding="utf-8")
    outside.write_text("outside", encoding="utf-8")
    (workspace / "link.txt").symlink_to(inside if target == "inside" else outside)

    with (
        WorkspaceGuard(workspace) as guard,
        pytest.raises(WorkspaceAccessError) as raised,
        guard.open_read("link.txt"),
    ):
        pass

    assert raised.value.reason is WorkspaceFailure.CONTAINMENT_VIOLATION


def test_guard_fails_closed_on_unknown_architecture(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceAccessError) as raised:
        WorkspaceGuard(tmp_path, architecture="mips64")

    assert raised.value.reason is WorkspaceFailure.UNSUPPORTED_PLATFORM


@pytest.mark.parametrize(
    ("error_number", "reason"),
    [
        (errno.ENOSYS, WorkspaceFailure.UNSUPPORTED_CONTAINMENT),
        (errno.EINVAL, WorkspaceFailure.UNSUPPORTED_CONTAINMENT),
        (errno.ELOOP, WorkspaceFailure.CONTAINMENT_VIOLATION),
        (errno.EXDEV, WorkspaceFailure.CONTAINMENT_VIOLATION),
    ],
)
def test_guard_normalizes_openat2_failures(
    tmp_path: Path, error_number: int, reason: WorkspaceFailure
) -> None:
    (tmp_path / "file.txt").write_text("data", encoding="utf-8")

    def fail_openat2(root_fd: int, path: str, flags: int) -> int:
        raise OSError(error_number, "injected")

    with (
        WorkspaceGuard(tmp_path, openat2=fail_openat2) as guard,
        pytest.raises(WorkspaceAccessError) as raised,
        guard.open_read("file.txt"),
    ):
        pass

    assert raised.value.reason is reason


def test_guard_fails_closed_if_pinned_root_identity_changes(tmp_path: Path) -> None:
    actual = os.stat(tmp_path)
    calls = 0

    def changed_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls <= 2:
            return os.fstat(descriptor)
        return SimpleNamespace(st_dev=actual.st_dev, st_ino=actual.st_ino + 1)

    with (
        WorkspaceGuard(tmp_path, fstat=changed_fstat) as guard,
        pytest.raises(WorkspaceAccessError) as raised,
        guard.open_read("file.txt"),
    ):
        pass

    assert raised.value.reason is WorkspaceFailure.ROOT_CHANGED


def test_close_is_idempotent_and_closed_guard_rejects_access(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)

    guard.close()
    guard.close()

    assert guard.closed is True
    with pytest.raises(WorkspaceAccessError) as raised, guard.open_read("file.txt"):
        pass
    assert raised.value.reason is WorkspaceFailure.CLOSED


def test_constructor_cleanup_failure_does_not_mask_pin_failure(tmp_path: Path) -> None:
    pinned_descriptors: list[int] = []
    real_close = os.close

    def fail_fstat(descriptor: int) -> os.stat_result:
        pinned_descriptors.append(descriptor)
        raise OSError(errno.EIO, "primary fstat failure")

    def release_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError(errno.ENOSPC, "cleanup close failure")

    with pytest.raises(WorkspaceAccessError) as raised:
        WorkspaceGuard(tmp_path, fstat=fail_fstat, close=release_then_fail)

    assert raised.value.reason is WorkspaceFailure.IO_ERROR
    assert isinstance(raised.value.__cause__, OSError)
    assert raised.value.__cause__.errno == errno.EIO
    with pytest.raises(OSError) as closed:
        os.fstat(pinned_descriptors[0])
    assert closed.value.errno == errno.EBADF


def test_root_close_failure_consumes_descriptor_ownership_once(tmp_path: Path) -> None:
    replacement_path = tmp_path / "replacement.txt"
    replacement_path.write_text("replacement", encoding="utf-8")
    released_descriptors: list[int] = []
    real_close = os.close

    def release_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        released_descriptors.append(descriptor)
        raise OSError(errno.EIO, "injected close failure")

    guard = WorkspaceGuard(tmp_path, close=release_then_fail)
    replacement: int | None = None
    try:
        with pytest.raises(WorkspaceAccessError) as raised:
            guard.close()
        assert raised.value.reason is WorkspaceFailure.IO_ERROR

        released_descriptor = released_descriptors[0]
        replacement = os.open(replacement_path, os.O_RDONLY)
        if replacement != released_descriptor:
            os.dup2(replacement, released_descriptor)
            real_close(replacement)
            replacement = released_descriptor

        guard.close()
        os.fstat(replacement)
    finally:
        if replacement is not None:
            try:
                real_close(replacement)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise


def test_guard_reads_only_from_descriptor_returned_by_openat2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file.txt"
    path.write_text("descriptor-only", encoding="utf-8")
    descriptor = os.open(path, os.O_RDONLY)

    def injected_openat2(root_fd: int, relative_path: str, flags: int) -> int:
        assert relative_path == "file.txt"
        return os.dup(descriptor)

    with WorkspaceGuard(tmp_path, openat2=injected_openat2) as guard:
        monkeypatch.setattr("shadow_code.policy.workspace.os.open", lambda *args: pytest.fail())
        with guard.open_read("file.txt") as opened:
            assert os.read(opened, 64) == b"descriptor-only"

    os.close(descriptor)


def test_invalid_returned_descriptor_is_a_typed_io_error(tmp_path: Path) -> None:
    with (
        WorkspaceGuard(tmp_path, openat2=lambda *_: -1) as guard,
        pytest.raises(WorkspaceAccessError) as raised,
        guard.open_read("file.txt"),
    ):
        pass

    assert raised.value.reason is WorkspaceFailure.IO_ERROR


def test_cleanup_error_is_typed_without_masking_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file.txt"
    path.write_text("data", encoding="utf-8")
    descriptors = [os.open(path, os.O_RDONLY), os.open(path, os.O_RDONLY)]
    returned = iter(descriptors)
    real_close = os.close

    def fail_result_close(descriptor: int) -> None:
        if descriptor in descriptors:
            raise OSError(errno.EIO, "injected close failure")
        real_close(descriptor)

    monkeypatch.setattr("shadow_code.policy.workspace.os.close", fail_result_close)
    with WorkspaceGuard(tmp_path, openat2=lambda *_: next(returned)) as guard:
        with pytest.raises(WorkspaceAccessError) as cleanup, guard.open_read("file.txt"):
            pass
        with pytest.raises(WorkspaceAccessError) as primary, guard.open_read("file.txt"):
            raise WorkspaceAccessError(WorkspaceFailure.INVALID_PATH, "primary")

    for descriptor in descriptors:
        real_close(descriptor)
    assert cleanup.value.reason is WorkspaceFailure.IO_ERROR
    assert primary.value.reason is WorkspaceFailure.INVALID_PATH
