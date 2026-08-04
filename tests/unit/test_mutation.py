"""Mutation planning and apply: snapshots, drift aborts, lock, cleanup."""

import fcntl
import hashlib
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from shadow_code import mutation
from shadow_code.domain.policy import WorkspaceAccessError
from shadow_code.mutation import (
    MutationError,
    MutationPlan,
    apply_mutation,
    build_edit_plan,
    build_write_plan,
    snapshot_file,
)
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.tools.catalog import EditFileArgs, WriteFileArgs


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[tuple[Path, WorkspaceGuard]]:
    with WorkspaceGuard(tmp_path) as guard:
        yield tmp_path, guard


def _write_args(path: str, content: str) -> WriteFileArgs:
    return WriteFileArgs(file_path=path, content=content)


def _edit_args(path: str, old: str, new: str) -> EditFileArgs:
    return EditFileArgs(file_path=path, old_text=old, new_text=new)


def _temp_files(root: Path) -> list[Path]:
    return list(root.glob("**/.shadow-tmp-*"))


def test_snapshot_missing_file_reports_absent(workspace: tuple[Path, WorkspaceGuard]) -> None:
    _, guard = workspace
    snapshot = snapshot_file(guard, "missing.txt")

    assert snapshot.exists is False
    assert snapshot.device is None
    assert snapshot.inode is None
    assert snapshot.mode is None
    assert snapshot.size is None
    assert snapshot.sha256 is None


def test_snapshot_existing_file_records_identity_and_digest(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    target = root / "note.txt"
    target.write_bytes(b"hello\n")
    os.chmod(target, 0o640)

    snapshot = snapshot_file(guard, "note.txt")
    direct = target.stat()

    assert snapshot.exists is True
    assert snapshot.device == direct.st_dev
    assert snapshot.inode == direct.st_ino
    assert snapshot.mode == stat.S_IMODE(direct.st_mode)
    assert snapshot.size == 6
    assert snapshot.sha256 == hashlib.sha256(b"hello\n").hexdigest()


def test_open_dir_resolves_root_and_nested_directories(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    (root / "sub").mkdir()

    for relative in ("", ".", "sub"):
        with guard.open_dir(relative) as dir_fd:
            assert stat.S_ISDIR(os.fstat(dir_fd).st_mode)


def test_open_dir_fails_closed_on_files_and_bad_paths(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    (root / "file.txt").write_text("x", encoding="utf-8")

    with pytest.raises(WorkspaceAccessError), guard.open_dir("file.txt"):
        pass
    with pytest.raises(WorkspaceAccessError), guard.open_dir("../outside"):
        pass
    with pytest.raises(WorkspaceAccessError), guard.open_dir("missing-dir"):
        pass


def test_write_plan_is_pure_and_marks_new_file(workspace: tuple[Path, WorkspaceGuard]) -> None:
    root, guard = workspace
    plan = build_write_plan(guard, _write_args("new.txt", "fresh content\n"))

    assert plan.operation == "write"
    assert plan.relative_path == "new.txt"
    assert plan.before.exists is False
    assert plan.new_size == len(b"fresh content\n")
    assert plan.new_sha256 == hashlib.sha256(b"fresh content\n").hexdigest()
    assert "new file" in plan.preview
    # Planning never writes.
    assert not (root / "new.txt").exists()


def test_write_new_file_end_to_end(workspace: tuple[Path, WorkspaceGuard]) -> None:
    root, guard = workspace
    args = _write_args("created.txt", "alpha\nbeta\n")
    plan = build_write_plan(guard, args)

    receipt = apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert (root / "created.txt").read_bytes() == b"alpha\nbeta\n"
    assert receipt.bytes_written == len(b"alpha\nbeta\n")
    assert receipt.before.exists is False
    assert receipt.after.exists is True
    assert receipt.after.sha256 == plan.new_sha256
    assert receipt.after.mode == 0o644
    assert _temp_files(root) == []


def test_overwrite_preserves_mode_and_previews_diff(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    target = root / "keep.txt"
    target.write_text("old line\n", encoding="utf-8")
    os.chmod(target, 0o640)

    args = _write_args("keep.txt", "new line\n")
    plan = build_write_plan(guard, args)
    assert plan.before.exists is True
    assert "--- a/keep.txt" in plan.preview
    assert "-old line" in plan.preview
    assert "+new line" in plan.preview

    receipt = apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert (root / "keep.txt").read_text(encoding="utf-8") == "new line\n"
    assert stat.S_IMODE((root / "keep.txt").stat().st_mode) == 0o640
    assert receipt.before.exists is True
    assert receipt.after.sha256 == plan.new_sha256


def test_edit_exact_text_success(workspace: tuple[Path, WorkspaceGuard]) -> None:
    root, guard = workspace
    (root / "code.py").write_text("value = 1\n", encoding="utf-8")

    args = _edit_args("code.py", "value = 1", "value = 2")
    plan = build_edit_plan(guard, args)
    assert plan.operation == "edit"
    assert "-value = 1" in plan.preview
    assert "+value = 2" in plan.preview

    old_bytes = (root / "code.py").read_bytes()
    new_bytes = old_bytes.replace(b"value = 1", b"value = 2", 1)
    receipt = apply_mutation(guard, plan, new_bytes)

    assert (root / "code.py").read_text(encoding="utf-8") == "value = 2\n"
    assert receipt.bytes_written == len(new_bytes)


def test_edit_zero_matches_fails_closed(workspace: tuple[Path, WorkspaceGuard]) -> None:
    root, guard = workspace
    (root / "a.txt").write_text("nothing here\n", encoding="utf-8")

    with pytest.raises(MutationError) as caught:
        build_edit_plan(guard, _edit_args("a.txt", "absent", "replacement"))

    assert caught.value.code == "no_match"
    assert (root / "a.txt").read_text(encoding="utf-8") == "nothing here\n"


def test_edit_missing_target_fails_closed(workspace: tuple[Path, WorkspaceGuard]) -> None:
    root, guard = workspace

    with pytest.raises(MutationError) as caught:
        build_edit_plan(guard, _edit_args("missing.txt", "a", "b"))

    assert caught.value.code == "no_match"
    assert not (root / "missing.txt").exists()


def test_edit_duplicate_matches_fail_closed_without_writes(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    (root / "dup.txt").write_text("foo and foo\n", encoding="utf-8")

    with pytest.raises(MutationError) as caught:
        build_edit_plan(guard, _edit_args("dup.txt", "foo", "bar"))

    assert caught.value.code == "ambiguous_match"
    assert "2 times" in str(caught.value)
    assert (root / "dup.txt").read_text(encoding="utf-8") == "foo and foo\n"
    assert _temp_files(root) == []


def test_content_drift_aborts_with_original_intact(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    (root / "drift.txt").write_text("planned state\n", encoding="utf-8")
    args = _write_args("drift.txt", "mutation\n")
    plan = build_write_plan(guard, args)

    # Tamper between plan build and apply: the snapshot is now stale.
    (root / "drift.txt").write_text("tampered state\n", encoding="utf-8")

    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "workspace_drift"
    assert (root / "drift.txt").read_text(encoding="utf-8") == "tampered state\n"
    assert _temp_files(root) == []


def test_new_file_race_aborts(workspace: tuple[Path, WorkspaceGuard]) -> None:
    root, guard = workspace
    args = _write_args("race.txt", "mine\n")
    plan = build_write_plan(guard, args)

    # The target appears between plan build and apply.
    (root / "race.txt").write_text("someone else\n", encoding="utf-8")

    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "workspace_drift"
    assert (root / "race.txt").read_text(encoding="utf-8") == "someone else\n"
    assert _temp_files(root) == []


def test_mode_drift_aborts(workspace: tuple[Path, WorkspaceGuard]) -> None:
    root, guard = workspace
    (root / "mode.txt").write_text("same\n", encoding="utf-8")
    args = _write_args("mode.txt", "changed\n")
    plan = build_write_plan(guard, args)

    os.chmod(root / "mode.txt", 0o600)

    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "workspace_drift"
    assert (root / "mode.txt").read_text(encoding="utf-8") == "same\n"
    assert _temp_files(root) == []


def test_oserror_before_rename_cleans_temp_and_keeps_original(
    workspace: tuple[Path, WorkspaceGuard], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, guard = workspace
    (root / "victim.txt").write_text("original\n", encoding="utf-8")
    args = _write_args("victim.txt", "replacement\n")
    plan = build_write_plan(guard, args)

    def fail_rename(*args: object, **kwargs: object) -> None:
        raise OSError("deterministic rename failure")

    monkeypatch.setattr(mutation.os, "rename", fail_rename)
    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "io_error"
    assert (root / "victim.txt").read_text(encoding="utf-8") == "original\n"
    assert _temp_files(root) == []


def test_apply_rejects_content_that_does_not_match_plan(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    plan = build_write_plan(guard, _write_args("plan.txt", "approved\n"))

    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, b"tampered content\n")

    assert caught.value.code == "invalid_plan"
    assert not (root / "plan.txt").exists()


def test_readback_mismatch_is_typed(
    workspace: tuple[Path, WorkspaceGuard], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, guard = workspace
    args = _write_args("readback.txt", "content\n")
    plan = build_write_plan(guard, args)

    real_snapshot = mutation.snapshot_file
    calls = {"count": 0}

    def forged(guard: WorkspaceGuard, path: str) -> mutation.FileSnapshot:
        calls["count"] += 1
        snapshot = real_snapshot(guard, path)
        if calls["count"] == 2:  # pre-commit passes; readback is forged
            return mutation.FileSnapshot(
                exists=snapshot.exists,
                device=snapshot.device,
                inode=snapshot.inode,
                mode=snapshot.mode,
                size=snapshot.size,
                sha256="0" * 64,
            )
        return snapshot

    monkeypatch.setattr(mutation, "snapshot_file", forged)
    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "readback_mismatch"


def test_mutation_lock_excludes_cooperating_writers(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    lock_path = root / ".shadow-code.lock"

    with guard.mutation_lock():
        assert lock_path.exists()
        probe = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe)

    # After release, a cooperating writer can take the lock immediately.
    probe = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)
    finally:
        os.close(probe)


def test_sequential_mutations_serialize_on_the_lock(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    first = _write_args("seq.txt", "one\n")
    second = _write_args("seq.txt", "two\n")

    plan_one = build_write_plan(guard, first)
    apply_mutation(guard, plan_one, first.content.encode("utf-8"))
    plan_two = build_write_plan(guard, second)
    apply_mutation(guard, plan_two, second.content.encode("utf-8"))

    assert (root / "seq.txt").read_text(encoding="utf-8") == "two\n"
    assert _temp_files(root) == []


def test_unicode_and_georgian_content_round_trip(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    content = "გამარჯობა — მსოფლიო\nსხვა ხაზი — emoji ✓\n"
    args = _write_args("unicode.txt", content)
    plan = build_write_plan(guard, args)
    receipt = apply_mutation(guard, plan, content.encode("utf-8"))

    assert (root / "unicode.txt").read_text(encoding="utf-8") == content
    assert receipt.bytes_written == len(content.encode("utf-8"))

    edit = _edit_args("unicode.txt", "მსოფლიო", "დედამიწა")
    edit_plan = build_edit_plan(guard, edit)
    old_bytes = (root / "unicode.txt").read_bytes()
    new_bytes = old_bytes.replace("მსოფლიო".encode(), "დედამიწა".encode(), 1)
    apply_mutation(guard, edit_plan, new_bytes)

    assert (root / "unicode.txt").read_text(encoding="utf-8") == content.replace(
        "მსოფლიო", "დედამიწა"
    )


def test_long_diff_preview_is_bounded(workspace: tuple[Path, WorkspaceGuard]) -> None:
    root, guard = workspace
    original = "".join(f"line {number} old\n" for number in range(200))
    (root / "big.txt").write_text(original, encoding="utf-8")

    updated = original.replace("old", "new")
    plan = build_write_plan(guard, _write_args("big.txt", updated))

    assert "diff truncated" in plan.preview
    assert len(plan.preview.splitlines()) <= 41  # 40 retained + truncation marker


def test_apply_in_nested_parent_directory(workspace: tuple[Path, WorkspaceGuard]) -> None:
    root, guard = workspace
    (root / "nested").mkdir()
    args = _write_args("nested/deep.txt", "deep\n")
    plan = build_write_plan(guard, args)
    apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert (root / "nested" / "deep.txt").read_text(encoding="utf-8") == "deep\n"
    assert _temp_files(root) == []


def test_mutation_plan_is_frozen(workspace: tuple[Path, WorkspaceGuard]) -> None:
    _, guard = workspace
    plan: MutationPlan = build_write_plan(guard, _write_args("frozen.txt", "x"))

    with pytest.raises(AttributeError):
        plan.preview = "tampered"  # type: ignore[misc]
