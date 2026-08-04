"""Deterministic interleaving harness for mutation verification boundaries.

Each test injects a workspace change (content, mode, identity, symlink,
deletion, parent replacement, or new-file race) immediately before a defined
verification boundary of ``apply_mutation`` — either between plan build and
apply, or inside the pre-commit re-snapshot via a wrapped ``snapshot_file`` —
and proves the observed change aborts the mutation with the original intact
and no temp files left behind.

These tests prove detection of observed races at the defined boundaries.
They do NOT claim atomic compare-and-swap against a hostile swap in the
final window between the pre-commit check and the rename; the declared
threat model is a stable workspace with cooperating writers.
"""

import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from shadow_code import mutation
from shadow_code.domain.policy import WorkspaceAccessError, WorkspaceFailure
from shadow_code.mutation import (
    MutationError,
    apply_mutation,
    build_write_plan,
)
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.tools.catalog import WriteFileArgs


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[tuple[Path, WorkspaceGuard]]:
    with WorkspaceGuard(tmp_path) as guard:
        yield tmp_path, guard


def _write_args(path: str, content: str) -> WriteFileArgs:
    return WriteFileArgs(file_path=path, content=content)


def _temp_files(root: Path) -> list[Path]:
    return list(root.glob("**/.shadow-tmp-*"))


def _inject_before_pre_commit_check(tamper) -> object:
    """Build a snapshot_file wrapper that runs `tamper` at the boundary.

    The first snapshot taken inside apply_mutation is the immediate
    pre-commit re-validation; running the tamper there places the change
    after the temp write but before the rename. The caller installs the
    wrapper via monkeypatch so the real function is restored afterwards.
    """
    real_snapshot = mutation.snapshot_file
    fired = {"done": False}

    def wrapped(guard: WorkspaceGuard, path: str) -> mutation.FileSnapshot:
        if not fired["done"]:
            fired["done"] = True
            tamper()
        return real_snapshot(guard, path)

    return wrapped


def test_content_change_before_pre_commit_boundary_aborts(
    workspace: tuple[Path, WorkspaceGuard], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, guard = workspace
    target = root / "victim.txt"
    target.write_text("planned\n", encoding="utf-8")
    args = _write_args("victim.txt", "approved\n")
    plan = build_write_plan(guard, args)

    def tamper() -> None:
        target.write_text("raced\n", encoding="utf-8")

    monkeypatch.setattr(mutation, "snapshot_file", _inject_before_pre_commit_check(tamper))
    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "workspace_drift"
    assert target.read_text(encoding="utf-8") == "raced\n"
    assert _temp_files(root) == []


def test_mode_change_before_apply_aborts(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    target = root / "mode.txt"
    target.write_text("same\n", encoding="utf-8")
    args = _write_args("mode.txt", "changed\n")
    plan = build_write_plan(guard, args)

    os.chmod(target, 0o600)

    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "workspace_drift"
    assert target.read_text(encoding="utf-8") == "same\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert _temp_files(root) == []


def test_target_replacement_with_new_inode_aborts(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    target = root / "swapped.txt"
    target.write_text("identical\n", encoding="utf-8")
    os.chmod(target, 0o644)
    planned_inode = target.stat().st_ino
    args = _write_args("swapped.txt", "approved\n")
    plan = build_write_plan(guard, args)

    # Replace the target with a new inode holding identical bytes and mode:
    # the identity half of the snapshot must still catch the swap.
    target.unlink()
    target.write_text("identical\n", encoding="utf-8")
    os.chmod(target, 0o644)
    assert target.stat().st_ino != planned_inode

    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "workspace_drift"
    assert target.read_text(encoding="utf-8") == "identical\n"
    assert _temp_files(root) == []


def test_symlink_swap_before_apply_aborts_and_sentinel_is_untouched(
    workspace: tuple[Path, WorkspaceGuard], tmp_path_factory: pytest.TempPathFactory
) -> None:
    root, guard = workspace
    target = root / "link.txt"
    target.write_text("planned\n", encoding="utf-8")
    sentinel = tmp_path_factory.mktemp("outside") / "sentinel.txt"
    sentinel.write_text("outside\n", encoding="utf-8")
    args = _write_args("link.txt", "approved\n")
    plan = build_write_plan(guard, args)

    # Swap the target for a symlink escaping the workspace between plan and
    # apply; containment must refuse the re-snapshot.
    target.unlink()
    target.symlink_to(sentinel)

    with pytest.raises(WorkspaceAccessError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.reason is WorkspaceFailure.CONTAINMENT_VIOLATION
    assert sentinel.read_text(encoding="utf-8") == "outside\n"
    assert target.is_symlink()
    assert _temp_files(root) == []


def test_target_deletion_before_apply_aborts(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    target = root / "gone.txt"
    target.write_text("planned\n", encoding="utf-8")
    args = _write_args("gone.txt", "approved\n")
    plan = build_write_plan(guard, args)
    assert plan.before.exists is True

    target.unlink()

    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "workspace_drift"
    assert not target.exists()
    assert _temp_files(root) == []


def test_parent_directory_replacement_aborts_and_never_lands(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    (root / "sub").mkdir()
    (root / "sub" / "file.txt").write_text("planned\n", encoding="utf-8")
    args = _write_args("sub/file.txt", "approved\n")
    plan = build_write_plan(guard, args)

    # Replace the parent directory between plan and apply: rename it away
    # and recreate it with different content at the planned path.
    (root / "sub").rename(root / "sub-old")
    (root / "sub").mkdir()
    (root / "sub" / "file.txt").write_text("recreated\n", encoding="utf-8")

    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "workspace_drift"
    # The approved content never lands anywhere: not in the recreated tree,
    # not in the renamed-away original.
    assert (root / "sub" / "file.txt").read_text(encoding="utf-8") == "recreated\n"
    assert (root / "sub-old" / "file.txt").read_text(encoding="utf-8") == "planned\n"
    assert _temp_files(root) == []


def test_new_file_race_at_pre_commit_boundary_aborts(
    workspace: tuple[Path, WorkspaceGuard], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, guard = workspace
    target = root / "race.txt"
    args = _write_args("race.txt", "mine\n")
    plan = build_write_plan(guard, args)
    assert plan.before.exists is False

    def tamper() -> None:
        target.write_text("someone else\n", encoding="utf-8")

    monkeypatch.setattr(mutation, "snapshot_file", _inject_before_pre_commit_check(tamper))
    with pytest.raises(MutationError) as caught:
        apply_mutation(guard, plan, args.content.encode("utf-8"))

    assert caught.value.code == "workspace_drift"
    assert target.read_text(encoding="utf-8") == "someone else\n"
    assert _temp_files(root) == []
