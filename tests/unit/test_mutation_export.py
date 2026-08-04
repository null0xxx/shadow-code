"""Strict-mode patch export: full diffs, isolated export dir, never applied."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from shadow_code.domain.policy import WorkspaceAccessError
from shadow_code.domain.tools import ToolCall
from shadow_code.mutation import (
    MutationError,
    build_edit_plan,
    build_write_plan,
    export_patch,
    render_patch,
)
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.tools.catalog import (
    EDIT_FILE_SPEC,
    WRITE_FILE_SPEC,
    EditFileArgs,
    WorkspaceContext,
    WriteFileArgs,
)

_EXPORTS_DIR = ".shadow-code-exports"


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[tuple[Path, WorkspaceGuard]]:
    with WorkspaceGuard(tmp_path) as guard:
        yield tmp_path, guard


def _write_args(path: str, content: str) -> WriteFileArgs:
    return WriteFileArgs(file_path=path, content=content)


def _exports(root: Path) -> list[Path]:
    exports_dir = root / _EXPORTS_DIR
    if not exports_dir.is_dir():
        return []
    return sorted(exports_dir.iterdir())


def _apply_patch(old_lines: list[str], patch_text: str) -> list[str]:
    """Minimal unified-diff applier for structural patch verification.

    Context and deletion lines must match the old content exactly; the
    result is the old content with the hunks applied. This avoids a patch
    binary dependency while still proving the exported diff is valid.
    """
    result = list(old_lines)
    lines = patch_text.splitlines(keepends=True)
    offset = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("@@"):
            index += 1
            continue
        old_start = int(line.split()[1].split(",")[0][1:])
        position = max(old_start - 1, 0) + offset
        index += 1
        while index < len(lines) and not lines[index].startswith(("@@", "diff ")):
            body = lines[index]
            tag, text = body[0], body[1:]
            if tag in {" ", "-"}:
                assert result[position] == text, f"patch context mismatch: {body!r}"
            if tag == " ":
                position += 1
            elif tag == "-":
                del result[position]
                offset -= 1
            elif tag == "+":
                result.insert(position, text)
                position += 1
                offset += 1
            index += 1
    return result


def test_render_patch_new_file_diffs_from_dev_null(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    _, guard = workspace
    new_bytes = b"alpha\nbeta\n"
    plan = build_write_plan(guard, _write_args("new.txt", new_bytes.decode()))

    patch_text = render_patch(plan, None, new_bytes)

    lines = patch_text.splitlines()
    assert lines[0] == "--- /dev/null"
    assert lines[1] == "+++ b/new.txt"
    assert "+alpha" in lines
    assert "+beta" in lines
    # The patch reconstructs the planned content exactly (structural
    # check, no patch binary needed).
    assert _apply_patch([], patch_text) == ["alpha\n", "beta\n"]


def test_render_patch_overwrite_is_a_full_untruncated_diff(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    original = "".join(f"line {number} old\n" for number in range(200))
    (root / "big.txt").write_text(original, encoding="utf-8")
    updated = original.replace("old", "new")
    plan = build_write_plan(guard, _write_args("big.txt", updated))
    assert "diff truncated" in plan.preview  # previews stay bounded

    patch_text = render_patch(plan, original.encode(), updated.encode())

    assert "diff truncated" not in patch_text
    lines = patch_text.splitlines()
    assert lines[0] == "--- a/big.txt"
    assert lines[1] == "+++ b/big.txt"
    assert _apply_patch(original.splitlines(keepends=True), patch_text) == (
        updated.splitlines(keepends=True)
    )
    # No timestamps: identical plans render identical patches.
    assert patch_text == render_patch(plan, original.encode(), updated.encode())


def test_render_patch_rejects_content_that_does_not_match_plan(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    _, guard = workspace
    plan = build_write_plan(guard, _write_args("plan.txt", "approved\n"))

    with pytest.raises(MutationError) as caught:
        render_patch(plan, None, b"tampered\n")

    assert caught.value.code == "invalid_plan"


def test_export_patch_creates_dir_and_returns_relative_path(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    plan = build_write_plan(guard, _write_args("made.txt", "content\n"))
    patch_text = render_patch(plan, None, b"content\n")

    relative = export_patch(guard, plan, patch_text)

    assert relative.startswith(f"{_EXPORTS_DIR}/")
    assert relative.endswith("-write-made.txt.patch")
    exported = root / relative
    assert exported.read_text(encoding="utf-8") == patch_text
    # The workspace target is never touched by an export.
    assert not (root / "made.txt").exists()


def test_export_patch_names_are_unique_within_one_second(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    plan = build_write_plan(guard, _write_args("same.txt", "content\n"))
    patch_text = render_patch(plan, None, b"content\n")

    first = export_patch(guard, plan, patch_text)
    second = export_patch(guard, plan, patch_text)

    assert first != second
    assert len(_exports(root)) == 2


def test_export_patch_fails_closed_on_symlinked_exports_dir(
    workspace: tuple[Path, WorkspaceGuard], tmp_path_factory: pytest.TempPathFactory
) -> None:
    root, guard = workspace
    outside = tmp_path_factory.mktemp("outside")
    (root / _EXPORTS_DIR).symlink_to(outside, target_is_directory=True)
    plan = build_write_plan(guard, _write_args("victim.txt", "content\n"))
    patch_text = render_patch(plan, None, b"content\n")

    with pytest.raises(WorkspaceAccessError):
        export_patch(guard, plan, patch_text)

    assert list(outside.iterdir()) == []
    assert not (root / "victim.txt").exists()


def test_export_mode_write_to_new_file_exports_patch_only(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    call = ToolCall(call_id="w1", name="write_file", arguments={"file_path": "fresh.txt"})
    context = WorkspaceContext(guard=guard, mutation_mode="export")

    result = WRITE_FILE_SPEC.handler(call, _write_args("fresh.txt", "exported only\n"), context)

    assert result.success
    assert "status: exported" in (result.output or "")
    assert "status: executed" not in (result.output or "")
    assert f"patch: {_EXPORTS_DIR}/" in (result.output or "")
    assert "NOT modified" in (result.output or "")
    assert not (root / "fresh.txt").exists()
    exports = _exports(root)
    assert len(exports) == 1
    patch_text = exports[0].read_text(encoding="utf-8")
    assert "+++ b/fresh.txt" in patch_text
    assert "+exported only" in patch_text


def test_export_mode_overwrite_leaves_target_unchanged_with_full_diff(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    target = root / "keep.txt"
    target.write_text("old line\n", encoding="utf-8")
    call = ToolCall(call_id="w2", name="write_file", arguments={"file_path": "keep.txt"})
    context = WorkspaceContext(guard=guard, mutation_mode="export")

    result = WRITE_FILE_SPEC.handler(call, _write_args("keep.txt", "new line\n"), context)

    assert result.success
    assert "status: exported" in (result.output or "")
    assert target.read_text(encoding="utf-8") == "old line\n"
    exports = _exports(root)
    assert len(exports) == 1
    patch_text = exports[0].read_text(encoding="utf-8")
    lines = patch_text.splitlines()
    assert lines[0] == "--- a/keep.txt"
    assert lines[1] == "+++ b/keep.txt"
    assert _apply_patch(["old line\n"], patch_text) == ["new line\n"]


def test_export_mode_edit_exports_patch_without_touching_target(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    target = root / "code.py"
    target.write_text("value = 1\n", encoding="utf-8")
    call = ToolCall(call_id="e1", name="edit_file", arguments={"file_path": "code.py"})
    context = WorkspaceContext(guard=guard, mutation_mode="export")

    result = EDIT_FILE_SPEC.handler(
        call, EditFileArgs(file_path="code.py", old_text="1", new_text="2"), context
    )

    assert result.success
    assert "status: exported" in (result.output or "")
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    exports = _exports(root)
    assert len(exports) == 1
    assert "-value = 1" in exports[0].read_text(encoding="utf-8")
    assert "+value = 2" in exports[0].read_text(encoding="utf-8")


def test_apply_mode_is_unchanged_when_mutation_mode_defaults(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    call = ToolCall(call_id="w3", name="write_file", arguments={"file_path": "direct.txt"})
    context = WorkspaceContext(guard=guard)

    result = WRITE_FILE_SPEC.handler(call, _write_args("direct.txt", "applied\n"), context)

    assert result.success
    assert "status: executed" in (result.output or "")
    assert (root / "direct.txt").read_text(encoding="utf-8") == "applied\n"
    assert _exports(root) == []
    # No stray temp or export artifacts on the apply path.
    assert list(root.glob("**/.shadow-tmp-*")) == []


def test_edit_plan_renders_export_patch_against_exact_old_bytes(
    workspace: tuple[Path, WorkspaceGuard],
) -> None:
    root, guard = workspace
    (root / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    plan = build_edit_plan(
        guard, EditFileArgs(file_path="notes.txt", old_text="two", new_text="TWO")
    )
    old_bytes = (root / "notes.txt").read_bytes()
    new_bytes = old_bytes.replace(b"two", b"TWO", 1)

    patch_text = render_patch(plan, old_bytes, new_bytes)

    old_lines = ["one\n", "two\n", "three\n"]
    assert _apply_patch(old_lines, patch_text) == ["one\n", "TWO\n", "three\n"]
