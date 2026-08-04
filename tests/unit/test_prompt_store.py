"""Tests for the content-addressed prompt snapshot store (WU-04)."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from shadow_code.prompt_compiler import PromptCompileError, PromptManager, compile_prompt
from shadow_code.prompt_store import PromptStore, PromptStoreError
from shadow_code.tools.catalog import BASH_SPEC, EDIT_FILE_SPEC, READ_FILE_SPEC, WRITE_FILE_SPEC
from shadow_code.tools.registry import ToolRegistry


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry((READ_FILE_SPEC, WRITE_FILE_SPEC, EDIT_FILE_SPEC, BASH_SPEC))


@pytest.fixture()
def store(tmp_path: Path) -> PromptStore:
    return PromptStore(tmp_path / "prompts")


def _compile(registry: ToolRegistry, tmp_path: Path, name: str, **kwargs: object):
    return compile_prompt(
        user_path=tmp_path / f"{name}-user.md",
        workspace_path=tmp_path / f"{name}-ws.md",
        registry=registry,
        **kwargs,  # type: ignore[arg-type]
    )


def test_save_load_roundtrip_reproduces_bytes_offline(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    user = tmp_path / "rt-user.md"
    workspace = tmp_path / "rt-ws.md"
    user.write_bytes(b"user rules\r\nwith CRLF\r\n")
    workspace.write_text("workspace rules\n", encoding="utf-8")
    compiled = _compile(registry, tmp_path, "rt")
    digest = store.save(compiled)

    # Delete the sources: the snapshot must stand on its own bytes.
    user.unlink()
    workspace.unlink()

    loaded = store.load(digest)
    assert loaded.digest == compiled.digest
    assert loaded.compiled_text == compiled.compiled_text
    assert loaded.sources == compiled.sources
    assert loaded.normalized_sources() == compiled.normalized_sources()
    assert loaded.registry_digest == compiled.registry_digest
    assert loaded.created_utc == compiled.created_utc


def test_save_is_content_addressed_and_idempotent(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    compiled = _compile(registry, tmp_path, "idem")
    first = store.save(compiled)
    snapshot_files = sorted((store.root / first).iterdir())
    mtimes = {path.name: path.stat().st_mtime_ns for path in snapshot_files}

    second = store.save(_compile(registry, tmp_path, "idem"))

    assert first == second == compiled.digest
    assert {path.name: path.stat().st_mtime_ns for path in snapshot_files} == mtimes


def test_load_accepts_unique_prefix(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    compiled = _compile(registry, tmp_path, "prefix")
    store.save(compiled)

    assert store.load(compiled.digest[:8]).digest == compiled.digest


def test_load_rejects_unknown_and_ambiguous_prefix(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    compiled = _compile(registry, tmp_path, "ambig")
    digest = store.save(compiled)

    with pytest.raises(PromptStoreError) as not_found:
        store.load("deadbeef")
    assert not_found.value.code == "snapshot_not_found"

    # A second directory sharing the first 8 hex chars makes the prefix ambiguous.
    twin = store.root / (digest[:8] + "f" * 56)
    shutil.copytree(store.root / digest, twin)
    with pytest.raises(PromptStoreError) as ambiguous:
        store.load(digest[:8])
    assert ambiguous.value.code == "ambiguous_digest"


def test_load_detects_corrupt_blob(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    compiled = _compile(registry, tmp_path, "corrupt")
    digest = store.save(compiled)

    blob = store.root / digest / "compiled.txt"
    data = bytearray(blob.read_bytes())
    data[0] ^= 0xFF
    blob.write_bytes(bytes(data))

    with pytest.raises(PromptStoreError) as excinfo:
        store.load(digest)
    assert excinfo.value.code == "corrupt_snapshot"


def test_load_detects_corrupt_source_metadata(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    compiled = _compile(registry, tmp_path, "meta")
    digest = store.save(compiled)

    metadata_path = store.root / digest / "sources.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["sources"][0]["data_b64"] = "!!!not-base64!!!"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PromptStoreError) as excinfo:
        store.load(digest)
    assert excinfo.value.code == "corrupt_snapshot"


def test_history_is_newest_first(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    (tmp_path / "hist-user.md").write_text("v1\n", encoding="utf-8")
    older = _compile(registry, tmp_path, "hist", now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    store.save(older)
    (tmp_path / "hist-user.md").write_text("v2\n", encoding="utf-8")
    newer = _compile(registry, tmp_path, "hist", now=datetime(2026, 1, 2, tzinfo=timezone.utc))
    store.save(newer)

    history = store.history()

    assert [snapshot.digest for snapshot in history] == [newer.digest, older.digest]


def test_history_skips_corrupt_snapshots(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    good = _compile(registry, tmp_path, "good")
    store.save(good)
    (tmp_path / "good-user.md").write_text("changed\n", encoding="utf-8")
    bad = _compile(registry, tmp_path, "good")
    store.save(bad)
    (store.root / bad.digest / "compiled.txt").write_bytes(b"tampered")

    assert [snapshot.digest for snapshot in store.history()] == [good.digest]
    with pytest.raises(PromptStoreError):
        store.load(bad.digest)


def test_set_active_switches_atomically_and_reports_previous(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    first = _compile(registry, tmp_path, "sw")
    store.save(first)
    assert store.get_active() is None

    assert store.set_active(first.digest) is None
    assert store.get_active() == first.digest

    (tmp_path / "sw-user.md").write_text("v2\n", encoding="utf-8")
    second = _compile(registry, tmp_path, "sw")
    store.save(second)
    assert store.set_active(second.digest[:12]) == first.digest
    assert store.get_active() == second.digest


def test_set_active_failure_leaves_previous_active(
    registry: ToolRegistry, store: PromptStore, tmp_path: Path
) -> None:
    good = _compile(registry, tmp_path, "fail")
    store.save(good)
    store.set_active(good.digest)

    (tmp_path / "fail-user.md").write_text("v2\n", encoding="utf-8")
    bad = _compile(registry, tmp_path, "fail")
    store.save(bad)
    (store.root / bad.digest / "compiled.txt").write_bytes(b"tampered")

    with pytest.raises(PromptStoreError):
        store.set_active(bad.digest)
    assert store.get_active() == good.digest

    with pytest.raises(PromptStoreError):
        store.set_active("deadbeef")
    assert store.get_active() == good.digest


def _bootstrap(registry: ToolRegistry, tmp_path: Path) -> PromptManager:
    manager, _ = PromptManager.bootstrap(
        registry=registry,
        store=PromptStore(tmp_path / "prompts"),
        user_path=tmp_path / "user.md",
        workspace_path=tmp_path / "ws.md",
    )
    return manager


def test_manager_rollback_restores_exact_snapshot(registry: ToolRegistry, tmp_path: Path) -> None:
    manager = _bootstrap(registry, tmp_path)
    original = manager.active

    (tmp_path / "user.md").write_text("new rules\n", encoding="utf-8")
    updated, _ = manager.reload()
    assert manager.active.digest == updated.digest != original.digest

    target, previous = manager.rollback(original.digest[:8])

    assert target.digest == original.digest
    assert previous == updated.digest
    assert manager.active.digest == original.digest
    assert manager.store.get_active() == original.digest


def test_manager_rollback_failure_keeps_active(registry: ToolRegistry, tmp_path: Path) -> None:
    manager = _bootstrap(registry, tmp_path)
    active = manager.active.digest

    with pytest.raises(PromptStoreError) as excinfo:
        manager.rollback("deadbeef")
    assert excinfo.value.code == "snapshot_not_found"
    assert manager.active.digest == active
    assert manager.store.get_active() == active


def test_manager_rollback_rejects_foreign_registry_snapshot(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    manager = _bootstrap(registry, tmp_path)
    active = manager.active.digest

    foreign = compile_prompt(
        user_path=None,
        workspace_path=None,
        registry=ToolRegistry((READ_FILE_SPEC,)),
    )
    manager.store.save(foreign)

    with pytest.raises(PromptStoreError) as excinfo:
        manager.rollback(foreign.digest)
    assert excinfo.value.code == "incompatible_snapshot"
    assert manager.active.digest == active
    assert manager.store.get_active() == active


def test_manager_watch_picks_up_overlay_edit(registry: ToolRegistry, tmp_path: Path) -> None:
    manager = _bootstrap(registry, tmp_path)
    original = manager.active.digest

    assert manager.watch() is None  # nothing changed on disk

    (tmp_path / "user.md").write_text("edited between turns\n", encoding="utf-8")
    watched = manager.watch()

    assert watched is not None
    compiled, previous = watched
    assert manager.active.digest == compiled.digest != original
    assert previous == original
    assert "edited between turns" in manager.active.compiled_text
    assert manager.store.get_active() == compiled.digest


def test_manager_watch_failure_keeps_previous_active(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    manager = _bootstrap(registry, tmp_path)
    original = manager.active.digest

    (tmp_path / "user.md").write_bytes(b"\xff\xfe not utf-8")

    with pytest.raises(PromptCompileError):
        manager.watch()
    assert manager.active.digest == original
    assert manager.store.get_active() == original
