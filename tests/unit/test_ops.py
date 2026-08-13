"""WU-12 operational polish: ops.py units plus /doctor, /backup, /restore wiring.

Ollama is always mocked; no test touches the network. Database backup and
restore run against real files in temporary directories so the round-trip,
dry-run, and tamper-fail-closed behavior are exercised end to end.
"""

import fcntl
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from shadow_code import ops


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class TestConfigResolution(unittest.TestCase):
    """Configuration precedence and corrupt-value fallback."""

    def test_env_override_is_labeled_with_its_source(self):
        settings = {s.name: s for s in ops.config_settings({"SHADOW_MODEL": "x:1b"})}
        self.assertEqual(settings["model"].source, "env:SHADOW_MODEL")
        self.assertEqual(settings["ollama_host"].source, "default")

    def test_blank_env_value_counts_as_default(self):
        settings = {s.name: s for s in ops.config_settings({"SHADOW_CTX": "  "})}
        self.assertEqual(settings["context_window"].source, "default")

    def test_corrupt_int_env_falls_back_and_records_issue(self):
        from shadow_code import config

        saved = list(config.CONFIG_ISSUES)
        config.CONFIG_ISSUES.clear()
        try:
            with patch.dict(os.environ, {"SHADOW_CTX": "not-a-number"}):
                value = config._env_int("SHADOW_CTX", 131072)
            self.assertEqual(value, 131072)
            self.assertEqual(len(config.CONFIG_ISSUES), 1)
            self.assertIn("not an integer", config.CONFIG_ISSUES[0])
            self.assertIn("SHADOW_CTX", config.CONFIG_ISSUES[0])
        finally:
            config.CONFIG_ISSUES.clear()
            config.CONFIG_ISSUES.extend(saved)

    def test_below_minimum_env_falls_back_and_records_issue(self):
        from shadow_code import config

        saved = list(config.CONFIG_ISSUES)
        config.CONFIG_ISSUES.clear()
        try:
            with patch.dict(os.environ, {"SHADOW_CTX": "0"}):
                value = config._env_int("SHADOW_CTX", 131072)
            self.assertEqual(value, 131072)
            self.assertIn("below 1", config.CONFIG_ISSUES[0])
        finally:
            config.CONFIG_ISSUES.clear()
            config.CONFIG_ISSUES.extend(saved)


class TestRedaction(unittest.TestCase):
    """Sentinel secrets never appear in diagnostic output."""

    def test_secret_looking_env_values_become_sentinels(self):
        env = {
            "MY_API_TOKEN": "tok-supersecret-value",
            "DB_PASSWORD": "hunter2-hunter2",
            "PLAIN_NAME": "not-a-secret-value",
            "SHADOW_MODEL": "short",  # allowlisted keys never count
        }
        sentinels = ops.collect_sentinels(env)
        self.assertIn("tok-supersecret-value", sentinels)
        self.assertIn("hunter2-hunter2", sentinels)
        self.assertNotIn("not-a-secret-value", sentinels)
        self.assertNotIn("short", sentinels)

    def test_shadow_redact_list_adds_explicit_sentinels(self):
        env = {"SHADOW_REDACT": "explicit-secret-1, tiny, explicit-secret-2"}
        sentinels = ops.collect_sentinels(env)
        self.assertIn("explicit-secret-1", sentinels)
        self.assertIn("explicit-secret-2", sentinels)
        self.assertNotIn("tiny", sentinels)  # below the minimum length

    def test_redact_replaces_longest_match_first(self):
        text = ops.redact("abc12345 and abc12345-longer", ("abc12345", "abc12345-longer"))
        self.assertEqual(text, "*** and ***")

    def test_doctor_output_never_contains_sentinels(self):
        sentinel = "sk-live-doctor-sentinel"
        facts = _doctor_facts_fixture(
            ollama_ok=False,
            ollama_message=f"Ollama error: auth failed for {sentinel}",
        )
        report = ops.doctor(facts, environ={"MY_API_TOKEN": sentinel}, run_cleanup=False)
        rendered = ops.render_doctor(report)
        self.assertNotIn(sentinel, rendered)
        self.assertIn("***", rendered)

    def test_backup_and_restore_receipts_never_contain_sentinels(self):
        sentinel = "sk-live-backup-sentinel"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write(root / f"{sentinel}-sessions.db", b"data")
            receipt = ops.backup_databases(
                sessions_path=str(source),
                events_path=None,
                dest_root=root / "backups",
                sentinels=(sentinel,),
            )
            rendered = ops.render_backup_receipt(receipt, (sentinel,))
            self.assertNotIn(sentinel, rendered)
            manifest = Path(receipt.manifest_path).read_text(encoding="utf-8")
            self.assertNotIn(sentinel, manifest)

            target = _write(root / "live.db", b"other")
            plan = ops.restore_databases(
                receipt.directory, sessions_path=str(target), events_path=None
            )
            rendered_plan = ops.render_restore_plan(plan, (sentinel,))
            self.assertNotIn(sentinel, rendered_plan)


class TestModelCapability(unittest.TestCase):
    """Unavailable Ollama, missing model, and tool-support diagnostics."""

    def test_reachable_model_has_no_next_step(self):
        self.assertEqual(ops.check_model_capability(True, "OK", model="m", base_url="http://x"), "")

    def test_unreachable_ollama_points_at_ollama_serve(self):
        step = ops.check_model_capability(
            False,
            "Cannot connect to Ollama at http://localhost:11434",
            model="m",
            base_url="http://localhost:11434",
        )
        self.assertIn("ollama serve", step)
        self.assertIn("http://localhost:11434", step)

    def test_missing_model_points_at_ollama_pull(self):
        step = ops.check_model_capability(
            False,
            "Model 'gemma4:2b' not found. Available: []",
            model="gemma4:2b",
            base_url="http://x",
        )
        self.assertIn("ollama pull gemma4:2b", step)

    def test_tool_incompatible_model_points_at_compat_path(self):
        step = ops.check_model_capability(
            False, "model does not support tool calls", model="m", base_url="http://x"
        )
        self.assertIn("SHADOW_LEGACY_MARKDOWN_TOOLS=1", step)

    def test_unknown_failure_is_generic_but_actionable(self):
        step = ops.check_model_capability(False, "weird", model="m", base_url="http://x")
        self.assertIn("weird", step)
        self.assertIn("ollama serve", step)


class TestCleanupStale(unittest.TestCase):
    """Post-crash temp orphan sweep; exports and live locks are respected."""

    def test_orphans_removed_exports_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / ".shadow-tmp-aaa", b"x")
            _write(root / "sub" / ".shadow-tmp-bbb", b"x")
            exported = _write(root / ".shadow-code-exports" / ".shadow-tmp-ccc", b"x")
            keep = _write(root / "normal.txt", b"x")

            result = ops.cleanup_stale(root)

            self.assertIsNone(result.skipped_reason)
            self.assertEqual(sorted(result.removed), [".shadow-tmp-aaa", "sub/.shadow-tmp-bbb"])
            self.assertFalse((root / ".shadow-tmp-aaa").exists())
            self.assertTrue(exported.exists())
            self.assertTrue(keep.exists())
            self.assertEqual(result.exports_count, 1)

    def test_symlink_orphans_are_never_unlinked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = _write(root / "real.txt", b"x")
            link = root / ".shadow-tmp-link"
            link.symlink_to(target)
            result = ops.cleanup_stale(root)
            self.assertEqual(result.removed, ())
            self.assertTrue(link.is_symlink())

    def test_live_mutation_lock_skips_the_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / ".shadow-tmp-orphan", b"x")
            lock = _write(root / ".shadow-code.lock", b"")
            fd = os.open(lock, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = ops.cleanup_stale(root)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            self.assertEqual(result.removed, ())
            self.assertEqual(result.skipped_reason, "a live instance holds the mutation lock")
            self.assertTrue(result.lock_present)
            self.assertTrue((root / ".shadow-tmp-orphan").exists())


class TestBackupRestore(unittest.TestCase):
    """Backup/restore round-trip, dry-run, and tamper fail-closed."""

    def _backup(self, root: Path) -> tuple[ops.BackupReceipt, Path, Path]:
        sessions = _write(root / "sessions.db", b"sessions-v1")
        events = _write(root / "events.db", b"events-v1")
        receipt = ops.backup_databases(
            sessions_path=str(sessions),
            events_path=str(events),
            dest_root=root / "backups",
        )
        return receipt, sessions, events

    def test_backup_writes_manifest_with_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, _, _ = self._backup(Path(tmp))
            manifest = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(len(manifest["entries"]), 2)
            sizes = {entry["label"]: entry["size"] for entry in manifest["entries"]}
            self.assertEqual(sizes, {"sessions": len(b"sessions-v1"), "events": len(b"events-v1")})
            for entry in manifest["entries"]:
                self.assertEqual(len(entry["sha256"]), 64)
            self.assertEqual(len(receipt.entries), 2)

    def test_backup_counts_prompt_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions = _write(root / "sessions.db", b"x")
            store = root / "prompts"
            (store / ("a" * 64)).mkdir(parents=True)
            (store / "not-a-digest").mkdir()
            receipt = ops.backup_databases(
                sessions_path=str(sessions),
                events_path=None,
                dest_root=root / "backups",
                prompt_store_dir=store,
            )
            self.assertEqual(receipt.prompt_snapshot_count, 1)

    def test_backup_without_databases_is_a_typed_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ops.OpsError) as caught:
                ops.backup_databases(sessions_path=None, events_path=None, dest_root=Path(tmp))
            self.assertEqual(caught.exception.code, "nothing_to_backup")

    def test_round_trip_restores_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, sessions, events = self._backup(Path(tmp))
            sessions.write_bytes(b"sessions-v2-corrupted")
            events.unlink()
            plan = ops.restore_databases(
                receipt.directory,
                sessions_path=str(sessions),
                events_path=str(events),
                apply=True,
            )
            self.assertTrue(plan.applied)
            self.assertEqual(sessions.read_bytes(), b"sessions-v1")
            self.assertEqual(events.read_bytes(), b"events-v1")

    def test_dry_run_reports_changes_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, sessions, _ = self._backup(Path(tmp))
            sessions.write_bytes(b"sessions-v2-corrupted")
            plan = ops.restore_databases(
                receipt.directory, sessions_path=str(sessions), events_path=None
            )
            self.assertFalse(plan.applied)
            changed = {a.label: a.would_change for a in plan.actions}
            self.assertEqual(changed, {"sessions": True})
            self.assertEqual(sessions.read_bytes(), b"sessions-v2-corrupted")

    def test_dry_run_marks_unchanged_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, sessions, _ = self._backup(Path(tmp))
            plan = ops.restore_databases(
                receipt.directory, sessions_path=str(sessions), events_path=None
            )
            self.assertFalse(any(a.would_change for a in plan.actions))

    def test_tampered_backup_fails_closed_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, sessions, _ = self._backup(Path(tmp))
            sessions.write_bytes(b"live-bytes")
            backup_file = Path(receipt.directory) / "sessions.db"
            backup_file.write_bytes(b"tampered")
            with self.assertRaises(ops.OpsError) as caught:
                ops.restore_databases(
                    receipt.directory,
                    sessions_path=str(sessions),
                    events_path=None,
                    apply=True,
                )
            self.assertEqual(caught.exception.code, "corrupt_backup")
            self.assertEqual(sessions.read_bytes(), b"live-bytes")

    def test_missing_manifest_is_a_typed_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ops.OpsError) as caught:
                ops.restore_databases(Path(tmp) / "nope")
            self.assertEqual(caught.exception.code, "backup_not_found")


def _doctor_facts_fixture(
    workspace: str | None = None,
    *,
    ollama_ok: bool = True,
    ollama_message: str = "OK",
    event_store=None,
    events_db_path: str | None = None,
) -> ops.DoctorFacts:
    root = workspace or tempfile.mkdtemp()
    return ops.DoctorFacts(
        workspace_root=root,
        workspace_device=1,
        workspace_inode=2,
        containment="openat2",
        granted=("filesystem.read", "filesystem.write", "process.execute"),
        withheld=(("network.access", "not supported by this build"),),
        sandbox_label="unconfined",
        mutation_mode="apply",
        model_name="test-model:latest",
        ollama_ok=ollama_ok,
        ollama_message=ollama_message,
        prompt_digest="d" * 64,
        prompt_layer_count=2,
        prompt_store_path=str(Path(root) / "prompts"),
        events_db_path=events_db_path,
        legacy_db_path=None,
        event_store=event_store,
    )


class TestDoctor(unittest.TestCase):
    """The diagnostic report: reachable, unreachable, missing model, redaction."""

    def test_healthy_runtime_reports_ok_with_tool_note(self):
        facts = _doctor_facts_fixture()
        report = ops.doctor(facts, environ={}, run_cleanup=False)
        self.assertTrue(report.ok)
        rendered = ops.render_doctor(report)
        self.assertIn("shadow-code doctor: OK", rendered)
        self.assertIn(ops.TOOL_SUPPORT_NOTE, rendered)

    def test_unreachable_ollama_fails_with_actionable_next_step(self):
        facts = _doctor_facts_fixture(
            ollama_ok=False,
            ollama_message="Cannot connect to Ollama at http://localhost:11434",
        )
        report = ops.doctor(facts, environ={}, run_cleanup=False)
        self.assertFalse(report.ok)
        rendered = ops.render_doctor(report)
        self.assertIn("ISSUES FOUND", rendered)
        self.assertIn("next step:", rendered)
        self.assertIn("ollama serve", rendered)

    def test_missing_model_next_step_names_ollama_pull(self):
        facts = _doctor_facts_fixture(
            ollama_ok=False,
            ollama_message="Model 'test-model:latest' not found. Available: ['other:7b']",
        )
        rendered = ops.render_doctor(ops.doctor(facts, environ={}, run_cleanup=False))
        self.assertIn("ollama pull test-model:latest", rendered)

    def test_withheld_capabilities_are_named(self):
        rendered = ops.render_doctor(
            ops.doctor(_doctor_facts_fixture(), environ={}, run_cleanup=False)
        )
        self.assertIn("withheld: network.access (not supported by this build)", rendered)

    def test_cleanup_runs_and_reports_orphan_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(Path(tmp) / ".shadow-tmp-orphan", b"x")
            facts = _doctor_facts_fixture(workspace=tmp)
            rendered = ops.render_doctor(ops.doctor(facts, environ={}))
            self.assertIn("removed 1 orphan(s)", rendered)
            self.assertFalse((Path(tmp) / ".shadow-tmp-orphan").exists())

    def test_run_cleanup_off_is_purely_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            orphan = _write(Path(tmp) / ".shadow-tmp-orphan", b"x")
            facts = _doctor_facts_fixture(workspace=tmp)
            rendered = ops.render_doctor(ops.doctor(facts, environ={}, run_cleanup=False))
            self.assertIn("not swept", rendered)
            self.assertTrue(orphan.exists())

    def test_live_event_store_reports_integrity_and_pending(self):
        from shadow_code.events import EventStore, NewEvent, SessionStartedPayload

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "events.db"
            store = EventStore(db_path)
            try:
                session_id = "s" * 32
                store.append(
                    session_id,
                    NewEvent(
                        "session_started",
                        SessionStartedPayload(model="m", cwd=tmp),
                    ),
                )
                facts = _doctor_facts_fixture(
                    workspace=tmp,
                    event_store=store,
                    events_db_path=str(db_path),
                )
                rendered = ops.render_doctor(ops.doctor(facts, environ={}))
            finally:
                store.close()
        self.assertIn("integrity: ok", rendered)
        self.assertIn("pending: none", rendered)
        self.assertIn("(current)", rendered)


class TestAuthoritySummary(unittest.TestCase):
    """The startup block names every boundary and unavailable capability."""

    def test_names_boundaries_and_withheld_capabilities(self):
        lines = ops.authority_summary(
            workspace_root="/work",
            device=8,
            inode=9,
            containment="openat2",
            granted=("filesystem.read", "process.execute"),
            withheld=(
                ("network.access", "not supported by this build"),
                ("mcp.invoke", "not supported by this build"),
            ),
            sandbox_label="unconfined",
            mutation_mode="apply",
            prompt_digest="f" * 64,
        )
        text = "\n".join(lines)
        self.assertIn("workspace=/work", text)
        self.assertIn("containment=openat2", text)
        self.assertIn("identity=8:9", text)
        self.assertIn("network.access(not supported by this build)", text)
        self.assertIn("mcp.invoke(not supported by this build)", text)
        self.assertIn("bash=unconfined", text)
        self.assertIn("approval=one-shot", text)

    def test_sentinels_are_redacted(self):
        lines = ops.authority_summary(
            workspace_root="/work/secret-workspace-name",
            device=None,
            inode=None,
            containment="openat2",
            granted=(),
            withheld=(),
            sandbox_label="unconfined",
            mutation_mode="apply",
            prompt_digest="f" * 64,
            sentinels=("secret-workspace-name",),
        )
        self.assertNotIn("secret-workspace-name", "\n".join(lines))
        self.assertIn("identity=unknown", lines[0])


def _slash_runtime(tmp: Path, *, databases: bool = True):
    """A minimal SessionRuntime wired for the ops slash commands."""
    from shadow_code.main import SessionRuntime

    rt = SessionRuntime(cwd=str(tmp))
    rt.ctx = MagicMock(cwd=str(tmp))
    guard = MagicMock()
    guard.identity.device = 1
    guard.identity.inode = 2
    rt.workspace_guard = guard
    rt.execution_context = MagicMock(
        workspace_root=str(tmp), sandbox_label="unconfined", mutation_mode="apply"
    )
    prompt_manager = MagicMock()
    prompt_manager.active.digest = "e" * 64
    prompt_manager.active.sources = [MagicMock(), MagicMock()]
    prompt_manager.store.root = str(tmp / "prompts")
    rt.prompt_manager = prompt_manager
    client = MagicMock()
    client.health_check.return_value = (True, "OK")
    rt.client = client
    rt.db_path = str(tmp / "sessions.db") if databases else None
    rt.events_db_path = tmp / "events.db" if databases else None
    rt.event_store = None
    return rt


class TestOpsSlashCommands(unittest.TestCase):
    """Wiring: /doctor, /backup, /restore through the shared dispatch."""

    def setUp(self):
        from shadow_code.main import _dispatch_slash_command

        self.dispatch = _dispatch_slash_command

    def _run(self, command: str, rt, confirm=None) -> str:
        lines: list[str] = []
        action = self.dispatch(command, rt, lines.append, confirm)
        self.assertEqual(action, "handled")
        return "\n".join(lines)

    def test_doctor_renders_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = _slash_runtime(Path(tmp), databases=False)
            output = self._run("/doctor", rt)
        self.assertIn("shadow-code doctor:", output)
        self.assertIn("configuration:", output)
        self.assertIn("capabilities:", output)

    def test_doctor_reports_unreachable_ollama(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = _slash_runtime(Path(tmp), databases=False)
            rt.client.health_check.return_value = (
                False,
                "Cannot connect to Ollama at http://localhost:11434",
            )
            output = self._run("/doctor", rt)
        self.assertIn("ISSUES FOUND", output)
        self.assertIn("ollama serve", output)

    def test_backup_copies_databases_and_prints_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as state:
            root = Path(tmp)
            _write(root / "sessions.db", b"sessions")
            _write(root / "events.db", b"events")
            rt = _slash_runtime(root)
            with patch.dict(os.environ, {"XDG_STATE_HOME": state}):
                output = self._run("/backup", rt)
            manifests = list((Path(state) / "shadow-code" / "backups").rglob("manifest.json"))
            self.assertEqual(len(manifests), 1)
        self.assertIn("backup:", output)
        self.assertIn("sessions.db", output)
        self.assertIn("events.db", output)

    def test_backup_without_databases_is_a_typed_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = _slash_runtime(Path(tmp), databases=False)
            output = self._run("/backup", rt)
        self.assertIn("[backup failed (nothing_to_backup)", output)

    def _backup_then_corrupt(self, root: Path, state: str) -> None:
        _write(root / "sessions.db", b"sessions-v1")
        rt = _slash_runtime(root)
        with patch.dict(os.environ, {"XDG_STATE_HOME": state}):
            self._run("/backup", rt)
        (root / "sessions.db").write_bytes(b"sessions-v2-live")

    def test_restore_previews_then_applies_on_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as state:
            root = Path(tmp)
            self._backup_then_corrupt(root, state)
            rt = _slash_runtime(root)
            prompts: list[str] = []

            def confirm(prompt: str) -> bool:
                prompts.append(prompt)
                return True

            with patch.dict(os.environ, {"XDG_STATE_HOME": state}):
                output = self._run("/restore", rt, confirm)
            restored = (root / "sessions.db").read_bytes()
        # The dry-run plan renders BEFORE the confirmation prompt fires.
        self.assertIn("restore plan (dry-run):", output)
        self.assertEqual(len(prompts), 1)
        self.assertIn("restore applied:", output)
        self.assertEqual(restored, b"sessions-v1")

    def test_restore_denial_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as state:
            root = Path(tmp)
            self._backup_then_corrupt(root, state)
            rt = _slash_runtime(root)
            with patch.dict(os.environ, {"XDG_STATE_HOME": state}):
                output = self._run("/restore", rt, lambda _prompt: False)
            live = (root / "sessions.db").read_bytes()
        self.assertIn("restore plan (dry-run):", output)
        self.assertIn("Restore NOT applied", output)
        self.assertEqual(live, b"sessions-v2-live")

    def test_restore_without_confirm_seam_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as state:
            root = Path(tmp)
            self._backup_then_corrupt(root, state)
            rt = _slash_runtime(root)
            with patch.dict(os.environ, {"XDG_STATE_HOME": state}):
                output = self._run("/restore", rt)
            live = (root / "sessions.db").read_bytes()
        self.assertIn("Restore NOT applied", output)
        self.assertEqual(live, b"sessions-v2-live")

    def test_restore_without_backups_guides_to_backup(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as state:
            rt = _slash_runtime(Path(tmp))
            with patch.dict(os.environ, {"XDG_STATE_HOME": state}):
                output = self._run("/restore", rt, lambda _prompt: True)
        self.assertIn("No backups found", output)

    def test_restore_unchanged_databases_skip_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as state:
            root = Path(tmp)
            _write(root / "sessions.db", b"sessions-v1")
            rt = _slash_runtime(root)
            with patch.dict(os.environ, {"XDG_STATE_HOME": state}):
                self._run("/backup", rt)

                def confirm(_prompt: str) -> bool:
                    raise AssertionError("confirmation must not fire when nothing changes")

                output = self._run("/restore", rt, confirm)
        self.assertIn("Nothing to restore", output)


if __name__ == "__main__":
    unittest.main()
