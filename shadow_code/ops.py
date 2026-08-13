"""Operational diagnostics, backup/restore, and crash recovery (WU-12).

Personal-CLI scale: the owner can answer "what is this thing doing on my
machine" with /doctor, back up and restore the local databases, preview
pending schema migrations, and recover from a crash without orphan temp
files. Every rendered diagnostic line passes through redact(), so configured
sentinel values (secret-looking environment variables plus the explicit
SHADOW_REDACT list) never reach the terminal or a backup manifest.

Diagnostics read; they do not mutate. The only two mutating operations in
this module are the documented exceptions:

  - cleanup_stale removes post-crash ``.shadow-tmp-*`` orphans (never the
    user's ``.shadow-code-exports/`` artifacts, and nothing at all while a
    live instance holds the workspace mutation lock);
  - restore_databases(apply=True) overwrites the live databases from a
    backup, only on an explicit apply flag. The backup is never deleted.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config as _config
from .events import SCHEMA_VERSION
from .process import _ENV_ALLOWLIST

_REDACTED = "***"
_MIN_SENTINEL_LEN = 6
_SECRET_KEY_MARKERS = (
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "apikey",
    "api_key",
    "auth",
    "private_key",
)
_TEMP_GLOB = ".shadow-tmp-*"
_EXPORTS_DIR_NAME = ".shadow-code-exports"
_LOCK_NAME = ".shadow-code.lock"

# Shown when a model cannot serve native tool calls; health_check() never
# probes tool support, so this guidance is attached to the doctor report.
TOOL_SUPPORT_NOTE = (
    "native tool calling requires a tool-capable model; if native calls fail "
    "at runtime, set SHADOW_LEGACY_MARKDOWN_TOOLS=1 for the compatibility "
    "path or switch to a newer tool-capable model"
)


class OpsError(Exception):
    """Typed, visible failure in an operational command."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- redaction ----------------------------------------------------------------


def collect_sentinels(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Values that must never appear in diagnostic output.

    Two sources: values of environment variables whose key looks secret
    (token/secret/password/...) and the explicit comma-separated
    SHADOW_REDACT list. Only non-trivial values (>= 6 chars) qualify, and
    the allowlisted process-environment keys are never treated as secrets.
    """
    origin = os.environ if env is None else env
    sentinels: set[str] = set()
    for key, value in origin.items():
        if key in _ENV_ALLOWLIST or not value or len(value) < _MIN_SENTINEL_LEN:
            continue
        lowered = key.lower()
        if any(marker in lowered for marker in _SECRET_KEY_MARKERS):
            sentinels.add(value)
    for entry in origin.get("SHADOW_REDACT", "").split(","):
        entry = entry.strip()
        if len(entry) >= _MIN_SENTINEL_LEN:
            sentinels.add(entry)
    return tuple(sorted(sentinels))


def redact(text: str, sentinels: tuple[str, ...]) -> str:
    """Replace every sentinel occurrence with ``***`` (longest first)."""
    for sentinel in sorted(sentinels, key=len, reverse=True):
        if sentinel:
            text = text.replace(sentinel, _REDACTED)
    return text


# -- configuration resolution --------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfigSetting:
    """One effective configuration value and where it came from."""

    name: str
    value: str
    source: str  # "env:SHADOW_MODEL" or "default"


# (label, env var, config attribute, is_flag)
_SETTINGS: tuple[tuple[str, str, str, bool], ...] = (
    ("model", "SHADOW_MODEL", "MODEL_NAME", False),
    ("ollama_host", "OLLAMA_HOST", "OLLAMA_BASE_URL", False),
    ("context_window", "SHADOW_CTX", "CONTEXT_WINDOW", False),
    ("max_output_tokens", "SHADOW_MAX_TOKENS", "MAX_OUTPUT_TOKENS", False),
    ("bash_strict", "SHADOW_BASH_STRICT", "BASH_STRICT", True),
    ("mutation_strict", "SHADOW_MUTATION_STRICT", "MUTATION_STRICT", True),
    ("legacy_markdown", "SHADOW_LEGACY_MARKDOWN_TOOLS", "LEGACY_MARKDOWN_TOOLS", True),
    ("tui", "SHADOW_TUI", "TUI_ENABLED", True),
    ("ascii", "SHADOW_ASCII", "ASCII_MODE", True),
    ("compaction_model", "SHADOW_COMPACTION_MODEL", "COMPACTION_MODEL", False),
)


def config_settings(environ: Mapping[str, str] | None = None) -> tuple[ConfigSetting, ...]:
    """Resolve every setting with its source: explicit env or built-in default."""
    origin = os.environ if environ is None else environ
    settings = []
    for label, env_name, attribute, is_flag in _SETTINGS:
        raw = getattr(_config, attribute)
        value = ("on" if raw else "off") if is_flag else str(raw)
        source = f"env:{env_name}" if origin.get(env_name, "").strip() else "default"
        settings.append(ConfigSetting(name=label, value=value, source=source))
    return tuple(settings)


# -- model capability guidance --------------------------------------------------


def check_model_capability(ok: bool, message: str, *, model: str, base_url: str) -> str:
    """Map a health_check outcome to an actionable next step ("" when ok)."""
    if ok:
        return ""
    if "Cannot connect" in message or "Connection" in message:
        return (
            f"Ollama is not reachable at {base_url}. Start it with `ollama serve` "
            "(or point OLLAMA_HOST at a running server), then retry."
        )
    if "not found" in message:
        return (
            f"Model '{model}' is not pulled. Run `ollama pull {model}` "
            "(or set SHADOW_MODEL to one of the available models), then retry."
        )
    if "tool" in message.lower():
        return (
            "The model does not support native tool calls. Pull a tool-capable "
            "model or set SHADOW_LEGACY_MARKDOWN_TOOLS=1 for the compatibility path."
        )
    return (
        f"Ollama check failed: {message}. Verify `ollama serve` is running and "
        "OLLAMA_HOST points at it."
    )


# -- stale-artifact recovery ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Outcome of a stale-artifact sweep."""

    removed: tuple[str, ...]  # workspace-relative paths removed
    skipped_reason: str | None
    lock_present: bool
    exports_count: int


def cleanup_stale(root: str | Path) -> CleanupResult:
    """Remove post-crash ``.shadow-tmp-*`` orphans beneath the workspace.

    The live mutation flow cleans its own temp files; these are leftovers of
    a killed process. ``.shadow-code-exports/`` holds user artifacts and is
    never touched. The ``.shadow-code.lock`` file is never removed either: if
    its advisory flock can be taken exclusively here, no live writer holds
    it, so sweeping is safe; if the flock cannot be taken, a live instance is
    running and nothing is removed.
    """
    root_path = Path(root)
    exports_dir = root_path / _EXPORTS_DIR_NAME
    exports_count = (
        sum(1 for entry in exports_dir.rglob("*") if entry.is_file()) if exports_dir.is_dir() else 0
    )
    lock_path = root_path / _LOCK_NAME
    lock_present = lock_path.is_file()
    if lock_present:
        try:
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
        except OSError as exc:
            return CleanupResult(
                (), f"cannot inspect mutation lock: {exc.strerror}", True, exports_count
            )
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return CleanupResult(
                    (), "a live instance holds the mutation lock", True, exports_count
                )
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    removed: list[str] = []
    for path in sorted(root_path.rglob(_TEMP_GLOB)):
        if exports_dir in path.parents:
            continue  # belt and braces: exports are user artifacts
        if not path.is_file() or path.is_symlink():
            continue  # temp commits are regular files; anything else stays
        try:
            path.unlink()
        except OSError:
            continue  # best effort; the report only lists what was removed
        removed.append(str(path.relative_to(root_path)))
    return CleanupResult(tuple(removed), None, lock_present, exports_count)


# -- database backup / restore ---------------------------------------------------


def default_backup_root() -> Path:
    """Backup root: $XDG_STATE_HOME/shadow-code/backups."""
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return base / "shadow-code" / "backups"


@dataclass(frozen=True, slots=True)
class BackupEntry:
    """One copied database file."""

    label: str
    file: str  # file name within the backup directory
    source: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BackupReceipt:
    """What a backup wrote; the manifest on disk mirrors this."""

    directory: str
    created_utc: str
    entries: tuple[BackupEntry, ...]
    prompt_snapshot_count: int
    manifest_path: str


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_with_fsync(source: Path, dest: Path) -> None:
    with open(source, "rb") as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def backup_databases(
    *,
    sessions_path: str | None,
    events_path: str | None,
    dest_root: str | Path,
    prompt_store_dir: str | Path | None = None,
    sentinels: tuple[str, ...] = (),
) -> BackupReceipt:
    """Copy the local databases into a timestamped, manifest-documented dir.

    Existing files only; a missing database is skipped, not an error. Copies
    are fsynced and recorded in manifest.json with size and sha256. Source
    paths in the manifest are redacted; digests never are.
    """
    dest = Path(dest_root) / f"shadow-code-backup-{_utc_stamp()}"
    dest.mkdir(parents=True, exist_ok=False)
    entries: list[BackupEntry] = []
    for label, raw_path in (("sessions", sessions_path), ("events", events_path)):
        if not raw_path:
            continue
        source = Path(raw_path)
        if not source.is_file():
            continue
        target = dest / f"{label}.db"
        try:
            _copy_with_fsync(source, target)
        except OSError as exc:
            raise OpsError("backup_failed", f"cannot copy {source}: {exc.strerror}") from exc
        entries.append(
            BackupEntry(
                label=label,
                file=target.name,
                source=str(source),
                size=target.stat().st_size,
                sha256=_sha256_path(target),
            )
        )
    if not entries:
        raise OpsError("nothing_to_backup", "no databases exist yet; nothing to back up")

    prompt_snapshot_count = 0
    if prompt_store_dir is not None:
        store = Path(prompt_store_dir)
        if store.is_dir():
            prompt_snapshot_count = sum(
                1 for child in store.iterdir() if child.is_dir() and len(child.name) == 64
            )

    manifest = {
        "version": 1,
        "created_utc": _utc_now(),
        "entries": [
            {
                "label": entry.label,
                "file": entry.file,
                "source": redact(entry.source, sentinels),
                "size": entry.size,
                "sha256": entry.sha256,
            }
            for entry in entries
        ],
        "prompt_store": {
            "path": redact(str(prompt_store_dir), sentinels) if prompt_store_dir else None,
            "snapshot_count": prompt_snapshot_count,
        },
    }
    manifest_path = dest / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return BackupReceipt(
        directory=str(dest),
        created_utc=str(manifest["created_utc"]),
        entries=tuple(entries),
        prompt_snapshot_count=prompt_snapshot_count,
        manifest_path=str(manifest_path),
    )


@dataclass(frozen=True, slots=True)
class RestoreAction:
    """One file a restore would (or did) overwrite."""

    label: str
    target: str
    backup_sha256: str
    current_sha256: str | None  # None when the live file does not exist
    would_change: bool


@dataclass(frozen=True, slots=True)
class RestorePlan:
    """Dry-run preview (applied=False) or executed restore (applied=True)."""

    backup_dir: str
    actions: tuple[RestoreAction, ...]
    applied: bool


def restore_databases(
    backup_dir: str | Path,
    *,
    sessions_path: str | None = None,
    events_path: str | None = None,
    apply: bool = False,
) -> RestorePlan:
    """Preview (default) or apply a restore from a backup directory.

    Dry-run compares digests and reports what would change; ``apply=True``
    verifies every backup file against the manifest BEFORE copying anything
    (a tampered backup fails closed and nothing is written), then overwrites
    the targets with fsync. The backup directory is never modified.
    """
    directory = Path(backup_dir)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OpsError(
            "backup_not_found", f"no manifest.json in {directory}; not a shadow-code backup"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise OpsError("corrupt_backup", f"cannot read manifest in {directory}: {exc}") from exc

    targets = {"sessions": sessions_path, "events": events_path}
    try:
        manifest_entries = list(manifest["entries"])
    except (KeyError, TypeError) as exc:
        raise OpsError("corrupt_backup", f"manifest in {directory} has no entries") from exc

    planned: list[tuple[str, str, Path, Path]] = []  # label, file, backup_file, target
    for entry in manifest_entries:
        label = str(entry.get("label", ""))
        target_raw = targets.get(label)
        if target_raw is None:
            continue  # caller did not ask to restore this database
        backup_file = directory / str(entry.get("file", ""))
        expected = str(entry.get("sha256", ""))
        try:
            actual = _sha256_path(backup_file)
        except OSError as exc:
            raise OpsError(
                "corrupt_backup", f"backup file {backup_file} is unreadable: {exc.strerror}"
            ) from exc
        if actual != expected:
            raise OpsError(
                "corrupt_backup",
                f"backup file {backup_file} does not match its manifest digest; refusing",
            )
        planned.append((label, expected, backup_file, Path(target_raw)))

    actions: list[RestoreAction] = []
    for label, digest, _backup_file, target in planned:
        current = _sha256_path(target) if target.is_file() else None
        actions.append(
            RestoreAction(
                label=label,
                target=str(target),
                backup_sha256=digest,
                current_sha256=current,
                would_change=current != digest,
            )
        )

    if apply:
        for _, _, backup_file, target in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                _copy_with_fsync(backup_file, target)
            except OSError as exc:
                raise OpsError(
                    "restore_failed", f"cannot restore {backup_file} -> {target}: {exc.strerror}"
                ) from exc
    return RestorePlan(
        backup_dir=str(directory), actions=tuple(actions), applied=apply and bool(planned)
    )


# -- doctor ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoctorFacts:
    """Live-runtime inputs for the diagnostic report (built by main())."""

    workspace_root: str
    workspace_device: int | None
    workspace_inode: int | None
    containment: str
    granted: tuple[str, ...]
    withheld: tuple[tuple[str, str], ...]  # (capability, reason)
    sandbox_label: str
    mutation_mode: str
    model_name: str
    ollama_ok: bool
    ollama_message: str
    prompt_digest: str
    prompt_layer_count: int
    prompt_store_path: str
    events_db_path: str | None
    legacy_db_path: str | None
    event_store: Any = None  # live EventStore when one is open
    mcp_servers: tuple[str, ...] = ()  # pre-rendered redacted MCP status lines


@dataclass(frozen=True, slots=True)
class DoctorSection:
    name: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DoctorReport:
    ok: bool
    sections: tuple[DoctorSection, ...]


def _read_schema_version(path: Path) -> int | None:
    """MAX(schema_migrations.version), read-only; None when unreadable.

    Read-only on purpose: opening the store through EventStore would APPLY
    pending migrations, and a diagnostic must never mutate.
    """
    quoted = urllib.parse.quote(str(path.absolute()))
    try:
        conn = sqlite3.connect(f"file:{quoted}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
        return int(row[0] or 0) if row else 0
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _read_session_count(path: Path) -> int | None:
    """Row count of the legacy sessions table, read-only; None on failure."""
    quoted = urllib.parse.quote(str(path.absolute()))
    try:
        conn = sqlite3.connect(f"file:{quoted}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def doctor(
    facts: DoctorFacts,
    *,
    environ: Mapping[str, str] | None = None,
    run_cleanup: bool = True,
) -> DoctorReport:
    """Assemble the redacted diagnostic report; never raises on bad state.

    Every line passes through redact() with the collected sentinels. When
    run_cleanup is set (the /doctor command), post-crash temp orphans are
    swept via cleanup_stale and the result is reported; set it off for pure
    read-only inspection.
    """
    sentinels = collect_sentinels(environ)
    ok = True
    sections: list[DoctorSection] = []

    def section(name: str, lines: list[str]) -> None:
        sections.append(DoctorSection(name, tuple(redact(line, sentinels) for line in lines)))

    config_lines = [
        f"{setting.name}={setting.value} ({setting.source})" for setting in config_settings(environ)
    ]
    for issue in _config.CONFIG_ISSUES:
        config_lines.append(f"issue: {issue}")
    section("configuration", config_lines)

    model_lines = [
        f"server: {facts.ollama_message if not facts.ollama_ok else 'reachable'}",
        f"model: {facts.model_name} {'available' if facts.ollama_ok else 'UNAVAILABLE'}",
    ]
    if facts.ollama_ok:
        model_lines.append(f"capability note: {TOOL_SUPPORT_NOTE}")
    else:
        ok = False
        model_lines.append(
            "next step: "
            + check_model_capability(
                facts.ollama_ok,
                facts.ollama_message,
                model=facts.model_name,
                base_url=str(getattr(_config, "OLLAMA_BASE_URL", "")),
            )
        )
    section("model", model_lines)

    identity = (
        f"device={facts.workspace_device} inode={facts.workspace_inode}"
        if facts.workspace_device is not None
        else "unavailable"
    )
    section(
        "workspace",
        [
            f"root: {facts.workspace_root}",
            f"identity: {identity}",
            f"containment: {facts.containment} (descriptor-relative, no symlinks)",
        ],
    )

    capability_lines = [f"granted: {', '.join(facts.granted) if facts.granted else 'none'}"]
    for capability, reason in facts.withheld:
        capability_lines.append(f"withheld: {capability} ({reason})")
    capability_lines.append(f"sandbox: {facts.sandbox_label}")
    capability_lines.append(f"mutations: {facts.mutation_mode}")
    capability_lines.append("approval: one-shot, digest-bound, no remembered grants")
    section("capabilities", capability_lines)

    mcp_lines = list(facts.mcp_servers) if facts.mcp_servers else ["no servers configured"]
    section("mcp servers", mcp_lines)

    section(
        "prompt",
        [
            f"snapshot: {facts.prompt_digest[:12]}",
            f"layers: {facts.prompt_layer_count}",
            f"store: {facts.prompt_store_path}",
        ],
    )

    event_lines: list[str] = [f"path: {facts.events_db_path or 'unavailable'}"]
    if facts.events_db_path and Path(facts.events_db_path).is_file():
        disk_version = _read_schema_version(Path(facts.events_db_path))
        if disk_version is None:
            event_lines.append("schema: unreadable (corrupt or locked database)")
            ok = False
        elif disk_version > SCHEMA_VERSION:
            event_lines.append(
                f"schema: v{disk_version} is NEWER than this build supports "
                f"(v{SCHEMA_VERSION}); upgrade shadow-code, downgrades are unsupported"
            )
            ok = False
        elif disk_version < SCHEMA_VERSION:
            event_lines.append(
                f"schema: v{disk_version}; pending migration(s) "
                f"v{disk_version} -> v{SCHEMA_VERSION} apply on next open"
            )
        else:
            event_lines.append(f"schema: v{disk_version} (current)")
    elif facts.events_db_path:
        event_lines.append(f"schema: not created yet (v{SCHEMA_VERSION} on first session)")
    store = facts.event_store
    if store is not None:
        try:
            issues = store.verify()
            event_lines.append(f"integrity: {len(issues)} issue(s)" if issues else "integrity: ok")
            for issue in issues[:5]:
                event_lines.append(f"issue: {issue}")
            latest = store.latest_session_id()
            if latest is not None:
                pending = store.pending_tool_calls(latest)
                event_lines.append(
                    f"pending: {len(pending)} unfinished call(s) in latest session"
                    if pending
                    else "pending: none"
                )
        except Exception as exc:  # a broken store must never break diagnostics
            event_lines.append(f"integrity: unavailable ({exc})")
    else:
        event_lines.append("integrity: no live store (run /events in a session)")
    section("event store", event_lines)

    legacy_lines: list[str] = [f"path: {facts.legacy_db_path or 'unavailable'}"]
    if facts.legacy_db_path and Path(facts.legacy_db_path).is_file():
        count = _read_session_count(Path(facts.legacy_db_path))
        legacy_lines.append(f"sessions: {count}" if count is not None else "sessions: unreadable")
    else:
        legacy_lines.append("sessions: no legacy database")
    section("legacy db", legacy_lines)

    artifact_lines: list[str] = []
    if run_cleanup:
        result = cleanup_stale(facts.workspace_root)
        if result.skipped_reason is not None:
            artifact_lines.append(f"stale temp files: not swept ({result.skipped_reason})")
        elif result.removed:
            artifact_lines.append(f"stale temp files: removed {len(result.removed)} orphan(s)")
            for removed in result.removed:
                artifact_lines.append(f"removed: {removed}")
        else:
            artifact_lines.append("stale temp files: none")
        artifact_lines.append(f"exports: {result.exports_count} file(s) (preserved)")
        artifact_lines.append(f"lock: {'present' if result.lock_present else 'absent'}")
    else:
        orphans = sorted(Path(facts.workspace_root).rglob(_TEMP_GLOB))
        artifact_lines.append(f"stale temp files: {len(orphans)} present (not swept)")
    section("workspace artifacts", artifact_lines)

    return DoctorReport(ok=ok, sections=tuple(sections))


def render_doctor(report: DoctorReport, theme: Any = None) -> str:
    """Render the report as plain text; theme is reserved for the TUI."""
    del theme  # plain-text rendering only in this unit
    lines = [f"shadow-code doctor: {'OK' if report.ok else 'ISSUES FOUND'}"]
    for section in report.sections:
        lines.append(f"{section.name}:")
        lines.extend(f"  {line}" for line in section.lines)
    return "\n".join(lines)


# -- startup authority summary ----------------------------------------------------


def authority_summary(
    *,
    workspace_root: str,
    device: int | None,
    inode: int | None,
    containment: str,
    granted: tuple[str, ...],
    withheld: tuple[tuple[str, str], ...],
    sandbox_label: str,
    mutation_mode: str,
    prompt_digest: str,
    sentinels: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Compact startup block naming every active boundary + withheld capability."""
    identity = f"{device}:{inode}" if device is not None else "unknown"
    granted_text = ",".join(granted) if granted else "none"
    withheld_text = (
        "; ".join(f"{capability}({reason})" for capability, reason in withheld) or "none"
    )
    lines = (
        f"authority: workspace={workspace_root} containment={containment} identity={identity}",
        f"capabilities: {granted_text} | withheld: {withheld_text}",
        f"policy: bash={sandbox_label} mutations={mutation_mode} "
        f"approval=one-shot prompt={prompt_digest[:12]}",
    )
    return tuple(redact(line, sentinels) for line in lines)


# -- receipt rendering -------------------------------------------------------------


def render_backup_receipt(receipt: BackupReceipt, sentinels: tuple[str, ...] = ()) -> str:
    """Render a backup receipt for the terminal (redacted)."""
    lines = [f"backup: {receipt.directory}"]
    for entry in receipt.entries:
        lines.append(f"  {entry.file:12} {entry.size:>10} bytes  sha256:{entry.sha256[:12]}")
    lines.append(f"  prompt snapshots: {receipt.prompt_snapshot_count}")
    lines.append(f"  manifest: {receipt.manifest_path}")
    return "\n".join(redact(line, sentinels) for line in lines)


def render_restore_plan(plan: RestorePlan, sentinels: tuple[str, ...] = ()) -> str:
    """Render a restore preview or applied restore for the terminal (redacted)."""
    header = "restore applied:" if plan.applied else "restore plan (dry-run):"
    lines = [f"{header} {plan.backup_dir}"]
    for action in plan.actions:
        backup_digest = action.backup_sha256[:12]
        current = action.current_sha256[:12] if action.current_sha256 else "missing"
        status = "would overwrite" if action.would_change else "unchanged"
        if plan.applied:
            status = "restored" if action.would_change else "already current"
        lines.append(
            f"  {action.label:9} -> {action.target}  "
            f"backup sha256:{backup_digest}  current sha256:{current}  [{status}]"
        )
    if not plan.applied:
        lines.append("no changes written; re-run with apply to write")
    else:
        lines.append("restart shadow-code to use the restored data")
    return "\n".join(redact(line, sentinels) for line in lines)
