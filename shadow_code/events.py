"""Append-only event authority (WU-06).

Every causal action needed for resume or audit is appended to a local SQLite
event log at $XDG_STATE_HOME/shadow-code/events.db. The store is append-only:
there are no update or delete APIs. Causally linked events are appended in a
single transaction, so a crash leaves either the whole group or nothing.

Key properties:
  - Schema versioning through schema_migrations; upgrades apply in order,
    opening a newer-than-supported database is a typed error, and there is
    no downgrade path.
  - Duplicate event ids are silently ignored, so terminal appends are
    idempotent and safe to retry.
  - rebuild_transcript projects events back into provider-shaped messages
    identical to what the live Conversation held.
  - pending_tool_calls detects proposed calls without a terminal result,
    which drives fail-closed resume reporting (never silent re-execution).
  - import_legacy_session copies legacy db.py sessions read-only; the legacy
    database is never written, migrated, or deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import urllib.parse
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from .domain.tools import FrozenModel

SCHEMA_VERSION = 1
PAYLOAD_VERSION = 1


class EventStoreError(Exception):
    """Typed, visible failure in the event store."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# -- Event payloads (payload_version=1) ---------------------------------------


class SessionStartedPayload(FrozenModel):
    model: str
    cwd: str


class UserMessagePayload(FrozenModel):
    content: str


class AssistantTextPayload(FrozenModel):
    content: str


class ToolCallProposedPayload(FrozenModel):
    call_id: str
    name: str
    arguments_json: str


class PolicyDecisionPayload(FrozenModel):
    call_id: str
    disposition: str
    reason: str


class ApprovalRequestedPayload(FrozenModel):
    call_id: str
    plan_digest: str
    preview: str


class ApprovalGrantedPayload(FrozenModel):
    call_id: str
    plan_digest: str


class ApprovalDeniedPayload(FrozenModel):
    call_id: str
    plan_digest: str


class ToolResultPayload(FrozenModel):
    call_id: str
    tool_name: str
    ok: bool
    output: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class TurnCompletedPayload(FrozenModel):
    prompt_digest: str


class SessionEndedPayload(FrozenModel):
    reason: str


class ImportedMessagePayload(FrozenModel):
    legacy_session_id: int
    role: str
    content: str
    timestamp: str


_PAYLOAD_MODELS: dict[str, type[FrozenModel]] = {
    "session_started": SessionStartedPayload,
    "user_message": UserMessagePayload,
    "assistant_text": AssistantTextPayload,
    "tool_call_proposed": ToolCallProposedPayload,
    "policy_decision": PolicyDecisionPayload,
    "approval_requested": ApprovalRequestedPayload,
    "approval_granted": ApprovalGrantedPayload,
    "approval_denied": ApprovalDeniedPayload,
    "tool_result": ToolResultPayload,
    "turn_completed": TurnCompletedPayload,
    "session_ended": SessionEndedPayload,
    "imported_message": ImportedMessagePayload,
}

# Event types that close a proposed tool call.
_TERMINAL_TYPES = ("tool_result", "approval_denied")
# Event types whose payloads reference a proposed call id.
_CALL_REF_TYPES = (
    "policy_decision",
    "approval_requested",
    "approval_granted",
    "approval_denied",
    "tool_result",
)


@dataclass(frozen=True, slots=True)
class NewEvent:
    """An event to append; event_id defaults to a random uuid."""

    type: str
    payload: FrozenModel
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class Event:
    """A persisted event row."""

    seq: int
    event_id: str
    session_id: str
    ts_utc: str
    type: str
    payload_version: int
    payload_json: str

    def parse_payload(self) -> FrozenModel:
        """Parse the payload into its versioned model; fail typed."""
        model = _PAYLOAD_MODELS.get(self.type)
        if model is None:
            raise EventStoreError(
                "unknown_event_type", f"event {self.event_id} has unknown type {self.type!r}"
            )
        if self.payload_version != PAYLOAD_VERSION:
            raise EventStoreError(
                "unknown_payload_version",
                f"event {self.event_id} has payload version {self.payload_version}",
            )
        try:
            return model.model_validate_json(self.payload_json)
        except ValidationError as exc:
            raise EventStoreError(
                "corrupt_payload", f"event {self.event_id} payload is invalid: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class PendingToolCall:
    """A proposed call with no terminal result; reported on resume."""

    call_id: str
    name: str
    proposed_utc: str
    plan_digest: str | None


def default_events_db_path() -> Path:
    """Event log path: $XDG_STATE_HOME/shadow-code/events.db."""
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return base / "shadow-code" / "events.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE events (
            seq             INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        TEXT    UNIQUE NOT NULL,
            session_id      TEXT    NOT NULL,
            ts_utc          TEXT    NOT NULL,
            type            TEXT    NOT NULL,
            payload_version INTEGER NOT NULL,
            payload_json    TEXT    NOT NULL
        )
        """,
        "CREATE INDEX idx_events_session ON events(session_id, seq)",
    ),
}


class EventStore:
    """Append-only SQLite event log. No update or delete APIs exist."""

    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path) if db_path is not None else default_events_db_path()
        self._path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._migrate()
        except EventStoreError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise EventStoreError(
                "open_failed", f"failed to open event store at {path}: {exc}"
            ) from exc

    @property
    def path(self) -> Path:
        return self._path

    # -- schema migrations (upgrade only) --

    def _migrate(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_utc TEXT NOT NULL)"
        )
        row = self._conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
        current = int(row["v"] or 0)
        if current > SCHEMA_VERSION:
            raise EventStoreError(
                "schema_newer_than_supported",
                f"event store schema is v{current}, this build supports up to "
                f"v{SCHEMA_VERSION}; downgrades are not supported",
            )
        with self._conn:
            for version in range(current + 1, SCHEMA_VERSION + 1):
                for statement in _MIGRATIONS[version]:
                    self._conn.execute(statement)
                self._conn.execute(
                    "INSERT INTO schema_migrations (version, applied_utc) VALUES (?, ?)",
                    (version, _utc_now()),
                )

    # -- append API (the only writes) --

    def append(self, session_id: str, event: NewEvent) -> int | None:
        """Append one event; returns its seq, or None for a duplicate id."""
        inserted = self.append_group(session_id, [event])
        return inserted[0] if inserted else None

    def append_group(self, session_id: str, events: Iterable[NewEvent]) -> list[int]:
        """Append events in ONE transaction; a crash persists none of them.

        Duplicate event ids are silently ignored (idempotent terminal
        append). Returns the seq values of the events actually inserted.
        """
        inserted: list[int] = []
        try:
            with self._conn:
                for event in events:
                    seq = self._insert_one(session_id, event)
                    if seq:
                        inserted.append(seq)
        except sqlite3.Error as exc:
            raise EventStoreError("append_failed", f"failed to append events: {exc}") from exc
        return inserted

    def _insert_one(self, session_id: str, event: NewEvent) -> int:
        model = _PAYLOAD_MODELS.get(event.type)
        if model is None:
            raise EventStoreError(
                "unknown_event_type", f"cannot append unknown event type {event.type!r}"
            )
        if type(event.payload) is not model:
            raise EventStoreError(
                "payload_mismatch",
                f"event type {event.type!r} requires {model.__name__}, "
                f"got {type(event.payload).__name__}",
            )
        event_id = event.event_id or uuid.uuid4().hex
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO events "
            "(event_id, session_id, ts_utc, type, payload_version, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event_id,
                session_id,
                _utc_now(),
                event.type,
                PAYLOAD_VERSION,
                event.payload.model_dump_json(),
            ),
        )
        return int(cursor.lastrowid or 0) if cursor.rowcount else 0

    # -- projections and queries --

    def events_for(self, session_id: str) -> list[Event]:
        """All events of a session, ordered by seq."""
        rows = self._conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY seq", (session_id,)
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def latest_session_id(self) -> str | None:
        """The session of the most recently appended event, if any."""
        row = self._conn.execute(
            "SELECT session_id FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return str(row["session_id"]) if row else None

    def rebuild_transcript(self, session_id: str) -> list[dict]:
        """Project events back into provider-shaped messages.

        Consecutive tool_call_proposed events form one assistant message
        with a tool_calls list, mirroring Conversation.add_assistant_tool_call;
        each tool_result becomes one tool-role message, mirroring
        Conversation.add_native_tool_result. Corrupt or unknown events fail
        typed; use verify() for non-fatal diagnostics.
        """
        messages: list[dict] = []
        proposed: list[dict] = []

        def flush_proposed() -> None:
            if proposed:
                messages.append({"role": "assistant", "content": "", "tool_calls": list(proposed)})
                proposed.clear()

        for event in self.events_for(session_id):
            if event.type == "tool_call_proposed":
                proposal = cast(ToolCallProposedPayload, event.parse_payload())
                proposed.append(
                    {
                        "call_id": proposal.call_id,
                        "name": proposal.name,
                        "arguments": json.loads(proposal.arguments_json),
                    }
                )
                continue
            payload = event.parse_payload()
            if isinstance(payload, UserMessagePayload):
                flush_proposed()
                messages.append({"role": "user", "content": payload.content})
            elif isinstance(payload, ImportedMessagePayload):
                flush_proposed()
                messages.append({"role": payload.role, "content": payload.content})
            elif isinstance(payload, AssistantTextPayload):
                flush_proposed()
                messages.append({"role": "assistant", "content": payload.content})
            elif isinstance(payload, ToolResultPayload):
                flush_proposed()
                if payload.ok:
                    content = payload.output or ""
                else:
                    content = f"[{payload.error_code}] {payload.error_message}"
                messages.append({"role": "tool", "content": content, "name": payload.tool_name})
            # policy/approval/turn/session events carry no transcript message.
        flush_proposed()
        return messages

    def pending_tool_calls(self, session_id: str) -> list[PendingToolCall]:
        """Proposed calls with no terminal result, in proposal order."""
        proposed: dict[str, PendingToolCall] = {}
        plan_digests: dict[str, str] = {}
        closed: set[str] = set()
        for event in self.events_for(session_id):
            payload = event.parse_payload()
            if isinstance(payload, ToolCallProposedPayload):
                proposed.setdefault(
                    payload.call_id,
                    PendingToolCall(
                        call_id=payload.call_id,
                        name=payload.name,
                        proposed_utc=event.ts_utc,
                        plan_digest=None,
                    ),
                )
            elif isinstance(payload, (ApprovalRequestedPayload, ApprovalGrantedPayload)):
                plan_digests[payload.call_id] = payload.plan_digest
            elif event.type in _TERMINAL_TYPES:
                call_id = getattr(payload, "call_id", "")
                closed.add(call_id)
        pending = []
        for call_id, pending_call in proposed.items():
            if call_id in closed:
                continue
            digest = plan_digests.get(call_id)
            if digest is not None and pending_call.plan_digest is None:
                pending_call = PendingToolCall(
                    call_id=pending_call.call_id,
                    name=pending_call.name,
                    proposed_utc=pending_call.proposed_utc,
                    plan_digest=digest,
                )
            pending.append(pending_call)
        return pending

    # -- integrity diagnostics --

    def verify(self, session_id: str | None = None) -> list[str]:
        """Non-fatal integrity check; returns issue strings (empty = OK).

        Checks sequence contiguity, payload parsing and versions, the
        per-session opening event, and that every call-id reference resolves
        to a proposed call in the same session.
        """
        issues: list[str] = []
        if session_id is None:
            rows = self._conn.execute("SELECT * FROM events ORDER BY seq").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY seq", (session_id,)
            ).fetchall()
        events = [self._row_to_event(row) for row in rows]

        seen_seqs = [event.seq for event in events]
        if len(set(seen_seqs)) != len(seen_seqs):
            issues.append("duplicate seq values present")
        if session_id is None and seen_seqs:
            missing = sorted(set(range(seen_seqs[0], seen_seqs[-1] + 1)) - set(seen_seqs))
            for seq in missing[:10]:
                issues.append(f"missing seq {seq}: event log has a gap")

        first_by_session: dict[str, Event] = {}
        proposed_by_session: dict[str, set[str]] = {}
        for event in events:
            first_by_session.setdefault(event.session_id, event)
            if event.type == "tool_call_proposed":
                try:
                    proposal = cast(ToolCallProposedPayload, event.parse_payload())
                except EventStoreError as exc:
                    issues.append(exc.message)
                    continue
                proposed_by_session.setdefault(event.session_id, set()).add(proposal.call_id)
                continue
            try:
                payload = event.parse_payload()
            except EventStoreError as exc:
                issues.append(exc.message)
                continue
            if event.type in _CALL_REF_TYPES:
                call_id = getattr(payload, "call_id", "")
                known = proposed_by_session.setdefault(event.session_id, set())
                if call_id not in known:
                    issues.append(
                        f"event {event.seq} ({event.type}) references unproposed "
                        f"call id {call_id!r}"
                    )

        for sid, first in first_by_session.items():
            if first.type != "session_started":
                issues.append(
                    f"session {sid} does not open with session_started "
                    f"(first event is {first.type})"
                )
        return issues

    # -- explicit legacy import (copy-only) --

    def import_legacy_session(self, legacy_db_path: str | Path, legacy_session_id: int) -> int:
        """Copy a legacy session's messages as imported_message events.

        The legacy database is opened read-only and never modified. The
        event session id is "legacy-<id>"; event ids are deterministic, so
        re-importing the same session is idempotent. Returns the number of
        events actually appended.
        """
        quoted = urllib.parse.quote(str(Path(legacy_db_path).absolute()))
        try:
            legacy = sqlite3.connect(f"file:{quoted}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise EventStoreError(
                "legacy_import_failed", f"cannot open legacy database read-only: {exc}"
            ) from exc
        try:
            rows = legacy.execute(
                "SELECT id, role, content, timestamp FROM messages "
                "WHERE session_id = ? ORDER BY id",
                (legacy_session_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise EventStoreError(
                "legacy_import_failed", f"cannot read legacy session: {exc}"
            ) from exc
        finally:
            legacy.close()

        def deterministic_id(label: str) -> str:
            digest = hashlib.sha256(
                f"legacy-import:{legacy_session_id}:{label}".encode()
            ).hexdigest()
            return digest

        session = f"legacy-{legacy_session_id}"
        events = [
            NewEvent(
                "session_started",
                SessionStartedPayload(model="legacy-import", cwd=""),
                event_id=deterministic_id("session"),
            )
        ]
        for message_id, role, content, timestamp in rows:
            events.append(
                NewEvent(
                    "imported_message",
                    ImportedMessagePayload(
                        legacy_session_id=legacy_session_id,
                        role=str(role),
                        content=str(content),
                        timestamp=str(timestamp),
                    ),
                    event_id=deterministic_id(f"message-{message_id}"),
                )
            )
        return len(self.append_group(session, events))

    # -- lifecycle --

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            seq=int(row["seq"]),
            event_id=str(row["event_id"]),
            session_id=str(row["session_id"]),
            ts_utc=str(row["ts_utc"]),
            type=str(row["type"]),
            payload_version=int(row["payload_version"]),
            payload_json=str(row["payload_json"]),
        )

    def close(self) -> None:
        import contextlib

        with contextlib.suppress(sqlite3.Error):
            self._conn.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
        self.close()
        return False
