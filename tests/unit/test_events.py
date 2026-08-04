"""Tests for the append-only event authority (WU-06)."""

import sqlite3
from pathlib import Path

import pytest

from shadow_code.db import Database
from shadow_code.events import (
    SCHEMA_VERSION,
    ApprovalDeniedPayload,
    EventStore,
    EventStoreError,
    NewEvent,
    PolicyDecisionPayload,
    SessionStartedPayload,
    ToolCallProposedPayload,
    ToolResultPayload,
    TurnCompletedPayload,
    UserMessagePayload,
    default_events_db_path,
)


@pytest.fixture()
def store(tmp_path: Path):
    with EventStore(tmp_path / "events.db") as event_store:
        yield event_store


def _user(text: str, event_id: str | None = None) -> NewEvent:
    return NewEvent("user_message", UserMessagePayload(content=text), event_id=event_id)


def _started(model: str = "test-model") -> NewEvent:
    return NewEvent("session_started", SessionStartedPayload(model=model, cwd="/tmp/ws"))


def _proposed(call_id: str, name: str = "read_file", args: str = "{}") -> NewEvent:
    return NewEvent(
        "tool_call_proposed",
        ToolCallProposedPayload(call_id=call_id, name=name, arguments_json=args),
    )


def _result(call_id: str, ok: bool = True) -> NewEvent:
    return NewEvent(
        "tool_result",
        ToolResultPayload(
            call_id=call_id,
            tool_name="read_file",
            ok=ok,
            output="content" if ok else None,
            error_code=None if ok else "policy_denied",
            error_message=None if ok else "denied",
        ),
    )


# -- creation, pragmas, migrations --


def test_fresh_database_creates_schema(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    with EventStore(path) as store:
        tables = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"events", "schema_migrations"} <= tables
        version = store._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        assert version == SCHEMA_VERSION
    assert path.is_file()


def test_wal_and_foreign_keys_pragmas(store: EventStore) -> None:
    journal = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    foreign_keys = store._conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert journal == "wal"
    assert foreign_keys == 1


def test_migration_applies_once_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    with EventStore(path) as store:
        store.append("s1", _started())
    with EventStore(path) as store:
        count = store._conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        assert count == SCHEMA_VERSION
        assert len(store.events_for("s1")) == 1


def test_newer_schema_on_open_is_a_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    with EventStore(path) as store:
        store._conn.execute(
            "INSERT INTO schema_migrations (version, applied_utc) VALUES (?, ?)",
            (SCHEMA_VERSION + 1, "2030-01-01T00:00:00+00:00"),
        )
        store._conn.commit()
    with pytest.raises(EventStoreError) as excinfo:
        EventStore(path)
    assert excinfo.value.code == "schema_newer_than_supported"


def test_unwritable_path_is_a_typed_open_error(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    with pytest.raises(EventStoreError) as excinfo:
        EventStore(blocked / "events.db")
    assert excinfo.value.code == "open_failed"


# -- append semantics --


def test_append_orders_events_by_seq(store: EventStore) -> None:
    first = store.append("s1", _started())
    second = store.append("s1", _user("hello"))
    third = store.append("s1", _user("world"))
    assert first is not None and second is not None and third is not None
    assert first < second < third
    events = store.events_for("s1")
    assert [event.seq for event in events] == [first, second, third]
    assert [event.type for event in events] == [
        "session_started",
        "user_message",
        "user_message",
    ]


def test_duplicate_event_id_is_silently_ignored(store: EventStore) -> None:
    event = _user("hello", event_id="fixed-id")
    assert store.append("s1", event) is not None
    assert store.append("s1", event) is None  # idempotent terminal append
    assert len(store.events_for("s1")) == 1


def test_append_group_is_atomic(store: EventStore) -> None:
    inserted = store.append_group("s1", [_started(), _user("a"), _user("b")])
    assert len(inserted) == 3
    assert len(store.events_for("s1")) == 3


def test_append_group_interruption_persists_nothing(
    store: EventStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = EventStore._insert_one
    calls = {"count": 0}

    def fail_on_second(self: EventStore, session_id: str, event: NewEvent) -> int:
        calls["count"] += 1
        if calls["count"] == 2:
            raise sqlite3.OperationalError("simulated crash mid-group")
        return original(self, session_id, event)

    monkeypatch.setattr(EventStore, "_insert_one", fail_on_second)
    with pytest.raises(EventStoreError) as excinfo:
        store.append_group("s1", [_started(), _user("a"), _user("b")])
    assert excinfo.value.code == "append_failed"
    assert store.events_for("s1") == []  # no partial projection checkpoint


def test_append_unknown_type_is_a_typed_error(store: EventStore) -> None:
    with pytest.raises(EventStoreError) as excinfo:
        store.append("s1", NewEvent("nonsense", UserMessagePayload(content="x")))
    assert excinfo.value.code == "unknown_event_type"


def test_append_payload_mismatch_is_a_typed_error(store: EventStore) -> None:
    with pytest.raises(EventStoreError) as excinfo:
        store.append("s1", NewEvent("user_message", SessionStartedPayload(model="m", cwd="c")))
    assert excinfo.value.code == "payload_mismatch"


def test_concurrent_reader_sees_committed_events_during_write(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    with EventStore(path) as writer:
        writer.append("s1", _started())
        # A second connection reads while the writer holds an open transaction.
        writer._conn.execute("BEGIN IMMEDIATE")
        try:
            reader = sqlite3.connect(str(path), timeout=5)
            try:
                count = reader.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            finally:
                reader.close()
        finally:
            writer._conn.rollback()
    assert count == 1  # WAL: readers never block behind the writer


# -- projections --


def test_rebuild_transcript_reproduces_provider_messages(store: EventStore) -> None:
    store.append_group(
        "s1",
        [
            _started(),
            _user("read the file"),
            _proposed("c1", "read_file", '{"file_path": "a.txt"}'),
            NewEvent(
                "policy_decision",
                PolicyDecisionPayload(call_id="c1", disposition="allow", reason="read_only"),
            ),
            _result("c1"),
            NewEvent(
                "turn_completed",
                TurnCompletedPayload(prompt_digest="ab" * 32),
            ),
        ],
    )
    transcript = store.rebuild_transcript("s1")
    assert transcript == [
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"call_id": "c1", "name": "read_file", "arguments": {"file_path": "a.txt"}}
            ],
        },
        {"role": "tool", "content": "content", "name": "read_file"},
    ]


def test_rebuild_transcript_batches_consecutive_proposals(store: EventStore) -> None:
    store.append_group(
        "s1",
        [
            _started(),
            _proposed("c1", "read_file", '{"file_path": "a.txt"}'),
            _proposed("c2", "read_file", '{"file_path": "b.txt"}'),
            _result("c1"),
            _result("c2"),
        ],
    )
    transcript = store.rebuild_transcript("s1")
    assert len(transcript) == 3
    assert [call["call_id"] for call in transcript[0]["tool_calls"]] == ["c1", "c2"]
    assert transcript[1] == {"role": "tool", "content": "content", "name": "read_file"}


def test_rebuild_transcript_formats_error_results(store: EventStore) -> None:
    store.append_group("s1", [_started(), _proposed("c1"), _result("c1", ok=False)])
    transcript = store.rebuild_transcript("s1")
    assert transcript[-1] == {
        "role": "tool",
        "content": "[policy_denied] denied",
        "name": "read_file",
    }


# -- pending detection --


def test_pending_tool_calls_detects_proposed_without_result(store: EventStore) -> None:
    store.append_group("s1", [_started(), _proposed("c1", "bash", '{"command": "id"}')])
    pending = store.pending_tool_calls("s1")
    assert len(pending) == 1
    assert pending[0].call_id == "c1"
    assert pending[0].name == "bash"
    assert pending[0].plan_digest is None


def test_pending_tool_calls_result_is_terminal(store: EventStore) -> None:
    store.append_group("s1", [_started(), _proposed("c1"), _result("c1")])
    assert store.pending_tool_calls("s1") == []


def test_pending_tool_calls_approval_denied_is_terminal(store: EventStore) -> None:
    store.append_group(
        "s1",
        [
            _started(),
            _proposed("c1", "bash", '{"command": "id"}'),
            NewEvent(
                "approval_denied",
                ApprovalDeniedPayload(call_id="c1", plan_digest="cd" * 32),
            ),
        ],
    )
    assert store.pending_tool_calls("s1") == []


def test_pending_tool_calls_carries_plan_digest(store: EventStore) -> None:
    from shadow_code.events import ApprovalRequestedPayload

    store.append_group(
        "s1",
        [
            _started(),
            _proposed("c1", "bash", '{"command": "id"}'),
            NewEvent(
                "approval_requested",
                ApprovalRequestedPayload(call_id="c1", plan_digest="ef" * 32, preview="p"),
            ),
        ],
    )
    pending = store.pending_tool_calls("s1")
    assert pending[0].plan_digest == "ef" * 32


# -- integrity diagnostics --


def test_verify_clean_log_has_no_issues(store: EventStore) -> None:
    store.append_group(
        "s1",
        [
            _started(),
            _user("hi"),
            _proposed("c1"),
            NewEvent(
                "policy_decision",
                PolicyDecisionPayload(call_id="c1", disposition="allow", reason="read_only"),
            ),
            _result("c1"),
        ],
    )
    assert store.verify() == []
    assert store.verify("s1") == []


def test_verify_flags_corrupt_payload(store: EventStore) -> None:
    store.append("s1", _started())
    store._conn.execute(
        "INSERT INTO events (event_id, session_id, ts_utc, type, payload_version,"
        " payload_json) VALUES ('bad', 's1', 'now', 'user_message', 1, '{not json')"
    )
    store._conn.commit()
    issues = store.verify()
    assert any("corrupt" in issue or "invalid" in issue for issue in issues)


def test_verify_flags_unknown_type_and_version(store: EventStore) -> None:
    store.append("s1", _started())
    store._conn.execute(
        "INSERT INTO events (event_id, session_id, ts_utc, type, payload_version,"
        " payload_json) VALUES ('x1', 's1', 'now', 'future_event', 1, '{}')"
    )
    store._conn.execute(
        "INSERT INTO events (event_id, session_id, ts_utc, type, payload_version,"
        " payload_json) VALUES ('x2', 's1', 'now', 'user_message', 99, '{}')"
    )
    store._conn.commit()
    issues = store.verify()
    assert any("unknown type" in issue for issue in issues)
    assert any("payload version 99" in issue for issue in issues)


def test_verify_flags_unresolved_call_reference(store: EventStore) -> None:
    store.append_group("s1", [_started(), _result("ghost-call")])
    issues = store.verify()
    assert any("unproposed call id 'ghost-call'" in issue for issue in issues)


def test_verify_flags_missing_session_started(store: EventStore) -> None:
    store.append("s1", _user("orphan"))
    issues = store.verify()
    assert any("does not open with session_started" in issue for issue in issues)


def test_verify_flags_sequence_gap(store: EventStore) -> None:
    store.append_group("s1", [_started(), _user("a"), _user("b")])
    store._conn.execute("DELETE FROM events WHERE seq = 2")  # simulated tampering
    store._conn.commit()
    issues = store.verify()
    assert any("missing seq 2" in issue for issue in issues)


# -- legacy import (explicit, copy-only) --


def test_import_legacy_session_copies_without_touching_source(
    store: EventStore, tmp_path: Path
) -> None:
    legacy_path = tmp_path / "legacy.db"
    with Database(str(legacy_path)) as legacy:
        legacy_id = legacy.create_session("test-model", name="old session")
        legacy.add_message(legacy_id, "user", "hello from the past")
        legacy.add_message(legacy_id, "assistant", "hi there")
    before_bytes = legacy_path.read_bytes()
    before_mtime = legacy_path.stat().st_mtime_ns

    count = store.import_legacy_session(legacy_path, legacy_id)

    assert count == 3  # session_started + two imported messages
    assert legacy_path.read_bytes() == before_bytes
    assert legacy_path.stat().st_mtime_ns == before_mtime

    events = store.events_for(f"legacy-{legacy_id}")
    assert [event.type for event in events] == [
        "session_started",
        "imported_message",
        "imported_message",
    ]
    transcript = store.rebuild_transcript(f"legacy-{legacy_id}")
    assert transcript == [
        {"role": "user", "content": "hello from the past"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_import_legacy_session_is_idempotent(store: EventStore, tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.db"
    with Database(str(legacy_path)) as legacy:
        legacy_id = legacy.create_session("test-model")
        legacy.add_message(legacy_id, "user", "once")
    assert store.import_legacy_session(legacy_path, legacy_id) == 2
    assert store.import_legacy_session(legacy_path, legacy_id) == 0
    assert len(store.events_for(f"legacy-{legacy_id}")) == 2


def test_import_legacy_session_missing_file_is_typed_error(
    store: EventStore, tmp_path: Path
) -> None:
    with pytest.raises(EventStoreError) as excinfo:
        store.import_legacy_session(tmp_path / "missing.db", 1)
    assert excinfo.value.code == "legacy_import_failed"


# -- path conventions --


def test_default_path_follows_xdg_state_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "/tmp/xdg-state-test")
    assert default_events_db_path() == Path("/tmp/xdg-state-test/shadow-code/events.db")
    monkeypatch.delenv("XDG_STATE_HOME")
    expected = Path.home() / ".local" / "state" / "shadow-code" / "events.db"
    assert default_events_db_path() == expected


def test_events_for_is_scoped_per_session(store: EventStore) -> None:
    store.append("s1", _started())
    store.append("s2", _started())
    store.append("s1", _user("one"))
    assert [event.type for event in store.events_for("s1")] == [
        "session_started",
        "user_message",
    ]
    assert store.latest_session_id() == "s1"
