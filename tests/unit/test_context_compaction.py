"""Tests for transactional context building and compression (WU-08)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from shadow_code import context_compaction as cc
from shadow_code.events import (
    ApprovalDeniedPayload,
    ApprovalGrantedPayload,
    ApprovalRequestedPayload,
    AssistantTextPayload,
    EventStore,
    NewEvent,
    PolicyDecisionPayload,
    SessionStartedPayload,
    ToolCallProposedPayload,
    ToolResultPayload,
    TurnCompletedPayload,
    UserMessagePayload,
)
from shadow_code.policy.workspace import WorkspaceGuard

SESSION = "s1"
DIGEST = "d" * 64


@pytest.fixture()
def store(tmp_path: Path):
    with EventStore(tmp_path / "events.db") as event_store:
        yield event_store


@pytest.fixture()
def guard(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello", encoding="utf-8")
    (workspace / "src").mkdir()
    with WorkspaceGuard(workspace) as workspace_guard:
        yield workspace_guard


# -- event builders (mirroring engine emission order) -----------------------


def _started() -> NewEvent:
    return NewEvent("session_started", SessionStartedPayload(model="m", cwd="/tmp/ws"))


def _user(text: str) -> NewEvent:
    return NewEvent("user_message", UserMessagePayload(content=text))


def _assistant(text: str) -> NewEvent:
    return NewEvent("assistant_text", AssistantTextPayload(content=text))


def _proposed(call_id: str, args: str = '{"file_path": "note.txt"}') -> NewEvent:
    return NewEvent(
        "tool_call_proposed",
        ToolCallProposedPayload(call_id=call_id, name="read_file", arguments_json=args),
    )


def _decision(call_id: str, disposition: str = "allow", reason: str = "readonly") -> NewEvent:
    return NewEvent(
        "policy_decision",
        PolicyDecisionPayload(call_id=call_id, disposition=disposition, reason=reason),
    )


def _approval_requested(call_id: str) -> NewEvent:
    return NewEvent(
        "approval_requested",
        ApprovalRequestedPayload(call_id=call_id, plan_digest=DIGEST, preview="p"),
    )


def _granted(call_id: str) -> NewEvent:
    return NewEvent("approval_granted", ApprovalGrantedPayload(call_id=call_id, plan_digest=DIGEST))


def _denied(call_id: str) -> NewEvent:
    return NewEvent("approval_denied", ApprovalDeniedPayload(call_id=call_id, plan_digest=DIGEST))


def _result(call_id: str, ok: bool = True, output: str = "file contents") -> NewEvent:
    return NewEvent(
        "tool_result",
        ToolResultPayload(
            call_id=call_id,
            tool_name="read_file",
            ok=ok,
            output=output if ok else None,
            error_code=None if ok else "approval_denied",
            error_message=None if ok else "denied",
        ),
    )


def _turn_completed() -> NewEvent:
    return NewEvent("turn_completed", TurnCompletedPayload(prompt_digest=DIGEST))


def _allowed_call(store: EventStore, call_id: str) -> None:
    store.append_group(SESSION, [_proposed(call_id)])
    store.append_group(SESSION, [_decision(call_id), _result(call_id)])


def _batch(store: EventStore, call_ids: list[str]) -> None:
    """A multi-call batch: proposals land together, then per-call chains."""
    store.append_group(SESSION, [_proposed(call_id) for call_id in call_ids])
    for call_id in call_ids:
        store.append_group(SESSION, [_decision(call_id), _result(call_id)])


def _uniform(_group: cc.CausalGroup) -> int:
    return 10


def _assert_protocol_valid(messages: list[dict]) -> None:
    open_calls = 0
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            open_calls += len(message["tool_calls"])
        elif message["role"] == "tool":
            assert open_calls > 0, f"orphan tool result: {message}"
            open_calls -= 1


# -- causal grouping ----------------------------------------------------------


def test_messages_are_terminal_singleton_groups(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("hi"), _assistant("hello")])
    groups = cc.group_events(store.events_for(SESSION))
    assert [group.kind for group in groups] == ["message", "message", "message"]
    assert all(group.terminal for group in groups)
    assert all(group.call_id is None for group in groups)
    assert groups[0].event_ids == (store.events_for(SESSION)[0].event_id,)


def test_tool_group_spans_proposal_through_result(store: EventStore) -> None:
    _allowed_call(store, "c1")
    groups = cc.group_events(store.events_for(SESSION))
    assert len(groups) == 1
    group = groups[0]
    assert group.kind == "tool_call"
    assert group.call_id == "c1"
    assert group.terminal
    assert len(group.event_ids) == 3  # proposal + decision + result
    assert group.seq_start < group.seq_end


def test_pending_proposal_is_non_terminal(store: EventStore) -> None:
    store.append_group(SESSION, [_proposed("c1")])
    (group,) = cc.group_events(store.events_for(SESSION))
    assert not group.terminal


def test_pending_approval_is_non_terminal(store: EventStore) -> None:
    store.append_group(SESSION, [_proposed("c1")])
    store.append_group(
        SESSION, [_decision("c1", "require_approval", "mutating"), _approval_requested("c1")]
    )
    (group,) = cc.group_events(store.events_for(SESSION))
    assert not group.terminal


def test_denied_call_group_is_terminal_and_spans_both_terminal_events(
    store: EventStore,
) -> None:
    store.append_group(SESSION, [_proposed("c1")])
    store.append_group(
        SESSION, [_decision("c1", "require_approval", "mutating"), _approval_requested("c1")]
    )
    store.append_group(SESSION, [_denied("c1"), _result("c1", ok=False)])
    (group,) = cc.group_events(store.events_for(SESSION))
    assert group.terminal
    assert len(group.event_ids) == 5  # proposal + decision + request + denial + result


def test_granted_call_group_is_terminal(store: EventStore) -> None:
    store.append_group(SESSION, [_proposed("c1")])
    store.append_group(
        SESSION, [_decision("c1", "require_approval", "mutating"), _approval_requested("c1")]
    )
    store.append_group(SESSION, [_granted("c1"), _result("c1")])
    (group,) = cc.group_events(store.events_for(SESSION))
    assert group.terminal


def test_orphan_call_reference_degrades_to_message_group(store: EventStore) -> None:
    store.append_group(SESSION, [_decision("ghost", "allow", "readonly")])
    (group,) = cc.group_events(store.events_for(SESSION))
    assert group.kind == "message"
    assert group.terminal


# -- closed-range selection ---------------------------------------------------


def test_selection_never_splits_a_tool_group(store: EventStore) -> None:
    store.append_group(SESSION, [_user("go")])
    _allowed_call(store, "c1")
    store.append_group(SESSION, [_assistant("done")])
    groups = cc.group_events(store.events_for(SESSION), _uniform)
    # Budget fits the user message plus part of the tool group: the tool
    # group must be taken whole or not at all.
    selected = cc.select_closed_range(groups, 15)
    assert [group.kind for group in selected] == ["message"]
    selected = cc.select_closed_range(groups, 20)
    assert [group.kind for group in selected] == ["message", "tool_call"]


def test_budget_cut_lands_on_group_boundary(store: EventStore) -> None:
    store.append_group(SESSION, [_user("a"), _assistant("b"), _user("c")])
    groups = cc.group_events(store.events_for(SESSION), _uniform)
    selected = cc.select_closed_range(groups, 25)
    assert len(selected) == 2
    assert selected == groups[:2]


def test_multi_call_batch_stays_whole(store: EventStore) -> None:
    _batch(store, ["c1", "c2", "c3"])
    groups = cc.group_events(store.events_for(SESSION), _uniform)
    assert len(groups) == 3
    assert groups[0].seq_start < groups[1].seq_start < groups[0].seq_end
    # Budget for one or two calls: the batch is all-or-nothing.
    assert cc.select_closed_range(groups, 10) == []
    assert cc.select_closed_range(groups, 20) == []
    # Budget for the whole batch: everything is selected.
    selected = cc.select_closed_range(groups, 30)
    assert selected == groups


def test_selection_stops_at_first_pending_group(store: EventStore) -> None:
    store.append_group(SESSION, [_user("go")])
    _allowed_call(store, "c1")
    store.append_group(SESSION, [_proposed("c2")])  # pending forever
    store.append_group(SESSION, [_assistant("later")])
    groups = cc.group_events(store.events_for(SESSION), _uniform)
    selected = cc.select_closed_range(groups, 10**6)
    assert [group.kind for group in selected] == ["message", "tool_call"]
    assert all(group.call_id != "c2" for group in selected)


def test_denied_call_group_is_selectable(store: EventStore) -> None:
    store.append_group(SESSION, [_proposed("c1")])
    store.append_group(SESSION, [_denied("c1"), _result("c1", ok=False)])
    groups = cc.group_events(store.events_for(SESSION), _uniform)
    selected = cc.select_closed_range(groups, 10)
    assert selected == groups


def test_token_estimator_error_is_typed(store: EventStore) -> None:
    store.append_group(SESSION, [_user("hi")])

    def broken(_group: cc.CausalGroup) -> int:
        raise RuntimeError("estimator exploded")

    with pytest.raises(cc.CompactionError) as excinfo:
        cc.group_events(store.events_for(SESSION), broken)
    assert excinfo.value.code == "estimator_failed"

    groups = cc.group_events(store.events_for(SESSION))
    with pytest.raises(cc.CompactionError) as excinfo:
        cc.select_closed_range(groups, 100, broken)
    assert excinfo.value.code == "estimator_failed"


def test_negative_token_estimate_is_typed(store: EventStore) -> None:
    store.append_group(SESSION, [_user("hi")])
    groups = cc.group_events(store.events_for(SESSION))
    with pytest.raises(cc.CompactionError) as excinfo:
        cc.select_closed_range(groups, 100, lambda _group: -1)
    assert excinfo.value.code == "estimator_failed"


# -- snapshot build + validate ------------------------------------------------


def _covered_groups(store: EventStore):
    groups = cc.group_events(store.events_for(SESSION))
    return cc.select_closed_range(groups, 10**9)


def test_build_snapshot_computes_range_count_and_digests(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("hi")])
    _allowed_call(store, "c1")
    selected = _covered_groups(store)
    snapshot = cc.build_snapshot(SESSION, selected, "Read note.txt.")
    assert snapshot.covered_seq_start == min(group.seq_start for group in selected)
    assert snapshot.covered_seq_end == max(group.seq_end for group in selected)
    assert snapshot.covered_group_count == len(selected)
    assert snapshot.covered_event_ids_digest == cc.covered_event_ids_digest(selected)
    assert snapshot.source_digest == cc.source_digest_for(snapshot.covered_event_ids_digest)
    assert snapshot.referenced_paths == ("note.txt",)


def test_build_snapshot_rejects_empty_selection() -> None:
    with pytest.raises(cc.CompactionError) as excinfo:
        cc.build_snapshot(SESSION, [], "nothing")
    assert excinfo.value.code == "empty_selection"


def test_validate_accepts_path_found_in_covered_payloads(store: EventStore) -> None:
    _allowed_call(store, "c1")  # arguments and output mention note.txt
    selected = _covered_groups(store)
    snapshot = cc.build_snapshot(SESSION, selected, "Read note.txt and summarized it.")
    assert cc.validate_snapshot(snapshot, selected, None) == []


def test_validate_accepts_path_beneath_workspace(store: EventStore, guard) -> None:
    store.append_group(SESSION, [_user("edit something")])
    selected = _covered_groups(store)
    snapshot = cc.build_snapshot(SESSION, selected, "Edited src and read note.txt.")
    assert cc.validate_snapshot(snapshot, selected, guard) == []


def test_validate_flags_hallucinated_path(store: EventStore, guard) -> None:
    store.append_group(SESSION, [_user("do work")])
    selected = _covered_groups(store)
    snapshot = cc.build_snapshot(SESSION, selected, "Edited src/ghost_module.py fully.")
    issues = cc.validate_snapshot(snapshot, selected, guard)
    assert any("ghost_module.py" in issue for issue in issues)


def test_validate_detects_digest_tampering(store: EventStore) -> None:
    _allowed_call(store, "c1")
    selected = _covered_groups(store)
    snapshot = cc.build_snapshot(SESSION, selected, "Read note.txt.")
    tampered = snapshot.model_copy(update={"covered_event_ids_digest": "0" * 64})
    issues = cc.validate_snapshot(tampered, selected, None)
    assert any("covered_event_ids_digest mismatch" in issue for issue in issues)
    tampered = snapshot.model_copy(update={"source_digest": "0" * 64})
    issues = cc.validate_snapshot(tampered, selected, None)
    assert any("source_digest mismatch" in issue for issue in issues)


def test_source_digest_detects_changed_projection_logic(
    store: EventStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allowed_call(store, "c1")
    selected = _covered_groups(store)
    snapshot = cc.build_snapshot(SESSION, selected, "Read note.txt.")
    monkeypatch.setattr(cc, "PROJECTION_LOGIC_VERSION", "other-logic-v999")
    issues = cc.validate_snapshot(snapshot, selected, None)
    assert any("source_digest mismatch" in issue for issue in issues)


def test_validate_flags_shape_issues(store: EventStore) -> None:
    store.append_group(SESSION, [_user("a"), _assistant("b")])
    selected = _covered_groups(store)
    snapshot = cc.build_snapshot(SESSION, selected, "summary")
    bad = snapshot.model_copy(update={"covered_group_count": 99})
    issues = cc.validate_snapshot(bad, selected, None)
    assert any("covered_group_count" in issue for issue in issues)
    bad = snapshot.model_copy(update={"covered_seq_end": snapshot.covered_seq_end + 1})
    issues = cc.validate_snapshot(bad, selected, None)
    assert any("covered_seq_end" in issue for issue in issues)
    bad = snapshot.model_copy(update={"summary_text": "   "})
    issues = cc.validate_snapshot(bad, selected, None)
    assert any("summary_text" in issue for issue in issues)


def test_summary_schema_failure_is_rejected() -> None:
    with pytest.raises(ValidationError):
        cc.SummarySnapshot(
            session_id=SESSION,
            covered_seq_start="not-an-int",
            covered_seq_end=2,
            covered_group_count=1,
            covered_event_ids_digest="x",
            source_digest="y",
            summary_text="z",
            referenced_paths=(),
            created_utc="now",
        )


def test_unicode_summary_and_paths_round_trip(store: EventStore) -> None:
    store.append_group(
        SESSION,
        [
            _user("编辑 docs/naïve.py"),
            NewEvent(
                "tool_result",
                ToolResultPayload(
                    call_id="c9",
                    tool_name="write_file",
                    ok=True,
                    output="wrote 設定/config.yml and docs/naïve.py",
                ),
            ),
        ],
    )
    selected = _covered_groups(store)
    summary = "修改了 docs/naïve.py 與 設定/config.yml。完成了。"
    snapshot = cc.build_snapshot(SESSION, selected, summary)
    assert "docs/naïve.py" in snapshot.referenced_paths
    assert "設定/config.yml" in snapshot.referenced_paths
    assert cc.validate_snapshot(snapshot, selected, None) == []
    cc.append_context_snapshot(store, snapshot)
    restored = cc.active_snapshot(store, SESSION)
    assert restored is not None
    assert restored.summary_text == summary
    assert restored.referenced_paths == snapshot.referenced_paths


# -- store integration ----------------------------------------------------------


def test_snapshot_event_round_trip_and_active_selection(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("hi")])
    _allowed_call(store, "c1")
    before = store.events_for(SESSION)
    transcript_before = store.rebuild_transcript(SESSION)
    selected = _covered_groups(store)
    snapshot = cc.build_snapshot(SESSION, selected, "Read note.txt.")
    cc.append_context_snapshot(store, snapshot)

    after = store.events_for(SESSION)
    assert len(after) == len(before) + 1  # one event appended, none modified
    assert after[-1].type == "context_snapshot"
    assert store.rebuild_transcript(SESSION) == transcript_before  # original events intact

    active = cc.active_snapshot(store, SESSION)
    assert active is not None
    assert active.session_id == SESSION
    assert active.covered_event_ids_digest == snapshot.covered_event_ids_digest


def test_active_snapshot_is_the_newest_non_overlapping(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("one")])
    first = cc.build_snapshot(SESSION, _covered_groups(store), "first")
    cc.append_context_snapshot(store, first)
    store.append_group(SESSION, [_user("two"), _turn_completed()])
    eligible = [
        group
        for group in cc.group_events(store.events_for(SESSION))
        if group.seq_start > first.covered_seq_end
    ]
    second = cc.build_snapshot(SESSION, eligible, "second")
    cc.append_context_snapshot(store, second)
    active = cc.active_snapshot(store, SESSION)
    assert active is not None
    assert active.summary_text == "second"


def test_overlapping_snapshot_append_is_rejected(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("hi")])
    selected = _covered_groups(store)
    cc.append_context_snapshot(store, cc.build_snapshot(SESSION, selected, "first"))
    overlapping = cc.build_snapshot(SESSION, selected, "again")
    with pytest.raises(cc.CompactionError) as excinfo:
        cc.append_context_snapshot(store, overlapping)
    assert excinfo.value.code == "snapshot_overlap"


def test_preexisting_overlapping_snapshots_fail_active_selection(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("hi")])
    selected = _covered_groups(store)
    snapshot = cc.build_snapshot(SESSION, selected, "first")
    cc.append_context_snapshot(store, snapshot)
    # Bypass the append check to simulate a corrupted history.
    fields = {k: v for k, v in snapshot.model_dump().items() if k != "session_id"}
    from shadow_code.events import ContextSnapshotPayload

    store.append(SESSION, NewEvent("context_snapshot", ContextSnapshotPayload(**fields)))
    with pytest.raises(cc.CompactionError) as excinfo:
        cc.active_snapshot(store, SESSION)
    assert excinfo.value.code == "snapshot_overlap"


def test_corrupt_snapshot_payload_is_typed(store: EventStore) -> None:
    store.append_group(SESSION, [_started()])
    store._conn.execute(
        "INSERT INTO events (event_id, session_id, ts_utc, type, payload_version, "
        "payload_json) VALUES ('bad', ?, 'now', 'context_snapshot', 1, '{\"x\": 1}')",
        (SESSION,),
    )
    store._conn.commit()
    with pytest.raises(cc.CompactionError) as excinfo:
        cc.active_snapshot(store, SESSION)
    assert excinfo.value.code == "snapshot_corrupt"


# -- provider projection --------------------------------------------------------


def test_projection_without_snapshot_equals_transcript(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("hi")])
    _allowed_call(store, "c1")
    store.append_group(SESSION, [_assistant("done")])
    assert cc.build_provider_messages(store, SESSION) == store.rebuild_transcript(SESSION)


def test_projection_with_snapshot_is_summary_plus_after_range(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("first")])
    _allowed_call(store, "c1")
    selected = _covered_groups(store)
    cc.append_context_snapshot(store, cc.build_snapshot(SESSION, selected, "summary text"))
    store.append_group(SESSION, [_user("second")])
    _allowed_call(store, "c2")
    store.append_group(SESSION, [_assistant("later")])

    messages = cc.build_provider_messages(store, SESSION)
    assert messages[0]["role"] == "assistant"
    assert "Compaction summary" in messages[0]["content"]
    assert "summary text" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "second"}
    # Covered events are gone from the projection; the log still has them.
    assert not any(
        message.get("content") == "first" for message in messages if message["role"] == "user"
    )
    _assert_protocol_valid(messages)


def test_projection_never_orphans_tool_results(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("go")])
    _batch(store, ["c1", "c2", "c3"])
    _allowed_call(store, "c4")
    selected = _covered_groups(store)
    cc.append_context_snapshot(store, cc.build_snapshot(SESSION, selected, "batch summary"))
    _batch(store, ["c5", "c6"])
    messages = cc.build_provider_messages(store, SESSION)
    _assert_protocol_valid(messages)


def test_projection_round_trips_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    with EventStore(path) as store:
        store.append_group(SESSION, [_started(), _user("hi")])
        _allowed_call(store, "c1")
        cc.append_context_snapshot(
            store, cc.build_snapshot(SESSION, _covered_groups(store), "resumable")
        )
        live = cc.build_provider_messages(store, SESSION)
    with EventStore(path) as reopened:
        assert cc.build_provider_messages(reopened, SESSION) == live


# -- emergency reduction ----------------------------------------------------------


def test_emergency_reduce_drops_only_complete_groups(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("first")])
    _allowed_call(store, "c1")
    selected = _covered_groups(store)
    cc.append_context_snapshot(store, cc.build_snapshot(SESSION, selected, "summary"))
    store.append_group(SESSION, [_proposed("c2")])  # pending: never droppable
    store.append_group(SESSION, [_user("second")])
    _allowed_call(store, "c3")
    store.append_group(SESSION, [_assistant("recent")])

    messages = cc.emergency_reduce(store, SESSION, keep_recent_groups=1)
    _assert_protocol_valid(messages)
    placeholders = [m for m in messages if "reduced to save context" in m.get("content", "")]
    assert len(placeholders) == 1
    assert "2 earlier complete group(s)" in placeholders[0]["content"]
    # The pending proposal survives untouched.
    assert any(
        message.get("tool_calls") and message["tool_calls"][0]["call_id"] == "c2"
        for message in messages
    )
    # The newest group is kept verbatim.
    assert messages[-1] == {"role": "assistant", "content": "recent"}
    # Dropped groups left no orphan results behind.
    assert not any(message["role"] == "tool" for message in messages)


def test_emergency_reduce_without_snapshot(store: EventStore) -> None:
    store.append_group(SESSION, [_user("one")])
    _allowed_call(store, "c1")
    store.append_group(SESSION, [_assistant("two")])
    messages = cc.emergency_reduce(store, SESSION, keep_recent_groups=1)
    _assert_protocol_valid(messages)
    assert "reduced to save context" in messages[0]["content"]
    assert messages[-1] == {"role": "assistant", "content": "two"}
    # The event log is untouched by reduction.
    assert len(store.events_for(SESSION)) == 5


def test_emergency_reduce_keeps_everything_within_window(store: EventStore) -> None:
    store.append_group(SESSION, [_user("one")])
    _allowed_call(store, "c1")
    messages = cc.emergency_reduce(store, SESSION, keep_recent_groups=99)
    assert not any("reduced to save context" in m.get("content", "") for m in messages)
    _assert_protocol_valid(messages)


# -- diagnostics ----------------------------------------------------------


def test_context_diagnostics_reports_groups_tokens_and_snapshot(store: EventStore) -> None:
    store.append_group(SESSION, [_started(), _user("hi")])
    _allowed_call(store, "c1")
    store.append_group(SESSION, [_proposed("c2")])  # pending
    diag = cc.context_diagnostics(store, SESSION)
    assert diag["groups_by_kind"] == {"message": 2, "tool_call": 2}
    assert diag["terminal_groups"] == 3
    assert diag["pending_groups"] == 1
    assert diag["active_snapshot"] is None
    assert diag["estimated_uncovered_tokens"] > 0
    assert diag["issues"] == []

    selected = cc.select_closed_range(cc.group_events(store.events_for(SESSION)), 10**9)
    cc.append_context_snapshot(store, cc.build_snapshot(SESSION, selected, "sum"))
    diag = cc.context_diagnostics(store, SESSION)
    snapshot = diag["active_snapshot"]
    assert snapshot is not None
    assert snapshot["covered_group_count"] == len(selected)
    assert 0 < snapshot["covered_group_percent"] < 100
    assert snapshot["source_digest"] == cc.source_digest_for(snapshot["covered_event_ids_digest"])


def test_context_diagnostics_estimator_error_is_typed(store: EventStore) -> None:
    store.append_group(SESSION, [_user("hi")])

    def broken(_group: cc.CausalGroup) -> int:
        raise RuntimeError("nope")

    with pytest.raises(cc.CompactionError) as excinfo:
        cc.context_diagnostics(store, SESSION, broken)
    assert excinfo.value.code == "estimator_failed"


# -- path extraction ----------------------------------------------------------


def test_extract_referenced_paths_skips_urls_versions_and_absolute() -> None:
    text = (
        "See https://example.com/docs/x.py, version 1.2.3, /abs/path.py, "
        "./rel/ok.py and plain words, e.g. this."
    )
    assert cc.extract_referenced_paths(text) == ("rel/ok.py",)
