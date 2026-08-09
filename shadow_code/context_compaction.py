"""Transactional context building and compression (WU-08).

Long sessions shrink model context WITHOUT losing causal tool groups or
source history. The event log is the untouched source of truth: compaction
never updates or deletes events. Instead, a closed range of complete causal
groups (a tool call's proposal -> decision -> approval -> result chain, or a
plain message) is condensed into a typed SummarySnapshot appended as a
``context_snapshot`` event. Provider messages are then rebuilt as one
synthetic summary message plus the projection of everything AFTER the
covered range.

Key properties:
  - A selected range never splits a call/decision/result chain: groups that
    overlap in sequence (a multi-call batch) are all-or-nothing, and a
    pending (non-terminal) group stops selection entirely.
  - A failed summary or failed validation creates NO active snapshot and
    leaves the conversation untouched.
  - The snapshot source digest binds the projection logic version to the
    covered event ids, so changed projection logic is detected on validate.
  - Emergency reduction is projection-level only: complete terminal groups
    are dropped in favor of a placeholder marker; pending groups are never
    touched; the original events always remain queryable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import cast

from .domain.policy import WorkspaceAccessError
from .events import (
    ContextSnapshotPayload,
    Event,
    EventStore,
    EventStoreError,
    NewEvent,
    project_events,
)
from .policy.workspace import WorkspaceGuard

# Bumped whenever the projection/grouping logic changes; bound into every
# snapshot's source_digest so stale snapshots fail validation.
PROJECTION_LOGIC_VERSION = "wu08-context-projection-v1"

_TERMINAL_TYPES = frozenset({"tool_result", "approval_denied"})
_CALL_REF_TYPES = frozenset(
    {
        "policy_decision",
        "approval_requested",
        "approval_granted",
        "approval_denied",
        "tool_result",
    }
)

# A referenced path is a workspace-relative token that either contains a
# slash or ends with a dot-extension of at least two word characters (so
# "note.txt" and "src/pkg/mod.py" match, but "e.g", "1.2.3" and URLs do
# not). \w is Unicode-aware, so non-ASCII paths are extracted too.
_TOKEN_RE = re.compile(r"[\w./-]+")
_EXTENSION_RE = re.compile(r"\.\w[\w\d]{1,9}$")


class CompactionError(Exception):
    """Typed, visible failure in context compaction."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class CausalGroup:
    """One indivisible unit of context: a message or a tool-call chain.

    ``text`` is the concatenated payload JSON of the group's events; it
    backs token estimation and hallucinated-reference validation.
    """

    kind: str  # "message" | "tool_call"
    event_ids: tuple[str, ...]
    seq_start: int
    seq_end: int
    call_id: str | None  # set for tool_call groups
    terminal: bool
    token_estimate: int
    text: str


TokenEstimator = Callable[[CausalGroup], int]


def default_token_estimator(group: CausalGroup) -> int:
    """Rough chars/4 heuristic; always positive, never provider-specific."""
    return max(1, len(group.text) // 4)


def _estimate(estimator: TokenEstimator, group: CausalGroup) -> int:
    """Run the injected estimator; failures surface as a typed error."""
    try:
        value = int(estimator(group))
    except CompactionError:
        raise
    except Exception as exc:
        raise CompactionError("estimator_failed", f"token estimator failed: {exc}") from exc
    if value < 0:
        raise CompactionError("estimator_failed", "token estimator returned a negative value")
    return value


@dataclass(slots=True)
class _GroupBuilder:
    kind: str
    call_id: str | None
    event_ids: list[str]
    seq_start: int
    seq_end: int
    terminal: bool
    texts: list[str]

    def add(self, event: Event) -> None:
        self.event_ids.append(event.event_id)
        self.seq_end = event.seq
        self.texts.append(event.payload_json)
        if event.type in _TERMINAL_TYPES:
            self.terminal = True


def _finalize(builder: _GroupBuilder, estimator: TokenEstimator) -> CausalGroup:
    group = CausalGroup(
        kind=builder.kind,
        event_ids=tuple(builder.event_ids),
        seq_start=builder.seq_start,
        seq_end=builder.seq_end,
        call_id=builder.call_id,
        terminal=builder.terminal,
        token_estimate=0,
        text="\n".join(builder.texts),
    )
    return replace(group, token_estimate=_estimate(estimator, group))


def group_events(
    events: Iterable[Event],
    estimator: TokenEstimator = default_token_estimator,
) -> list[CausalGroup]:
    """Build causal groups from an event stream, in seq_start order.

    A tool_call group spans its tool_call_proposed through every following
    event that references the same call id (policy decisions, approvals,
    results) up to and including its terminal tool_result/approval_denied.
    A proposed call without a terminal event stays NON-terminal (pending)
    and can never be selected for compaction. Every other event is a
    terminal singleton "message" group. A call-reference event with no open
    proposal degrades to a singleton; verify() reports that corruption.
    """
    builders: list[_GroupBuilder] = []
    open_builders: dict[str, _GroupBuilder] = {}
    for event in events:
        if event.type == "tool_call_proposed":
            # Sequential admission means a new proposal closes any finished
            # builders still open (defensive; call ids are unique per turn).
            for call_id in [cid for cid, b in open_builders.items() if b.terminal]:
                del open_builders[call_id]
            call_id = str(getattr(event.parse_payload(), "call_id", event.event_id))
            builder = _GroupBuilder(
                kind="tool_call",
                call_id=call_id,
                event_ids=[event.event_id],
                seq_start=event.seq,
                seq_end=event.seq,
                terminal=False,
                texts=[event.payload_json],
            )
            open_builders[call_id] = builder
            builders.append(builder)
        elif event.type in _CALL_REF_TYPES:
            call_id = str(getattr(event.parse_payload(), "call_id", ""))
            open_builder = open_builders.get(call_id)
            if open_builder is None:
                builders.append(
                    _GroupBuilder(
                        kind="message",
                        call_id=None,
                        event_ids=[event.event_id],
                        seq_start=event.seq,
                        seq_end=event.seq,
                        terminal=True,
                        texts=[event.payload_json],
                    )
                )
            else:
                open_builder.add(event)
        else:
            builders.append(
                _GroupBuilder(
                    kind="message",
                    call_id=None,
                    event_ids=[event.event_id],
                    seq_start=event.seq,
                    seq_end=event.seq,
                    terminal=True,
                    texts=[event.payload_json],
                )
            )
    return [_finalize(builder, estimator) for builder in builders]


def select_closed_range(
    groups: list[CausalGroup],
    token_budget: int,
    estimator: TokenEstimator | None = None,
) -> list[CausalGroup]:
    """Oldest-first selection of complete terminal groups within budget.

    NEVER splits a group, never selects a pending one, and stops at the
    first non-terminal group. Groups whose sequence ranges overlap (a
    multi-call batch whose proposals land together) are all-or-nothing:
    either the whole overlapping closure fits the remaining budget, or
    selection stops before it. The default estimator reads each group's
    precomputed token_estimate; an injected estimator failure is typed.
    """
    estimate = estimator or (lambda group: group.token_estimate)
    selected: list[CausalGroup] = []
    used = 0
    index = 0
    budget = max(0, token_budget)
    while index < len(groups):
        end = index + 1
        closure_end = groups[index].seq_end
        closure_tokens = _estimate(estimate, groups[index])
        while end < len(groups) and groups[end].seq_start <= closure_end:
            closure_end = max(closure_end, groups[end].seq_end)
            closure_tokens += _estimate(estimate, groups[end])
            end += 1
        closure = groups[index:end]
        if any(not group.terminal for group in closure):
            break  # pending chain: never selectable, and nothing past it either
        if used + closure_tokens > budget:
            break
        selected.extend(closure)
        used += closure_tokens
        index = end
    return selected


class SummarySnapshot(ContextSnapshotPayload):
    """A validated compaction summary plus its owning session id."""

    session_id: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def covered_event_ids_digest(groups: Iterable[CausalGroup]) -> str:
    """sha256 over the ordered covered event ids (stream order)."""
    return _sha256("\n".join(event_id for group in groups for event_id in group.event_ids))


def source_digest_for(ids_digest: str) -> str:
    """Bind the projection logic version to the covered ids digest."""
    return _sha256(f"{PROJECTION_LOGIC_VERSION}:{ids_digest}")


def extract_referenced_paths(summary_text: str) -> tuple[str, ...]:
    """Workspace-relative path-looking tokens in the summary, in order."""
    paths: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(summary_text):
        token = match.group(0)
        if "://" in token:
            continue  # URLs are out of scope
        token = token.removeprefix("./").strip(".")
        if not token or token.startswith("/"):
            continue  # absolute paths are out of scope
        if ".." in token.split("/"):
            continue
        if ("/" in token or _EXTENSION_RE.search(token)) and token not in seen:
            seen.add(token)
            paths.append(token)
    return tuple(paths)


def build_snapshot(
    session_id: str,
    groups: list[CausalGroup],
    summary_text: str,
) -> SummarySnapshot:
    """Build the typed snapshot over a selected closed range of groups."""
    if not groups:
        raise CompactionError("empty_selection", "cannot snapshot an empty group selection")
    ids_digest = covered_event_ids_digest(groups)
    return SummarySnapshot(
        session_id=session_id,
        covered_seq_start=min(group.seq_start for group in groups),
        covered_seq_end=max(group.seq_end for group in groups),
        covered_group_count=len(groups),
        covered_event_ids_digest=ids_digest,
        source_digest=source_digest_for(ids_digest),
        summary_text=summary_text,
        referenced_paths=extract_referenced_paths(summary_text),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )


def _exists_beneath(guard: WorkspaceGuard | None, path: str) -> bool:
    if guard is None:
        return False
    try:
        with guard.open_read(path):
            return True
    except WorkspaceAccessError:
        pass
    try:
        with guard.open_dir(path):
            return True
    except WorkspaceAccessError:
        return False


def validate_snapshot(
    snapshot: SummarySnapshot,
    groups: list[CausalGroup],
    workspace_guard: WorkspaceGuard | None,
) -> list[str]:
    """Validate a snapshot against the groups it claims to cover.

    Returns issue strings; an empty list means valid. Checks shape, digest
    recomputation (detects both tampering and changed projection logic via
    the version-bound source digest), and referenced paths: a path that
    appears NEITHER in the covered groups' payloads NOR beneath the
    workspace is a hallucinated reference.
    """
    issues: list[str] = []
    if not groups:
        issues.append("snapshot covers no groups")
        return issues
    if snapshot.covered_seq_start > snapshot.covered_seq_end:
        issues.append(
            f"covered range inverted: {snapshot.covered_seq_start} > {snapshot.covered_seq_end}"
        )
    if snapshot.covered_seq_start != min(group.seq_start for group in groups):
        issues.append("covered_seq_start does not match the selected groups")
    if snapshot.covered_seq_end != max(group.seq_end for group in groups):
        issues.append("covered_seq_end does not match the selected groups")
    if snapshot.covered_group_count != len(groups):
        issues.append(f"covered_group_count {snapshot.covered_group_count} != {len(groups)} groups")
    if not snapshot.summary_text.strip():
        issues.append("summary_text is empty")
    ids_digest = covered_event_ids_digest(groups)
    if snapshot.covered_event_ids_digest != ids_digest:
        issues.append("covered_event_ids_digest mismatch (recomputed from groups)")
    if snapshot.source_digest != source_digest_for(ids_digest):
        issues.append("source_digest mismatch (projection logic changed or snapshot tampered)")
    for path in snapshot.referenced_paths:
        in_payloads = any(path in group.text for group in groups)
        if not in_payloads and not _exists_beneath(workspace_guard, path):
            issues.append(
                f"referenced path {path!r} appears neither in the covered "
                "payloads nor beneath the workspace"
            )
    return issues


def _ranges_overlap(a: SummarySnapshot, b: SummarySnapshot) -> bool:
    return a.covered_seq_start <= b.covered_seq_end and b.covered_seq_start <= a.covered_seq_end


def active_snapshot(store: EventStore, session_id: str) -> SummarySnapshot | None:
    """The newest context snapshot of a session, or None.

    Typed failures: a corrupt snapshot payload raises
    CompactionError("snapshot_corrupt"); two snapshots with OVERLAPPING
    covered ranges raise CompactionError("snapshot_overlap") -- there must
    never be two overlapping actives.
    """
    snapshots: list[SummarySnapshot] = []
    for event in store.events_for(session_id):
        if event.type != "context_snapshot":
            continue
        try:
            payload = cast(ContextSnapshotPayload, event.parse_payload())
        except EventStoreError as exc:
            raise CompactionError("snapshot_corrupt", exc.message) from exc
        snapshots.append(SummarySnapshot(session_id=session_id, **payload.model_dump()))
    for previous, newer in zip(snapshots, snapshots[1:], strict=False):
        if _ranges_overlap(previous, newer):
            raise CompactionError(
                "snapshot_overlap",
                f"snapshots overlap: seq {previous.covered_seq_start}-"
                f"{previous.covered_seq_end} and {newer.covered_seq_start}-"
                f"{newer.covered_seq_end}",
            )
    return snapshots[-1] if snapshots else None


def append_context_snapshot(store: EventStore, snapshot: SummarySnapshot) -> None:
    """Append a validated snapshot as a context_snapshot event.

    The covered range must not overlap the current active snapshot; an
    overlap is a typed error and nothing is appended. The event log stays
    append-only -- this ADDS one event and never modifies history.
    """
    active = active_snapshot(store, snapshot.session_id)
    if active is not None and _ranges_overlap(active, snapshot):
        raise CompactionError(
            "snapshot_overlap",
            f"covered range {snapshot.covered_seq_start}-{snapshot.covered_seq_end} "
            f"overlaps the active snapshot {active.covered_seq_start}-"
            f"{active.covered_seq_end}",
        )
    fields = {key: value for key, value in snapshot.model_dump().items() if key != "session_id"}
    store.append(
        snapshot.session_id,
        NewEvent("context_snapshot", ContextSnapshotPayload(**fields)),
    )


def _summary_message(snapshot: SummarySnapshot) -> dict:
    return {
        "role": "assistant",
        "content": (
            "[Compaction summary: earlier context was condensed into this "
            "snapshot; the original events remain in the event store.]\n\n" + snapshot.summary_text
        ),
    }


def build_provider_messages(store: EventStore, session_id: str) -> list[dict]:
    """The active provider projection: summary + events after the range.

    With no snapshot this equals rebuild_transcript. With one, it is one
    synthetic assistant message holding the summary text, followed by the
    projection of events AFTER the covered range. Because a covered range
    never splits a call chain, tool results are never orphaned: any result
    after the range has its proposal after the range too.
    """
    events = store.events_for(session_id)
    snapshot = active_snapshot(store, session_id)
    if snapshot is None:
        return project_events(events)
    after = [event for event in events if event.seq > snapshot.covered_seq_end]
    return [_summary_message(snapshot)] + project_events(after)


def emergency_reduce(
    store: EventStore,
    session_id: str,
    keep_recent_groups: int,
) -> list[dict]:
    """Projection-level emergency reduction; the event log is NOT touched.

    Groups already covered by the active snapshot stay condensed into the
    summary message. Beyond it, when more groups exist than
    ``keep_recent_groups``, the oldest COMPLETE TERMINAL groups are dropped
    from the projection and replaced by a single placeholder marker.
    Pending (non-terminal) groups are never dropped. This is purely a view:
    the original events always remain queryable via events_for.
    """
    events = store.events_for(session_id)
    snapshot = active_snapshot(store, session_id)
    covered_end = snapshot.covered_seq_end if snapshot is not None else 0
    after = [
        event for event in events if event.seq > covered_end and event.type != "context_snapshot"
    ]
    by_id = {event.event_id: event for event in after}
    groups = group_events(after)

    messages: list[dict] = [_summary_message(snapshot)] if snapshot else []
    keep_from = max(0, len(groups) - max(0, keep_recent_groups))
    dropped = 0
    for index, group in enumerate(groups):
        if index < keep_from and group.terminal:
            dropped += 1
            continue
        messages.extend(project_events(by_id[event_id] for event_id in group.event_ids))
    if dropped:
        messages.insert(
            1 if snapshot else 0,
            {
                "role": "user",
                "content": (
                    f"[{dropped} earlier complete group(s) reduced to save "
                    "context; the full record remains in the event store.]"
                ),
            },
        )
    return messages


def context_diagnostics(
    store: EventStore,
    session_id: str,
    estimator: TokenEstimator | None = None,
) -> dict:
    """Context state for /context: group counts, tokens, snapshot, issues."""
    estimate = estimator or default_token_estimator
    events = store.events_for(session_id)
    groups = group_events(events)
    snapshot = active_snapshot(store, session_id)
    covered_end = snapshot.covered_seq_end if snapshot is not None else 0
    uncovered = [group for group in groups if group.seq_start > covered_end]
    by_kind = {"message": 0, "tool_call": 0}
    for group in groups:
        by_kind[group.kind] = by_kind.get(group.kind, 0) + 1
    snapshot_info = None
    if snapshot is not None:
        percent = round(100.0 * snapshot.covered_group_count / len(groups), 1) if groups else 0.0
        snapshot_info = {
            "covered_seq_start": snapshot.covered_seq_start,
            "covered_seq_end": snapshot.covered_seq_end,
            "covered_group_count": snapshot.covered_group_count,
            "covered_group_percent": percent,
            "covered_event_ids_digest": snapshot.covered_event_ids_digest,
            "source_digest": snapshot.source_digest,
            "created_utc": snapshot.created_utc,
        }
    return {
        "session_id": session_id,
        "total_events": len(events),
        "groups_total": len(groups),
        "groups_by_kind": by_kind,
        "terminal_groups": sum(1 for group in groups if group.terminal),
        "pending_groups": sum(1 for group in groups if not group.terminal),
        "estimated_uncovered_tokens": sum(_estimate(estimate, g) for g in uncovered),
        "active_snapshot": snapshot_info,
        "issues": store.verify(session_id),
    }
