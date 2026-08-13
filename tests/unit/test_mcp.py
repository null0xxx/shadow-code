"""Focused tests for the isolated MCP adapter (WU-13).

All tests use the disposable fake stdio JSON-RPC server in
tests/fixtures/mcp_fake_server.py -- no network, no real MCP servers.
Covers the roadmap list: namespace collision, malformed schema, malicious
descriptions, environment filtering, disconnect, timeout, cancellation,
oversized output, server restart (no auto-reconnect), per-server
isolation, plus the runtime scenario (approve once, deny once, cancel
once, cleanup and persisted events).
"""

import json
import os
import queue
import sys
from pathlib import Path

import pytest

from shadow_code import mcp
from shadow_code.domain.approval import ApprovalAuthority
from shadow_code.domain.policy import PolicyFacts
from shadow_code.domain.tools import Capability, ToolError
from shadow_code.engine import AgentEngine, EngineState, ProviderRound
from shadow_code.events import (
    EventStore,
    McpServerPayload,
    NewEvent,
    SessionStartedPayload,
)
from shadow_code.executor import execute_validated_call
from shadow_code.main import _authority_facts, _granted_capabilities
from shadow_code.ops import collect_sentinels
from shadow_code.policy.engine import PolicyEngine
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.tools.catalog import WorkspaceContext
from shadow_code.tools.registry import ToolRegistry

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "mcp_fake_server.py"
BUILTIN_NAMES = frozenset({"read_file", "write_file", "edit_file", "bash"})


def _entry(mode: str, **overrides: object) -> dict:
    entry: dict = {
        "command": sys.executable,
        "args": [str(FIXTURE)],
        "env": {"FAKE_MCP_MODE": mode},
    }
    entry.update(overrides)
    return entry


def _config_path(tmp_path: Path, servers: dict) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": servers}), encoding="utf-8")
    return path


def _discover(tmp_path: Path, servers: dict, environ: dict | None = None) -> mcp.McpManager:
    return mcp.McpManager.discover(
        _config_path(tmp_path, servers),
        reserved_names=BUILTIN_NAMES,
        environ={} if environ is None else environ,
    )


# -- configuration ---------------------------------------------------------------


def test_missing_config_file_is_silent(tmp_path: Path) -> None:
    configs, issues = mcp.load_mcp_config(tmp_path / "absent.json")
    assert configs == () and issues == ()


def test_malformed_config_never_crashes(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("{not json", encoding="utf-8")
    configs, issues = mcp.load_mcp_config(path)
    assert configs == () and len(issues) == 1

    path.write_text(json.dumps({"servers": [], "extra": 1}), encoding="utf-8")
    configs, issues = mcp.load_mcp_config(path)
    assert configs == () and issues


def test_invalid_server_entries_are_rejected_with_issues(tmp_path: Path) -> None:
    servers = {
        "Bad Name": _entry("echo"),
        "unknown_keys": {**_entry("echo"), "surprise": True},
        "bad_enabled": {**_entry("echo"), "enabled": "yes"},
        "bad_command": {**_entry("echo"), "command": "  "},
        "bad_args": {**_entry("echo"), "args": [1]},
        "bad_env": {**_entry("echo"), "env": {"1BAD": "x"}},
        "entry_not_object": "nope",
    }
    configs, issues = mcp.load_mcp_config(_config_path(tmp_path, servers))
    assert configs == ()
    assert len(issues) == len(servers)


def test_disabled_server_is_unregistered(tmp_path: Path) -> None:
    manager = _discover(tmp_path, {"off": {**_entry("echo"), "enabled": False}})
    assert manager.specs == ()
    assert manager.statuses == ()
    assert manager.ready_servers == ()


def test_timeout_defaults_fallback_and_clamp(tmp_path: Path) -> None:
    servers = {
        "defaults": _entry("echo"),
        "corrupt": {**_entry("echo"), "call_timeout_seconds": "soon"},
        "huge": {**_entry("echo"), "call_timeout_seconds": 99999},
    }
    configs, issues = mcp.load_mcp_config(_config_path(tmp_path, servers))
    by_name = {config.name: config for config in configs}
    assert by_name["defaults"].call_timeout_seconds == mcp.DEFAULT_CALL_TIMEOUT_SECONDS
    assert by_name["defaults"].connect_timeout_seconds == mcp.DEFAULT_CONNECT_TIMEOUT_SECONDS
    assert by_name["corrupt"].call_timeout_seconds == mcp.DEFAULT_CALL_TIMEOUT_SECONDS
    assert by_name["huge"].call_timeout_seconds == mcp.MAX_CALL_TIMEOUT_SECONDS
    assert any("corrupt" in issue for issue in issues)
    assert any("huge" in issue for issue in issues)


# -- discovery, namespacing, sanitization ------------------------------------------


def test_discovery_registers_namespaced_specs(tmp_path: Path) -> None:
    manager = _discover(tmp_path, {"echo": _entry("echo")})
    try:
        assert manager.ready_servers == ("echo",)
        names = [spec.name for spec in manager.specs]
        assert names == ["mcp_echo__echo"]
        spec = manager.specs[0]
        assert spec.capability is Capability.MCP_INVOKE
        assert spec.side_effects.value == "unknown"
        assert spec.handler is not None
        # The combined registry accepts built-ins + MCP tools unchanged.
        registry = ToolRegistry((*ToolRegistry([]).specs, *manager.specs))
        assert "mcp_echo__echo" in registry.names
    finally:
        assert manager.close() == ("echo",)
        assert manager.close() == ()  # idempotent


def test_builtin_names_can_never_be_shadowed(tmp_path: Path) -> None:
    # Structural: every MCP name carries the mcp_<server>__ prefix, so it
    # cannot equal a built-in. Duplicate tool names from one server are
    # rejected as collisions.
    manager = _discover(tmp_path, {"dupe": _entry("collision")})
    try:
        names = [spec.name for spec in manager.specs]
        assert names == ["mcp_dupe__dupe"]
        assert any("collision" in issue for issue in manager.issues)
        assert not set(names) & BUILTIN_NAMES
    finally:
        manager.close()


def test_reserved_names_rejected_in_spec_building() -> None:
    config = mcp.McpServerConfig(name="x", command="true", args=())
    client = mcp.McpClient(config, {})
    specs, issues = mcp.build_server_specs(
        config,
        client,
        [{"name": "tool", "inputSchema": {"type": "object", "properties": {}}}],
        frozenset({"mcp_x__tool"}),
    )
    assert specs == ()
    assert any("collision" in issue for issue in issues)


def test_malformed_schemas_are_rejected(tmp_path: Path) -> None:
    manager = _discover(tmp_path, {"bad": _entry("malformed")})
    try:
        names = sorted(spec.name for spec in manager.specs)
        # Schema-less tools are legal (empty object); broken or unsupported
        # schemas and invalid names are rejected with visible issues.
        assert names == ["mcp_bad__good", "mcp_bad__no_schema"]
        joined = " ".join(manager.issues)
        assert "bad_schema" in joined
        assert "array_prop" in joined
        assert "Bad Name" in joined
    finally:
        manager.close()


def test_extra_arguments_fail_closed(tmp_path: Path) -> None:
    manager = _discover(tmp_path, {"bad": _entry("malformed")})
    try:
        registry = ToolRegistry(manager.specs)
        outcome = registry.validate_call(
            {"call_id": "c1", "name": "mcp_bad__no_schema", "arguments": {"surprise": 1}}
        )
        assert isinstance(outcome, ToolError)
        assert outcome.code == "invalid_arguments"
    finally:
        manager.close()


def test_malicious_description_is_neutralized(tmp_path: Path) -> None:
    manager = _discover(tmp_path, {"evil": _entry("evil")})
    try:
        spec = manager.specs[0]
        description = spec.description
        assert description.startswith("[MCP evil] ")
        for char in ("\x1b", "\x00", "\x07", "\n", "\r"):
            assert char not in description
        assert len(description) <= 500  # length cap plus prefix and marker
        assert description.endswith("[truncated]")
    finally:
        manager.close()


def test_environment_filtering(tmp_path: Path) -> None:
    sentinel = "aws-secret-value-12345"
    environ = {
        "AWS_SECRET_ACCESS_KEY": sentinel,
        "PATH": "/usr/bin",
        "HOME": "/nonexistent",
    }
    manager = _discover(tmp_path, {"probe": _entry("env_keys")}, environ=environ)
    try:
        client = manager.client_for("probe")
        assert client is not None
        visible = client.call_tool("env_keys", {}).splitlines()
        # Only allowlisted keys plus explicitly configured env reach the server.
        assert "AWS_SECRET_ACCESS_KEY" not in visible
        assert "PATH" in visible
        assert "FAKE_MCP_MODE" in visible
    finally:
        manager.close()


def test_report_lines_are_redacted(tmp_path: Path) -> None:
    sentinel = "hunter2-super-secret"
    environ = {"MY_API_TOKEN": sentinel, "PATH": "/usr/bin"}
    manager = _discover(
        tmp_path,
        {"echo": {**_entry("echo"), "args": [str(FIXTURE), sentinel]}},
        environ=environ,
    )
    try:
        lines = manager.report_lines(collect_sentinels(environ))
        assert lines
        assert all(sentinel not in line for line in lines)
        assert any("***" in line for line in lines)
        assert any("env keys=['FAKE_MCP_MODE']" in line for line in lines)
    finally:
        manager.close()


# -- failure modes -------------------------------------------------------------------


def _only_client(manager: mcp.McpManager) -> mcp.McpClient:
    clients = [manager.client_for(name) for name in manager.ready_servers]
    assert len(clients) == 1 and clients[0] is not None
    return clients[0]


def test_disconnect_fails_closed_without_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawns = 0
    real_popen = mcp.subprocess.Popen

    def counting_popen(*args: object, **kwargs: object) -> object:
        nonlocal spawns
        spawns += 1
        return real_popen(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(mcp.subprocess, "Popen", counting_popen)
    manager = _discover(tmp_path, {"fragile": _entry("die_on_call")})
    try:
        client = _only_client(manager)
        assert spawns == 1
        with pytest.raises(mcp.McpError) as first:
            client.call_tool("echo", {"text": "hi"})
        assert first.value.code == "mcp_disconnected"
        # No auto-reconnect: the next call fails immediately, no new spawn.
        with pytest.raises(mcp.McpError) as second:
            client.call_tool("echo", {"text": "hi"})
        assert second.value.code == "mcp_disconnected"
        assert spawns == 1
        assert not client.alive
    finally:
        manager.close()


def test_timeout_kills_the_process_group(tmp_path: Path) -> None:
    manager = _discover(tmp_path, {"slow": _entry("hang", call_timeout_seconds=1)})
    try:
        client = _only_client(manager)
        process = client._process
        assert process is not None
        pid = process.pid
        with pytest.raises(mcp.McpTimeoutError):
            client.call_tool("echo", {"text": "hi"})
        assert not client.alive
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        manager.close()


def test_cancellation_kills_the_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _discover(tmp_path, {"echo": _entry("echo")})
    try:
        client = _only_client(manager)
        process = client._process
        assert process is not None
        pid = process.pid

        def interrupted_get(*args: object, **kwargs: object) -> object:
            raise KeyboardInterrupt

        monkeypatch.setattr(queue.Queue, "get", interrupted_get)
        with pytest.raises(KeyboardInterrupt):
            client.call_tool("echo", {"text": "hi"})
        assert not client.alive
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        manager.close()


def test_oversized_output_is_bounded(tmp_path: Path) -> None:
    manager = _discover(tmp_path, {"big": _entry("oversize", max_output_chars=1000)})
    try:
        registry = ToolRegistry(manager.specs)
        validated = registry.validate_call(
            {"call_id": "c1", "name": "mcp_big__echo", "arguments": {"text": "hi"}}
        )
        assert not isinstance(validated, ToolError)
        result = execute_validated_call(validated, None)
        assert result.success
        assert result.output is not None
        assert len(result.output) <= 1000
        assert "truncated" in result.output
    finally:
        manager.close()


def test_per_server_isolation(tmp_path: Path) -> None:
    manager = _discover(tmp_path, {"alpha": _entry("echo"), "beta": _entry("die_on_call")})
    try:
        assert set(manager.ready_servers) == {"alpha", "beta"}
        beta = manager.client_for("beta")
        alpha = manager.client_for("alpha")
        assert alpha is not None and beta is not None
        with pytest.raises(mcp.McpError):
            beta.call_tool("echo", {"text": "boom"})
        assert not beta.alive
        # The surviving server keeps working untouched.
        assert alpha.call_tool("echo", {"text": "still here"}) == "still here"
    finally:
        manager.close()


def test_unavailable_server_never_blocks_discovery(tmp_path: Path) -> None:
    servers = {
        "missing": {"command": "/nonexistent/mcp-server-binary"},
        "echo": _entry("echo"),
    }
    manager = _discover(tmp_path, servers)
    try:
        assert manager.ready_servers == ("echo",)
        states = {status.name: status.state for status in manager.statuses}
        assert states == {"missing": "unavailable", "echo": "ready"}
    finally:
        manager.close()


# -- engine / events integration (the runtime scenario) -------------------------------


def _engine_harness(
    tmp_path: Path, manager: mcp.McpManager, decisions: list[bool]
) -> tuple[AgentEngine, EventStore, str]:
    store = EventStore(tmp_path / "events.db")
    session = "mcp-session"
    store.append(session, NewEvent("session_started", SessionStartedPayload(model="t", cwd="")))
    guard = WorkspaceGuard(str(tmp_path))
    registry = ToolRegistry(manager.specs)
    policy = PolicyEngine(PolicyFacts(frozenset({Capability.MCP_INVOKE}), guard.identity))
    remaining = iter(decisions)
    engine = AgentEngine(
        registry,
        policy,
        WorkspaceContext(guard=guard),
        ApprovalAuthority(),
        consent=lambda plan: next(remaining),
        event_store=store,
        event_session_id=session,
    )
    return engine, store, session


def test_approve_once_deny_once_events_replay(tmp_path: Path) -> None:
    environ = {"MY_API_TOKEN": "hunter2-super-secret", "PATH": "/usr/bin"}
    manager = _discover(tmp_path, {"echo": _entry("echo")}, environ=environ)
    store = None
    try:
        engine, store, session = _engine_harness(tmp_path, manager, [True, False])
        rounds = iter(
            [
                ProviderRound(
                    text="",
                    native_calls=(
                        {
                            "call_id": "m1",
                            "name": "mcp_echo__echo",
                            "arguments": {"text": "hello"},
                        },
                    ),
                ),
                ProviderRound(
                    text="",
                    native_calls=(
                        {
                            "call_id": "m2",
                            "name": "mcp_echo__echo",
                            "arguments": {"text": "denied"},
                        },
                    ),
                ),
                ProviderRound(text="done"),
            ]
        )
        outcome = engine.run_turn(lambda: next(rounds))
        assert outcome.status is EngineState.COMPLETED
        assert outcome.calls_executed == 1
        approved, denied = outcome.results
        assert approved.success and approved.output == "hello"
        assert not denied.success
        assert denied.error is not None and denied.error.code == "approval_denied"

        event_types = [event.type for event in store.events_for(session)]
        assert event_types == [
            "session_started",
            "tool_call_proposed",
            "policy_decision",
            "approval_requested",
            "approval_granted",
            "tool_result",
            "tool_call_proposed",
            "policy_decision",
            "approval_requested",
            "approval_denied",
            "tool_result",
        ]
        assert store.verify(session) == []
        assert store.pending_tool_calls(session) == []
        # Replay matches the built-in shape exactly: one assistant message
        # per proposal batch, one tool message per result.
        transcript = store.rebuild_transcript(session)
        assert transcript[0]["role"] == "assistant"
        assert transcript[1] == {"role": "tool", "content": "hello", "name": "mcp_echo__echo"}
        # Filtered secrets never enter event payloads.
        assert "hunter2-super-secret" not in json.dumps(
            [event.payload_json for event in store.events_for(session)]
        )
    finally:
        if store is not None:
            store.close()
        manager.close()


def test_cancel_during_mcp_call_ends_turn_and_kills_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _discover(tmp_path, {"echo": _entry("echo")})
    store = None
    try:
        engine, store, session = _engine_harness(tmp_path, manager, [True])
        client = _only_client(manager)
        process = client._process
        assert process is not None
        pid = process.pid
        rounds = iter(
            [
                ProviderRound(
                    text="",
                    native_calls=(
                        {
                            "call_id": "m1",
                            "name": "mcp_echo__echo",
                            "arguments": {"text": "hi"},
                        },
                    ),
                )
            ]
        )

        def interrupted_get(*args: object, **kwargs: object) -> object:
            raise KeyboardInterrupt

        monkeypatch.setattr(queue.Queue, "get", interrupted_get)
        outcome = engine.run_turn(lambda: next(rounds))
        assert outcome.status is EngineState.CANCELLED
        assert not client.alive
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        # The cancelled call stays pending in the log -- never re-executed.
        assert [call.call_id for call in store.pending_tool_calls(session)] == ["m1"]
    finally:
        if store is not None:
            store.close()
        manager.close()


def test_mcp_server_lifecycle_events_roundtrip(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    try:
        session = "lifecycle"
        store.append_group(
            session,
            [
                NewEvent("session_started", SessionStartedPayload(model="t", cwd="")),
                NewEvent(
                    "mcp_server",
                    McpServerPayload(server="echo", status="ready", detail="1 tool(s)"),
                ),
                NewEvent(
                    "mcp_server",
                    McpServerPayload(
                        server="echo", status="closed", detail="process group terminated"
                    ),
                ),
            ],
        )
        events = store.events_for(session)
        statuses = [
            (event.parse_payload().server, event.parse_payload().status)  # type: ignore[attr-defined]
            for event in events
            if event.type == "mcp_server"
        ]
        assert statuses == [("echo", "ready"), ("echo", "closed")]
        assert store.verify(session) == []
    finally:
        store.close()


# -- authority wiring (main.py) -------------------------------------------------------


def test_mcp_capability_granted_only_when_active() -> None:
    granted = _granted_capabilities(False, "unconfined", mcp_active=True)
    assert Capability.MCP_INVOKE in granted
    inactive = _granted_capabilities(False, "unconfined", mcp_active=False)
    assert Capability.MCP_INVOKE not in inactive

    _, withheld = _authority_facts("unconfined", mcp_active=False)
    assert (Capability.MCP_INVOKE.value, "no MCP servers configured") in withheld
    granted_active, withheld_active = _authority_facts("unconfined", mcp_active=True)
    assert Capability.MCP_INVOKE.value in granted_active
    assert all(capability != Capability.MCP_INVOKE.value for capability, _ in withheld_active)
