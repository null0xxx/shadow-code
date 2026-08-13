"""Isolated MCP adapter (WU-13): one stdio transport, untrusted by default.

A hand-rolled, dependency-free MCP client speaks newline-delimited
JSON-RPC 2.0 over stdio (the MCP stdio transport): ``initialize`` ->
``notifications/initialized`` -> ``tools/list`` -> ``tools/call``. Each
configured server runs as an owned subprocess in its own process group,
with the same discipline as the bash executor: an allowlisted minimal
environment (secrets never reach the server unless explicitly configured),
a timeout that kills the whole group, and bounded output that truncates
with a marker instead of crashing.

Trust model:

  - No server is trusted by default. Every discovered tool registers as a
    namespaced spec (``mcp_<server>__<tool>``) with capability
    ``mcp.invoke`` and side effects UNKNOWN, so the policy engine requires
    a fresh one-shot approval for EVERY call, exactly like bash.
  - Namespacing is structural: built-in tool names can never be shadowed,
    and any residual collision (duplicate tool names from one server) is
    rejected at discovery time.
  - Schemas are data, never instructions: descriptions are stripped of
    control characters and length-capped; malformed input schemas and
    non-scalar properties are rejected with a visible issue.
  - A server that fails to initialize is reported unavailable and never
    blocks startup. There is NO auto-reconnect: once the owned process
    dies or times out, every later call fails closed with a typed error
    and the dead client stays dead until the next session.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from .domain.tools import (
    Capability,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolError,
    ToolHandler,
    ToolResult,
    ToolSpec,
)
from .ops import collect_sentinels, redact
from .process import _BoundedStreamReader, _kill_process_group, build_process_env

PROTOCOL_VERSION = "2025-06-18"

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
MAX_CONNECT_TIMEOUT_SECONDS = 60.0
DEFAULT_CALL_TIMEOUT_SECONDS = 60.0
MAX_CALL_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_OUTPUT_CHARS = 20_000
MAX_OUTPUT_CHARS_LIMIT = 100_000

_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_DESCRIPTION_MAX_CHARS = 400
_PROPERTY_DESCRIPTION_MAX_CHARS = 200
_STDERR_BUDGET_BYTES = 8 * 1024

_SERVER_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_NAMESPACED_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONFIG_KEYS = frozenset(
    {
        "command",
        "args",
        "env",
        "enabled",
        "connect_timeout_seconds",
        "call_timeout_seconds",
        "max_output_chars",
    }
)
_SCALAR_JSON_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}
_TYPE_DEFAULTS: dict[str, object] = {"string": "", "integer": 0, "number": 0.0, "boolean": False}


class McpError(Exception):
    """Typed MCP failure; the code becomes the ToolError code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class McpTimeoutError(McpError):
    def __init__(self, message: str) -> None:
        super().__init__("mcp_timeout", message)


class McpDisconnectError(McpError):
    def __init__(self, message: str) -> None:
        super().__init__("mcp_disconnected", message)


class McpProtocolError(McpError):
    def __init__(self, message: str) -> None:
        super().__init__("mcp_protocol_error", message)


class McpServerError(McpError):
    def __init__(self, message: str) -> None:
        super().__init__("mcp_server_error", message)


# -- configuration -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """One validated server entry from mcp.json."""

    name: str
    command: str
    args: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    call_timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


def default_mcp_config_path() -> Path:
    """Config path: $XDG_CONFIG_HOME/shadow-code/mcp.json (or ~/.config)."""
    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "shadow-code" / "mcp.json"


def _bounded_timeout(
    value: object, default: float, cap: float, label: str, name: str, issues: list[str]
) -> float:
    """Parse a per-server timeout; corrupt values degrade to the default."""
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        issues.append(f"mcp server {name}: invalid {label}; using default {default:g}s")
        return default
    if value > cap:
        issues.append(f"mcp server {name}: {label} above {cap:g}s; clamped")
        return cap
    return float(value)


def _parse_server(name: object, entry: object, issues: list[str]) -> McpServerConfig | None:
    """Validate one server entry; None means rejected or disabled."""
    label = name if isinstance(name, str) else repr(name)
    if not isinstance(name, str) or not _SERVER_NAME_PATTERN.fullmatch(name):
        issues.append(f"mcp server {label}: invalid name (want {_SERVER_NAME_PATTERN.pattern})")
        return None
    if not isinstance(entry, dict):
        issues.append(f"mcp server {name}: entry must be an object")
        return None
    unknown = sorted(set(entry) - _CONFIG_KEYS)
    if unknown:
        issues.append(f"mcp server {name}: unknown keys {unknown}")
        return None
    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        issues.append(f"mcp server {name}: enabled must be a boolean")
        return None
    if not enabled:
        return None  # explicitly disabled: silently unregistered
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        issues.append(f"mcp server {name}: command must be a non-empty string")
        return None
    args = entry.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        issues.append(f"mcp server {name}: args must be a list of strings")
        return None
    env = entry.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and _ENV_KEY_PATTERN.fullmatch(key) and isinstance(value, str)
        for key, value in env.items()
    ):
        issues.append(f"mcp server {name}: env must map valid variable names to strings")
        return None
    max_output = entry.get("max_output_chars", DEFAULT_MAX_OUTPUT_CHARS)
    if not isinstance(max_output, int) or isinstance(max_output, bool) or max_output <= 0:
        issues.append(
            f"mcp server {name}: invalid max_output_chars; using {DEFAULT_MAX_OUTPUT_CHARS}"
        )
        max_output = DEFAULT_MAX_OUTPUT_CHARS
    max_output = min(max_output, MAX_OUTPUT_CHARS_LIMIT)
    return McpServerConfig(
        name=name,
        command=command,
        args=tuple(args),
        env=dict(env),
        connect_timeout_seconds=_bounded_timeout(
            entry.get("connect_timeout_seconds"),
            DEFAULT_CONNECT_TIMEOUT_SECONDS,
            MAX_CONNECT_TIMEOUT_SECONDS,
            "connect_timeout_seconds",
            name,
            issues,
        ),
        call_timeout_seconds=_bounded_timeout(
            entry.get("call_timeout_seconds"),
            DEFAULT_CALL_TIMEOUT_SECONDS,
            MAX_CALL_TIMEOUT_SECONDS,
            "call_timeout_seconds",
            name,
            issues,
        ),
        max_output_chars=max_output,
    )


def load_mcp_config(path: Path) -> tuple[tuple[McpServerConfig, ...], tuple[str, ...]]:
    """Load enabled server configs; a bad file yields issues, never a crash."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return (), ()
    except (OSError, json.JSONDecodeError) as exc:
        return (), (f"mcp config {path} is unreadable: {exc}",)
    if not isinstance(raw, dict) or set(raw) - {"servers"}:
        return (), (f"mcp config {path}: root must be an object with only a 'servers' key",)
    servers = raw.get("servers")
    if not isinstance(servers, dict):
        return (), (f"mcp config {path}: 'servers' must be an object",)
    issues: list[str] = []
    configs = []
    for name, entry in servers.items():
        config = _parse_server(name, entry, issues)
        if config is not None:
            configs.append(config)
    return tuple(configs), tuple(issues)


def build_server_env(
    config_env: Mapping[str, str], source: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Allowlisted minimal env plus the explicitly configured overrides.

    Only the process-env allowlist survives from the real environment; API
    keys and tokens are dropped unless the owner put them in mcp.json.
    """
    env = build_process_env(source)
    env.update(config_env)
    return dict(sorted(env.items()))


# -- schema sanitization ---------------------------------------------------------


def _sanitize_text(value: object, *, max_chars: int, fallback: str = "(no description)") -> str:
    """Neutralize untrusted text: strip control chars, collapse, cap length."""
    raw = value if isinstance(value, str) else ""
    cleaned = "".join(char if char.isprintable() else " " for char in raw)
    collapsed = " ".join(cleaned.split())
    if not collapsed:
        return fallback
    if len(collapsed) > max_chars:
        return collapsed[:max_chars] + " [truncated]"
    return collapsed


def _namespaced_name(server: str, tool: object) -> str | None:
    if not isinstance(tool, str):
        return None
    candidate = f"mcp_{server}__{tool}"
    return candidate if _NAMESPACED_NAME_PATTERN.fullmatch(candidate) else None


def _build_args_model(label: str, schema: object) -> type[BaseModel]:
    """Build a strict closed scalar args model from an MCP inputSchema.

    Only flat scalar properties are supported in this unit; anything else
    (arrays, objects, nested schemas, anyOf) is rejected as unsupported.
    Optional properties receive the type's zero default so the registry's
    flat-schema projection stays closed and exact.
    """
    if schema is None:
        schema = {}
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise ValueError("inputSchema must be a JSON object schema")
    properties = schema.get("properties", {})
    required_raw = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required_raw, list):
        raise ValueError("inputSchema properties/required are malformed")
    required = {item for item in required_raw if isinstance(item, str)}
    fields: dict[str, Any] = {}
    for prop, prop_schema in properties.items():
        if not isinstance(prop, str) or not prop.isidentifier() or prop.startswith("_"):
            raise ValueError(f"property {prop!r} is not a plain identifier")
        if not isinstance(prop_schema, dict):
            raise ValueError(f"property {prop!r} schema is malformed")
        json_type = prop_schema.get("type")
        if json_type not in _SCALAR_JSON_TYPES:
            raise ValueError(f"property {prop!r} has unsupported type {json_type!r}")
        description = _sanitize_text(
            prop_schema.get("description"), max_chars=_PROPERTY_DESCRIPTION_MAX_CHARS
        )
        if prop in required:
            fields[prop] = (_SCALAR_JSON_TYPES[json_type], Field(description=description))
        else:
            fields[prop] = (
                _SCALAR_JSON_TYPES[json_type],
                Field(default=_TYPE_DEFAULTS[json_type], description=description),
            )
    return create_model(
        f"McpArgs_{label}",
        __config__=ConfigDict(extra="forbid", frozen=True, strict=True),
        **fields,
    )


# -- stdio client -----------------------------------------------------------------


class McpClient:
    """Owns one stdio MCP server process group for its whole lifetime.

    A background thread parses newline-delimited JSON-RPC responses and
    dispatches them by id; server-initiated messages are never served. Any
    protocol violation, EOF, timeout, or cancellation breaks the client
    permanently and kills the owned process group -- there is no reconnect.
    """

    def __init__(self, config: McpServerConfig, env: Mapping[str, str]) -> None:
        self._config = config
        self._env = dict(env)
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_reader: _BoundedStreamReader | None = None
        self._lock = threading.Lock()
        self._pending: dict[int, queue.Queue[object]] = {}
        self._next_id = 0
        self._broken: McpError | None = None

    @property
    def server_name(self) -> str:
        return self._config.name

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None and self._broken is None

    # -- lifecycle --

    def connect(self) -> None:
        """Spawn the server and run the initialize handshake; fail typed."""
        try:
            process = subprocess.Popen(  # nosec B603 - owner-configured command, approval-gated per call
                [self._config.command, *self._config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env,
                start_new_session=True,
            )
        except OSError as exc:
            raise McpDisconnectError(f"cannot spawn server: {exc}") from exc
        self._process = process
        self._stderr_reader = _BoundedStreamReader(
            cast(IO[bytes], process.stderr), _STDERR_BUDGET_BYTES
        )
        self._stderr_reader.start()
        threading.Thread(target=self._read_loop, daemon=True).start()
        try:
            result = self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "shadow-code", "version": "0.1.0"},
                },
                timeout=self._config.connect_timeout_seconds,
            )
            if not isinstance(result.get("protocolVersion"), str):
                raise McpProtocolError("initialize result has no protocolVersion")
            self._notify("notifications/initialized", {})
        except McpError:
            self.close()
            raise

    def close(self) -> None:
        """Idempotently break the client and kill the owned process group."""
        self._fail(McpDisconnectError("client closed"))
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            _kill_process_group(process)
        else:
            process.wait()
        if self._stderr_reader is not None:
            with suppress(Exception):
                self._stderr_reader.finish()

    # -- protocol --

    def list_tools(self) -> list[object]:
        result = self._request("tools/list", {}, timeout=self._config.connect_timeout_seconds)
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise McpProtocolError("tools/list result has no tool list")
        return tools

    def call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> str:
        """One tools/call; returns joined text content, fail typed."""
        result = self._request(
            "tools/call",
            {"name": tool_name, "arguments": dict(arguments)},
            timeout=self._config.call_timeout_seconds,
        )
        content = result.get("content")
        if not isinstance(content, list):
            raise McpProtocolError("tools/call result has no content list")
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                parts.append(text if isinstance(text, str) else "")
            else:
                parts.append("[unsupported content item omitted]")
        text = "\n".join(parts).strip("\n") or "(empty result)"
        if result.get("isError"):
            raise McpServerError(text[:500])
        return text

    # -- internals --

    def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    def _write(self, message: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise McpDisconnectError("server process is not running")
        try:
            process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
            process.stdin.flush()
        except (OSError, ValueError) as exc:
            error = McpDisconnectError(f"cannot write to server: {exc}")
            self._fail(error)
            raise error from exc

    def _request(self, method: str, params: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
        if self._broken is not None:
            raise self._broken
        process = self._process
        if process is None or process.poll() is not None:
            error = McpDisconnectError("server process is not running")
            self._fail(error)
            raise error
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            waiter: queue.Queue[object] = queue.Queue(maxsize=1)
            self._pending[request_id] = waiter
        try:
            self._write(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)}
            )
            outcome = waiter.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                self._pending.pop(request_id, None)
            self.close()  # a timed-out server is never trusted again
            raise McpTimeoutError(f"{method}: no response within {timeout:g}s") from None
        except KeyboardInterrupt:
            self.close()  # cancellation kills the owned process group
            raise
        except McpError:
            with self._lock:
                self._pending.pop(request_id, None)
            raise
        if isinstance(outcome, McpError):
            raise outcome
        return self._extract_result(cast(dict[str, Any], outcome))

    @staticmethod
    def _extract_result(message: dict[str, Any]) -> dict[str, Any]:
        error = message.get("error")
        if error is not None:
            if isinstance(error, dict):
                raise McpServerError(
                    f"server error {error.get('code', '?')}: {str(error.get('message', ''))[:200]}"
                )
            raise McpProtocolError("malformed error member")
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpProtocolError("response has no result object")
        return result

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        stream = process.stdout
        while True:
            try:
                line = stream.readline(_MAX_MESSAGE_BYTES + 1)
            except (OSError, ValueError):
                break
            if not line:
                break  # EOF: the server exited or closed stdout
            if len(line) > _MAX_MESSAGE_BYTES:
                self._fail(McpProtocolError("message exceeds the size cap"))
                return
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._fail(McpProtocolError("malformed JSON-RPC message"))
                return
            if not isinstance(message, dict):
                self._fail(McpProtocolError("JSON-RPC message is not an object"))
                return
            message_id = message.get("id")
            if message_id is None:
                continue  # server notification/request: never served
            with self._lock:
                waiter = self._pending.pop(message_id, None)
            if waiter is not None:
                waiter.put(message)
        self._fail(McpDisconnectError("server closed the connection"))

    def _fail(self, error: McpError) -> None:
        with self._lock:
            if self._broken is None:
                self._broken = error
            waiters = list(self._pending.values())
            self._pending.clear()
        for waiter in waiters:
            waiter.put(error)


# -- tool specs ---------------------------------------------------------------------


def _make_handler(client: McpClient, tool_name: str) -> ToolHandler:
    def handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
        del context  # MCP handlers need no workspace context
        try:
            text = client.call_tool(tool_name, arguments.model_dump(mode="json"))
        except McpError as error:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.name,
                error=ToolError(code=error.code, message=error.message),
            )
        return ToolResult(call_id=call.call_id, tool_name=call.name, output=text)

    return handler


def build_server_specs(
    config: McpServerConfig,
    client: McpClient,
    tools_payload: list[object],
    reserved: frozenset[str],
) -> tuple[tuple[ToolSpec, ...], tuple[str, ...]]:
    """Sanitize discovered tools into namespaced specs; reject, never fail."""
    specs: list[ToolSpec] = []
    issues: list[str] = []
    seen = set(reserved)
    for entry in tools_payload:
        label = entry.get("name") if isinstance(entry, dict) else None
        name = _namespaced_name(config.name, label)
        if name is None or not isinstance(entry, dict):
            issues.append(f"mcp server {config.name}: tool {label!r} rejected (malformed name)")
            continue
        if name in seen:
            issues.append(
                f"mcp server {config.name}: tool {label!r} rejected (name collision: {name})"
            )
            continue
        try:
            args_model = _build_args_model(f"{config.name}_{label}", entry.get("inputSchema"))
        except ValueError as exc:
            issues.append(f"mcp server {config.name}: tool {label!r} rejected ({exc})")
            continue
        description = _sanitize_text(entry.get("description"), max_chars=_DESCRIPTION_MAX_CHARS)
        specs.append(
            ToolSpec(
                name=name,
                version="1",
                description=f"[MCP {config.name}] {description}",
                args_model=args_model,
                handler=_make_handler(client, str(label)),
                capability=Capability.MCP_INVOKE,
                risk=RiskLevel.HIGH,
                side_effects=SideEffect.UNKNOWN,
                timeout_seconds=config.call_timeout_seconds,
                max_output_chars=config.max_output_chars,
                idempotency=False,
                parallel_safety=False,
                renderer_hint="text",
            )
        )
        seen.add(name)
    return tuple(specs), tuple(issues)


# -- manager ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class McpServerStatus:
    """Discovery outcome for one configured server."""

    name: str
    state: str  # "ready" | "unavailable" | "empty"
    detail: str
    tools: tuple[str, ...] = ()


class McpManager:
    """Per-session owner of every configured MCP client and its specs.

    Discovery never raises: a server that cannot spawn, handshake, or list
    tools is recorded unavailable and startup continues. close() kills every
    owned process group exactly once and reports which servers it closed.
    """

    def __init__(self) -> None:
        self._clients: dict[str, McpClient] = {}
        self._configs: dict[str, McpServerConfig] = {}
        self._statuses: list[McpServerStatus] = []
        self._specs: list[ToolSpec] = []
        self._issues: list[str] = []
        self._closed = False

    @classmethod
    def discover(
        cls,
        config_path: Path,
        *,
        reserved_names: frozenset[str],
        environ: Mapping[str, str] | None = None,
    ) -> McpManager:
        """Load mcp.json, connect every enabled server, sanitize its tools."""
        manager = cls()
        configs, issues = load_mcp_config(config_path)
        manager._issues.extend(issues)
        reserved = set(reserved_names)
        for config in configs:
            manager._configs[config.name] = config
            client = McpClient(config, build_server_env(config.env, source=environ))
            try:
                client.connect()
                tools_payload = client.list_tools()
            except McpError as error:
                client.close()
                manager._statuses.append(
                    McpServerStatus(config.name, "unavailable", f"{error.code}: {error.message}")
                )
                continue
            specs, spec_issues = build_server_specs(
                config, client, tools_payload, frozenset(reserved)
            )
            manager._issues.extend(spec_issues)
            if not specs:
                client.close()
                manager._statuses.append(
                    McpServerStatus(config.name, "empty", "no usable tools discovered")
                )
                continue
            manager._clients[config.name] = client
            manager._specs.extend(specs)
            reserved.update(spec.name for spec in specs)
            manager._statuses.append(
                McpServerStatus(
                    config.name,
                    "ready",
                    f"{len(specs)} tool(s)",
                    tuple(spec.name for spec in specs),
                )
            )
        return manager

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs)

    @property
    def statuses(self) -> tuple[McpServerStatus, ...]:
        return tuple(self._statuses)

    @property
    def issues(self) -> tuple[str, ...]:
        return tuple(self._issues)

    @property
    def ready_servers(self) -> tuple[str, ...]:
        return tuple(status.name for status in self._statuses if status.state == "ready")

    def client_for(self, server: str) -> McpClient | None:
        return self._clients.get(server)

    def report_lines(self, sentinels: tuple[str, ...] | None = None) -> tuple[str, ...]:
        """Redacted human-facing status lines for startup and /doctor.

        Configured env VALUES are never shown -- only key names; every line
        additionally passes through sentinel redaction.
        """
        active = collect_sentinels() if sentinels is None else sentinels
        lines = [redact(issue, active) for issue in self._issues]
        for status in self._statuses:
            tools = f": {', '.join(status.tools)}" if status.tools else ""
            line = f"server {status.name}: {status.state} ({status.detail}){tools}"
            config = self._configs.get(status.name)
            if config is not None:
                argv = " ".join([config.command, *config.args])
                line += f"; command={argv}; env keys={sorted(config.env)}"
            lines.append(redact(line, active))
        return tuple(lines)

    def close(self) -> tuple[str, ...]:
        """Kill every owned process group once; return the closed server names."""
        if self._closed:
            return ()
        self._closed = True
        closed = []
        for name, client in self._clients.items():
            client.close()
            closed.append(name)
        return tuple(closed)
