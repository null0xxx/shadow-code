"""Controlled process execution for the approval-gated bash tool.

Shell execution here is explicit host authority: every command runs
UNCONFINED (no sandbox is applied in this unit) in its own process group,
with a minimal allowlisted environment, a predictable timeout that kills
the whole group, bounded output capture that records removed byte counts,
and deterministic fact strings that approvals are bound to.
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import IO, cast

_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "SHELL",
)

_TERMINATE_GRACE_SECONDS = 0.5
_READ_CHUNK_BYTES = 8192


def build_process_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a minimal allowlisted environment, deterministic and secret-free.

    Only allowlisted keys present in the source (os.environ by default) are
    kept; everything else — API keys, tokens, and any other secrets — is
    dropped. Keys are sorted so the result is deterministic.
    """
    origin = os.environ if source is None else source
    return {key: origin[key] for key in _ENV_ALLOWLIST if key in origin}


def env_digest(env: Mapping[str, str]) -> str:
    """SHA-256 hex over the canonical JSON of the sorted environment mapping."""
    encoded = json.dumps(
        dict(sorted(env.items())),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def resolve_shell() -> str:
    """Return the absolute path of the shell used to run commands."""
    return shutil.which("bash") or shutil.which("sh") or "/bin/sh"


def detect_sandbox() -> str:
    """Return the available sandbox label: "bwrap", "firejail", or "unconfined".

    Informational only: commands are NOT wrapped in this unit. The label
    feeds the approval-plan facts and the UI labeling so unconfined
    execution is always visible to the approver.
    """
    if shutil.which("bwrap"):
        return "bwrap"
    if shutil.which("firejail"):
        return "firejail"
    return "unconfined"


def classify_command(command: str) -> frozenset[str]:
    """Lexically classify shell features for the approval preview.

    This is a COSMETIC scan that feeds the human-facing preview; it is not
    a safety control and must never be treated as one.
    """
    features: set[str] = set()
    if "$(" in command or "`" in command:
        features.add("substitution")
    if ">" in command or "<" in command:
        features.add("redirection")
    if "&&" in command or "||" in command or ";" in command:
        features.add("chain")
    if "|" in command.replace("||", ""):
        features.add("pipe")
    if "&" in command.replace("&&", ""):
        features.add("background")
    return frozenset(features)


def execution_facts(env: Mapping[str, str], cwd: str, sandbox_label: str) -> str:
    """Canonical JSON of the facts an approval for process execution binds to."""
    facts = {
        "shell": resolve_shell(),
        "cwd": cwd,
        "env_digest": env_digest(env),
        "sandbox": sandbox_label,
    }
    return json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """Result of a finished or terminated process execution."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    stdout_removed_bytes: int
    stderr_removed_bytes: int


class _BoundedStreamReader:
    """Drain a stream, retaining up to a byte budget and counting the rest."""

    def __init__(self, stream: IO[bytes], byte_budget: int) -> None:
        self._stream = stream
        self._byte_budget = byte_budget
        self._retained = bytearray()
        self.removed_bytes = 0
        self._thread = threading.Thread(target=self._drain, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _drain(self) -> None:
        while True:
            chunk = self._stream.read(_READ_CHUNK_BYTES)
            if not chunk:
                return
            room = max(self._byte_budget - len(self._retained), 0)
            self._retained.extend(chunk[:room])
            self.removed_bytes += len(chunk) - min(len(chunk), room)

    def finish(self) -> str:
        self._thread.join()
        return self._retained.decode("utf-8", errors="replace")


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    """SIGTERM the process group, then SIGKILL it after a short grace period."""
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_process(
    command: str,
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: int,
    max_output_chars: int,
) -> ProcessOutcome:
    """Run a shell command unconfined with timeout, group kill, and bounded output.

    The command runs via the resolved shell in its own session (own process
    group). Each output stream is captured up to max_output_chars * 4 bytes
    (UTF-8 worst case); bytes beyond the budget are drained, discarded, and
    counted. On timeout the whole process group is SIGTERMed then SIGKILLed
    and partial output is returned. KeyboardInterrupt kills the group the
    same way and re-raises so the CLI layer can treat it as cancellation.
    """
    byte_budget = max_output_chars * 4
    process = subprocess.Popen(  # noqa: S603  # nosec B603 - intentional approval-gated shell execution
        [resolve_shell(), "-c", command],
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    # stdout/stderr are pipes by construction above; cast narrows the Optional.
    stdout_reader = _BoundedStreamReader(cast(IO[bytes], process.stdout), byte_budget)
    stderr_reader = _BoundedStreamReader(cast(IO[bytes], process.stderr), byte_budget)
    stdout_reader.start()
    stderr_reader.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process)
    except KeyboardInterrupt:
        _kill_process_group(process)
        raise

    return ProcessOutcome(
        exit_code=None if timed_out else process.returncode,
        stdout=stdout_reader.finish(),
        stderr=stderr_reader.finish(),
        timed_out=timed_out,
        stdout_removed_bytes=stdout_reader.removed_bytes,
        stderr_removed_bytes=stderr_reader.removed_bytes,
    )
