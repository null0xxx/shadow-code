"""Deterministic tests for controlled process execution (shadow_code.process)."""

import os
import subprocess
import time
from pathlib import Path

import pytest

from shadow_code.process import (
    build_process_env,
    classify_command,
    detect_sandbox,
    env_digest,
    execution_facts,
    resolve_shell,
    run_process,
)

_SENTINEL_ENV_KEY = "SHADOW_TEST_SENTINEL_ENTRY"


def _child_pid(path: Path) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return int(path.read_text(encoding="utf-8").strip())
        time.sleep(0.01)
    raise AssertionError(f"pid file {path} was never written")


def _assert_pid_dead(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    raise AssertionError(f"process {pid} is still alive")


class TestBuildProcessEnv:
    def test_allowlist_keeps_present_keys_and_drops_secrets(self) -> None:
        source = {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            _SENTINEL_ENV_KEY: "sentinel-value",
            "AWS_SECRET_ACCESS_KEY": "sentinel-value",
            "RANDOM_OTHER": "x",
        }

        env = build_process_env(source)

        assert env == {"HOME": "/home/test", "PATH": "/usr/bin"}

    def test_missing_allowlist_keys_are_omitted(self) -> None:
        assert build_process_env({"PATH": "/bin"}) == {"PATH": "/bin"}
        assert build_process_env({}) == {}

    def test_real_environment_is_filtered(self) -> None:
        env = build_process_env()

        assert _SENTINEL_ENV_KEY not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env
        assert set(env) <= {
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
        }


class TestEnvDigest:
    def test_deterministic_regardless_of_order(self) -> None:
        first = env_digest({"PATH": "/bin", "HOME": "/h"})
        second = env_digest({"HOME": "/h", "PATH": "/bin"})

        assert first == second
        assert len(first) == 64

    def test_sensitive_to_changes(self) -> None:
        assert env_digest({"PATH": "/bin"}) != env_digest({"PATH": "/usr/bin"})
        assert env_digest({"PATH": "/bin"}) != env_digest({"PATH": "/bin", "HOME": "/h"})


class TestExecutionFacts:
    def test_canonical_json_with_all_fact_keys(self) -> None:
        import json

        facts = execution_facts({"PATH": "/bin"}, "/workspace", "unconfined")

        parsed = json.loads(facts)
        assert parsed["cwd"] == "/workspace"
        assert parsed["sandbox"] == "unconfined"
        assert parsed["shell"] == resolve_shell()
        assert parsed["env_digest"] == env_digest({"PATH": "/bin"})
        assert facts == execution_facts({"PATH": "/bin"}, "/workspace", "unconfined")

    def test_sensitive_to_sandbox_label_and_cwd(self) -> None:
        base = execution_facts({}, "/w", "unconfined")

        assert base != execution_facts({}, "/w", "bwrap")
        assert base != execution_facts({}, "/other", "unconfined")


class TestClassifyCommand:
    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("echo ok", frozenset()),
            ("echo $(id)", frozenset({"substitution"})),
            ("echo `id`", frozenset({"substitution"})),
            ("echo hi > out.txt", frozenset({"redirection"})),
            ("echo hi >> out.txt", frozenset({"redirection"})),
            ("cat < in.txt", frozenset({"redirection"})),
            ("ls | grep x", frozenset({"pipe"})),
            ("ls && pwd", frozenset({"chain"})),
            ("ls || pwd", frozenset({"chain"})),
            ("ls; pwd", frozenset({"chain"})),
            ("sleep 5 &", frozenset({"background"})),
            (
                "cat $(ls) > out | grep x && true &",
                frozenset({"substitution", "redirection", "pipe", "chain", "background"}),
            ),
        ],
    )
    def test_feature_flags(self, command: str, expected: frozenset[str]) -> None:
        assert classify_command(command) == expected


class TestDetectSandbox:
    def test_label_is_a_known_value(self) -> None:
        assert detect_sandbox() in {"bwrap", "firejail", "unconfined"}

    def test_unconfined_without_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)

        assert detect_sandbox() == "unconfined"

    def test_bwrap_preferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "shutil.which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None
        )

        assert detect_sandbox() == "bwrap"


class TestRunProcess:
    def test_captures_stdout_and_exit_code(self, tmp_path: Path) -> None:
        outcome = run_process(
            "printf hello",
            cwd=str(tmp_path),
            env={},
            timeout_seconds=10,
            max_output_chars=1000,
        )

        assert outcome.exit_code == 0
        assert outcome.stdout == "hello"
        assert outcome.stderr == ""
        assert outcome.timed_out is False
        assert outcome.stdout_removed_bytes == 0

    def test_non_zero_exit_is_captured(self, tmp_path: Path) -> None:
        outcome = run_process(
            "exit 3", cwd=str(tmp_path), env={}, timeout_seconds=10, max_output_chars=1000
        )

        assert outcome.exit_code == 3
        assert outcome.timed_out is False

    def test_captures_stderr(self, tmp_path: Path) -> None:
        outcome = run_process(
            "echo oops >&2",
            cwd=str(tmp_path),
            env={},
            timeout_seconds=10,
            max_output_chars=1000,
        )

        assert outcome.exit_code == 0
        assert outcome.stderr.strip() == "oops"

    def test_runs_in_given_cwd(self, tmp_path: Path) -> None:
        outcome = run_process(
            "pwd", cwd=str(tmp_path), env={}, timeout_seconds=10, max_output_chars=4096
        )

        assert os.path.realpath(outcome.stdout.strip()) == os.path.realpath(tmp_path)

    def test_environment_is_exactly_the_given_mapping(self, tmp_path: Path) -> None:
        source = {"PATH": "/usr/bin:/bin", _SENTINEL_ENV_KEY: "sentinel-value"}
        env = build_process_env(source)
        assert _SENTINEL_ENV_KEY not in env

        outcome = run_process(
            "env", cwd=str(tmp_path), env=env, timeout_seconds=10, max_output_chars=4096
        )

        assert outcome.exit_code == 0
        assert _SENTINEL_ENV_KEY not in outcome.stdout
        assert "sentinel-value" not in outcome.stdout

    def test_timeout_kills_whole_process_group(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "child.pid"

        started = time.monotonic()
        outcome = run_process(
            "sleep 60 & echo $! > child.pid; wait",
            cwd=str(tmp_path),
            env={},
            timeout_seconds=1,
            max_output_chars=1000,
        )
        elapsed = time.monotonic() - started

        assert outcome.timed_out is True
        assert outcome.exit_code is None
        assert elapsed < 10
        _assert_pid_dead(_child_pid(pid_file))

    def test_truncation_records_removed_bytes(self, tmp_path: Path) -> None:
        env = build_process_env({"PATH": os.environ.get("PATH", "/usr/bin:/bin")})

        outcome = run_process(
            "head -c 100000 /dev/zero | tr '\\0' a",
            cwd=str(tmp_path),
            env=env,
            timeout_seconds=30,
            max_output_chars=100,
        )

        assert outcome.exit_code == 0
        assert len(outcome.stdout.encode()) <= 100 * 4
        assert outcome.stdout_removed_bytes > 0
        retained = len(outcome.stdout.encode())
        assert retained + outcome.stdout_removed_bytes == 100_000

    def test_invalid_utf8_is_replaced_not_raised(self, tmp_path: Path) -> None:
        outcome = run_process(
            "printf '\\377\\376'",
            cwd=str(tmp_path),
            env={},
            timeout_seconds=10,
            max_output_chars=1000,
        )

        assert outcome.exit_code == 0
        assert outcome.stdout == "��"

    def test_cancellation_kills_group_and_reraises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pid_file = tmp_path / "shell.pid"
        original_wait = subprocess.Popen.wait
        calls = 0

        def interrupting_wait(self: subprocess.Popen[bytes], timeout: object = None) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                # Give the shell a deterministic chance to write its pid,
                # then simulate Ctrl+C arriving mid-wait.
                _child_pid(pid_file)
                raise KeyboardInterrupt
            return original_wait(self, timeout)

        monkeypatch.setattr(subprocess.Popen, "wait", interrupting_wait)

        with pytest.raises(KeyboardInterrupt):
            run_process(
                "echo $$ > shell.pid; sleep 60",
                cwd=str(tmp_path),
                env={},
                timeout_seconds=60,
                max_output_chars=1000,
            )

        _assert_pid_dead(_child_pid(pid_file))
