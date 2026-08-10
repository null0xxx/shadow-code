"""Integration: SHADOW_TUI main()-driven headless TUI sessions (WU-09/WU-10).

Drives the real main() composition root with the TUI forced on, a fake
Ollama client, prompt_toolkit pipe input, and a dummy output -- no real
terminal, no real Ollama. Covers: a streamed text turn reaching the
transcript, approvals answered through the single-focus widget keys, a
full multi-call tool lifecycle (read + approved write + denied bash), and
a provider failure mid-turn with the event store left consistent.
"""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

import shadow_code.main as main_module
import shadow_code.tui as tui_module
from shadow_code.events import EventStore
from shadow_code.policy.workspace import WorkspaceGuard
from shadow_code.tui import TuiApp, TuiTheme

_ASCII_THEME = TuiTheme(colors=False, ascii_mode=True)


class _FakeClient:
    """Scripted Ollama client: rounds are (chunks, calls) or exceptions."""

    def __init__(self, rounds):
        self._rounds = iter(rounds)
        self.last_prompt_tokens = 10
        self.last_eval_tokens = 5
        self.last_tool_calls: list[dict] = []

    def health_check(self):
        return True, "OK"

    def chat_stream(self, messages, system, model=None, tools=None):
        item = next(self._rounds)
        if isinstance(item, BaseException):
            raise item
        chunks, self.last_tool_calls = item
        yield from chunks


def _wait_for(predicate, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _run_tui_session(test, inputs, rounds, workspace: str, *, approve_script=(), enable_db=False):
    """Run main() headless through the TUI; return (app, event store).

    A controller thread answers each approval with the scripted key
    ("y"/"n") once the single-focus widget is armed, then sends /exit only
    after the turn finished -- approval focus would otherwise swallow the
    typed characters.

    enable_db=True keeps the real legacy Database (patched to a workspace
    path) so thread-affinity regressions surface instead of being hidden.
    """
    client = _FakeClient(rounds)
    events_db = Path(workspace) / "events.db"
    sessions_db = Path(workspace) / "sessions.db"
    apps: list[TuiApp] = []
    original_run = TuiApp.run

    def spy_run(self):
        apps.append(self)
        return original_run(self)

    state_home = Path(workspace) / "state"
    config_home = Path(workspace) / "config"
    state_home.mkdir()
    config_home.mkdir()

    db_patches = (
        patch.object(main_module, "default_db_path", return_value=str(sessions_db))
        if enable_db
        else patch.object(main_module, "_HAS_DB", False)
    )

    with (
        patch.object(main_module, "_want_tui", return_value=True),
        patch.object(main_module, "_RICH", False),
        patch.object(main_module, "_HAS_REPL", False),
        db_patches,
        patch.object(main_module, "OllamaClient", return_value=client),
        patch.object(
            main_module,
            "WorkspaceGuard",
            side_effect=lambda root, **kw: WorkspaceGuard(workspace),
        ),
        patch.object(main_module, "_register_optional_tools"),
        patch.object(main_module.tool_reg, "register"),
        patch.object(main_module, "default_store_dir", return_value=Path(workspace) / "prompts"),
        patch.object(main_module, "default_events_db_path", return_value=events_db),
        patch.object(
            tui_module.TuiTheme,
            "from_env",
            classmethod(lambda cls, env=None: _ASCII_THEME),
        ),
        patch.object(TuiApp, "run", spy_run),
        patch.dict(
            os.environ,
            {"XDG_STATE_HOME": str(state_home), "XDG_CONFIG_HOME": str(config_home)},
        ),
        create_pipe_input() as pipe,
    ):
        for text in inputs:
            pipe.send_text(text + "\r")

        def controller() -> None:
            # _approval_count is monotonic: waiting for the count (not for a
            # pending edge) cannot miss a fast resolve-then-rearm sequence
            # between two sequential approvals in one batch.
            for index, answer in enumerate(approve_script, start=1):

                def armed(index: int = index) -> bool:
                    return bool(
                        apps and apps[0]._approval_count >= index and apps[0]._approval_pending
                    )

                if not _wait_for(armed):
                    return
                pipe.send_text(answer)
            # FIFO: /exit queues after the user message and is dispatched
            # once the turn ends; the pipe processes the final answer key
            # (which restores input focus) before these characters.
            pipe.send_text("/exit\r")

        threading.Thread(target=controller, daemon=True).start()
        errors: list[BaseException] = []

        def target() -> None:
            try:
                main_module.main(tui_input=pipe, tui_output=DummyOutput())
            except BaseException as error:  # surfaced below
                errors.append(error)

        driver = threading.Thread(target=target, daemon=True)
        driver.start()
        driver.join(timeout=90)
        test.assertFalse(driver.is_alive(), "TUI session did not exit cleanly")
        if errors:
            raise errors[0]

    test.assertEqual(len(apps), 1)
    return apps[0], events_db


def _event_types(events_db: Path, test: unittest.TestCase) -> list[str]:
    store = EventStore(events_db)
    try:
        session_id = store.latest_session_id()
        test.assertIsNotNone(session_id)
        issues = store.verify(session_id)
        test.assertEqual(issues, [])
        return [event.type for event in store.events_for(session_id)]
    finally:
        store.close()


class TuiIntegrationCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = self._tmp.name

    def test_streamed_text_turn_reaches_transcript(self):
        app, events_db = _run_tui_session(
            self,
            ["hello there"],
            [(["Hel", "lo", " world"], [])],
            self.workspace,
        )
        rendered = app.model.render(_ASCII_THEME)
        self.assertIn("> hello there", rendered)
        self.assertIn("Hello world", rendered)
        types = _event_types(events_db, self)
        for expected in (
            "session_started",
            "user_message",
            "assistant_text",
            "turn_completed",
            "session_ended",
        ):
            self.assertIn(expected, types)

    def test_approval_round_trip_through_widget(self):
        bash_call = {
            "call_id": "c1",
            "name": "bash",
            "arguments": {"command": "echo wu09-approval"},
        }
        app, events_db = _run_tui_session(
            self,
            ["run echo"],
            [(["I will run it."], [bash_call]), (["done"], [])],
            self.workspace,
            approve_script=("y",),
        )
        rendered = app.model.render(_ASCII_THEME)
        self.assertEqual(app._approval_count, 1)
        self.assertIn("step 1", rendered)
        self.assertIn("$ echo wu09-approval", rendered)
        self.assertIn("ok", rendered)
        self.assertIn("done", rendered)
        types = _event_types(events_db, self)
        for expected in ("approval_requested", "approval_granted", "tool_result"):
            self.assertIn(expected, types)

    def test_tool_lifecycle_multi_call_turn(self):
        """One round: read (allowed) + write (approved) + bash (denied)."""
        (Path(self.workspace) / "target.txt").write_text("original\n", encoding="utf-8")
        read_call = {
            "call_id": "r1",
            "name": "read_file",
            "arguments": {"file_path": "target.txt"},
        }
        write_call = {
            "call_id": "w1",
            "name": "write_file",
            "arguments": {"file_path": "target.txt", "content": "changed by wu10\n"},
        }
        bash_call = {
            "call_id": "b1",
            "name": "bash",
            "arguments": {"command": "touch bash-ran.txt"},
        }
        app, events_db = _run_tui_session(
            self,
            ["inspect and change things"],
            [
                (["I will inspect and change things."], [read_call, write_call, bash_call]),
                (["all done"], []),
            ],
            self.workspace,
            approve_script=("y", "n"),  # write approved, bash denied
        )
        rendered = app.model.render(_ASCII_THEME)

        # Fresh single-focus control per side-effecting call.
        self.assertEqual(app._approval_count, 2)

        # Full lifecycle visible in one grouped round.
        self.assertIn("step 1", rendered)
        self.assertIn("read_file", rendered)
        self.assertIn("target.txt", rendered)
        self.assertIn("write_file", rendered)
        self.assertIn("status: executed", rendered)
        self.assertIn("$ touch bash-ran.txt", rendered)
        self.assertIn("denied — not retried", rendered)
        self.assertIn("hint: denied by user — not retried", rendered)
        self.assertIn("all done", rendered)

        # Evidence on disk: the approved write landed; the denied bash did not.
        target = Path(self.workspace) / "target.txt"
        self.assertEqual(target.read_text(encoding="utf-8"), "changed by wu10\n")
        self.assertFalse((Path(self.workspace) / "bash-ran.txt").exists())

        types = _event_types(events_db, self)
        self.assertEqual(types.count("approval_requested"), 2)
        self.assertIn("approval_granted", types)
        self.assertIn("approval_denied", types)

    def test_provider_failure_mid_turn_keeps_store_consistent(self):
        app, events_db = _run_tui_session(
            self,
            ["boom"],
            [RuntimeError("ollama down")],
            self.workspace,
        )
        rendered = app.model.render(_ASCII_THEME)
        self.assertIn("[Error:", rendered)
        self.assertIn("ollama down", rendered)
        types = _event_types(events_db, self)
        self.assertIn("session_started", types)
        self.assertIn("user_message", types)
        self.assertIn("session_ended", types)

    def test_turn_with_legacy_db_enabled_has_no_thread_errors(self):
        """Regression: legacy Database is thread-affine like the event store.

        The TUI must swap in a worker-thread connection; before the fix the
        turn failed with "Failed to add message: SQLite objects created in
        a thread can only be used in that same thread".
        """
        app, _ = _run_tui_session(
            self,
            ["hello db"],
            [(["persisted", " reply"], [])],
            self.workspace,
            enable_db=True,
        )
        rendered = app.model.render(_ASCII_THEME)
        self.assertNotIn("Failed to add message", rendered)
        self.assertIn("persisted reply", rendered)

        from shadow_code.db import Database

        check = Database(str(Path(self.workspace) / "sessions.db"))
        try:
            sessions = check.list_sessions()
            self.assertEqual(len(sessions), 1)
            messages = check.get_session(sessions[0]["id"])["messages"]
            roles = [row["role"] for row in messages]
            self.assertEqual(roles, ["user", "assistant"])
        finally:
            check.close()


if __name__ == "__main__":
    unittest.main()
