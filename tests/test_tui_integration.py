"""Integration: SHADOW_TUI main()-driven headless TUI sessions (WU-09).

Drives the real main() composition root with the TUI forced on, a fake
Ollama client, prompt_toolkit pipe input, and a dummy output -- no real
terminal, no real Ollama. Covers: a streamed text turn reaching the
transcript, a full approval round-trip through the input area, and a
provider failure mid-turn with the event store left consistent.
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


def _run_tui_session(test: unittest.TestCase, inputs, rounds, workspace: str, *, approve=False):
    """Run main() headless through the TUI; return (app, event store)."""
    client = _FakeClient(rounds)
    events_db = Path(workspace) / "events.db"
    apps: list[TuiApp] = []
    original_run = TuiApp.run

    def spy_run(self):
        apps.append(self)
        return original_run(self)

    state_home = Path(workspace) / "state"
    config_home = Path(workspace) / "config"
    state_home.mkdir()
    config_home.mkdir()

    with (
        patch.object(main_module, "_want_tui", return_value=True),
        patch.object(main_module, "_RICH", False),
        patch.object(main_module, "_HAS_REPL", False),
        patch.object(main_module, "_HAS_DB", False),
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
        if approve:
            # Answer only after the approval bridge is armed; a "y" sent
            # earlier would be consumed as a regular queued message.
            def answer_when_prompted() -> None:
                for _ in range(6000):
                    if apps and apps[0]._approval_pending:
                        pipe.send_text("y\r")
                        return
                    time.sleep(0.01)

            threading.Thread(target=answer_when_prompted, daemon=True).start()
        errors: list[BaseException] = []

        def target() -> None:
            try:
                main_module.main(tui_input=pipe, tui_output=DummyOutput())
            except BaseException as error:  # surfaced below
                errors.append(error)

        driver = threading.Thread(target=target, daemon=True)
        driver.start()
        driver.join(timeout=60)
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
            ["hello there", "/exit"],
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

    def test_approval_round_trip_through_input_area(self):
        bash_call = {
            "call_id": "c1",
            "name": "bash",
            "arguments": {"command": "echo wu09-approval"},
        }
        app, events_db = _run_tui_session(
            self,
            ["run echo", "/exit"],
            [(["I will run it."], [bash_call]), (["done"], [])],
            self.workspace,
            approve=True,
        )
        rendered = app.model.render(_ASCII_THEME)
        self.assertIn("Action requires approval:", rendered)
        self.assertIn("Approve this exact action? [y/N]", rendered)
        self.assertIn("[Approved]", rendered)
        self.assertIn("[bash] ok", rendered)
        self.assertIn("done", rendered)
        types = _event_types(events_db, self)
        for expected in ("approval_requested", "approval_granted", "tool_result"):
            self.assertIn(expected, types)

    def test_provider_failure_mid_turn_keeps_store_consistent(self):
        app, events_db = _run_tui_session(
            self,
            ["boom", "/exit"],
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


if __name__ == "__main__":
    unittest.main()
