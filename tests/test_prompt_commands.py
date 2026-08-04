"""End-to-end tests for the /prompt command surface and source watch (WU-04)."""

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from shadow_code.prompt_compiler import compile_prompt
from shadow_code.tools.catalog import BASH_SPEC, EDIT_FILE_SPEC, READ_FILE_SPEC, WRITE_FILE_SPEC
from shadow_code.tools.registry import ToolRegistry


def _runtime_registry() -> ToolRegistry:
    return ToolRegistry((READ_FILE_SPEC, WRITE_FILE_SPEC, EDIT_FILE_SPEC, BASH_SPEC))


class PromptCommandCase(unittest.TestCase):
    """Drive main() with a fake client and isolated XDG state/config dirs."""

    def setUp(self) -> None:
        import shadow_code.main as main_module

        self.main_module = main_module
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.state_home = root / "state"
        self.config_home = root / "config"
        self.user_overlay = self.config_home / "shadow-code" / "prompt.md"
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.prompts: list[str] = []
        self.inputs: list[str] = []
        self.on_input = None

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _compiled_with(self, overlay_text: str | None):
        if overlay_text is not None:
            self.user_overlay.parent.mkdir(parents=True, exist_ok=True)
            self.user_overlay.write_text(overlay_text, encoding="utf-8")
        return compile_prompt(
            user_path=self.user_overlay,
            workspace_path=self.workspace / ".shadow-code" / "prompt.md",
            registry=_runtime_registry(),
        )

    def _run_main(self) -> str:
        client = MagicMock()
        client.health_check.return_value = (True, "OK")
        client.last_prompt_tokens = 0
        client.last_eval_tokens = 0
        client.last_tool_calls = []

        def chat_stream(messages, system, model=None, tools=None):
            self.prompts.append(system)
            client.last_tool_calls = []
            return iter(["ok"])

        client.chat_stream.side_effect = chat_stream

        def fake_input(prompt=""):
            if self.on_input is not None:
                self.on_input(len(self.prompts))
            if not self.inputs:
                raise EOFError
            return self.inputs.pop(0)

        main_module = self.main_module
        with (
            patch.object(main_module, "_RICH", False),
            patch.object(main_module, "_HAS_REPL", False),
            patch.object(main_module, "_HAS_DB", False),
            patch.object(main_module, "OllamaClient", return_value=client),
            patch.object(main_module, "_register_optional_tools"),
            patch.object(main_module.tool_reg, "register"),
            patch.object(main_module.signal, "signal"),
            patch("os.getcwd", return_value=str(self.workspace)),
            patch.dict(
                os.environ,
                {
                    "XDG_STATE_HOME": str(self.state_home),
                    "XDG_CONFIG_HOME": str(self.config_home),
                },
            ),
            patch("builtins.input", side_effect=fake_input),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            main_module.main()
        return stdout.getvalue()

    def test_startup_prints_snapshot_and_streams_compiled_prompt(self) -> None:
        expected = self._compiled_with(None)
        self.inputs = ["/exit"]

        output = self._run_main()

        self.assertIn(f"prompt snapshot: {expected.digest[:12]}", output)
        self.assertEqual(self.prompts, [])  # /exit never reaches the model

    def test_prompt_sources_and_validate(self) -> None:
        self._compiled_with("house style rules\n")
        self.inputs = ["/prompt sources", "/prompt validate", "/exit"]

        output = self._run_main()

        self.assertIn("builtin", output)
        self.assertIn("registry", output)
        self.assertIn(str(self.user_overlay), output)
        self.assertIn("prompt OK", output)

    def test_watch_picks_up_overlay_edit_on_next_turn(self) -> None:
        before = self._compiled_with("version one\n")
        expected_after = self._compiled_with("version two\n")
        self._compiled_with("version one\n")  # restore the pre-edit overlay

        def edit_between_turns(turn_count: int) -> None:
            if turn_count == 1:  # first request used the original prompt
                self.user_overlay.write_text("version two\n", encoding="utf-8")

        self.on_input = edit_between_turns
        self.inputs = ["first", "second", "/exit"]

        output = self._run_main()

        self.assertEqual(len(self.prompts), 2)
        self.assertEqual(self.prompts[0], before.compiled_text)
        self.assertEqual(self.prompts[1], expected_after.compiled_text)
        self.assertNotEqual(before.digest, expected_after.digest)
        self.assertIn(
            f"prompt: active {before.digest[:12]} -> {expected_after.digest[:12]}", output
        )

    def test_watch_failure_keeps_previous_active(self) -> None:
        before = self._compiled_with("good\n")

        def break_overlay(turn_count: int) -> None:
            if turn_count == 1:
                self.user_overlay.write_bytes(b"\xff\xfe broken")

        self.on_input = break_overlay
        self.inputs = ["first", "second", "/exit"]

        output = self._run_main()

        self.assertEqual(len(self.prompts), 2)
        self.assertEqual(self.prompts[0], before.compiled_text)
        self.assertEqual(self.prompts[1], before.compiled_text)
        self.assertIn("prompt warning", output)
        self.assertIn(before.digest[:12], output)

    def test_rollback_restores_previous_snapshot_and_audits(self) -> None:
        original = self._compiled_with("original rules\n")
        updated = self._compiled_with("updated rules\n")
        self._compiled_with("original rules\n")  # restore the pre-edit overlay

        def swap_overlay(turn_count: int) -> None:
            if turn_count == 0 and not self._swapped:
                self._swapped = True
                self.user_overlay.write_text("updated rules\n", encoding="utf-8")

        self._swapped = False
        self.on_input = swap_overlay
        self.inputs = [
            "/prompt reload",
            "/prompt history",
            f"/prompt rollback {original.digest[:12]}",
            "hello",
            "/prompt diff",
            "/exit",
        ]

        output = self._run_main()

        self.assertIn(f"prompt: active {original.digest[:12]} -> {updated.digest[:12]}", output)
        self.assertIn(f"prompt: active {updated.digest[:12]} -> {original.digest[:12]}", output)
        # The turn after the rollback uses the restored snapshot.
        self.assertEqual(self.prompts[-1], original.compiled_text)
        self.assertIn("updated rules", output)  # diff shows the delta

    def test_rollback_unknown_digest_fails_and_keeps_active(self) -> None:
        expected = self._compiled_with(None)
        self.inputs = ["/prompt rollback deadbeef", "hello", "/exit"]

        output = self._run_main()

        self.assertIn("failed [snapshot_not_found]", output)
        self.assertEqual(self.prompts[-1], expected.compiled_text)

    def test_prompt_edit_creates_overlay_and_reloads(self) -> None:
        self.inputs = ["/prompt edit", "/exit"]

        with patch.dict(os.environ, {"EDITOR": "true"}):
            output = self._run_main()

        self.assertTrue(self.user_overlay.is_file())
        self.assertIn("user prompt overlay", self.user_overlay.read_text(encoding="utf-8"))
        self.assertIn("prompt snapshot:", output)

    def test_prompt_show_prints_active_text(self) -> None:
        self._compiled_with(None)
        self.inputs = ["/prompt show", "/exit"]

        output = self._run_main()

        self.assertIn("You are Shadow", output)
        self.assertIn("# Available Tools", output)


if __name__ == "__main__":
    unittest.main()
