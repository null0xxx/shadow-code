"""Tests for prompt.py -- system prompt integrity checks."""

import unittest

import shadow_code.prompt as prompt_module

SYSTEM_PROMPT = prompt_module.SYSTEM_PROMPT


def _render_system_prompt(**options: bool) -> str:
    renderer = getattr(prompt_module, "render_system_prompt", None)
    assert callable(renderer), "runtime prompt selection is not implemented"
    return renderer(**options)


class TestSystemPrompt(unittest.TestCase):
    """Test the static SYSTEM_PROMPT constant."""

    def test_is_string(self):
        self.assertIsInstance(SYSTEM_PROMPT, str)

    def test_not_empty(self):
        self.assertGreater(len(SYSTEM_PROMPT), 500)

    # --- Required sections ---

    def test_has_doing_tasks(self):
        self.assertIn("# Doing Tasks", SYSTEM_PROMPT)

    def test_has_safety_section(self):
        self.assertIn("# Safety", SYSTEM_PROMPT)

    def test_has_output_quality(self):
        self.assertIn("Output Quality", SYSTEM_PROMPT)

    def test_has_common_mistakes(self):
        self.assertIn("Common Mistakes", SYSTEM_PROMPT)

    def test_has_tool_calling_format(self):
        self.assertIn("Tool Calling Format", SYSTEM_PROMPT)

    # --- Language rule ---

    def test_has_language_rule(self):
        self.assertIn("same language", SYSTEM_PROMPT.lower())

    def test_has_georgian_rule(self):
        self.assertIn("Georgian", SYSTEM_PROMPT)

    # --- Safety rules ---

    def test_has_git_safety(self):
        self.assertIn("--no-verify", SYSTEM_PROMPT)
        self.assertIn("destructive", SYSTEM_PROMPT.lower())

    def test_no_dynamic_content(self):
        """System prompt must be 100% static."""
        self.assertNotIn("{os.", SYSTEM_PROMPT)
        self.assertNotIn("{datetime", SYSTEM_PROMPT)
        self.assertNotIn("{platform", SYSTEM_PROMPT)
        self.assertNotIn("{sys.", SYSTEM_PROMPT)

    # --- Code quality ---

    def test_has_complete_code_instruction(self):
        self.assertIn("COMPLETE", SYSTEM_PROMPT)
        self.assertIn("PRODUCTION-READY", SYSTEM_PROMPT)

    def test_has_edit_file_warning(self):
        self.assertIn("old_string", SYSTEM_PROMPT)
        self.assertIn("line numbers", SYSTEM_PROMPT.lower())

    # --- Tool descriptions NOT in prompt (moved to native API) ---

    def test_has_tool_call_format_markdown(self):
        """Gemma 3 uses markdown tool_call format in prompt."""
        self.assertIn("```tool_call", SYSTEM_PROMPT)

    def test_tool_schemas_exist(self):
        """Verify TOOL_SCHEMAS is defined in ollama_client."""
        from shadow_code.ollama_client import TOOL_SCHEMAS

        self.assertIsInstance(TOOL_SCHEMAS, list)
        self.assertGreater(len(TOOL_SCHEMAS), 5)
        names = [t["function"]["name"] for t in TOOL_SCHEMAS]
        self.assertIn("bash", names)
        self.assertIn("read_file", names)
        self.assertIn("edit_file", names)
        self.assertIn("write_file", names)
        self.assertIn("grep", names)

    def test_text_only_prompt_has_no_executable_tool_protocol(self):
        prompt = _render_system_prompt(native_tools=False, legacy_markdown_tools=False)

        self.assertNotIn("```tool_call", prompt)
        self.assertIn("No executable tool protocol is active", prompt)

    def test_native_prompt_uses_only_provider_native_tools(self):
        prompt = _render_system_prompt(native_tools=True, legacy_markdown_tools=False)

        self.assertNotIn("```tool_call", prompt)
        self.assertIn("provider-native tool calls", prompt)

    def test_explicit_legacy_prompt_retains_markdown_protocol(self):
        prompt = _render_system_prompt(native_tools=False, legacy_markdown_tools=True)

        self.assertIn("```tool_call", prompt)


if __name__ == "__main__":
    unittest.main()
