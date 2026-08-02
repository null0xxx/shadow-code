"""get_language_rules tool — fetch EAS language-specific rules on demand.

The model calls this tool when it is about to write/edit code in a specific
language and wants to align with project-wide engineering standards.

Two query modes:
  - by extension: {"extension": ".py"}
  - by name:      {"name": "python"} or {"name": "common-security"}

Returns a compact summary (~1.5KB) by default, or the full rule if
{"full": true} is passed.
"""

from __future__ import annotations

from ..rules_loader import (
    KNOWN_RULES,
    is_rules_root_available,
    list_available_rules,
    load_rule_full,
    load_rule_summary,
    rule_for_extension,
)
from .base import BaseTool, ToolResult


class GetLanguageRulesTool(BaseTool):
    name = "get_language_rules"

    def __init__(self, ctx):  # ctx: ToolContext (unused but kept for symmetry)
        self.ctx = ctx

    def validate(self, params: dict) -> str | None:
        if "extension" not in params and "name" not in params:
            return (
                "Provide either 'extension' (e.g. '.py') or 'name' "
                "(e.g. 'python', 'common-security')"
            )
        if "extension" in params and not isinstance(params["extension"], str):
            return "'extension' must be a string"
        if "name" in params and not isinstance(params["name"], str):
            return "'name' must be a string"
        if "full" in params and not isinstance(params["full"], bool):
            return "'full' must be a boolean"
        return None

    def execute(self, params: dict) -> ToolResult:
        if not is_rules_root_available():
            return ToolResult(
                False,
                "EAS rules directory ~/.copilot/rules/ is not installed on this machine.",
            )

        full_mode: bool = bool(params.get("full", False))
        rule_name: str | None = None

        if "name" in params:
            rule_name = params["name"].strip().lower()
            if rule_name not in KNOWN_RULES:
                avail = ", ".join(sorted(list_available_rules()))
                return ToolResult(False, f"Unknown rule '{rule_name}'. Available: {avail}")
        elif "extension" in params:
            ext = params["extension"]
            rule_name = rule_for_extension(ext)
            if rule_name is None:
                return ToolResult(
                    False,
                    (
                        f"No language rule mapped for extension '{ext}'. "
                        f"Try one of: {', '.join(sorted(list_available_rules()))}"
                    ),
                )

        if rule_name is None:
            return ToolResult(False, "Provide either 'extension' or 'name'.")

        loader = load_rule_full if full_mode else load_rule_summary
        content = loader(rule_name)
        if content is None:
            return ToolResult(False, f"Rule '{rule_name}' could not be loaded from disk.")

        header = f"=== EAS Rule: {rule_name} ({'full' if full_mode else 'summary'}) ===\n"
        return ToolResult(True, header + content)
