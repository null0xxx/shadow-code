"""Tests for opt-in language-rule loading from an isolated rules root."""

from pathlib import Path

import pytest

from shadow_code import rules_loader
from shadow_code.tool_context import ToolContext
from shadow_code.tools.get_language_rules import GetLanguageRulesTool


@pytest.fixture(autouse=True)
def isolated_rules_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "rules"
    monkeypatch.setattr(rules_loader, "RULES_ROOT", root)
    rules_loader.load_rule_full.cache_clear()
    rules_loader.load_rule_summary.cache_clear()
    yield root
    rules_loader.load_rule_full.cache_clear()
    rules_loader.load_rule_summary.cache_clear()


def test_rule_name_mapping_is_case_insensitive() -> None:
    assert rules_loader.rule_for_extension("PY") == "python"
    assert rules_loader.rule_for_extension(".TsX") == "typescript"
    assert rules_loader.rule_for_filename("src/main.go") == "go"
    assert rules_loader.rule_for_filename("Makefile") is None
    assert rules_loader.rule_for_extension("") is None
    assert rules_loader.rule_for_extension("unknown") is None


def test_full_and_summary_loading_are_cached_and_structured(
    isolated_rules_root: Path,
) -> None:
    isolated_rules_root.mkdir()
    content = """---
title: Python
---
# Python

Short guidance.
- Validate inputs
```python
dangerous_example()
```
> Keep failures visible.
"""
    (isolated_rules_root / "python.md").write_text(content, encoding="utf-8")

    assert rules_loader.is_rules_root_available() is True
    assert rules_loader.list_available_rules() == ["python"]
    assert rules_loader.load_rule_full(" PYTHON ") == content
    summary = rules_loader.load_rule_summary("python")
    assert summary is not None
    assert "# Python" in summary
    assert "Validate inputs" in summary
    assert "dangerous_example" not in summary
    assert "Keep failures visible" in summary

    (isolated_rules_root / "python.md").write_text("changed", encoding="utf-8")
    assert rules_loader.load_rule_full("python") == content


def test_summary_truncates_with_visible_marker(isolated_rules_root: Path) -> None:
    isolated_rules_root.mkdir()
    (isolated_rules_root / "python.md").write_text(
        "# Python\n" + "- actionable guidance\n" * 20, encoding="utf-8"
    )

    summary = rules_loader.load_rule_summary("python", max_chars=60)

    assert summary is not None
    assert summary.endswith("[...truncated; ask for full rule if needed]")
    assert len(summary) > 60


def test_missing_unknown_and_unreadable_rules_return_none(
    isolated_rules_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert rules_loader.list_available_rules() == []
    assert rules_loader.load_rule_full("unknown") is None
    assert rules_loader.load_rule_full("python") is None

    isolated_rules_root.mkdir()
    rule = isolated_rules_root / "python.md"
    rule.write_text("# Python", encoding="utf-8")
    rules_loader.load_rule_full.cache_clear()
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    assert rules_loader.load_rule_full("python") is None


def test_language_rules_tool_validates_and_loads_both_modes(
    isolated_rules_root: Path,
) -> None:
    isolated_rules_root.mkdir()
    (isolated_rules_root / "python.md").write_text("# Python\n- Test behavior\n", encoding="utf-8")
    tool = GetLanguageRulesTool(ToolContext(str(isolated_rules_root.parent)))

    assert tool.validate({}) is not None
    assert tool.validate({"extension": 1}) == "'extension' must be a string"
    assert tool.validate({"name": 1}) == "'name' must be a string"
    assert tool.validate({"name": "python", "full": "yes"}) == "'full' must be a boolean"
    assert tool.validate({"extension": ".py"}) is None

    summary = tool.execute({"extension": ".py"})
    full = tool.execute({"name": "python", "full": True})
    unknown = tool.execute({"name": "unknown"})
    unmapped = tool.execute({"extension": ".wat"})

    assert summary.success is True
    assert "(summary)" in summary.output
    assert full.success is True
    assert "(full)" in full.output
    assert unknown.success is False
    assert "Unknown rule" in unknown.output
    assert unmapped.success is False
    assert "No language rule mapped" in unmapped.output


def test_language_rules_tool_reports_missing_install_and_missing_file(
    isolated_rules_root: Path,
) -> None:
    tool = GetLanguageRulesTool(ToolContext(str(isolated_rules_root.parent)))
    unavailable = tool.execute({"name": "python"})
    isolated_rules_root.mkdir()
    missing = tool.execute({"name": "python"})

    assert unavailable.success is False
    assert "not installed" in unavailable.output
    assert missing.success is False
    assert "could not be loaded" in missing.output
