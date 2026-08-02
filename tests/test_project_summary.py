"""Behavior tests for the legacy read-only project summary tool.

These tests exercise only temporary project trees. They document discovery
behavior, not workspace-containment guarantees for the legacy tool.
"""

import json
from pathlib import Path

from shadow_code.tool_context import ToolContext
from shadow_code.tools.project_summary import ProjectSummaryTool


def test_project_summary_detects_structure_dependencies_tests_and_git(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\ndependencies = ["requests>=2"]\n', encoding="utf-8"
    )
    (project / "package.json").write_text(
        json.dumps({"dependencies": {"react": "1", "vite": "2"}}), encoding="utf-8"
    )
    (project / "Makefile").write_text("test:\n\tpytest\n", encoding="utf-8")
    (project / "next.config.ts").write_text("export default {}\n", encoding="utf-8")
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (project / "tests").mkdir()
    (project / "tests" / "test_app.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text("ref: refs/heads/feature/demo\n", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "ignored.js").write_text("ignored\n", encoding="utf-8")

    result = ProjectSummaryTool(ToolContext(str(tmp_path))).execute({"path": "demo"})

    assert result.success is True
    assert "Project: demo" in result.output
    assert "Language: JavaScript/TypeScript, Python" in result.output
    assert "Framework: Make, Next.js" in result.output
    assert "dependencies =" in result.output
    assert "src/main.py" in result.output
    assert "tests/ directory (1 test files)" in result.output
    assert "branch=feature/demo" in result.output
    assert "node_modules" not in result.output


def test_project_summary_uses_package_dependencies_and_default_root(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps({"dependencies": {f"dep-{index}": "1" for index in range(12)}}),
        encoding="utf-8",
    )
    (tmp_path / "index.js").write_text("console.log('ok')\n", encoding="utf-8")

    result = ProjectSummaryTool(ToolContext(str(tmp_path))).execute({})

    assert result.success is True
    assert "dep-0" in result.output
    assert "dep-9" in result.output
    assert "dep-10" not in result.output
    assert "Entry points: index.js" in result.output


def test_project_summary_handles_invalid_and_unrecognized_directories(tmp_path: Path) -> None:
    tool = ProjectSummaryTool(ToolContext(str(tmp_path)))

    missing = tool.execute({"path": "missing"})
    empty = tmp_path / "empty"
    empty.mkdir()
    unrecognized = tool.execute({"path": str(empty)})

    assert missing.success is False
    assert "Not a directory" in missing.output
    assert unrecognized.success is True
    assert "No project structure detected" in unrecognized.output


def test_project_summary_tolerates_malformed_metadata_and_detached_git(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{broken", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("0123456789abcdef\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")

    result = ProjectSummaryTool(ToolContext(str(tmp_path))).execute({})

    assert result.success is True
    assert "Dependencies:" not in result.output
    assert "Git: git repo" in result.output
