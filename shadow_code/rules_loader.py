"""Dynamic loader for EAS language rules.

Rules live in ~/.copilot/rules/ as part of the Elite Agent System (EAS).
Each rule file is 2-3KB of language-specific guidance (style, security delta,
patterns, testing).

Strategy: opt-in loading via the get_language_rules tool, NOT auto-injection.
Why?
  - Auto-injection on every edit_file/write_file call would bloat context
  - The model decides when it actually needs the rules (e.g. before non-trivial code)
  - Cached per session = KV cache hits after first lookup
  - User can also ask for rules explicitly ("show me Python rules")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Resolve rules root: ~/.copilot/rules/ — produced by EAS install.
RULES_ROOT = Path.home() / ".copilot" / "rules"

# Map file extension (lowercase, with leading dot) -> rule name (basename without .md)
EXTENSION_TO_RULE: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # TypeScript / JavaScript
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".mjs": "typescript",
    ".cjs": "typescript",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Java
    ".java": "java",
    # C#
    ".cs": "csharp",
    # C / C++
    ".c": "cpp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",
    # PHP
    ".php": "php",
    # Shell
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}

# Names accepted as direct lookup keys (in addition to extensions).
KNOWN_RULES: set[str] = {
    "common-security",
    "python",
    "typescript",
    "go",
    "rust",
    "java",
    "csharp",
    "cpp",
    "kotlin",
    "php",
    "shell",
}


def rule_for_extension(ext: str) -> str | None:
    """Return canonical rule name for a file extension, or None if unknown."""
    if not ext:
        return None
    if not ext.startswith("."):
        ext = "." + ext
    return EXTENSION_TO_RULE.get(ext.lower())


def rule_for_filename(filename: str) -> str | None:
    """Return rule name based on a filename's extension."""
    suffix = Path(filename).suffix
    return rule_for_extension(suffix) if suffix else None


@lru_cache(maxsize=32)
def load_rule_full(name: str) -> str | None:
    """Read the full rule markdown file from ~/.copilot/rules/<name>.md.

    Returns None if the file doesn't exist or name is unknown.
    Cached for the lifetime of the process.
    """
    name = name.strip().lower()
    if name not in KNOWN_RULES:
        return None
    path = RULES_ROOT / f"{name}.md"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


@lru_cache(maxsize=32)
def load_rule_summary(name: str, max_chars: int = 1500) -> str | None:
    """Return a compact summary of a rule (~1.5KB).

    Strategy:
      1. Take the first paragraph (until the first '---' or empty section).
      2. Append all top-level bullet points (lines starting with '- ').
      3. Truncate to max_chars with a clear marker.

    For most rules this captures the essentials: heading, scope, and the
    actionable bullet list. Detailed prose and examples are omitted.
    """
    full = load_rule_full(name)
    if full is None:
        return None

    # Skip front-matter (everything up to and including the first '---' separator)
    lines = full.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            body_start = i + 1
            break

    out: list[str] = []
    in_code_block = False
    total = 0

    for line in lines[body_start:]:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            # Skip code examples in summary mode (keep summary terse)
            continue
        if in_code_block:
            continue
        # Keep headings, bullets, blockquotes, and short prose paragraphs
        if (
            stripped.startswith("#")
            or stripped.startswith("- ")
            or stripped.startswith("* ")
            or stripped.startswith("> ")
            or stripped == ""
            or len(stripped) <= 200
        ):
            out.append(line)
            total += len(line) + 1

        if total >= max_chars:
            break

    summary = "\n".join(out).rstrip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + "\n\n[...truncated; ask for full rule if needed]"
    return summary


def list_available_rules() -> list[str]:
    """Return the sorted list of rule names that are currently installed."""
    if not RULES_ROOT.is_dir():
        return []
    rules = []
    for path in sorted(RULES_ROOT.glob("*.md")):
        rules.append(path.stem)
    return rules


def is_rules_root_available() -> bool:
    """Whether the EAS rules directory is present on this machine."""
    return RULES_ROOT.is_dir()
