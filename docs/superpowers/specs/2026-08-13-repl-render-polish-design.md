# REPL Render Polish — Design (2026-08-13)

Status: approved by user. Scope: fix the line-REPL rendering gaps found from real user sessions.
Roadmap invariants involved: ANSI neutralization, fence-is-text, one-shot digest-bound approval,
NO_COLOR/ASCII degradation, versioned event payloads, TUI/REPL parity.

## Evidence (from events.db real sessions + code review)

- REPL assistant text prints raw (ui.py `_prefix_text`); ```html fences render as plain text walls.
- REPL approval block is raw `print` (main.py `_request_approval`); the unified-diff preview has no colors.
  TUI already has both (render_markdown_lite, render_diff).
- Model emitted fake ```json {"tool_call": ...}``` fences as text; user saw unexplained JSON (correct
  safety behavior, missing UX notice).
- Model rewrote calc.py 3x; behavior steering belongs in the prompt base layer, measured via eval.
- Pre-existing gap: REPL has NO terminal sanitization at all (StreamDisplay.feed writes raw bytes;
  sanitize_terminal_text exists only in the TUI path and tui_tools imports wcwidth, a [full]-only dep).

## PR plan (isolated rollback boundaries)

### PR 1 — foundation + Part A (this branch: feat/repl-markdown-renderer)
- Extract `sanitize_terminal_text` + its regexes from tui_tools.py into dependency-free
  `shadow_code/terminal_text.py`; re-export from tui_tools (TUI + existing tests keep working).
- REPL final render only: `ui.render_response` returns rich Group(Markdown(sanitized text), token line).
  Streaming preview (`render_streaming*`) stays plain — no per-delta markdown (flicker, O(n^2)).
- Syntax highlighting comes free via rich Markdown (pygments ships with rich). No new dependencies.
- NO_COLOR / non-TTY: rich Console auto-degrades; add tests. Georgian text round-trip test.
- Stored `AssistantTextPayload.content` stays byte-identical (sanitize at render only).

### PR 2 — Part B: colored REPL approval panel
- Rich Panel for `_request_approval` (both call sites: main.py line-REPL + legacy admit path).
- Rich-native 8-line diff classifier in ui.py (add/del/hunk); do NOT import tui_tools.
- `plan.preview` string unchanged (digest-safe); classification at render time only.
- Bash previews (sandbox/features lines) pass through unstyled. Plain fallback byte-identical to today.

### PR 3 — Part C: fake tool-call fence detector
- Pure detector (```json/untagged fence parsing to a dict with "tool_call" or {"tool","params"} shape);
  ```tool_call fences excluded (already handled by the legacy protocol error path).
- Post-turn hook in `_handle_user_message` (serves REPL + TUI). Display-only; never feeds the parser.

### PR 4 — Part D: prompt steering + eval measurement
- Behavioral steering ("finish multi-file tasks with multiple native calls", "prefer edit_file over
  full-file rewrites") in the builtin base layer of prompt.py — NOT in the registry doc generator
  (parity contract pinned by contract tests).
- Declare expected delta first; run WU-11 live harness before/after on gemma4-cline:32k; prompt_digest
  is the provenance link. Corpus v1 untouched. Note: old prompt snapshots become non-rollbackable.

### PR 5 (recommended) — StreamDisplay write-side sanitization
- Sanitize at the StreamDisplay write points (write-side only; never touch full_response).
  Separate PR because it modifies safety-invariant code.

## Explicitly cut (YAGNI)
- Per-delta Markdown re-render; preview-string format changes; mid-stream fence detection;
  engine-level dedup of rewrites (duplicate budget owns it); TUI pygments highlighting (later).

## Gates
pytest (938+ green), ruff check/format, mypy, bandit. Live smoke with gemma4-cline:32k per PR.
