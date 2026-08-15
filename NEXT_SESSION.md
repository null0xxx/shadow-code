# Project Status

Shadow Code is feature-complete against docs/architecture-roadmap.md (14/14 work units merged)
plus the post-roadmap polish and visual identity line. This file is updated at the end of each
working session.

## Current state (2026-08-15)

- **main**: roadmap 14/14 + render polish (#39–#47) + packaging (#49) + multi-file enablement
  (#51) + retry-loop steering (#53) + thinking mode (#55) + visual identity (#57) +
  block-letter banner (#59). 1057 tests + 15 subtests green; CI on Python 3.10/3.11/3.12.
- **Install**: `./scripts/install.sh` (never with sudo) → `~/.local/bin/shadow-code`.
  Uninstall preserves data; `--purge` asks first.
- **Run**: `shadow-code` (default model gemma4-cline:32k). TUI: `SHADOW_TUI=1`.
  Thinking mode: `SHADOW_THINK=1` (requires a think-capable model).
- **Eval**: `python -m shadow_code.eval --model gemma4-cline:32k` (corpus v2, 16 scenarios).

## Gates

`pytest tests -q` · `ruff check shadow_code tests` · `ruff format --check shadow_code tests` ·
`mypy shadow_code` · `bandit -c pyproject.toml -r shadow_code/ -q`
(also run pytest under `env -u TERM` — CI runners have no TERM; render tests pin TERM=dumb)

## Known model limitations (measured, not bugs)

- duplicate_loop: the small model sometimes repeats a denied/errored call until the
  duplicate guard terminates the turn. 10-run measurement: steering stays within noise.
- strict-patch-export: fails under every prompt — export mode leaves the target unchanged
  BY DESIGN; the model verifies with a read, sees old content, and retries (live-reproduced,
  3/3 single-scenario). Stable capability limit of gemma4-cline:32k.

## Housekeeping notes

- Live eval runs create `/tmp/shadow-eval-*` dirs that trip
  `test_eval_runner::test_run_scenario_without_base_dir_uses_disposable_workspace` —
  clean them before running pytest; never run pytest parallel to a live eval.
- The installed launcher and repo `.venv` are non-editable builds; verify unmerged
  code via `.venv/bin/python -m shadow_code.main` from the source tree.
- Terminal visual verification: capture with `script -qec` (pty), render ANSI → PNG with
  /tmp/render_term.py (pyte + PIL), view with an image reader.
- `.atl/` is untracked scratch; leave it.

## Open decisions (user-owned)

- Remembered approvals ("approve all writes this session"): NOT implemented — the one-shot
  approval is the final security boundary against prompt injection. Requires an explicit
  product decision with a threat-model review before any work starts.
