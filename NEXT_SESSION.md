# Project Status

Shadow Code is feature-complete against docs/architecture-roadmap.md (14/14 work units merged)
plus the post-roadmap polish line. This file is updated at the end of each working session.

## Current state (2026-08-15)

- **main**: all roadmap units + render polish (#39–#47), packaging fix (#49),
  multi-file enablement (#51). 991 tests + 10 subtests green; CI on Python 3.10/3.11/3.12.
- **Install**: `./scripts/install.sh` (never with sudo) → `~/.local/bin/shadow-code`.
  Uninstall preserves data; `--purge` asks first.
- **Run**: `shadow-code` (default model gemma4-cline:32k). TUI: `SHADOW_TUI=1`.
- **Eval**: `python -m shadow_code.eval --model gemma4-cline:32k` (corpus v2, 16 scenarios).

## Gates

`pytest tests -q` · `ruff check shadow_code tests` · `ruff format --check shadow_code tests` ·
`mypy shadow_code` · `bandit -c pyproject.toml -r shadow_code/ -q`
(also run pytest under `env -u TERM` — CI runners have no TERM; render tests pin TERM=dumb)

## Known model limitations (measured, not bugs)

- duplicate_loop: the small model sometimes repeats a denied/errored call until the
  duplicate guard terminates the turn. Which scenario it lands on varies per run
  (safety pass rate fluctuates 62–88%).
- strict-patch-export fails under every prompt tested (stable model limit).

## Housekeeping notes

- Live eval runs create `/tmp/shadow-eval-*` dirs that trip
  `test_eval_runner::test_run_scenario_without_base_dir_uses_disposable_workspace` —
  clean them before running pytest.
- The installed launcher and repo `.venv` are non-editable builds; verify unmerged
  code via `.venv/bin/python -m shadow_code.main` from the source tree.
- `.atl/` is untracked scratch; leave it.
