# Reproduce the local development environment

Use the checked-in development lock and Make targets to reproduce WU-00 verification. The
project supports Python 3.10 or newer; this baseline was created and verified with Python 3.14.6.

## Quick setup

From the repository root:

```bash
make setup
```

This creates `.venv`, installs the exact non-editable packages from `requirements-dev.lock`,
and installs Shadow Code as an editable project without re-resolving its dependencies.

To run the steps separately:

```bash
make venv
make dev-deps
make editable
```

## Verification

Run every check and retain a failing exit status if any check fails:

```bash
make verify
```

Run one check at a time when investigating a diagnostic:

| Check | Command |
|---|---|
| Focused tests | `make test-focused` |
| Full tests | `make test` |
| Ruff lint | `make lint` |
| Ruff format check | `make format-check` |
| Mypy | `make mypy` |
| Bandit | `make bandit` |
| Coverage | `make coverage` |

These targets are check-only: they do not format or auto-fix source. The existing pre-commit
configuration includes mutating Ruff hooks, so it is not the WU-00 verification entry point.

## Current baseline diagnostics

The WU-00 baseline is not green. Its recorded diagnostics are:

| Check | Baseline result |
|---|---|
| Pytest | Collection blocked because `tests/test_parser.py` imports missing `shadow_code.parser.TOOL_CALL_RE` |
| Ruff lint | 25 findings |
| Ruff format check | 11 files would be reformatted |
| Mypy | 1 error |
| Bandit | 1 Low-severity finding |

Pytest collection also blocks coverage from producing an approval-quality result. These failures
are reproducibility evidence, not approval of the current behavior, safety, or code quality.

## Baseline runtime evidence

The text-only CLI boundary was exercised against the installed local model with disposable
`HOME` and working directories:

```bash
HOME=<disposable-home> \
SHADOW_MODEL=gemma4:e4b-it-qat \
SHADOW_CTX=32768 \
SHADOW_MAX_TOKENS=128 \
.venv/bin/shadow-code
```

Input was limited to `Reply with exactly BASELINE_OK. Do not call or describe any tool.` followed
by `/exit`. The process exited with status `0`, returned `BASELINE_OK`, emitted no tool-call or
execution event, and changed no tracked production or test source. The isolated session created
only `.shadow-code/prompt_history` and `.shadow-code/sessions.db` under the disposable home.

## Rollback boundary

Remove `.venv`, `requirements-dev.lock`, `Makefile`, and this document. No production code or
runtime behavior belongs to this work unit.
