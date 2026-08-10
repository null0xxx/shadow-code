# shadow_code/eval/corpus.py -- Versioned scenario corpus (WU-11)
#
# The corpus is DATA: one JSON file per scenario under corpus/<version>/,
# runnable unchanged for every model family. No model-specific fields or
# special-casing live here; a scenario describes a workspace, a prompt, and
# structured expectations, and both the deterministic suite and the live
# harness interpret exactly the same files.

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ..domain.tools import FrozenModel

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
DEFAULT_CORPUS_VERSION = "v1"

SAFETY_INVARIANT = "safety_invariant"
CAPABILITY = "capability"
ScenarioCategory = Literal["safety_invariant", "capability"]

# The roadmap WU-11 scenario list; corpus tests assert all are present.
REQUIRED_SCENARIO_IDS = frozenset(
    {
        # safety_invariant
        "injection-repo-content",
        "symlink-escape",
        "denied-command",
        "cancellation",
        "malformed-call",
        "repeated-call-termination",
        "strict-patch-export",
        "stale-preview",
        # capability
        "read-only-orientation",
        "targeted-read",
        "exact-edit",
        "new-file",
        "focused-test",
        "multi-step-recovery",
        "context-compression-continuation",
    }
)


class CorpusError(Exception):
    """Typed corpus loading failure (missing file, bad schema, bad ids)."""


class ScenarioExpect(FrozenModel):
    """Structured, machine-checkable expectations for one scenario.

    ``files_changed`` maps a workspace-relative path to one content
    predicate or a list of them (all must hold): ``"text"`` requires the
    substring in the final content, ``"!text"`` forbids it.
    ``error_codes`` must ALL appear among tool results;
    ``error_codes_any`` requires AT LEAST ONE (for environment-dependent
    typed aborts such as drift detected as ``workspace_drift`` or
    ``no_match``). ``forbidden_effects`` names harness-checked invariants:
    ``workspace_unchanged``, ``outside_sentinel_unchanged``,
    ``no_handler_runs``, ``no_process_execution``, ``policy_unchanged``,
    ``file_unchanged:<path>``, ``file_absent:<path>``.
    """

    tools_used: tuple[str, ...] = ()
    tools_order: tuple[str, ...] = ()
    paths_touched: tuple[str, ...] = ()
    files_changed: dict[str, str | list[str]] = Field(default_factory=dict)
    denial_expected: bool = False
    error_codes: tuple[str, ...] = ()
    error_codes_any: tuple[str, ...] = ()
    expected_status: str = "completed"
    expected_reason: str = ""
    max_calls: int | None = None
    exported_patch: bool = False
    forbidden_effects: tuple[str, ...] = ()


class TranscriptRound(FrozenModel):
    """One scripted provider round: plain text plus raw call envelopes.

    The deterministic suite always drives the engine with the scenario's
    transcript (an attack script for safety invariants, an ideal script for
    capability checks). The live harness uses it only for scenarios whose
    trigger is engine-facing (``live_driver == "transcript"``); everywhere
    else the real model produces the rounds.
    """

    text: str = ""
    calls: tuple[Mapping[str, Any], ...] = ()


class Scenario(FrozenModel):
    """One versioned evaluation scenario; the same file drives every model.

    ``outside_files`` are created OUTSIDE the workspace (escape sentinels);
    ``symlinks`` maps a workspace-relative link to its target (typically an
    outside sentinel). ``auto_approve`` lets capability scenarios past the
    one-shot approval gate; safety-invariant scenarios deny by default.
    ``tamper_on_approval`` simulates a concurrent writer: the harness consent
    callback rewrites those files before approving, so drift must abort the
    mutation instead of silently landing on stale content.
    ``cancel_after_rounds`` trips the cancellation seam after that many
    admitted rounds (0 = never).
    """

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    version: int = Field(ge=1)
    category: ScenarioCategory
    title: str = Field(min_length=1)
    workspace_files: dict[str, str] = Field(default_factory=dict)
    outside_files: dict[str, str] = Field(default_factory=dict)
    symlinks: dict[str, str] = Field(default_factory=dict)
    prompt: str = Field(min_length=1)
    expect: ScenarioExpect
    auto_approve: bool = False
    mutation_mode: Literal["apply", "export"] = "apply"
    tamper_on_approval: dict[str, str] = Field(default_factory=dict)
    cancel_after_rounds: int = 0
    live_driver: Literal["model", "transcript"] = "model"
    transcript: tuple[TranscriptRound, ...] = ()
    tags: tuple[str, ...] = ()


def _load_one(path: Path) -> Scenario:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CorpusError(f"cannot read scenario {path}: {error}") from error
    try:
        scenario = Scenario.model_validate_json(text, strict=True)
    except ValueError as error:
        raise CorpusError(f"invalid scenario {path}: {error}") from error
    if path.stem != scenario.id:
        raise CorpusError(f"scenario file {path.name} must be named after its id {scenario.id!r}")
    return scenario


def load_corpus(
    version: str = DEFAULT_CORPUS_VERSION, corpus_dir: Path = CORPUS_DIR
) -> tuple[Scenario, ...]:
    """Load every scenario of one corpus version, sorted by id."""
    version_dir = corpus_dir / version
    if not version_dir.is_dir():
        raise CorpusError(f"corpus version directory missing: {version_dir}")
    scenarios = tuple(
        sorted(
            (_load_one(path) for path in version_dir.glob("*.json")),
            key=lambda scenario: scenario.id,
        )
    )
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise CorpusError("duplicate scenario ids in corpus")
    return scenarios


def corpus_digest(scenarios: tuple[Scenario, ...]) -> str:
    """SHA-256 over the canonical JSON of the whole loaded corpus."""
    encoded = json.dumps(
        [scenario.model_dump(mode="json") for scenario in scenarios],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()
