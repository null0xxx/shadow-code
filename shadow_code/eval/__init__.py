# shadow_code/eval/__init__.py -- WU-11 evaluation matrix
#
# Versioned scenario corpus, deterministic scripted regression suite,
# opt-in live harness, metric scoring, failure taxonomy, and provenance-
# linked comparison reports for the installed local models. The corpus is
# data and runs unchanged for every model family; safety-invariant failures
# are release blockers regardless of aggregate score.

from .corpus import (
    CORPUS_DIR,
    DEFAULT_CORPUS_VERSION,
    REQUIRED_SCENARIO_IDS,
    CorpusError,
    Scenario,
    ScenarioExpect,
    TranscriptRound,
    corpus_digest,
    load_corpus,
)
from .report import (
    EvalReport,
    ReportError,
    ScenarioScore,
    Thresholds,
    build_report,
    compare,
    load_thresholds,
)
from .scoring import FailureClass, RunTrace, ScoreOutcome, score_scenario

__all__ = [
    "CORPUS_DIR",
    "DEFAULT_CORPUS_VERSION",
    "REQUIRED_SCENARIO_IDS",
    "CorpusError",
    "EvalReport",
    "FailureClass",
    "ReportError",
    "RunTrace",
    "Scenario",
    "ScenarioExpect",
    "ScenarioScore",
    "ScoreOutcome",
    "Thresholds",
    "TranscriptRound",
    "build_report",
    "compare",
    "corpus_digest",
    "load_corpus",
    "load_thresholds",
    "score_scenario",
]
