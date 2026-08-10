# shadow_code/eval/report.py -- Versioned evaluation reports (WU-11)
#
# Every score links back to the redacted raw events digest and the exact
# model tag, prompt digest, registry digest, and corpus version. Thresholds
# are declared BEFORE any tuning in thresholds.json and travel inside the
# report; a safety-invariant failure blocks release regardless of the
# aggregate pass rate.

import json
from pathlib import Path

from pydantic import Field

from ..domain.tools import FrozenModel
from .corpus import SAFETY_INVARIANT

THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.json"
REPORT_VERSION = 1


class ReportError(Exception):
    """Typed report/threshold loading failure."""


class CategoryThreshold(FrozenModel):
    min_pass_rate: float = Field(ge=0.0, le=1.0)


class Thresholds(FrozenModel):
    """Release thresholds declared before tuning; immutable once published."""

    version: int = Field(ge=1)
    declared_utc: str
    note: str = ""
    categories: dict[str, CategoryThreshold]
    safety_failure_blocks_release: bool = True


def load_thresholds(path: Path = THRESHOLDS_PATH) -> Thresholds:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Thresholds.model_validate(raw, strict=True)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ReportError(f"cannot load thresholds from {path}: {error}") from error


class ScenarioScore(FrozenModel):
    """One scored scenario run with full provenance links."""

    scenario_id: str
    category: str
    passed: bool
    failure_classes: tuple[str, ...]
    metrics: dict[str, float]
    notes: tuple[str, ...] = ()
    latency_ms: float
    events_digest: str
    model_tag: str
    prompt_digest: str
    registry_digest: str
    corpus_version: str


class EvalReport(FrozenModel):
    """Aggregate report for one model over one corpus version."""

    report_version: int = REPORT_VERSION
    generated_utc: str
    model_tag: str
    corpus_version: str
    corpus_digest: str
    prompt_digest: str
    registry_digest: str
    thresholds: Thresholds
    scores: tuple[ScenarioScore, ...]
    pass_rates: dict[str, float]
    threshold_results: dict[str, bool]
    blocked: bool
    blockers: tuple[str, ...]


def build_report(
    scores: tuple[ScenarioScore, ...],
    thresholds: Thresholds,
    *,
    generated_utc: str,
    model_tag: str,
    corpus_version: str,
    corpus_digest: str,
    prompt_digest: str,
    registry_digest: str,
) -> EvalReport:
    """Compute pass rates, evaluate thresholds, and list release blockers."""
    pass_rates: dict[str, float] = {}
    threshold_results: dict[str, bool] = {}
    blockers: list[str] = []
    for category, threshold in thresholds.categories.items():
        scoped = [score for score in scores if score.category == category]
        if not scoped:
            # Subset runs evaluate only the categories they measured.
            continue
        pass_rates[category] = sum(1 for score in scoped if score.passed) / len(scoped)
        ok = pass_rates[category] >= threshold.min_pass_rate
        threshold_results[category] = ok
        if not ok:
            blockers.append(
                f"{category}: pass rate {pass_rates[category]:.0%} below "
                f"declared threshold {threshold.min_pass_rate:.0%}"
            )

    safety_failures = [
        score.scenario_id
        for score in scores
        if score.category == SAFETY_INVARIANT and not score.passed
    ]
    if thresholds.safety_failure_blocks_release and safety_failures:
        blockers.extend(
            f"safety-invariant failure blocks release: {scenario_id}"
            for scenario_id in safety_failures
        )

    return EvalReport(
        generated_utc=generated_utc,
        model_tag=model_tag,
        corpus_version=corpus_version,
        corpus_digest=corpus_digest,
        prompt_digest=prompt_digest,
        registry_digest=registry_digest,
        thresholds=thresholds,
        scores=scores,
        pass_rates=pass_rates,
        threshold_results=threshold_results,
        blocked=bool(blockers),
        blockers=tuple(blockers),
    )


def compare(report_a: EvalReport, report_b: EvalReport) -> str:
    """Render a plain-text comparison table for two model reports."""
    by_id_b = {score.scenario_id: score for score in report_b.scores}
    name_width = max([len("scenario")] + [len(score.scenario_id) for score in report_a.scores])
    header = (
        f"{'scenario':<{name_width}}  {'category':<16}  "
        f"{report_a.model_tag:<20}  {report_b.model_tag:<20}"
    )
    lines = [
        f"corpus {report_a.corpus_version} "
        f"(A digest {report_a.corpus_digest[:12]}, B digest {report_b.corpus_digest[:12]})",
        header,
        "-" * len(header),
    ]
    for score_a in report_a.scores:
        score_b = by_id_b.get(score_a.scenario_id)

        def cell(score: ScenarioScore | None) -> str:
            if score is None:
                return "N/A"
            if score.passed:
                return "PASS"
            return "FAIL(" + ",".join(score.failure_classes) + ")"

        lines.append(
            f"{score_a.scenario_id:<{name_width}}  {score_a.category:<16}  "
            f"{cell(score_a):<20}  {cell(score_b):<20}"
        )
    lines.append("-" * len(header))
    for category in report_a.pass_rates:
        rate_a = report_a.pass_rates.get(category, 0.0)
        rate_b = report_b.pass_rates.get(category, 0.0)
        lines.append(f"pass rate {category}: {rate_a:.0%} vs {rate_b:.0%}")
    lines.append(
        f"release: {report_a.model_tag} {'BLOCKED' if report_a.blocked else 'ok'}, "
        f"{report_b.model_tag} {'BLOCKED' if report_b.blocked else 'ok'}"
    )
    return "\n".join(lines)
