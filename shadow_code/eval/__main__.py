# shadow_code/eval/__main__.py -- Opt-in live evaluation harness (WU-11)
#
#   python -m shadow_code.eval --model gemma4-cline:32k [--scenario id ...]
#       [--report path] [--ollama-url url] [--list]
#
# Runs the versioned corpus against a REAL local Ollama in disposable
# temporary workspaces only (deleted after every scenario), records and
# redacts the raw event stream, scores the metrics, and writes a versioned
# report to eval/reports/<model>-<utc>.json. Consent DENIES by default;
# capability scenarios carry auto_approve: true in the corpus. The harness
# never weakens validation, budgets, or approvals, and CI never calls this.

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import MODEL_OPTIONS, OLLAMA_BASE_URL
from .corpus import DEFAULT_CORPUS_VERSION, CorpusError, corpus_digest, load_corpus
from .report import ReportError, build_report, load_thresholds
from .runner import OllamaEvalProvider, run_scenario

_REPORTS_DIR = Path("eval") / "reports"


def _slug(model_tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_tag)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="shadow_code.eval",
        description="Opt-in live tool-calling evaluation against a local Ollama.",
    )
    parser.add_argument("--model", help="Exact Ollama model tag, e.g. gemma4-cline:32k")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Run only this scenario id (repeatable); default runs the whole corpus",
    )
    parser.add_argument("--report", help="Report output path (default eval/reports/)")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL)
    parser.add_argument("--corpus-version", default=DEFAULT_CORPUS_VERSION)
    parser.add_argument("--list", action="store_true", help="List corpus scenarios and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - opt-in CLI
    args = _parse_args(argv)
    try:
        scenarios = load_corpus(args.corpus_version)
        thresholds = load_thresholds()
    except (CorpusError, ReportError) as error:
        print(f"eval: {error}", file=sys.stderr)
        return 2

    if args.list:
        for scenario in scenarios:
            print(f"{scenario.id}  [{scenario.category}]  {scenario.title}")
        return 0
    if not args.model:
        print("eval: --model is required (or use --list)", file=sys.stderr)
        return 2

    selected = scenarios
    if args.scenario:
        wanted = set(args.scenario)
        unknown = wanted - {scenario.id for scenario in scenarios}
        if unknown:
            print(f"eval: unknown scenarios: {sorted(unknown)}", file=sys.stderr)
            return 2
        selected = tuple(scenario for scenario in scenarios if scenario.id in wanted)

    # Model context comes from the model's own Modelfile; num_ctx from the
    # interactive config (128K) would override it, so it is dropped here.
    options = {key: value for key, value in MODEL_OPTIONS.items() if key != "num_ctx"}
    provider = OllamaEvalProvider(args.ollama_url, options)

    scores = []
    for scenario in selected:
        print(f"[eval] {scenario.id} ...", flush=True)
        try:
            result = run_scenario(
                scenario,
                provider,
                args.model,
                corpus_version=args.corpus_version,
            )
        except Exception as error:
            print(f"eval: scenario {scenario.id} failed to run: {error}", file=sys.stderr)
            return 2
        verdict = "PASS" if result.score.passed else f"FAIL {result.score.failure_classes}"
        print(f"[eval] {scenario.id}: {verdict} ({result.score.latency_ms:.0f} ms)")
        scores.append(result.score)

    generated_utc = datetime.now(timezone.utc).isoformat()
    report = build_report(
        tuple(scores),
        thresholds,
        generated_utc=generated_utc,
        model_tag=args.model,
        corpus_version=args.corpus_version,
        corpus_digest=corpus_digest(scenarios),
        prompt_digest=scores[0].prompt_digest if scores else "",
        registry_digest=scores[0].registry_digest if scores else "",
    )

    report_path = (
        Path(args.report)
        if args.report
        else _REPORTS_DIR / f"{_slug(args.model)}-{generated_utc.replace(':', '-')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"[eval] report written to {report_path}")

    for category, rate in report.pass_rates.items():
        print(f"[eval] pass rate {category}: {rate:.0%}")
    if report.blocked:
        print("[eval] RELEASE BLOCKED:")
        for blocker in report.blockers:
            print(f"  - {blocker}")
        return 1
    print("[eval] thresholds satisfied")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
