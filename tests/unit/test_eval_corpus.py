"""Corpus integrity tests (WU-11): schema, ids, required set, thresholds."""

import json
from pathlib import Path

import pytest

from shadow_code.eval.corpus import (
    CORPUS_DIR,
    REQUIRED_SCENARIO_IDS,
    CorpusError,
    corpus_digest,
    load_corpus,
)
from shadow_code.eval.report import load_thresholds


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_corpus_loads_all_scenarios(corpus) -> None:
    assert len(corpus) == len(REQUIRED_SCENARIO_IDS) == 15


def test_scenario_ids_are_unique(corpus) -> None:
    ids = [scenario.id for scenario in corpus]
    assert len(ids) == len(set(ids))


def test_all_required_scenarios_present(corpus) -> None:
    assert {scenario.id for scenario in corpus} == REQUIRED_SCENARIO_IDS


def test_category_split(corpus) -> None:
    safety = [s.id for s in corpus if s.category == "safety_invariant"]
    capability = [s.id for s in corpus if s.category == "capability"]
    assert len(safety) == 8
    assert len(capability) == 7


def test_every_scenario_has_transcript_and_prompt(corpus) -> None:
    for scenario in corpus:
        assert scenario.prompt.strip(), scenario.id
        assert scenario.transcript, scenario.id


def test_safety_scenarios_never_auto_approve_without_tamper_reason(corpus) -> None:
    for scenario in corpus:
        if scenario.category != "safety_invariant" or not scenario.auto_approve:
            continue
        # auto-approval on a safety scenario exists only to drive the
        # mutation paths (export/drift), never to bypass a denial check.
        assert scenario.mutation_mode == "export" or scenario.tamper_on_approval, scenario.id


def test_corpus_digest_is_stable(corpus) -> None:
    assert corpus_digest(corpus) == corpus_digest(load_corpus())


def test_schema_rejects_unknown_fields(tmp_path: Path) -> None:
    version_dir = tmp_path / "vX"
    version_dir.mkdir()
    (version_dir / "bogus.json").write_text(
        json.dumps(
            {
                "id": "bogus",
                "version": 1,
                "category": "capability",
                "title": "bogus",
                "prompt": "hi",
                "expect": {},
                "bogus_field": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CorpusError):
        load_corpus("vX", corpus_dir=tmp_path)


def test_filename_must_match_scenario_id(tmp_path: Path) -> None:
    version_dir = tmp_path / "vX"
    version_dir.mkdir()
    (version_dir / "wrong-name.json").write_text(
        json.dumps(
            {
                "id": "right-name",
                "version": 1,
                "category": "capability",
                "title": "mismatch",
                "prompt": "hi",
                "expect": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="named after its id"):
        load_corpus("vX", corpus_dir=tmp_path)


def test_missing_version_directory_fails_typed(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        load_corpus("vNope", corpus_dir=tmp_path)


def test_thresholds_file_is_valid_and_declared() -> None:
    thresholds = load_thresholds()
    assert thresholds.version >= 1
    assert thresholds.declared_utc
    assert thresholds.safety_failure_blocks_release is True
    assert thresholds.categories["safety_invariant"].min_pass_rate == 1.0
    assert 0.0 < thresholds.categories["capability"].min_pass_rate <= 1.0


def test_corpus_directory_is_versioned() -> None:
    assert (CORPUS_DIR / "v1").is_dir()
