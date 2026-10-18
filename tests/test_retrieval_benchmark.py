from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from spectro_ai.models import RetrievedEvidence
from spectro_ai.providers import ExtractionSettings, RagSettings

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spectro_ai_benchmark_retrieval", ROOT / "scripts" / "benchmark_retrieval.py"
)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)
rank_metrics = benchmark.rank_metrics
run_benchmark = benchmark.run_benchmark
validate_labels = benchmark.validate_labels


def _item(name: str, score: float = 0.0) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=name,
        source_id="S01",
        document_version="version-1",
        excerpt=f"excerpt {name}",
        review_status="reviewed",
        retrieval_score=score,
    )


class _Retriever:
    def __init__(self) -> None:
        self.items = [_item(f"evidence-{number}", 1 / (number + 1)) for number in range(12)]
        self.candidate_calls = 0

    def index_metadata(self) -> dict:
        return {
            "revision": "index-1",
            "manifest_version": "manifest-1",
            "embedding_identity": "bm25-only",
            "chunking_version": 3,
            "files": [{"source_id": "S01", "document_version": "version-1"}],
            "chunks": [],
        }

    def search(self, use_case, limit=8):
        return self.items[:limit]

    def search_candidates(self, use_case, limit=30):
        self.candidate_calls += 1
        return self.items[:limit]


def test_unreviewed_metrics_are_unavailable_and_labels_cannot_be_fabricated():
    snapshot = _Retriever().index_metadata()
    cases = {
        "status": "provisional_unreviewed",
        "cases": [{"id": "q", "query": "q", "relevant": []}],
    }
    assert not validate_labels(cases, snapshot)
    assert rank_metrics(["x"], set(), False)["mrr"] is None
    cases["cases"][0]["relevant"] = [{"evidence_id": "x"}]
    try:
        validate_labels(cases, snapshot)
    except ValueError as exc:
        assert "cannot contain relevance" in str(exc)
    else:
        raise AssertionError("unreviewed labels must be rejected")


def test_stale_reviewed_labels_are_rejected():
    cases = {
        "status": "human_reviewed",
        "cases": [
            {
                "id": "q",
                "relevant": [
                    {
                        "source_id": "S01",
                        "source_version": "old",
                        "content_sha256": "hash",
                        "evidence_id": "e",
                    }
                ],
            }
        ],
    }
    try:
        validate_labels(cases, _Retriever().index_metadata())
    except ValueError as exc:
        assert "Stale" in str(exc)
    else:
        raise AssertionError("stale labels must be rejected")


def test_benchmark_scores_candidates_beyond_legacy_cutoff_and_stable_ties():
    cases = {
        "status": "provisional_unreviewed",
        "cases": [
            {
                "id": "q",
                "query": "query",
                "relevant": [],
                "validated_requirements": {
                    "measurement_objective": "query",
                    "target_analyte_or_property": "query",
                    "sample_matrix": "query",
                },
            }
        ],
    }
    settings = RagSettings(reranker="cross_encoder", reranker_candidates=12)
    retriever = _Retriever()
    report = run_benchmark(
        cases,
        retriever,
        settings=settings,
        scorer=lambda query, passages: ([0.0] * len(passages), "fake-revision"),
        repeats=2,
    )
    result = report["results"][0]["modes"]
    assert len(result["widened_rrf"]["candidate_ids"]) == 12
    assert result["widened_reranked"]["evidence_ids"] == result["widened_rrf"]["evidence_ids"]
    assert report["metrics_status"] == "provisional_metrics_unavailable"
    assert report["latency_ms"]["warm_repeats_by_mode"] == {
        "legacy": 2,
        "widened_rrf": 2,
        "widened_reranked": 2,
    }
    assert all(
        report["latency_ms"]["warm_search_mean_by_mode"][mode] is not None
        for mode in ("legacy", "widened_rrf", "widened_reranked")
    )
    assert retriever.candidate_calls == 1 + 2 * 2


def test_default_scorer_receives_benchmark_settings(monkeypatch):
    settings = RagSettings(reranker="cross_encoder", reranker_candidates=12)
    calls = []

    def fake_rerank(actual_settings, query, passages):
        calls.append((actual_settings, query, passages))
        return [0.0] * len(passages), "fake-revision"

    monkeypatch.setattr(benchmark, "rerank_scores", fake_rerank)
    run_benchmark(
        {
            "status": "provisional_unreviewed",
            "cases": [{"id": "q", "query": "query", "relevant": []}],
        },
        _Retriever(),
        settings=settings,
        repeats=1,
    )
    assert calls
    assert calls[0][0] is settings
    assert calls[0][1].startswith("source text: query")


def test_reviewed_label_matches_exact_source_version_when_source_id_repeats():
    snapshot = _Retriever().index_metadata()
    snapshot["chunks"] = [
        {
            "content_sha256": "old-hash",
            "provenance": [
                {"evidence_id": "old-evidence", "source_id": "S01", "document_version": "old"}
            ],
        },
        {
            "content_sha256": "new-hash",
            "provenance": [
                {"evidence_id": "new-evidence", "source_id": "S01", "document_version": "new"}
            ],
        },
    ]
    cases = {
        "status": "human_reviewed",
        "cases": [
            {
                "id": "q",
                "relevant": [
                    {
                        "evidence_id": "old-evidence",
                        "source_id": "S01",
                        "source_version": "old",
                        "content_sha256": "old-hash",
                    }
                ],
            }
        ],
    }
    assert validate_labels(cases, snapshot)


def test_incomplete_human_review_keeps_all_metrics_unavailable():
    cases = {
        "status": "human_reviewed",
        "review_complete": False,
        "cases": [{"id": "q", "query": "query", "relevant": []}],
    }
    assert validate_labels(cases, _Retriever().index_metadata()) is False
    assert rank_metrics(["e1"], {"e1"}, False)["recall_at_8"] is None


class _ReplacingIndexRetriever(_Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.settings = RagSettings()
        self.metadata_calls = 0
        self.pinned_revisions: list[str] = []

    def index_metadata(self) -> dict:
        self.metadata_calls += 1
        return {**super().index_metadata(), "revision": f"index-{self.metadata_calls}"}

    def _search_many(self, queries, limit, pinned_index=None, **kwargs):
        assert pinned_index is not None
        self.pinned_revisions.append(pinned_index["revision"])
        return self.items[:limit]


def test_all_modes_share_one_pinned_index_snapshot():
    retriever = _ReplacingIndexRetriever()
    cases = {
        "status": "provisional_unreviewed",
        "cases": [{"id": "q", "query": "query", "relevant": []}],
    }
    run_benchmark(
        cases,
        retriever,
        settings=RagSettings(reranker="cross_encoder", reranker_candidates=12),
        scorer=lambda query, passages: ([0.0] * len(passages), "fake-revision"),
        repeats=2,
    )
    assert retriever.metadata_calls == 1
    assert retriever.pinned_revisions
    assert set(retriever.pinned_revisions) == {"index-1"}


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [True]])
def test_invalid_reranker_scores_are_rejected(scores):
    with pytest.raises(ValueError, match="finite scalar"):
        benchmark._ordered([_item("evidence-1")], "query", lambda *_: (scores, None))


def test_metrics_use_fixed_precision_denominator_and_reviewed_recall():
    metrics = rank_metrics(["a", "x", "b"], {"a", "b"}, True, ["a", "x", "b", "b"])
    assert metrics == {
        "precision_at_5": 0.4,
        "recall_at_8": 1.0,
        "recall_at_10": 1.0,
        "mrr": 1.0,
        "candidate_recall": 1.0,
    }
    assert rank_metrics([], set(), True) == {
        "precision_at_5": 0.0,
        "recall_at_8": None,
        "recall_at_10": None,
        "mrr": None,
        "candidate_recall": None,
    }


class _QuestionRetriever(_Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.settings = RagSettings()
        self.question_calls = []

    def _search_many(self, queries, limit, pinned_index=None, **kwargs):
        return self.items[:limit]

    def search_question_candidates(
        self, use_case, questions, limit=30, timeout=30, include_anchor=True
    ):
        self.question_calls.append((list(questions), include_anchor))
        self.last_query_diagnostics = {
            "question_evidence": {question: [self.items[0].evidence_id] for question in questions},
            "candidate_count": min(limit, len(self.items)),
        }
        return self.items[:limit]

    def _search_questions(
        self,
        use_case,
        questions,
        limit,
        timeout,
        *,
        include_anchor,
        candidate_mode,
        pinned_index=None,
    ):
        return self.search_question_candidates(
            use_case, questions, limit=limit, timeout=timeout, include_anchor=include_anchor
        )


def test_query_modes_share_pinned_candidates_and_mocked_generation():
    cases = {
        "status": "provisional_unreviewed",
        "cases": [
            {
                "id": "q",
                "query": "query",
                "relevant": [],
                "validated_requirements": {
                    "measurement_objective": "query",
                    "target_analyte_or_property": "query",
                    "sample_matrix": "query",
                },
            }
        ],
    }
    retriever = _QuestionRetriever()
    report = run_benchmark(
        cases,
        retriever,
        settings=RagSettings(reranker="none", reranker_candidates=12),
        multi_query=True,
        query_generator=lambda _use_case: object(),
        query_validator=lambda *_args, **_kwargs: (
            [type("Q", (), {"question": "complementary"})()],
            [],
        ),
        repeats=1,
    )
    modes = report["results"][0]["modes"]
    assert modes["single_query"]["status"] == "available"
    assert modes["grouped_pooled"]["status"] == "available"
    assert modes["multi_query"]["generation"]["calls"] == 1
    assert modes["multi_query"]["metrics"]["recall_at_8"] is None
    assert modes["multi_query"]["information_need_coverage"] is None
    assert modes["multi_query"]["context_bytes"]["total_serialized_context_bytes"] > 0
    assert modes["multi_query"]["warm_latency_ms"]["scope"] == "retrieval_only_reusing_plan"
    assert retriever.question_calls[0][0][0] == "query"
    assert retriever.question_calls[0][1] is False
    assert retriever.question_calls[-1] == (["complementary"], True)
    assert report["resource_usage"]["process_cpu_seconds"] >= 0


def test_information_need_coverage_requires_complete_reviewed_labels():
    snapshot = _Retriever().index_metadata()
    snapshot["chunks"] = [
        {
            "content_sha256": "hash",
            "provenance": [
                {"evidence_id": "e1", "source_id": "S01", "document_version": "version-1"}
            ],
        }
    ]
    cases = {
        "status": "human_reviewed",
        "cases": [
            {
                "id": "q",
                "query": "query",
                "relevant": [],
                "needs_reviewed": True,
                "information_needs": [
                    {
                        "id": "limitations",
                        "relevant": [
                            {
                                "evidence_id": "e1",
                                "source_id": "S01",
                                "source_version": "version-1",
                                "content_sha256": "hash",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    assert validate_labels(cases, snapshot)
    assert benchmark._need_coverage(["e1"], cases["cases"][0], True) == 1.0
    cases["cases"][0]["information_needs"][0]["relevant"][0]["source_version"] = "stale"
    with pytest.raises(ValueError, match="Stale"):
        validate_labels(cases, snapshot)


def test_insufficient_case_skips_generated_call():
    calls = []
    report = run_benchmark(
        {
            "status": "provisional_unreviewed",
            "cases": [{"id": "q", "query": "query", "relevant": []}],
        },
        _Retriever(),
        settings=RagSettings(reranker="none"),
        multi_query=True,
        query_generator=lambda case: calls.append(case),
        repeats=1,
    )
    mode = report["results"][0]["modes"]["multi_query"]
    assert mode["status"] == "skipped_insufficient_information"
    assert mode["generation"]["calls"] == 0
    assert mode["metrics"] is None
    assert mode["retrieval_only"] is True
    assert report["aggregate_metrics"]["multi_query"]["recall_at_8"] is None
    assert calls == []


def test_default_generator_captures_usage_per_case(monkeypatch):
    import spectro_ai.providers as providers

    monkeypatch.setattr(
        ExtractionSettings,
        "from_env",
        classmethod(lambda cls: ExtractionSettings(provider="ollama")),
    )
    call_number = 0

    def generate(_case, _settings, *, timeout, usage):
        nonlocal call_number
        call_number += 1
        usage.update(prompt_tokens=call_number * 10, completion_tokens=call_number)
        return object()

    monkeypatch.setattr(providers, "generate_evidence_query_plan", generate)
    cases = {
        "status": "provisional_unreviewed",
        "cases": [
            {
                "id": f"q{number}",
                "query": "query",
                "relevant": [],
                "validated_requirements": {
                    "measurement_objective": "query",
                    "target_analyte_or_property": "query",
                    "sample_matrix": "query",
                },
            }
            for number in range(2)
        ],
    }
    report = run_benchmark(
        cases,
        _QuestionRetriever(),
        settings=RagSettings(reranker="none"),
        multi_query=True,
        query_validator=lambda *_args, **_kwargs: (
            [type("Q", (), {"question": "complementary"})()],
            [],
        ),
        repeats=1,
    )

    generations = [row["modes"]["multi_query"]["generation"] for row in report["results"]]
    assert [(entry["prompt_tokens"], entry["completion_tokens"]) for entry in generations] == [
        (10, 1),
        (20, 2),
    ]


def test_generation_timeout_records_baseline_fallback():
    def timeout(_case):
        raise TimeoutError("local model request timed out")

    report = run_benchmark(
        {
            "status": "provisional_unreviewed",
            "cases": [
                {
                    "id": "q",
                    "query": "query",
                    "relevant": [],
                    "validated_requirements": {
                        "measurement_objective": "query",
                        "target_analyte_or_property": "query",
                        "sample_matrix": "query",
                    },
                }
            ],
        },
        _QuestionRetriever(),
        settings=RagSettings(reranker="none"),
        multi_query=True,
        query_generator=timeout,
        repeats=1,
    )
    modes = report["results"][0]["modes"]
    assert modes["multi_query"]["status"] == "baseline_fallback"
    assert modes["multi_query"]["evidence_ids"] == modes["grouped_baseline"]["evidence_ids"]
    assert modes["multi_query"]["generation"]["fallback_reason"] == "TimeoutError"


def test_reviewed_aggregate_accepts_disabled_modes_with_null_metrics():
    aggregate = benchmark._aggregate(
        [
            {
                "modes": {
                    "legacy": {"metrics": {"recall_at_8": 1.0}},
                    "multi_query": {"status": "disabled", "metrics": None},
                }
            }
        ],
        reviewed=True,
    )
    assert aggregate["legacy"]["recall_at_8"] == 1.0
    assert aggregate["multi_query"]["recall_at_8"] is None


def test_paired_metrics_exclude_insufficient_cases_and_include_fallbacks():
    def row(case_id, baseline, multi_status, multi):
        return {
            "id": case_id,
            "modes": {
                "grouped_baseline": {"status": "available", "metrics": baseline},
                "multi_query": {"status": multi_status, "metrics": multi},
            },
        }

    rows = [
        row("available", {"recall_at_8": 0.2}, "available", {"recall_at_8": 0.8}),
        row("fallback", {"recall_at_8": 0.6}, "baseline_fallback", {"recall_at_8": 0.4}),
        row("skipped", {"recall_at_8": 1.0}, "skipped_insufficient_information", None),
    ]
    case_ids, aggregates = benchmark._paired_aggregate(rows, reviewed=True)

    assert case_ids == ["available", "fallback"]
    assert aggregates["grouped_baseline"]["recall_at_8"] == 0.4
    assert aggregates["multi_query"]["recall_at_8"] == pytest.approx(0.6)


def test_fallback_timing_does_not_mutate_shared_baseline_timings():
    baseline = {
        "latency_ms": {"search": 5.0, "rerank": None},
        "warm_latency_ms": {"search_mean": 5.0, "rerank_mean": None},
    }
    benchmark._finish_mode_timings(baseline)
    fallback = {**baseline, "generation": {"latency_ms": 10.0}}
    benchmark._finish_mode_timings(fallback)

    assert baseline["latency_ms"]["end_to_end"] == 5.0
    assert fallback["latency_ms"]["end_to_end"] == 15.0
    assert baseline["warm_latency_ms"]["end_to_end_mean"] == 5.0
    assert fallback["warm_latency_ms"]["end_to_end_mean"] == 5.0
    assert fallback["latency_ms"] is not baseline["latency_ms"]
    assert fallback["warm_latency_ms"] is not baseline["warm_latency_ms"]
