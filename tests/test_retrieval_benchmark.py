from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from spectro_ai.models import RetrievedEvidence
from spectro_ai.providers import RagSettings

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
        "cases": [{"id": "q", "query": "query", "relevant": []}],
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


class _ReplacingIndexRetriever(_Retriever):
    def __init__(self) -> None:
        super().__init__()
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
        "recall_at_10": 1.0,
        "mrr": 1.0,
        "candidate_recall": 1.0,
    }
    assert rank_metrics([], set(), True) == {
        "precision_at_5": 0.0,
        "recall_at_10": None,
        "mrr": None,
        "candidate_recall": None,
    }
