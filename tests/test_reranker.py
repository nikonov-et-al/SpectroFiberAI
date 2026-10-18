from types import SimpleNamespace

import numpy as np
import pytest

from spectro_ai import providers
from spectro_ai.evidence import EvidenceRetriever
from spectro_ai.models import AssessmentReport, CustomerUseCase, RetrievedEvidence
from spectro_ai.providers import ProviderError, RagSettings


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, tokens):
        return " ".join(f"t{token}" for token in tokens)

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2


class Scorer:
    tokenizer = Tokenizer()
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="resolved-sha"))

    def predict(self, pairs, **kwargs):
        assert kwargs["convert_to_numpy"] is True
        for query, passage in pairs:
            assert len(self.tokenizer.encode(query)) <= 128
            assert (
                len(self.tokenizer.encode(query)) + len(self.tokenizer.encode(passage)) + 3 <= 512
            )
        return np.arange(len(pairs), dtype=float)


def test_rerank_scores_bounds_inputs_and_returns_resolved_revision(monkeypatch):
    monkeypatch.setattr(providers, "_cross_encoder", lambda settings: Scorer())
    scores, revision = providers.rerank_scores(RagSettings(), "q " * 200, ["p " * 600])
    assert scores == [0]
    assert revision == "resolved-sha"


@pytest.mark.parametrize("values", [[1, float("nan")], [True, 1], [[1], [2]], [1]])
def test_rerank_scores_rejects_invalid_model_output(monkeypatch, values):
    scorer = Scorer()
    scorer.predict = lambda *args, **kwargs: values
    monkeypatch.setattr(providers, "_cross_encoder", lambda settings: scorer)
    with pytest.raises(ProviderError, match="Reranker inference failed"):
        providers.rerank_scores(RagSettings(), "query", ["one", "two"])


class CandidateRetriever(EvidenceRetriever):
    def __init__(self, settings):
        super().__init__(settings=settings, mode="bm25", context_budget=12)

    def index_metadata(self):
        return {"chunks": []}

    def _search(self, query_text, limit, pinned_index=None, *, apply_context_budget=True):
        return [
            RetrievedEvidence(
                evidence_id=f"e{index:02}",
                source_id="source",
                excerpt=f"x{index}",
                review_status="reviewed",
                retrieval_score=float(30 - index),
            )
            for index in range(min(30, limit))
        ]


def test_reranker_sees_candidates_beyond_old_context_cutoff(monkeypatch):
    seen = []

    def scores(settings, query, passages):
        seen.extend(passages)
        return [1.0 if passage == "x10" else 0.0 for passage in passages], "model-sha"

    monkeypatch.setattr(providers, "rerank_scores", scores)
    retriever = CandidateRetriever(RagSettings(reranker="cross_encoder"))
    diagnostics = []
    selected = retriever._search_many(
        ["complete customer query"], 2, {"chunks": []}, diagnostics=diagnostics
    )
    assert len(seen) == 30
    assert selected[0].excerpt == "x10"
    assert selected[0].retrieval_score == pytest.approx(1 / 71)
    assert selected[0].rerank_score == 1
    assert sum(len(item.excerpt.encode()) for item in selected) <= 12
    assert diagnostics[0]["status"] == "applied"


def test_candidate_api_skips_final_context_budget_and_deduplicates(monkeypatch):
    retriever = CandidateRetriever(RagSettings())
    hits = retriever._search_many(["query"], 30, {"chunks": []}, candidate_mode=True)
    assert len(hits) == 30
    assert len({hit.evidence_id for hit in hits}) == len(hits)


def test_reranker_failure_falls_back_to_fused_order_with_diagnostic(monkeypatch):
    def fail(*args, **kwargs):
        raise ProviderError("model cache missing")

    monkeypatch.setattr(providers, "rerank_scores", fail)
    retriever = CandidateRetriever(RagSettings(reranker="cross_encoder"))
    diagnostics = []
    selected = retriever._search_many(["query"], 2, {"chunks": []}, diagnostics=diagnostics)
    assert [item.evidence_id for item in selected] == ["e00", "e01"]
    assert all(item.rerank_score is None for item in selected)
    assert diagnostics[0]["status"] == "fallback"
    assert "cache missing" in diagnostics[0]["failure_reason"]


@pytest.mark.parametrize(
    "env,value",
    [("SPECTRO_RAG_RERANKER", "unknown"), ("SPECTRO_RAG_RERANK_CANDIDATES", "0")],
)
def test_reranker_settings_are_validated(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(ProviderError):
        RagSettings.from_env()


def test_empty_passages_do_not_load_model(monkeypatch):
    monkeypatch.setattr(providers, "_cross_encoder", lambda settings: pytest.fail("loaded"))
    assert providers.rerank_scores(RagSettings(reranker="cross_encoder"), "query", []) == ([], None)


def test_explicit_setup_allows_download_even_when_ref_exists(monkeypatch):
    requested = []
    model = Scorer()
    monkeypatch.setattr(
        providers,
        "_cross_encoder",
        lambda settings, *, local_files_only: requested.append(local_files_only) or model,
    )
    monkeypatch.setattr(providers, "_resolved_reranker_revision", lambda settings, loaded: "sha")
    assert providers.setup_reranker(RagSettings()) == "sha"
    assert requested == [False]


def test_disabled_path_skips_model_and_keeps_fused_order(monkeypatch):
    monkeypatch.setattr(providers, "rerank_scores", lambda *args: pytest.fail("scored"))
    retriever = CandidateRetriever(RagSettings())
    diagnostics = []
    selected = retriever._search_many(["query"], 2, {"chunks": []}, diagnostics=diagnostics)
    assert [item.evidence_id for item in selected] == ["e00", "e01"]
    assert diagnostics == [{"status": "disabled"}]


def test_equal_rerank_scores_preserve_fused_order(monkeypatch):
    monkeypatch.setattr(
        providers, "rerank_scores", lambda settings, query, passages: ([0.0] * len(passages), "rev")
    )
    retriever = CandidateRetriever(RagSettings(reranker="cross_encoder"))
    selected = retriever._search_many(["query"], 3, {"chunks": []})
    assert [item.evidence_id for item in selected] == ["e00", "e01", "e02"]


def test_empty_pool_skips_model(monkeypatch):
    monkeypatch.setattr(providers, "rerank_scores", lambda *args: pytest.fail("scored"))
    retriever = CandidateRetriever(RagSettings(reranker="cross_encoder"))
    retriever._search = lambda *args, **kwargs: []
    diagnostics = []
    assert retriever._search_many(["query"], 1, {"chunks": []}, diagnostics=diagnostics) == []
    assert diagnostics == [{"status": "skipped", "candidate_count": 0}]


def test_old_report_payload_defaults_rerank_fields():
    case = CustomerUseCase(source_text="old report")
    payload = {
        "extracted_requirements": case.model_dump(),
        "proposed_feasibility_experiment": {},
        "status": "insufficient_information",
        "retrieved_evidence": [
            {
                "evidence_id": "e1",
                "source_id": "s1",
                "excerpt": "text",
                "review_status": "needs_review",
            }
        ],
    }
    report = AssessmentReport.model_validate(payload)
    assert report.retrieval_diagnostics == []
    assert report.retrieved_evidence[0].rerank_score is None


def test_pinned_snapshot_keeps_initial_and_claim_diagnostics_isolated(monkeypatch):
    from spectro_ai.evidence import _PinnedEvidenceRetriever

    monkeypatch.setattr(
        providers, "rerank_scores", lambda settings, query, passages: ([0.0] * len(passages), "rev")
    )
    base = CandidateRetriever(RagSettings(reranker="cross_encoder"))
    case = CustomerUseCase(
        measurement_objective="measure",
        target_analyte_or_property="water",
        sample_matrix="powder",
    )
    first = _PinnedEvidenceRetriever(base, {"chunks": []})
    first.search(case, limit=1)
    first.search_claim(case, "supporting claim", limit=1)
    second = _PinnedEvidenceRetriever(base, {"chunks": []})
    second.search(case, limit=1)
    assert [item["status"] for item in first.retrieval_diagnostics] == ["applied", "applied"]
    assert [item["status"] for item in second.retrieval_diagnostics] == ["applied"]
