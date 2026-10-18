import json

import pytest

from spectro_ai import providers
from spectro_ai.evidence import EvidenceRetriever, _PinnedEvidenceRetriever
from spectro_ai.models import (
    CustomerUseCase,
    EvidenceQueryPlan,
    EvidenceQuestion,
    RetrievedEvidence,
)
from spectro_ai.providers import (
    ExtractionSettings,
    RagSettings,
    assess_use_case,
    validate_evidence_query_plan,
)
from spectro_ai.reporting import render_markdown


def _case() -> CustomerUseCase:
    return CustomerUseCase(
        source_text="We measure water concentration in milk.",
        measurement_objective="measure water concentration",
        target_analyte_or_property="water concentration",
        sample_matrix="milk",
    )


def test_query_plan_is_strict_and_bounded() -> None:
    question = EvidenceQuestion(
        question="How is water concentration measured in milk?",
        anchor_fields=["target_analyte_or_property"],
    )
    assert EvidenceQueryPlan(questions=[question])
    with pytest.raises(ValueError):
        EvidenceQuestion(question="", anchor_fields=["sample_matrix"])
    with pytest.raises(ValueError):
        EvidenceQueryPlan(questions=[question] * 4)
    with pytest.raises(ValueError):
        EvidenceQuestion(question=23, anchor_fields=["sample_matrix"])


def test_query_validation_rejects_drift_and_duplicate_questions() -> None:
    plan = EvidenceQueryPlan(
        questions=[
            EvidenceQuestion(
                question="Which methods measure water concentration in milk?",
                anchor_fields=["target_analyte_or_property", "sample_matrix"],
            ),
            EvidenceQuestion(
                question="Which method measures water concentration in milk at 50 °C?",
                anchor_fields=["sample_matrix"],
            ),
            EvidenceQuestion(
                question="Which methods measure water concentration in milk!",
                anchor_fields=["sample_matrix"],
            ),
        ]
    )
    accepted, reasons = validate_evidence_query_plan(plan, _case())
    assert len(accepted) == 1
    assert {"unsupported_quantity", "duplicate_question"} <= set(reasons)
    drift = EvidenceQueryPlan(
        questions=[
            EvidenceQuestion(
                question="What does Raman spectroscopy measure?",
                anchor_fields=["sample_matrix"],
            )
        ]
    )
    assert validate_evidence_query_plan(drift, _case())[1] == ["anchor_not_present"]


def test_query_generator_makes_one_bounded_untrusted_context_call(monkeypatch) -> None:
    calls = []
    response = {
        "message": {
            "content": json.dumps(
                {
                    "questions": [
                        {
                            "question": "Which sources discuss water concentration in milk?",
                            "anchor_fields": ["sample_matrix"],
                        }
                    ]
                }
            )
        },
        "prompt_eval_count": 100,
        "eval_count": 20,
    }

    def request(settings, route, payload, **kwargs):
        calls.append((route, payload, kwargs))
        return response

    monkeypatch.setattr(providers, "ollama_request", request)
    usage = {}
    plan = providers.generate_evidence_query_plan(
        _case(), ExtractionSettings(), timeout=12, usage=usage
    )
    assert len(calls) == 1
    assert calls[0][0] == "/api/chat"
    assert calls[0][2] == {"timeout": 12, "lock_timeout": 12}
    assert calls[0][1]["options"]["temperature"] == 0
    assert calls[0][1]["options"]["num_predict"] == 768
    assert calls[0][1]["format"]["$defs"]["EvidenceQuestion"]["properties"]["anchor_fields"][
        "items"
    ]["enum"] == [
        "measurement_objective",
        "target_analyte_or_property",
        "sample_matrix",
    ]
    serialized = calls[0][1]["messages"][1]["content"]
    assert '"source_text_untrusted"' in serialized
    prompt_text = calls[0][1]["messages"][0]["content"]
    assert "answerable from technical literature" in prompt_text
    assert "Do not ask the corpus to provide customer-specific missing facts" in prompt_text
    assert usage == {"call_count": 1, "prompt_tokens": 100, "completion_tokens": 20}
    assert plan.questions[0].question.startswith("Which sources")


def test_question_search_uses_best_reciprocal_rank_and_round_robin(monkeypatch) -> None:
    retriever = EvidenceRetriever(settings=RagSettings(reranker="none"), mode="bm25")
    use_case = _case()
    entries = {
        "A": RetrievedEvidence(
            evidence_id="A", source_id="S1", excerpt="water", review_status="reviewed"
        ),
        "B": RetrievedEvidence(
            evidence_id="B", source_id="S2", excerpt="milk", review_status="reviewed"
        ),
        "C": RetrievedEvidence(
            evidence_id="C", source_id="S3", excerpt="protein", review_status="reviewed"
        ),
    }
    chunks = [
        {
            "chunk_id": key,
            "provenance": [
                {"evidence_id": key, "source_id": f"S{index}", "filename": f"{key}.txt"}
            ],
        }
        for index, key in enumerate(entries, 1)
    ]
    index = {"chunks": chunks}
    retriever.index_metadata = lambda: index
    retriever._search = lambda query, *args, **kwargs: {
        "first": [entries["A"], entries["B"]],
        "second": [entries["C"], entries["A"]],
    }[query]

    result = retriever.search_question_candidates(
        use_case, ["first", "second"], limit=2, include_anchor=False
    )
    assert [item.evidence_id for item in result] == ["A", "C"]
    assert result[0].retrieval_score == pytest.approx(1 / 61)
    assert retriever.last_query_diagnostics["question_evidence"] == {
        "first": ["A", "B"],
        "second": ["C", "A"],
    }


def test_selector_keeps_fused_score_and_refills_after_document_cap(monkeypatch) -> None:
    retriever = EvidenceRetriever(settings=RagSettings(reranker="none"), mode="bm25")
    items = [
        RetrievedEvidence(
            evidence_id=f"E{index}",
            source_id=source,
            excerpt=f"passage {index}",
            review_status="reviewed",
            retrieval_score=score,
            rerank_score=100 - index,
        )
        for index, (source, score) in enumerate(
            [("S1", 0.9), ("S1", 0.8), ("S1", 0.7), ("S2", 0.6)], 1
        )
    ]
    index = {
        "chunks": [
            {
                "chunk_id": item.evidence_id,
                "provenance": [
                    {
                        "evidence_id": item.evidence_id,
                        "source_id": item.source_id,
                        "filename": item.source_id + ".txt",
                    }
                ],
            }
            for item in items
        ]
    }
    monkeypatch.setattr(retriever, "index_metadata", lambda: index)
    selected = retriever.select_question_candidates(items, limit=4)
    assert [item.evidence_id for item in selected] == ["E1", "E2", "E4", "E3"]
    assert selected[0].retrieval_score == 0.9


def test_pinned_question_search_reranks_once_after_fusion(monkeypatch) -> None:
    retriever = EvidenceRetriever(settings=RagSettings(reranker="cross_encoder"), mode="bm25")
    evidence = [
        RetrievedEvidence(
            evidence_id=key,
            source_id=f"S{index}",
            excerpt=key,
            review_status="reviewed",
        )
        for index, key in enumerate(("A", "B", "C"), 1)
    ]
    index = {
        "chunks": [
            {
                "chunk_id": item.evidence_id,
                "provenance": [
                    {
                        "evidence_id": item.evidence_id,
                        "source_id": item.source_id,
                        "filename": item.source_id + ".txt",
                    }
                ],
            }
            for item in evidence
        ]
    }
    retriever._search = lambda query, *args, **kwargs: evidence[:2]
    calls = []

    def rerank(settings, query, passages):
        calls.append((query, passages))
        return [3.0, 2.0], "local-revision"

    monkeypatch.setattr(providers, "rerank_scores", rerank)
    pinned = _PinnedEvidenceRetriever(retriever, index)
    result = pinned.search_questions(_case(), ["water", "milk"], limit=2, include_anchor=False)
    assert len(calls) == 1
    assert calls[0][1] == ["A", "B"]
    assert len(result) == 2
    assert pinned.retrieval_diagnostics[0]["status"] == "applied"


def test_openai_planner_calls_once_and_rejects_oversized_input_before_call(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []
    payload = json.dumps({"questions": []})
    monkeypatch.setattr(
        providers,
        "_openai_json",
        lambda settings, messages, schema, **kwargs: (
            calls.append((messages, schema, kwargs)) or payload
        ),
    )
    settings = ExtractionSettings(provider="openai", model="test-model")
    providers.generate_evidence_query_plan(_case(), settings, timeout=5)
    assert len(calls) == 1
    assert calls[0][2]["temperature"] == 0
    calls.clear()
    large_case = _case().model_copy(update={"source_text": "x" * 5000})
    with pytest.raises(providers.ProviderError, match="context budget"):
        providers.generate_evidence_query_plan(
            large_case, ExtractionSettings(provider="openai", model="test", context=4096)
        )
    assert calls == []


def test_insufficient_case_skips_query_generation() -> None:
    calls = []
    report = assess_use_case(
        CustomerUseCase(measurement_objective="measure water"),
        retriever=type("Source", (), {"settings": RagSettings(multi_query=True)})(),
        query_generator=lambda case: calls.append(case) or EvidenceQueryPlan(),
    )
    assert report.status == "insufficient_information"
    assert report.query_diagnostics is None
    assert calls == []


def test_empty_plan_uses_exact_baseline_retrieval(monkeypatch) -> None:
    evidence = RetrievedEvidence(
        evidence_id="E1", source_id="S1", excerpt="water in milk", review_status="reviewed"
    )

    class Source:
        settings = RagSettings(multi_query=True)

        def __init__(self):
            self.calls = []

        def snapshot(self):
            return self

        def index_metadata(self):
            return {"manifest_version": "manifest-1", "revision": "revision-1"}

        def search_questions(self, *args, **kwargs):
            self.calls.append("questions")
            return []

        def search(self, *args, **kwargs):
            self.calls.append("baseline")
            return [evidence]

    class Assessor:
        def __init__(self):
            self.seen_case = None

        def assess(self, use_case, items):
            self.seen_case = use_case
            return providers._TechnicalDraft(status="insufficient_information")

    source, assessor = Source(), Assessor()
    monkeypatch.setattr(providers, "_verified_retrieval", lambda _source, items: (items, "set-1"))
    report = providers.assess_use_case(
        _case(),
        retriever=source,
        assessor=assessor,
        query_generator=lambda _case: EvidenceQueryPlan(),
    )
    assert source.calls == ["baseline"]
    assert assessor.seen_case.measurement_objective == "measure water concentration"
    assert report.query_diagnostics.status == "fallback"
    assert "no_useful_questions" in report.query_diagnostics.reason_codes


def test_search_deadline_falls_back_but_embedding_error_propagates(monkeypatch) -> None:
    evidence = RetrievedEvidence(
        evidence_id="E1", source_id="S1", excerpt="water in milk", review_status="reviewed"
    )
    question = EvidenceQueryPlan(
        questions=[
            EvidenceQuestion(
                question="Which sources discuss water concentration in milk?",
                anchor_fields=["sample_matrix"],
            )
        ]
    )

    class Source:
        settings = RagSettings(multi_query=True)

        def __init__(self, failure):
            self.failure, self.calls = failure, []

        def snapshot(self):
            return self

        def index_metadata(self):
            return {"manifest_version": "manifest-1", "revision": "revision-1"}

        def search_questions(self, *args, **kwargs):
            self.calls.append("questions")
            raise self.failure

        def search(self, *args, **kwargs):
            self.calls.append("baseline")
            return [evidence]

    monkeypatch.setattr(providers, "_verified_retrieval", lambda _source, items: (items, "set-1"))
    timeout_source = Source(providers.RetrievalDeadlineExceeded("deadline"))
    report = providers.assess_use_case(
        _case(),
        retriever=timeout_source,
        assessor=type(
            "Assessor",
            (),
            {
                "assess": lambda self, *_: providers._TechnicalDraft(
                    status="insufficient_information"
                )
            },
        )(),
        query_generator=lambda _case: question,
    )
    assert timeout_source.calls == ["questions", "baseline"]
    assert "search_deadline" in report.query_diagnostics.reason_codes

    provider_source = Source(providers.ProviderError("embedding unavailable"))
    with pytest.raises(providers.ProviderError, match="Grounded assessment failed"):
        providers.assess_use_case(
            _case(),
            retriever=provider_source,
            assessor=type(
                "Assessor",
                (),
                {
                    "assess": lambda self, *_: providers._TechnicalDraft(
                        status="insufficient_information"
                    )
                },
            )(),
            query_generator=lambda _case: question,
        )
    assert provider_source.calls == ["questions"]


@pytest.mark.parametrize("parent_expansion", [False, True])
def test_sqlite_multi_query_assessment_preserves_pinned_evidence_and_exports(
    tmp_path, monkeypatch, parent_expansion
) -> None:
    import yaml

    from spectro_ai.evidence import evidence_context_blocks_bytes

    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    monkeypatch.setattr(providers, "embedding_identity", lambda settings, **kwargs: "mock-digest")
    monkeypatch.setattr(
        providers,
        "embed_texts",
        lambda settings, texts, **kwargs: [[1.0, 0.0] for _ in texts],
    )
    repo = tmp_path / "repo"
    materials, documents = repo / "materials", repo / "documents"
    materials.mkdir(parents=True)
    documents.mkdir()
    source_text = (
        "Near-infrared spectroscopy measures water concentration in milk samples. "
        "The method requires validation for the specific application."
    )
    (materials / "reference.txt").write_text(source_text, encoding="utf-8")
    manifest = repo / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "root": "materials",
                "sources": [
                    {
                        "source_id": "S01",
                        "filename": "reference.txt",
                        "title": "Synthetic water in milk reference",
                        "source_type": "technical_reference",
                        "scope": "technical",
                        "review_status": "reviewed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    retriever = EvidenceRetriever(
        tmp_path / "index.sqlite3",
        manifest,
        settings=RagSettings(
            embedding_model="mock-embedding",
            multi_query=True,
            reranker="none",
        ),
        documents_path=documents,
        parent_expansion=parent_expansion,
        context_budget=1000,
    )
    retriever.update_knowledge_base()
    initial_revision = retriever.knowledge_status()["revision"]
    use_case = _case()
    plan = EvidenceQueryPlan(
        questions=[
            EvidenceQuestion(
                question="Which sources describe water concentration in milk?",
                anchor_fields=["target_analyte_or_property", "sample_matrix"],
            )
        ]
    )

    class Assessor:
        def __init__(self):
            self.seen_case = None
            self.seen_evidence = None

        def assess(self, case, retrieved):
            self.seen_case, self.seen_evidence = case, retrieved
            return providers._TechnicalDraft(status="insufficient_information")

    assessor = Assessor()
    snapshot_calls = []
    original_snapshot = retriever.snapshot

    def snapshot():
        snapshot_calls.append(True)
        return original_snapshot()

    retriever.snapshot = snapshot
    report = assess_use_case(
        use_case,
        retriever=retriever,
        assessor=assessor,
        query_generator=lambda case: plan,
    )
    assert snapshot_calls == [True]
    assert report.knowledge_index_revision == initial_revision
    assert report.retrieved_evidence
    assert all(
        item.source_id == "S01" and item.review_status == "reviewed"
        for item in report.retrieved_evidence
    )
    assert assessor.seen_case.measurement_objective == use_case.measurement_objective
    assert assessor.seen_case.target_analyte_or_property == use_case.target_analyte_or_property
    assert assessor.seen_case.sample_matrix == use_case.sample_matrix
    assert assessor.seen_case.source_text == use_case.source_text
    assert not hasattr(assessor.seen_case, "query_diagnostics")
    charged_bytes = sum(
        len(item.excerpt.encode("utf-8")) + evidence_context_blocks_bytes(item.context_blocks)
        for item in report.retrieved_evidence
    )
    assert charged_bytes <= retriever.context_budget
    assert report.query_diagnostics.planner_call_count == 1
    assert report.query_diagnostics.unique_document_count == 1
    assert report.model_validate_json(report.model_dump_json())
    assert plan.questions[0].question not in render_markdown(report)
    assert plan.questions[0].question in render_markdown(report, include_diagnostics=True)


@pytest.mark.parametrize(
    "failure",
    [
        providers.ProviderError("invalid JSON"),
        providers.ProviderError("planner timeout"),
    ],
)
def test_query_generation_failure_falls_back_to_grouped_search(monkeypatch, failure) -> None:
    evidence = RetrievedEvidence(
        evidence_id="E1", source_id="S1", excerpt="water in milk", review_status="reviewed"
    )

    class Source:
        settings = RagSettings(multi_query=True)

        def __init__(self):
            self.calls = []

        def snapshot(self):
            return self

        def index_metadata(self):
            return {"manifest_version": "manifest-1", "revision": "revision-1"}

        def search(self, *args, **kwargs):
            self.calls.append("baseline")
            return [evidence]

    monkeypatch.setattr(providers, "_verified_retrieval", lambda _source, items: (items, "set-1"))
    source = Source()
    report = assess_use_case(
        _case(),
        retriever=source,
        assessor=type(
            "Assessor",
            (),
            {
                "assess": lambda self, *_: providers._TechnicalDraft(
                    status="insufficient_information"
                )
            },
        )(),
        query_generator=lambda _case: (_ for _ in ()).throw(failure),
    )
    assert source.calls == ["baseline"]
    assert report.query_diagnostics.status == "fallback"
    assert report.query_diagnostics.planner_call_count == 1
