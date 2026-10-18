import pytest
from test_assessment import FakeAssessor, FakeRetriever, case, draft, evidence

from spectro_ai.evidence import EvidenceRetriever
from spectro_ai.models import EvidenceClaim
from spectro_ai.providers import RagSettings, assess_use_case


def test_unsupported_essential_rationale_still_rejects_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    generated = draft()
    generated.candidate_techniques[0].evidence_claims = [
        EvidenceClaim(
            claim="NIR detects protein in milk.",
            evidence_id="S01-p17",
            quote="NIR detects protein in milk.",
        ),
        EvidenceClaim(
            claim="Fiber-optic probes support remote measurements.",
            evidence_id="S01-p17",
            quote="Fiber-optic probes support remote measurements.",
        ),
    ]
    report = assess_use_case(case(), retriever=FakeRetriever(), assessor=FakeAssessor(generated))
    assert report.candidate_techniques == []
    assert report.status != "provisional"


def test_assessment_uses_pinned_retrieval_and_records_index_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = FakeRetriever()
    snapshots = []

    def snapshot():
        snapshots.append(True)
        return source

    source.snapshot = snapshot
    report = assess_use_case(case(), retriever=source, assessor=FakeAssessor())
    assert snapshots == [True]
    assert report.knowledge_index_revision == "manifest-v1"
    assert report.candidate_techniques[0].supporting_evidence_ids == ["S01-p17"]
    assert report.retrieved_evidence[0].evidence_id == "S01-p17"


def test_d01_assessment_keeps_supported_claim_and_index_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    method_quote = (
        "Near-infrared (NIR) measurements can quantify API concentration in powder blend samples."
    )
    source = evidence(excerpt=(method_quote + " Fiber-optic probes support remote measurements."))
    retriever = FakeRetriever(source)
    generated = draft()
    generated.candidate_techniques[0].evidence_claims = [
        EvidenceClaim(
            claim="NIR quantifies API concentration in powder blend samples.",
            evidence_id=source.evidence_id,
            quote=method_quote,
        ),
        EvidenceClaim(
            claim="The customer's installed probe is suitable.",
            evidence_id=source.evidence_id,
            quote="The customer's installed probe is suitable.",
        ),
    ]
    d01_case = case().model_copy(
        update={
            "source_text": None,
            "measurement_objective": "Measure blend potency at the production feed frame.",
            "target_analyte_or_property": "Active Pharmaceutical Ingredient (API) concentration",
            "sample_matrix": "powder blend",
            "field_quotes": {},
        }
    )
    report = assess_use_case(d01_case, retriever=retriever, assessor=FakeAssessor(generated))
    assert report.status == "provisional"
    candidate = report.candidate_techniques[0]
    assert candidate.supporting_evidence_ids == [source.evidence_id]
    assert any(method_quote in claim.quote for claim in candidate.evidence_claims)
    assert all(item.evidence_id == source.evidence_id for item in report.retrieved_evidence)
    assert report.knowledge_index_revision == "manifest-v1"


def test_d01_sqlite_hybrid_assessment_cites_pinned_revision_after_update(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spectro_ai import providers

    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    monkeypatch.setattr(providers, "embedding_identity", lambda settings: "mock-digest")
    monkeypatch.setattr(
        providers, "embed_texts", lambda settings, texts: [[1.0, 0.0] for _ in texts]
    )
    repo = tmp_path / "repo"
    materials = repo / "materials"
    documents = repo / "documents"
    materials.mkdir(parents=True)
    documents.mkdir()
    reference = materials / "d01.txt"
    first_quote = (
        "Near-infrared (NIR) spectroscopy measures active pharmaceutical ingredient "
        "concentration in powder blend samples."
    )
    reference.write_text(first_quote, encoding="utf-8")
    import yaml

    manifest = repo / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "root": "materials",
                "sources": [
                    {
                        "source_id": "S01",
                        "filename": "d01.txt",
                        "title": "Synthetic D01 technical evidence",
                        "source_type": "technical_reference",
                        "scope": "technical",
                        "review_status": "needs_source_review",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    retriever = EvidenceRetriever(
        tmp_path / "index.sqlite3",
        manifest,
        settings=RagSettings(embedding_model="mock-embedding"),
        documents_path=documents,
    )
    retriever.update_knowledge_base()

    class EvidenceEchoAssessor:
        seen = []

        def assess(self, use_case, retrieved):
            self.seen.append(retrieved)
            result = draft()
            candidate = result.candidate_techniques[0]
            candidate.advantages = [retrieved[0].excerpt]
            candidate.limitations = []
            candidate.supporting_evidence_ids = [retrieved[0].evidence_id]
            candidate.evidence_claims = [
                EvidenceClaim(
                    claim="NIR measures the customer target in this matrix.",
                    evidence_id=retrieved[0].evidence_id,
                    quote=retrieved[0].excerpt,
                )
            ]
            return result

    use_case = case().model_copy(
        update={
            "source_text": None,
            "measurement_objective": "Measure blend potency at the production feed frame.",
            "target_analyte_or_property": "Active Pharmaceutical Ingredient concentration",
            "sample_matrix": "powder blend",
            "field_quotes": {},
        }
    )
    assessor = EvidenceEchoAssessor()
    first_revision = retriever.knowledge_status()["revision"]
    first_report = assess_use_case(use_case, retriever=retriever, assessor=assessor)
    assert first_report.status == "provisional"
    assert first_report.knowledge_index_revision == first_revision
    assert first_report.retrieved_evidence[0].excerpt.endswith(first_quote)
    assert first_report.candidate_techniques[0].supporting_evidence_ids == [
        first_report.retrieved_evidence[0].evidence_id
    ]
    assert assessor.seen[0][0].evidence_id == first_report.retrieved_evidence[0].evidence_id

    second_quote = (
        "Near-infrared (NIR) spectroscopy quantifies active pharmaceutical ingredient "
        "concentration in powder blend material."
    )
    reference.write_text(second_quote, encoding="utf-8")
    retriever.update_knowledge_base()
    second_revision = retriever.knowledge_status()["revision"]
    second_report = assess_use_case(use_case, retriever=retriever, assessor=assessor)
    assert second_revision != first_revision
    assert second_report.knowledge_index_revision == second_revision
    assert second_report.retrieved_evidence[0].excerpt.endswith(second_quote)
    assert (
        second_report.retrieved_evidence[0].evidence_id
        != first_report.retrieved_evidence[0].evidence_id
    )
