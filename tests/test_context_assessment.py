from __future__ import annotations

import json

import pytest

from spectro_ai.evidence import EvidenceRetriever
from spectro_ai.models import CustomerUseCase, EvidenceClaim, FeasibilityPlan
from spectro_ai.providers import (
    ExtractionSettings,
    OllamaTechnicalAssessor,
    ProviderError,
    RagSettings,
    _TechnicalDraft,
    _verified_retrieval,
    assess_use_case,
)
from spectro_ai.reporting import render_markdown

_POSITIVE = (
    "Near-infrared diffuse reflection spectroscopy can support measurement of protein in milk "
    "and high-turbidity slurry samples using a probe and representative references, after the "
    "sample is homogenized and its surface is prepared; results require held-out validation "
    "before any performance interpretation."
)
_LIMITATION = (
    "NIR diffuse reflection is not suitable for protein measurement in high-turbidity slurry "
    "when the probe is not in tight contact with the sample; this limitation applies to the "
    "reported study conditions."
)
_TABLE_ROW = (
    "NIR diffuse reflection | protein in milk and high-turbidity slurry | transmission | "
    "25 °C; 2 bar"
)


@pytest.fixture
def context_index(tmp_path, monkeypatch):
    import spectro_ai.providers as providers

    monkeypatch.setenv("SPECTRO_RAG_CHUNK_SIZE", "220")
    monkeypatch.setenv("SPECTRO_RAG_CHUNK_OVERLAP", "20")
    monkeypatch.setenv("SPECTRO_RAG_PARENT_EXPANSION", "false")
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    monkeypatch.setattr(
        providers,
        "embed_texts",
        lambda settings, texts: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(providers, "embedding_identity", lambda settings: "offline-test-model")
    root = tmp_path / "sources"
    root.mkdir()
    source = root / "context.md"
    source.write_text(
        "# Protein measurement\n\n"
        f"{_POSITIVE}\n\n"
        "| Method | Measurand and sample | Geometry | Operating condition |\n"
        "| --- | --- | --- | --- |\n"
        f"| {_TABLE_ROW}\n\n"
        f"{_LIMITATION}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        "schema_version: 1\n"
        "root: sources\n"
        "sources:\n"
        "  - source_id: S01\n"
        "    filename: context.md\n"
        "    title: Synthetic context fixture\n"
        "    source_type: technical_reference\n"
        "    scope: technical\n"
        "    review_status: reviewed_for_phase2\n",
        encoding="utf-8",
    )
    retriever = EvidenceRetriever(
        index_path=tmp_path / "index.sqlite3",
        manifest_path=manifest,
        mode="bm25",
        settings=RagSettings(evidence_count=1, reranker="none"),
        context_budget=20000,
        parent_expansion=True,
    )
    retriever.update_knowledge_base(force_rebuild=True)
    return retriever, source


class _MockAssessor:
    def __init__(self, *, wrong_context_id: bool = False, title_only: bool = False):
        self.settings = ExtractionSettings()
        self.wrong_context_id = wrong_context_id
        self.title_only = title_only
        self.seen = []

    def assess(self, use_case: CustomerUseCase, evidence):
        self.seen = evidence
        item = evidence[0]
        row = next(block for block in item.context_blocks if "transmission" in block.text)
        limit = next(block for block in item.context_blocks if "not suitable" in block.text)
        claims = [
            EvidenceClaim(
                claim="The source associates NIR with the table row.",
                evidence_id=item.evidence_id,
                quote="Synthetic context fixture" if self.title_only else row.text,
                context_block_id=(
                    "tampered-block"
                    if self.wrong_context_id
                    else None
                    if self.title_only
                    else row.block_id
                ),
            ),
            EvidenceClaim(
                claim="The source records an operating limitation.",
                evidence_id=item.evidence_id,
                quote=limit.text,
                context_block_id=limit.block_id,
            ),
        ]
        return _TechnicalDraft(
            status="provisional",
            candidate_techniques=[
                {
                    "spectroscopy_method": "NIR diffuse reflection",
                    "advantages": ["The table reports an NIR diffuse reflection measurement."],
                    "limitations": ["The source records a sample limitation."],
                    "supporting_evidence_ids": [item.evidence_id],
                    "evidence_claims": claims,
                    "preliminary_suitability_assessment": "provisional_candidate",
                }
            ],
            recommendation_rationale="A source-grounded comparison remains provisional.",
            proposed_feasibility_experiment=FeasibilityPlan(
                objective="Compare measurements with independent references."
            ),
        )


def _case(matrix: str = "milk") -> CustomerUseCase:
    return CustomerUseCase(
        source_text=f"Measure protein in {matrix} samples.",
        measurement_objective="measure protein",
        target_analyte_or_property="protein",
        sample_matrix=matrix,
    )


def test_ingestion_retrieval_context_assessment_and_exports_preserve_provenance(
    context_index,
):
    retriever, source_path = context_index
    case = _case()
    retriever.parent_expansion = False
    without_context = retriever.search(case, limit=1)
    assert without_context and without_context[0].context_blocks == []

    retriever.parent_expansion = True
    assessor = _MockAssessor()
    report = assess_use_case(case, retriever=retriever, assessor=assessor)
    assert len(assessor.seen) == 1
    assert len(report.retrieved_evidence) == 1
    evidence = report.retrieved_evidence[0]
    assert evidence.context_blocks
    assert len(report.candidate_techniques) == 1
    candidate = report.candidate_techniques[0]
    assert candidate.supporting_evidence_ids == [evidence.evidence_id]
    context_claim = next(
        claim for claim in candidate.evidence_claims if claim.context_block_id is not None
    )
    canonical = next(
        block
        for block in evidence.context_blocks
        if block.block_id == context_claim.context_block_id
    )
    expected_line = (
        source_path.read_text(encoding="utf-8").splitlines().index(f"| {_TABLE_ROW}") + 1
    )
    assert canonical.source_locations[0].line_start == expected_line
    assert canonical.source_locations[0].document_version == evidence.document_version
    assert any("25 °C" in block.text and "2 bar" in block.text for block in evidence.context_blocks)
    assert any("not suitable" in block.text for block in evidence.context_blocks)
    attached_payload = json.loads(
        OllamaTechnicalAssessor._messages(case, report.retrieved_evidence)[1]["content"]
    )["retrieved_sources_untrusted_data"]
    assert len(attached_payload) == 1
    assert (
        attached_payload[0]["context_relation"] == "same source context; not independent evidence"
    )
    assert len(attached_payload[0]["context_blocks"]) == len(evidence.context_blocks)
    bare_payload = json.loads(
        OllamaTechnicalAssessor._messages(case, without_context)[1]["content"]
    )["retrieved_sources_untrusted_data"]
    assert len(bare_payload) == 1
    assert "context_blocks" not in bare_payload[0]

    customer = render_markdown(report)
    diagnostic = render_markdown(report, include_diagnostics=True)
    assert "Retrieved source context" in customer
    assert "not independent corroboration" in customer
    assert str(expected_line) in customer
    assert "Same-source context" in diagnostic
    assert json.loads(report.model_dump_json())["retrieved_evidence"][0]["context_blocks"]


def test_same_source_context_limitation_blocks_conflicting_customer_case(context_index):
    retriever, _ = context_index

    class FixedSnapshotRetriever:
        settings = retriever.settings
        retrieval_diagnostics = []

        def index_metadata(self):
            return retriever.index_metadata()

        def search(self, use_case, limit=8):
            return retriever.search(_case(), limit=1)

    report = assess_use_case(
        _case("high-turbidity slurry"),
        retriever=FixedSnapshotRetriever(),
        assessor=_MockAssessor(),
    )
    assert report.candidate_techniques == []
    assert any(
        "direct conflicting technical statement" in warning
        for warning in report.assessment_warnings
    )
    assert len(report.retrieved_evidence) == 1
    assert any(
        "not suitable" in block.text for block in report.retrieved_evidence[0].context_blocks
    )


def test_context_block_tampering_and_wrong_quote_id_are_rejected(context_index):
    retriever, _ = context_index
    evidence = retriever.search(_case(), limit=1)
    item = evidence[0]
    original = item.context_blocks[0]
    tampered = item.model_copy(
        update={"context_blocks": [original.model_copy(update={"text": "forged source content"})]}
    )
    with pytest.raises(ProviderError, match="context block"):
        _verified_retrieval(retriever, [tampered])

    report = assess_use_case(
        _case(), retriever=retriever, assessor=_MockAssessor(wrong_context_id=True)
    )
    assert report.candidate_techniques == []
    assert any(
        "Removed unsupported or invalidly cited claims" in warning
        for warning in report.assessment_warnings
    )


def test_source_title_alone_cannot_support_a_context_quote(context_index):
    retriever, _ = context_index
    report = assess_use_case(
        _case(),
        retriever=retriever,
        assessor=_MockAssessor(title_only=True),
    )
    assert report.candidate_techniques == []


def test_claim_search_evidence_is_verified_and_refit_to_the_provider_budget(context_index):
    retriever, _ = context_index
    case = _case()
    candidates = retriever.search(case, limit=12)
    first, additional = (
        candidates[0],
        next(item for item in candidates[1:] if item.excerpt not in candidates[0].excerpt),
    )
    quote = additional.excerpt[:80].strip()
    assert quote and quote not in first.excerpt

    class ClaimSearchRetriever:
        settings = retriever.settings
        retrieval_diagnostics = []

        def index_metadata(self):
            return retriever.index_metadata()

        def search(self, use_case, limit=8):
            return [first]

        def search_claim(self, use_case, claim, limit=4):
            return [additional]

    class FittingAssessor:
        settings = ExtractionSettings()

        def __init__(self):
            self.fit_calls = 0

        def fit_evidence(self, use_case, evidence):
            self.fit_calls += 1
            return evidence, []

        def assess(self, use_case, evidence):
            return _TechnicalDraft(
                status="provisional",
                candidate_techniques=[
                    {
                        "spectroscopy_method": "NIR diffuse reflection",
                        "advantages": ["A retrieved source excerpt is provided for review."],
                        "limitations": [],
                        "supporting_evidence_ids": [first.evidence_id],
                        "evidence_claims": [
                            EvidenceClaim(
                                claim="The source quote should trigger claim-specific lookup.",
                                evidence_id=first.evidence_id,
                                quote=quote,
                            )
                        ],
                        "preliminary_suitability_assessment": "provisional_candidate",
                    }
                ],
                recommendation_rationale="This test checks bounded claim retrieval only.",
                proposed_feasibility_experiment=FeasibilityPlan(
                    objective="Compare measurements with independent references."
                ),
            )

    assessor = FittingAssessor()
    report = assess_use_case(case, retriever=ClaimSearchRetriever(), assessor=assessor)
    assert assessor.fit_calls == 2
    assert additional.evidence_id in {item.evidence_id for item in report.retrieved_evidence}
    context_ids = [
        block.block_id for item in report.retrieved_evidence for block in item.context_blocks
    ]
    assert len(context_ids) == len(set(context_ids))


def test_later_claim_refit_cannot_remove_context_from_an_accepted_claim(context_index):
    retriever, _ = context_index
    case = _case()
    first = retriever.search(case, limit=1)[0]
    additional = next(
        item
        for item in retriever.search(case, limit=12)
        if item.evidence_id != first.evidence_id and item.excerpt not in first.excerpt
    )
    quote = additional.excerpt[:80].strip()
    assert quote and quote not in first.excerpt

    class ClaimSearchRetriever:
        settings = retriever.settings
        retrieval_diagnostics = []

        def index_metadata(self):
            return retriever.index_metadata()

        def search(self, use_case, limit=8):
            return [first]

        def search_claim(self, use_case, claim, limit=4):
            return [additional]

    class ContextTrimmingAssessor:
        settings = ExtractionSettings()

        def __init__(self):
            self.fit_calls = 0

        def fit_evidence(self, use_case, evidence):
            self.fit_calls += 1
            if self.fit_calls == 1:
                return evidence, []
            return [
                evidence[0].model_copy(update={"context_blocks": []}),
                *evidence[1:],
            ], []

        def assess(self, use_case, evidence):
            draft = _MockAssessor().assess(use_case, evidence)
            second = draft.candidate_techniques[0].model_copy(
                update={
                    "evidence_claims": [
                        EvidenceClaim(
                            claim="A later claim requests another source excerpt.",
                            evidence_id=first.evidence_id,
                            quote=quote,
                        )
                    ]
                }
            )
            return draft.model_copy(
                update={"candidate_techniques": [draft.candidate_techniques[0], second]}
            )

    assessor = ContextTrimmingAssessor()
    report = assess_use_case(case, retriever=ClaimSearchRetriever(), assessor=assessor)
    assert assessor.fit_calls == 2
    assert len(report.candidate_techniques) == 1
    assert report.candidate_techniques[0].evidence_claims
    assert any(
        block.block_id == report.candidate_techniques[0].evidence_claims[0].context_block_id
        for block in report.retrieved_evidence[0].context_blocks
    )
    assert any(
        "could not fit without changing earlier verified evidence" in warning
        for warning in report.assessment_warnings
    )


def test_provider_budget_drops_whole_context_before_child_excerpt():
    from spectro_ai.models import EvidenceContextBlock, RetrievedEvidence, SourceLocation

    location = SourceLocation(
        source_id="S01",
        filename="fixture.md",
        document_version="v1",
        locator="L1-L1",
        line_start=1,
        line_end=1,
    )
    child = RetrievedEvidence(
        evidence_id="e1",
        source_id="S01",
        document_version="v1",
        page_or_section="L1-L1",
        excerpt="short child excerpt",
        review_status="reviewed_for_phase2",
        source_scope="technical",
        context_blocks=[
            EvidenceContextBlock(
                block_id="b1",
                document_id="d1",
                section_id="s1",
                source_id="S01",
                filename="fixture.md",
                document_version="v1",
                text="large complete optional context block " * 1000,
                source_locations=[location],
                review_status="reviewed_for_phase2",
                source_scope="technical",
            )
        ],
    )
    assessor = OllamaTechnicalAssessor(ExtractionSettings(context=16000, output_tokens=128))
    retained, dropped = assessor.fit_evidence(_case(), [child])
    assert retained == [child.model_copy(update={"context_blocks": []})]
    assert dropped == ["context:e1:b1"]
