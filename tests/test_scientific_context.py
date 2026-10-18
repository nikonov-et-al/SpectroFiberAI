from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from spectro_ai.evidence import EvidenceRetriever
from spectro_ai.models import CustomerUseCase, EvidenceClaim, FeasibilityPlan, SourceLocation
from spectro_ai.providers import (
    ExtractionSettings,
    OllamaTechnicalAssessor,
    RagSettings,
    _TechnicalDraft,
    _verified_retrieval,
    assess_use_case,
)
from spectro_ai.reporting import render_markdown

_ALPHA_PAGE = (
    "Synthetic fixture only. Instrument Alpha was evaluated for protein in milk. "
    "NIR diffuse reflection can support measurement of protein in milk samples under "
    "the reported study conditions. "
    "Table columns are instrument, analyte, sample, geometry, operating condition, and result. "
    "Alpha | protein | milk | diffuse reflection | 20 °C; 1 bar | Reported value: 0.84 AU. "
    "Limitation: The value is specific to the homogenized test sample and does not establish "
    "performance on other sample preparations."
)
_BETA_PAGE = (
    "Synthetic fixture only. Instrument Beta was evaluated for protein in milk. "
    "NIR diffuse reflection can support measurement of protein in milk samples under "
    "the reported study conditions. "
    "Table columns are instrument, analyte, sample, geometry, operating condition, and result. "
    "Beta | protein | milk | diffuse reflection | 40 °C; 5 bar | Reported value: 0.84 AU. "
    "Limitation: The value is specific to the sterilized test sample and does not establish "
    "performance on other sample preparations."
)
_VALUE_QUOTE = "Reported value: 0.84 AU."


@pytest.fixture
def repeated_pdf_index(tmp_path, monkeypatch):
    import spectro_ai.providers as providers

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakePdfReader:
        is_encrypted = False

        def __init__(self, _path):
            self.pages = [FakePage(_ALPHA_PAGE), FakePage(_BETA_PAGE)]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakePdfReader))
    monkeypatch.setenv("SPECTRO_PDF_BACKEND", "pypdf")
    monkeypatch.setenv("SPECTRO_RAG_CHUNK_SIZE", "220")
    monkeypatch.setenv("SPECTRO_RAG_CHUNK_OVERLAP", "20")
    monkeypatch.setenv("SPECTRO_RAG_PARENT_EXPANSION", "true")
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    monkeypatch.setattr(
        providers,
        "embed_texts",
        lambda settings, texts: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(providers, "embedding_identity", lambda settings: "offline-test-model")

    root = tmp_path / "sources"
    root.mkdir()
    (root / "repeated.pdf").write_bytes(b"synthetic fixture; fake PdfReader supplies pages")
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        "schema_version: 1\n"
        "root: sources\n"
        "sources:\n"
        "  - source_id: S99\n"
        "    filename: repeated.pdf\n"
        "    title: Synthetic repeated-value fixture only\n"
        "    source_type: technical_reference\n"
        "    scope: technical\n"
        "    review_status: reviewed_for_phase2\n",
        encoding="utf-8",
    )
    retriever = EvidenceRetriever(
        index_path=tmp_path / "index.sqlite3",
        manifest_path=manifest,
        mode="bm25",
        settings=RagSettings(evidence_count=8, reranker="none"),
        context_budget=30000,
        parent_expansion=True,
    )
    retriever.update_knowledge_base(force_rebuild=True)
    return retriever


def _case() -> CustomerUseCase:
    return CustomerUseCase(
        source_text="Instrument Alpha was evaluated.",
        measurement_objective="measure protein",
        target_analyte_or_property="protein",
        sample_matrix="milk",
    )


class _FixedChildRetriever:
    def __init__(self, retriever, child):
        self.retriever = retriever
        self.child = child
        self.settings = retriever.settings
        self.retrieval_diagnostics = []

    def index_metadata(self):
        return self.retriever.index_metadata()

    def search(self, use_case, limit=8):
        return [self.child]


class _PdfContextAssessor:
    settings = ExtractionSettings(context=30000)

    def __init__(self, wrong_page_block_id=None):
        self.wrong_page_block_id = wrong_page_block_id
        self.messages = None

    def fit_evidence(self, use_case, evidence):
        return OllamaTechnicalAssessor(self.settings).fit_evidence(use_case, evidence)

    def assess(self, use_case, evidence):
        self.messages = OllamaTechnicalAssessor._messages(use_case, evidence)
        item = evidence[0]
        row = next(block for block in item.context_blocks if _VALUE_QUOTE in block.text)
        claims = [
            EvidenceClaim(
                claim="The source identifies a supported NIR diffuse reflection use.",
                evidence_id=item.evidence_id,
                quote=item.excerpt,
            ),
            EvidenceClaim(
                claim="The source reports the repeated value for the selected page.",
                evidence_id=item.evidence_id,
                quote=_VALUE_QUOTE,
                context_block_id=self.wrong_page_block_id or row.block_id,
            ),
        ]
        return _TechnicalDraft(
            status="provisional",
            candidate_techniques=[
                {
                    "spectroscopy_method": "NIR diffuse reflection",
                    "advantages": ["The source reports a synthetic fixture example."],
                    "limitations": [],
                    "supporting_evidence_ids": [item.evidence_id],
                    "evidence_claims": claims,
                    "preliminary_suitability_assessment": "provisional_candidate",
                }
            ],
            recommendation_rationale="Synthetic fixture only; no customer recommendation.",
            proposed_feasibility_experiment=FeasibilityPlan(
                objective="Compare measurements with independent references."
            ),
        )


def test_repeated_pdf_value_stays_bound_to_the_selected_page_and_block(repeated_pdf_index):
    retriever = repeated_pdf_index
    case = _case()
    results = retriever.search(case, limit=1)
    child = next(
        item
        for item in results
        if any(location.page_number == 1 for location in item.source_locations)
        and any(_VALUE_QUOTE in block.text for block in item.context_blocks)
        and "NIR diffuse reflection can support measurement of protein in milk" in item.excerpt
    )
    row = next(block for block in child.context_blocks if _VALUE_QUOTE in block.text)
    assert row.source_locations[0].page_number == 1
    assert "1 bar" in row.text
    assert "5 bar" not in row.text
    assert row.review_status == child.review_status

    report_assessor = _PdfContextAssessor()
    report = assess_use_case(
        case,
        retriever=_FixedChildRetriever(retriever, child),
        assessor=report_assessor,
    )
    assert len(report.candidate_techniques) == 1
    value_claim = next(
        claim
        for claim in report.candidate_techniques[0].evidence_claims
        if claim.quote == _VALUE_QUOTE
    )
    assert value_claim.context_block_id == row.block_id
    json_report = json.loads(report.model_dump_json())
    exported_row = next(
        block
        for block in json_report["retrieved_evidence"][0]["context_blocks"]
        if block["block_id"] == value_claim.context_block_id
    )
    assert exported_row["source_locations"][0]["page_number"] == 1
    assert exported_row["source_locations"][0]["document_version"] == child.document_version
    assert report_assessor.messages is not None
    prompt_source = json.loads(report_assessor.messages[1]["content"])[
        "retrieved_sources_untrusted_data"
    ][0]
    assert prompt_source["context_relation"] == "same source context; not independent evidence"
    assert any(block["block_id"] == row.block_id for block in prompt_source["context_blocks"])
    markdown = render_markdown(report)
    assert row.block_id in markdown
    assert "p.1" in markdown
    assert "synthetic fixture" in markdown.casefold()

    metadata = retriever.index_metadata()
    page_two = next(
        block
        for block in metadata["source_blocks"]
        if _VALUE_QUOTE in block["text"]
        and any(location["page_number"] == 2 for location in block.get("locations", []))
    )
    wrong_page_assessor = _PdfContextAssessor(wrong_page_block_id=page_two["block_id"])
    wrong_page_report = assess_use_case(
        case,
        retriever=_FixedChildRetriever(retriever, child),
        assessor=wrong_page_assessor,
    )
    assert len(wrong_page_report.candidate_techniques) == 1
    assert all(
        claim.quote != _VALUE_QUOTE
        for claim in wrong_page_report.candidate_techniques[0].evidence_claims
    )
    assert any(
        "Removed unsupported or invalidly cited claims" in warning
        for warning in wrong_page_report.assessment_warnings
    )

    tampered = child.model_copy(
        update={
            "context_blocks": [
                row.model_copy(
                    update={
                        "source_locations": [
                            SourceLocation.model_validate(location)
                            for location in page_two.get("locations", [])
                        ]
                    }
                ),
                *[block for block in child.context_blocks if block.block_id != row.block_id],
            ]
        }
    )
    with pytest.raises(ValueError, match="does not match"):
        _verified_retrieval(retriever, [tampered])
