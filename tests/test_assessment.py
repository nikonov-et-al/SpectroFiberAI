import json

import pytest

from spectro_ai.intake import correct_requirements
from spectro_ai.models import (
    AssessmentReport,
    CustomerUseCase,
    FeasibilityPlan,
    RetrievedEvidence,
    TechniqueCandidate,
)
from spectro_ai.providers import (
    ExtractionSettings,
    OllamaTechnicalAssessor,
    ProviderError,
    _supports_candidate,
    _TechnicalDraft,
    assess_use_case,
)


def case() -> CustomerUseCase:
    return CustomerUseCase(
        source_text="We monitor protein in milk.",
        measurement_objective="monitor protein",
        target_analyte_or_property="protein",
        sample_matrix="milk",
        field_quotes={
            "measurement_objective": "monitor protein",
            "target_analyte_or_property": "protein",
            "sample_matrix": "milk",
        },
    )


def evidence(
    *,
    scope: str = "technical_reference; technical",
    excerpt: str = "NIR diffuse reflection can measure protein in opaque milk samples.",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="S01-p17",
        source_id="S01",
        document_version="hash-v1",
        page_or_section="p.17",
        excerpt=excerpt,
        review_status="needs_page_review",
        retrieval_score=1.2,
        source_scope=scope,
    )


class FakeRetriever:
    def __init__(self, item: RetrievedEvidence | None = None) -> None:
        self.item = item or evidence()
        self.searched: list[CustomerUseCase] = []

    def search(self, use_case: CustomerUseCase, limit: int = 8) -> list[RetrievedEvidence]:
        self.searched.append(use_case)
        return [self.item] if self.item else []

    def index_metadata(self) -> dict:
        return {
            "manifest_version": "manifest-v1",
            "chunks": [
                {
                    "text": self.item.excerpt,
                    "provenance": [
                        {
                            "evidence_id": self.item.evidence_id,
                            "source_id": self.item.source_id,
                            "document_version": self.item.document_version,
                            "page_or_section": self.item.page_or_section,
                            "review_status": self.item.review_status,
                            "source_scope": self.item.source_scope,
                        }
                    ],
                }
            ]
            if self.item
            else [],
        }


def draft(*, evidence_ids: list[str] | None = None, method: str = "NIR") -> _TechnicalDraft:
    return _TechnicalDraft(
        status="provisional",
        suggested_technical_questions=[
            "Which protein range must the study distinguish, and why it changes calibration."
        ],
        candidate_techniques=[
            {
                "spectroscopy_method": method,
                "measurement_geometry": None,
                "advantages": [
                    "NIR diffuse reflection can measure protein in opaque milk samples."
                ],
                "limitations": [
                    "NIR diffuse reflection can measure protein in opaque milk samples."
                ],
                "supporting_evidence_ids": evidence_ids or ["S01-p17"],
                "evidence_claims": [
                    {
                        "claim": (
                            "NIR diffuse reflection can measure protein in opaque milk samples."
                        ),
                        "evidence_id": "S01-p17",
                        "quote": (
                            "NIR diffuse reflection can measure protein in opaque milk samples."
                        ),
                    }
                ],
                "preliminary_suitability_assessment": "provisional_candidate",
            }
        ],
        recommendation_rationale=(
            "Provisional source-supported study; customer performance remains unverified."
        ),
        proposed_feasibility_experiment=FeasibilityPlan(
            objective="Compare candidate response against customer-defined requirements.",
            reference_measurement="Independent laboratory protein results.",
            experimental_steps=["Collect representative samples."],
            success_criteria=["Agree acceptance criteria with customer before testing."],
            success_actions=["Continue only if held-out samples meet agreed criteria."],
            failure_actions=["Revise method or stop if criteria are not met."],
        ),
        assumptions_and_uncertainties=["Source case may not transfer to this customer."],
    )


class FakeAssessor:
    def __init__(self, value: _TechnicalDraft | None = None) -> None:
        parsed = value or draft()
        self.value = AssessmentReport(
            extracted_requirements=case(),
            missing_information=[],
            suggested_technical_questions=parsed.suggested_technical_questions,
            candidate_techniques=parsed.candidate_techniques,
            recommendation_rationale=parsed.recommendation_rationale,
            proposed_feasibility_experiment=parsed.proposed_feasibility_experiment,
            assumptions_and_uncertainties=parsed.assumptions_and_uncertainties,
            status=parsed.status,
        )
        self.seen: list[CustomerUseCase] = []

    def assess(
        self, use_case: CustomerUseCase, retrieved: list[RetrievedEvidence]
    ) -> AssessmentReport:
        self.seen.append(use_case)
        return self.value


def test_grounded_assessment_binds_candidate_to_validated_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = FakeRetriever()
    assessor = FakeAssessor()
    report = assess_use_case(case(), retriever=source, assessor=assessor)
    assert report.status == "provisional"
    assert report.assessment_method == "ollama_grounded"
    assert report.assessment_provider == "ollama"
    assert report.evidence_set_version != "manifest-v1"
    assert report.candidate_techniques[0].supporting_evidence_ids == ["S01-p17"]
    assert (
        report.extracted_requirements.target_analyte_or_property
        == case().target_analyte_or_property
    )
    assert report.extracted_requirements.unknown_or_missing_information
    assert case().unknown_or_missing_information == []
    assert len(source.searched) == 1
    assert source.searched[0].target_analyte_or_property == "protein"
    assert source.searched[0].unknown_or_missing_information


def test_unknown_or_unrelated_candidate_evidence_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    report = assess_use_case(
        case(),
        retriever=FakeRetriever(),
        assessor=FakeAssessor(draft(evidence_ids=["made-up-id"])),
    )
    assert report.candidate_techniques == []
    assert report.status == "insufficient_information"
    assert any("absent from retrieved sources" in warning for warning in report.assessment_warnings)
    assert "NIR" not in (report.recommendation_rationale or "")
    assert report.proposed_feasibility_experiment.proposed_measurement_technique is None

    unrelated = FakeRetriever(
        evidence(excerpt="NIR spectroscopy measures concentration changes in mineral samples.")
    )
    report = assess_use_case(case(), retriever=unrelated, assessor=FakeAssessor())
    assert report.candidate_techniques == []
    assert any("Removed NIR" in warning for warning in report.assessment_warnings)


def test_combined_method_requires_lexical_support_for_each_component() -> None:
    combined = TechniqueCandidate(spectroscopy_method="NIR-Raman fusion")
    one_component = evidence(excerpt="Raman detects protein in milk samples.")
    both_components = evidence(
        excerpt="NIR and Raman are compared for protein measurements in milk samples."
    )
    assert not _supports_candidate(case(), combined, [one_component])
    assert _supports_candidate(case(), combined, [both_components])


def test_context_fit_keeps_multiple_whole_excerpts_with_output_reserve() -> None:
    excerpts = [
        evidence(
            excerpt=(f"Technical excerpt {index}: " + "protein milk validation " * 52)
        ).model_copy(update={"evidence_id": f"S01-p{index}21"})
        for index in range(8)
    ]
    assessor = OllamaTechnicalAssessor(ExtractionSettings())
    retained, dropped = assessor.fit_evidence(case(), excerpts)
    assert len(retained) >= 3
    assert len(retained) + len(dropped) == len(excerpts)
    assert [item.evidence_id for item in retained] == [
        item.evidence_id for item in excerpts[: len(retained)]
    ]


def test_business_only_sources_cannot_support_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    business = FakeRetriever(evidence(scope="business_context"))
    report = assess_use_case(case(), retriever=business, assessor=FakeAssessor())
    assert report.candidate_techniques == []
    assert report.status == "outside_supported_scope"
    assert report.assessment_method == "evidence_abstention"


def test_empty_retrieval_abstains_instead_of_using_baseline_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")

    class EmptyRetriever(FakeRetriever):
        def __init__(self) -> None:
            self.item = None
            self.searched = []

    report = assess_use_case(case(), retriever=EmptyRetriever(), assessor=FakeAssessor())
    assert report.candidate_techniques == []
    assert report.status == "outside_supported_scope"


def test_provider_failure_does_not_switch_to_offline_recommendations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")

    class FailingAssessor:
        def assess(
            self, use_case: CustomerUseCase, retrieved: list[RetrievedEvidence]
        ) -> AssessmentReport:
            raise ProviderError("model timed out")

    corrected = correct_requirements(case(), {"target_analyte_or_property": "casein"})
    with pytest.raises(ProviderError, match="Grounded assessment failed"):
        assess_use_case(
            corrected,
            retriever=FakeRetriever(),
            assessor=FailingAssessor(),
        )


def test_retrieval_and_assessor_both_receive_corrected_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    corrected = correct_requirements(case(), {"target_analyte_or_property": "casein"})
    source = FakeRetriever(evidence(excerpt="NIR can measure casein in milk samples."))
    assessor = FakeAssessor(draft())
    report = assess_use_case(corrected, retriever=source, assessor=assessor)
    assert source.searched[0].target_analyte_or_property == "casein"
    assert assessor.seen[0].target_analyte_or_property == "casein"
    assert report.extracted_requirements.target_analyte_or_property == "casein"


def test_assessment_refreshes_missing_fields_after_expert_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    incomplete = case().model_copy(
        update={
            "target_analyte_or_property": None,
            "unknown_or_missing_information": ["target_analyte_or_property"],
        }
    )
    corrected = correct_requirements(incomplete, {"target_analyte_or_property": "protein"})
    report = assess_use_case(
        corrected,
        retriever=FakeRetriever(),
        assessor=FakeAssessor(draft()),
    )
    assert "target_analyte_or_property" not in report.missing_information
    assert (
        "target_analyte_or_property"
        not in report.extracted_requirements.unknown_or_missing_information
    )


def test_grounded_plan_replaces_invented_thresholds_and_requires_heldout_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    generated = draft()
    generated.proposed_feasibility_experiment.success_criteria = [
        "R² > 0.90",
        "Every candidate must pass.",
    ]
    generated.proposed_feasibility_experiment.evaluation_metrics = ["RMSE"]
    generated.proposed_feasibility_experiment.hypothesis = "Fusion will be best."
    generated.proposed_feasibility_experiment.success_actions = ["Select production hardware."]
    report = assess_use_case(
        case(),
        retriever=FakeRetriever(),
        assessor=FakeAssessor(generated),
    )
    plan = report.proposed_feasibility_experiment
    assert plan.success_criteria == [
        "Customer-defined acceptance criteria are TBD and must be agreed before testing."
    ]
    assert "0.90" not in " ".join(plan.evaluation_metrics)
    assert "no result is assumed" in plan.hypothesis
    assert any("held-out" in step for step in plan.experimental_steps)
    assert "Select production hardware." not in plan.success_actions
    assert any(
        "No customer-defined acceptance criteria" in item for item in report.assessment_warnings
    )
    stated = case().model_copy(update={"required_accuracy": "within ±0.2% per customer protocol"})
    stated_report = assess_use_case(
        stated,
        retriever=FakeRetriever(),
        assessor=FakeAssessor(draft()),
    )
    assert stated_report.proposed_feasibility_experiment.success_criteria == [
        "within ±0.2% per customer protocol"
    ]


def test_assessor_cannot_replace_trusted_customer_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    original = case()

    class TamperingAssessor(FakeAssessor):
        def assess(
            self, use_case: CustomerUseCase, retrieved: list[RetrievedEvidence]
        ) -> AssessmentReport:
            report = super().assess(use_case, retrieved)
            report.extracted_requirements.target_analyte_or_property = "invented analyte"
            return report

    report = assess_use_case(
        original,
        retriever=FakeRetriever(),
        assessor=TamperingAssessor(),
    )
    assert report.extracted_requirements.target_analyte_or_property == "protein"


def test_assessor_schema_and_prompt_exclude_service_owned_case_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = draft().model_dump_json()
    captured: list[dict] = []

    def transport(settings: object, route: str, payload: dict, **kwargs: object) -> dict:
        captured.append(payload)
        return {"message": {"content": response}, "done_reason": "stop"}

    monkeypatch.setattr("spectro_ai.providers.ollama_request", transport)
    assessor = OllamaTechnicalAssessor(ExtractionSettings())
    result = assessor.assess(case(), [evidence()])
    assert result.candidate_techniques[0].spectroscopy_method == "NIR"
    payload = captured[0]
    system = payload["messages"][0]["content"]
    schema = payload["format"]
    properties = schema["properties"]
    assert "Authoritative output JSON schema" in system
    assert "exact evidence IDs" in system
    assert "extracted_requirements" not in properties
    assert "retrieved_evidence" not in properties
    assert "supporting_evidence" not in schema["$defs"]["_CandidateDraft"]["properties"]
    assert "Evidence" not in schema["$defs"]
    assert "Name each specific method" in system
    user_payload = json.loads(payload["messages"][1]["content"])
    customer = user_payload["customer_case_untrusted_data"]
    assert customer["source_text_untrusted"] == case().source_text
    assert customer["validated_customer_facts"]["target_analyte_or_property"] == "protein"
    assert customer["unknown_fields"]
    assert "field_quotes" not in customer
    source = user_payload["retrieved_sources_untrusted_data"][0]
    assert source["page_or_section"] == "p.17"
    assert source["review_status"] == "needs_page_review"
    assert "retrieval_score" not in source
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0


def test_assessor_malformed_json_gets_one_structural_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [
            {"message": {"content": "not-json"}},
            {"message": {"content": draft().model_dump_json()}},
        ]
    )
    calls = []
    monkeypatch.setattr(
        "spectro_ai.providers.ollama_request",
        lambda *args, **kwargs: (calls.append(args), next(replies))[1],
    )
    assert OllamaTechnicalAssessor(ExtractionSettings()).assess(case(), [evidence()])
    assert len(calls) == 2


def test_assessor_timeout_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spectro_ai.providers.ollama_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProviderError("timeout")),
    )
    with pytest.raises(ProviderError, match="timeout"):
        OllamaTechnicalAssessor(ExtractionSettings()).assess(case(), [evidence()])


def test_export_contains_source_page_excerpt_scope_review_and_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    report = assess_use_case(case(), retriever=FakeRetriever(), assessor=FakeAssessor())
    from spectro_ai.reporting import render_markdown

    markdown = render_markdown(report, include_diagnostics=True)
    assert "S01 p.17" in markdown
    assert "NIR diffuse reflection can measure protein in opaque milk samples." in markdown
    assert "technical\\_reference; technical" in markdown
    assert "needs\\_page\\_review" in markdown
    assert "report grounded-ollama-v1" in markdown
