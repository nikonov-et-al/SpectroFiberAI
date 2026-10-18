import pytest

from spectro_ai.models import (
    ApplicationSupportAssessment,
    AssessmentReport,
    CustomerUseCase,
    EvidenceAssessment,
    EvidenceClaim,
    EvidenceConditionComparison,
    EvidenceConflict,
    EvidenceContextBlock,
    EvidenceObservation,
    EvidenceReference,
    ExploratoryBridge,
    FeasibilityPlan,
    RetrievedEvidence,
    SourceBlock,
    SourceLocation,
    TechniqueCandidate,
)
from spectro_ai.providers import ProviderError, assess_use_case


class Retriever:
    def __init__(self, item, repair_items=None):
        self.item = item
        self.repair_items = repair_items or []

    def search(self, use_case, limit=8):
        return [self.item]

    def index_metadata(self):
        return {
            "manifest_version": "manifest-v1",
            "files": [
                {
                    "source_id": block.source_id,
                    "filename": block.filename,
                    "document_version": block.document_version,
                    "document_id": block.document_id,
                }
                for item in [self.item, *self.repair_items]
                for block in item.context_blocks
            ],
            "sections": [
                {
                    "section_id": block.section_id,
                    "document_id": block.document_id,
                    "source_id": block.source_id,
                    "filename": block.filename,
                    "document_version": block.document_version,
                    "block_ids": [block.block_id],
                }
                for item in [self.item, *self.repair_items]
                for block in item.context_blocks
            ],
            "source_blocks": [
                SourceBlock(
                    block_id=block.block_id,
                    document_id=block.document_id,
                    section_id=block.section_id,
                    source_id=block.source_id,
                    filename=block.filename,
                    document_version=block.document_version,
                    ordinal=0,
                    text=block.text,
                    location=block.source_locations[0],
                    locations=block.source_locations,
                    review_status=block.review_status,
                    source_scope=block.source_scope,
                ).model_dump(mode="json")
                for item in [self.item, *self.repair_items]
                for block in item.context_blocks
            ],
            "chunks": [
                {
                    "text": item.excerpt,
                    "provenance": [
                        {
                            "evidence_id": item.evidence_id,
                            "source_id": item.source_id,
                            "document_version": item.document_version,
                            "page_or_section": item.page_or_section,
                            "review_status": item.review_status,
                            "source_scope": item.source_scope,
                            "document_id": item.context_blocks[0].document_id
                            if item.context_blocks
                            else None,
                            "section_id": item.context_blocks[0].section_id
                            if item.context_blocks
                            else None,
                        }
                    ],
                }
                for item in [self.item, *self.repair_items]
            ],
        }

    def search_claim(self, use_case, query, *, limit):
        return self.repair_items


class Synthesis:
    def __init__(self, candidate):
        self.candidate = candidate
        self.assessment_inputs = []

    def assess(self, use_case, evidence, *, evidence_assessment):
        self.assessment_inputs.append(evidence_assessment)
        return AssessmentReport(
            extracted_requirements=use_case,
            proposed_feasibility_experiment=FeasibilityPlan(
                proposed_measurement_technique="FTIR chosen by draft",
                experimental_steps=["Draft suggestion."],
            ),
            candidate_techniques=[self.candidate],
            status="provisional",
        )


def _case(*, target="API concentration", matrix="powder blend", temperature=None):
    return CustomerUseCase(
        source_text="Measure API in powder before tablet pressing.",
        measurement_objective="Monitor API powder feed-frame blend",
        target_analyte_or_property=target,
        sample_matrix=matrix,
        temperature=temperature,
    )


def _evidence(text: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="S02-p14",
        source_id="S02",
        document_version="v1",
        page_or_section="p.14",
        excerpt=text,
        review_status="needs_page_review",
        source_scope="technical",
    )


def _draft_candidate(method: str) -> TechniqueCandidate:
    return TechniqueCandidate(
        spectroscopy_method=method,
        advantages=["DRAFT ASSERTION MUST NOT ESCAPE"],
        evidence_claims=[EvidenceClaim(evidence_id="S02-p14", quote="API", claim="Works for D01")],
    )


def _assessment(
    evidence,
    *,
    basis,
    matrix_comparison,
    exploratory=False,
    method="FTIR",
    target_value="API",
    matrix_value="polymer",
    target_comparison="match",
    extra_comparisons=(),
):
    quote = evidence.excerpt
    observation = EvidenceObservation(
        observation_id="obs-key",
        scenario_id="scenario-key",
        source_statement=quote,
        references=[EvidenceReference(evidence_id=evidence.evidence_id, quote=quote)],
        statement_kind="reported_result",
        method=method,
        condition_comparisons=[
            EvidenceConditionComparison(
                dimension="target_property",
                source_value=target_value,
                customer_fields=["target_analyte_or_property"],
                comparison=target_comparison,
                rationale="Source names API.",
                supporting_references=[
                    EvidenceReference(evidence_id=evidence.evidence_id, quote=quote)
                ],
            ),
            EvidenceConditionComparison(
                dimension="matrix_state",
                source_value=matrix_value,
                customer_fields=["sample_matrix"],
                comparison=matrix_comparison,
                rationale="Compare source matrix with validated customer matrix.",
                supporting_references=[
                    EvidenceReference(evidence_id=evidence.evidence_id, quote=quote)
                ],
            ),
        ],
        entailment="supported",
        applicability="related_application",
        transfer_limitations=["Hot polymer and powder feed-frame conditions differ."],
    )
    observation.condition_comparisons.extend(extra_comparisons)
    bridge = None
    if exploratory:
        bridge = ExploratoryBridge(
            source_premises=[EvidenceReference(evidence_id=evidence.evidence_id, quote=quote)],
            customer_fields=["target_analyte_or_property", "sample_matrix"],
            hypothesis=(
                "A powder measurement may respond to API concentration under feed-frame conditions."
            ),
            inference=(
                "FTIR molecular absorption could provide a testable signal, but polymer evidence "
                "does not establish powder performance."
            ),
            application_differences=[
                "Source describes hot API in polymer; customer has powder blend."
            ],
            unknown_gaps=["Powder optical path and representative sampling remain unknown."],
            possible_failure_mechanisms=[
                "Scattering and powder heterogeneity may obscure useful signal."
            ],
            checks=["Expert review and representative powder study are required."],
        )
    support = ApplicationSupportAssessment(
        support_assessment_id="support-key",
        method=method,
        basis=basis,
        observation_ids=["obs-key"],
        bounded_conclusion="FTIR was used to observe API in a hot polymer extrusion process.",
        exploratory_bridge=bridge,
        rejection_reason="Only API overlap; source application differs."
        if basis == "unsupported"
        else None,
    )
    return EvidenceAssessment(
        observations=[observation],
        support_assessments=[support],
        input_fingerprint="model-key",
        assessor_version="test",
    )


def test_s02_hot_polymer_ft_ir_alone_cannot_promote_d01_powder_experiment(monkeypatch):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = _evidence("FTIR measured API in hot polymer during extrusion.")
    synthesis = Synthesis(_draft_candidate("FTIR"))
    report = assess_use_case(
        _case(),
        retriever=Retriever(source),
        assessor=synthesis,
        evidence_assessor=lambda case, items, **kwargs: _assessment(
            source, basis="unsupported", matrix_comparison="different"
        ),
    )
    assert report.candidate_techniques == []
    assert report.proposed_feasibility_experiment.proposed_measurement_technique is None
    assert "DRAFT ASSERTION MUST NOT ESCAPE" not in str(report)
    assert (
        report.evidence_assessment.observations[0].condition_comparisons[1].comparison
        == "different"
    )


def test_synthesis_budget_abstains_while_preserving_assessed_evidence(monkeypatch):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = _evidence("FTIR measured API in hot polymer during extrusion.")

    class OverflowSynthesis:
        def fit_evidence(self, use_case, retrieved):
            return retrieved, []

        def assess(self, *args, **kwargs):
            raise ProviderError("Estimated assessment synthesis exceeds configured context budget")

    report = assess_use_case(
        _case(),
        retriever=Retriever(source),
        assessor=OverflowSynthesis(),
        evidence_assessor=lambda *_args, **_kwargs: _assessment(
            source, basis="reasoned_exploratory", matrix_comparison="different", exploratory=True
        ),
    )
    assert report.candidate_techniques == []
    assert report.evidence_assessment is not None
    assert report.retrieved_evidence == [source]
    assert "Estimated synthesis context budget exceeded" in report.recommendation_rationale


def test_lexical_overlap_cannot_override_application_supported_label(monkeypatch):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = _evidence("FTIR measured API in hot polymer during extrusion.")
    report = assess_use_case(
        _case(),
        retriever=Retriever(source),
        assessor=Synthesis(_draft_candidate("FTIR")),
        evidence_assessor=lambda case, items, **kwargs: _assessment(
            source, basis="application_supported", matrix_comparison="different"
        ),
    )
    assert report.candidate_techniques == []
    assert any(
        "no supported application observation" in item
        for item in report.evidence_assessment.diagnostics.warnings
    )


def test_novel_application_can_be_considered_exploratorily_without_selection(monkeypatch):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = _evidence("FTIR measured API in hot polymer during extrusion.")
    synthesis = Synthesis(_draft_candidate("FTIR"))
    report = assess_use_case(
        _case(),
        retriever=Retriever(source),
        assessor=synthesis,
        evidence_assessor=lambda case, items, **kwargs: _assessment(
            source, basis="reasoned_exploratory", matrix_comparison="different", exploratory=True
        ),
    )
    candidate = report.candidate_techniques[0]
    assert candidate.consideration_basis == "reasoned_exploratory"
    assert "Engineering reasoning (exploratory)" in candidate.advantages[0]
    assert report.proposed_feasibility_experiment.proposed_measurement_technique is None
    assert any(
        "expert review" in step.casefold()
        for step in report.proposed_feasibility_experiment.experimental_steps
    )
    assert "DRAFT ASSERTION MUST NOT ESCAPE" not in str(report)


def test_superficial_novel_application_bridge_is_rejected(monkeypatch):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = _evidence("FTIR measured API in hot polymer during extrusion.")
    invalid = _assessment(
        source, basis="reasoned_exploratory", matrix_comparison="different", exploratory=False
    )
    report = assess_use_case(
        _case(),
        retriever=Retriever(source),
        assessor=Synthesis(_draft_candidate("FTIR")),
        evidence_assessor=lambda case, items, **kwargs: invalid,
    )
    assert report.candidate_techniques == []
    assert any(
        "exploratory bridge incomplete" in item
        for item in report.evidence_assessment.diagnostics.warnings
    )


@pytest.mark.parametrize(
    ("method", "source_id", "case", "quote", "target", "matrix", "target_cmp", "matrix_cmp"),
    [
        (
            "Raman",
            "source-alpha",
            _case(target="protein", matrix="grain powder"),
            "Raman measured caffeine in grain powder.",
            "caffeine",
            "grain powder",
            "different",
            "match",
        ),
        (
            "method-X",
            "source-beta",
            _case(target="API", matrix="powder blend"),
            "Method-X measured API in hot polymer.",
            "API",
            "hot polymer",
            "match",
            "different",
        ),
    ],
)
def test_application_support_requires_relevant_target_and_matrix_alignment(
    monkeypatch, method, source_id, case, quote, target, matrix, target_cmp, matrix_cmp
):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = _evidence(quote).model_copy(
        update={"evidence_id": f"{source_id}-p1", "source_id": source_id}
    )
    assessment = _assessment(
        source,
        basis="application_supported",
        matrix_comparison=matrix_cmp,
        method=method,
        target_value=target,
        matrix_value=matrix,
        target_comparison=target_cmp,
    )
    report = assess_use_case(
        case,
        retriever=Retriever(source),
        assessor=Synthesis(_draft_candidate(method)),
        evidence_assessor=lambda *_args, **_kwargs: assessment,
    )
    assert report.candidate_techniques == []
    comparisons = report.evidence_assessment.observations[0].condition_comparisons
    assert [item.comparison for item in comparisons] == [target_cmp, matrix_cmp]


def test_same_method_with_incompatible_operating_condition_is_withheld(monkeypatch):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    quote = "FTIR measured API at 180 C in a polymer melt."
    source = _evidence(quote)
    comparison = EvidenceConditionComparison(
        dimension="operating_conditions",
        source_value="180 C",
        customer_fields=["temperature"],
        comparison="different",
        rationale="The documented temperature conflicts with the customer condition.",
        supporting_references=[EvidenceReference(evidence_id=source.evidence_id, quote=quote)],
    )
    report = assess_use_case(
        _case(temperature="25 C"),
        retriever=Retriever(source),
        assessor=Synthesis(_draft_candidate("FTIR")),
        evidence_assessor=lambda *_args, **_kwargs: _assessment(
            source,
            basis="application_supported",
            matrix_comparison="different",
            extra_comparisons=[comparison],
        ),
    )
    assert report.candidate_techniques == []
    assert report.proposed_feasibility_experiment.proposed_measurement_technique is None


def test_unavailable_decisive_context_withholds_candidate_but_supplied_context_can_support(
    monkeypatch,
):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = _evidence("FTIR assessed API concentration.").model_copy(
        update={"context_omissions": ["Adjacent application conditions were not supplied."]}
    )
    missing = EvidenceObservation(
        observation_id="missing-context",
        scenario_id="application-unknown",
        source_statement=source.excerpt,
        references=[EvidenceReference(evidence_id=source.evidence_id, quote=source.excerpt)],
        statement_kind="proposed_capability",
        method="FTIR",
        condition_comparisons=[
            EvidenceConditionComparison(
                dimension="matrix_state",
                comparison="unknown",
                rationale="The excerpt does not identify the sample matrix.",
            )
        ],
        entailment="supported",
        applicability="insufficient_context",
    )
    missing_assessment = EvidenceAssessment(
        observations=[missing],
        support_assessments=[
            ApplicationSupportAssessment(
                support_assessment_id="unknown-support",
                method="FTIR",
                basis="unsupported",
                observation_ids=["missing-context"],
                bounded_conclusion="Application conditions are unavailable.",
                rejection_reason="The decisive matrix qualification was not supplied.",
            )
        ],
        input_fingerprint="model-key",
        assessor_version="test",
    )
    missing_report = assess_use_case(
        _case(),
        retriever=Retriever(source),
        assessor=Synthesis(_draft_candidate("FTIR")),
        evidence_assessor=lambda *_args, **_kwargs: missing_assessment,
    )
    assert missing_report.candidate_techniques == []
    assert missing_report.evidence_assessment.observations[0].context_availability == "child_only"

    block_text = "The capability applies to API concentration in powder blend."
    location = SourceLocation(
        source_id=source.source_id,
        filename="source.pdf",
        document_version="v1",
        locator="p.2, block 3",
        page_number=2,
    )
    context_block = EvidenceContextBlock(
        block_id="ctx-1",
        document_id="doc-1",
        section_id="section-1",
        source_id=source.source_id,
        filename="source.pdf",
        document_version="v1",
        text=block_text,
        source_locations=[location],
    )
    complete_source = source.model_copy(update={"context_blocks": [context_block]})
    refs = [
        EvidenceReference(evidence_id=source.evidence_id, quote=source.excerpt),
        EvidenceReference(
            evidence_id=source.evidence_id, quote=block_text, context_block_id="ctx-1"
        ),
    ]
    supported = EvidenceObservation(
        observation_id="supported-capability",
        scenario_id="powder-application",
        source_statement=block_text,
        references=refs,
        statement_kind="proposed_capability",
        method="FTIR",
        condition_comparisons=[
            EvidenceConditionComparison(
                dimension="target_property",
                source_value="API concentration",
                customer_fields=["target_analyte_or_property"],
                comparison="match",
                rationale="The supplied context names the target property.",
                supporting_references=[refs[1]],
            ),
            EvidenceConditionComparison(
                dimension="matrix_state",
                source_value="powder blend",
                customer_fields=["sample_matrix"],
                comparison="match",
                rationale="The supplied context names the matrix.",
                supporting_references=[refs[1]],
            ),
        ],
        entailment="supported",
        applicability="direct_application",
    )
    complete_assessment = EvidenceAssessment(
        observations=[supported],
        support_assessments=[
            ApplicationSupportAssessment(
                support_assessment_id="context-support",
                method="FTIR",
                basis="application_supported",
                observation_ids=["supported-capability"],
                bounded_conclusion=(
                    "Source describes a capability for API concentration in powder blend."
                ),
            )
        ],
        input_fingerprint="model-key",
        assessor_version="test",
    )
    complete_report = assess_use_case(
        _case(),
        retriever=Retriever(complete_source),
        assessor=Synthesis(_draft_candidate("FTIR")),
        evidence_assessor=lambda *_args, **_kwargs: complete_assessment,
    )
    assert complete_report.candidate_techniques[0].consideration_basis == "application_supported"
    assert (
        complete_report.evidence_assessment.observations[0].context_availability
        == "attached_context"
    )
    assert (
        "compatibility remain unestablished"
        in complete_report.candidate_techniques[0].required_assumptions[0]
    )


def test_claim_repair_reassesses_all_sources_before_accepting_any_candidate(monkeypatch):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    customer = CustomerUseCase(
        source_text="Measure protein in milk powder.",
        measurement_objective="measure protein",
        target_analyte_or_property="protein",
        sample_matrix="milk powder",
    )
    shared_location = SourceLocation(
        source_id="S01",
        filename="source.pdf",
        document_version="v1",
        locator="p.1, block 2",
        page_number=1,
    )
    shared_context = EvidenceContextBlock(
        block_id="S01-context-2",
        document_id="S01-doc",
        section_id="S01-section",
        source_id="S01",
        filename="source.pdf",
        document_version="v1",
        text="Same source section context.",
        source_locations=[shared_location],
    )
    original = _evidence("NIR measured protein in milk powder.").model_copy(
        update={
            "evidence_id": "S01-original",
            "source_id": "S01",
            "context_blocks": [shared_context],
        }
    )
    added = _evidence("NIR did not support quantitative measurement in milk powder.").model_copy(
        update={
            "evidence_id": "S02-repair",
            "source_id": "S01",
            "context_blocks": [shared_context],
        }
    )
    synthesis = Synthesis(
        TechniqueCandidate(
            spectroscopy_method="NIR",
            evidence_claims=[
                EvidenceClaim(
                    evidence_id="S02-repair",
                    quote="NIR did not support quantitative measurement in milk powder.",
                    claim="NIR may not support quantitative measurement.",
                )
            ],
        )
    )
    assessments = []

    def judge(case, items, *, allow_retry=True):
        assessments.append(items)
        if len(items) == 1:
            return _assessment(
                original,
                basis="application_supported",
                matrix_comparison="match",
                method="NIR",
                target_value="protein",
                matrix_value="milk powder",
            )
        first_quote = original.excerpt
        second_quote = added.excerpt
        first = EvidenceObservation(
            observation_id="old-observation",
            scenario_id="milk-powder",
            source_statement=first_quote,
            references=[EvidenceReference(evidence_id=original.evidence_id, quote=first_quote)],
            statement_kind="reported_result",
            method="NIR",
            condition_comparisons=[
                EvidenceConditionComparison(
                    dimension="matrix_state",
                    source_value="milk powder",
                    customer_fields=["sample_matrix"],
                    comparison="match",
                    rationale="Source and customer matrix align.",
                    supporting_references=[
                        EvidenceReference(evidence_id=original.evidence_id, quote=first_quote)
                    ],
                )
            ],
            entailment="supported",
            applicability="direct_application",
        )
        second = EvidenceObservation(
            observation_id="new-limitation",
            scenario_id="milk-powder",
            source_statement=second_quote,
            references=[EvidenceReference(evidence_id=added.evidence_id, quote=second_quote)],
            statement_kind="limitation",
            method="NIR",
            entailment="supported",
            applicability="related_application",
        )
        return EvidenceAssessment(
            observations=[first, second],
            support_assessments=[
                ApplicationSupportAssessment(
                    support_assessment_id="old-support",
                    method="NIR",
                    basis="application_supported",
                    observation_ids=["old-observation"],
                    bounded_conclusion="Source reports NIR measurement of protein in milk powder.",
                )
            ],
            conflicts=[
                EvidenceConflict(
                    observation_ids=["old-observation", "new-limitation"],
                    description="Reports conflict on quantitative use in this matrix.",
                    references=[
                        EvidenceReference(evidence_id=original.evidence_id, quote=first_quote),
                        EvidenceReference(evidence_id=added.evidence_id, quote=second_quote),
                    ],
                )
            ],
            input_fingerprint="model-key",
            assessor_version="test",
        )

    report = assess_use_case(
        customer,
        retriever=Retriever(original, [added]),
        assessor=synthesis,
        evidence_assessor=judge,
    )
    assert len(assessments) == 2, report.evidence_assessment.diagnostics.model_dump()
    assert len(synthesis.assessment_inputs) == 2
    assert report.candidate_techniques == []
    assert report.retrieved_evidence[0].model_dump(mode="json") == original.model_dump(mode="json")
    assert report.retrieved_evidence[1].context_blocks == []
    assert any(
        "Repeated context blocks" in item for item in report.retrieved_evidence[1].context_omissions
    )
    assert report.evidence_assessment.conflicts
