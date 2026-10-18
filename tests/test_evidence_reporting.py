from spectro_ai.models import (
    ApplicationSupportAssessment,
    AssessmentReport,
    CustomerUseCase,
    EvidenceAssessment,
    EvidenceConditionComparison,
    EvidenceConflict,
    EvidenceCoverage,
    EvidenceObservation,
    EvidenceReference,
    ExploratoryBridge,
    FeasibilityPlan,
    RetrievedEvidence,
    TechniqueCandidate,
)
from spectro_ai.reporting import render_markdown


def _report() -> AssessmentReport:
    source = RetrievedEvidence(
        evidence_id="ev-1",
        source_id="S02",
        document_version="rev-4",
        page_or_section="p. 14",
        excerpt="FTIR measured API concentration in hot polymer during extrusion.",
        review_status="needs_source_review",
        source_scope="technical_primary",
        context_omissions=["Parent section not retrieved"],
    )
    ref = EvidenceReference(
        evidence_id="ev-1",
        quote="FTIR measured API concentration in hot polymer",
    )
    observation = EvidenceObservation(
        observation_id="obs-1",
        scenario_id="scenario-1",
        source_statement="FTIR measured API concentration in hot polymer during extrusion.",
        references=[ref],
        statement_kind="reported_result",
        method="FTIR",
        geometry="ATR",
        condition_comparisons=[
            EvidenceConditionComparison(
                dimension="matrix_state",
                source_value="hot polymer",
                customer_fields=["sample_matrix"],
                comparison="different",
                rationale="Source describes hot polymer; customer describes powder.",
                supporting_references=[EvidenceReference(evidence_id="ev-1", quote="hot polymer")],
            )
        ],
        entailment="supported",
        applicability="related_application",
        transfer_limitations=["Polymer extrusion differs from powder feed-frame monitoring."],
        context_availability="child_only",
    )
    support = ApplicationSupportAssessment(
        support_assessment_id="support-1",
        method="FTIR",
        basis="reasoned_exploratory",
        observation_ids=["obs-1"],
        bounded_conclusion=(
            "FTIR is an exploratory possibility; the source does not establish powder "
            "feed-frame performance."
        ),
        exploratory_bridge=ExploratoryBridge(
            source_premises=[ref],
            customer_fields=["sample_matrix", "measurement_objective"],
            hypothesis="Test whether a suitable FTIR geometry can measure the target in powder.",
            inference=(
                "Source demonstrates polymer measurement only; transfer to powder remains a "
                "hypothesis."
            ),
            application_differences=["Hot polymer extrusion versus powder feed frame."],
            unknown_gaps=["Customer probe compatibility is unknown."],
            possible_failure_mechanisms=[
                "Different matrix and physical state may change the signal."
            ],
            checks=["Check a representative powder sample against a reference method."],
        ),
    )
    assessment = EvidenceAssessment(
        observations=[observation],
        support_assessments=[support],
        conflicts=[
            EvidenceConflict(
                observation_ids=["obs-1", "obs-2"],
                description="Source excerpts describe different sample conditions.",
                references=[ref],
            )
        ],
        coverage=[
            EvidenceCoverage(
                evidence_id="ev-1",
                disposition="assessed",
                context_availability="child_only",
                omissions=["Parent section not retrieved"],
            )
        ],
        input_fingerprint="fingerprint-123",
        assessor_version="ea-v1",
        model_identity="local-model",
    )
    return AssessmentReport(
        extracted_requirements=CustomerUseCase(
            sample_matrix="powder feed-frame", measurement_objective="measure API concentration"
        ),
        candidate_techniques=[
            TechniqueCandidate(
                spectroscopy_method="FTIR",
                support_assessment_id="support-1",
                consideration_basis="reasoned_exploratory",
            )
        ],
        proposed_feasibility_experiment=FeasibilityPlan(),
        status="provisional",
        retrieved_evidence=[source],
        evidence_assessment=assessment,
    )


def test_customer_export_shows_exploratory_basis_transfer_unknowns_and_review_warning():
    markdown = render_markdown(_report())

    assert "reasoned\\_exploratory" in markdown
    assert "Engineering reasoning" in markdown
    assert "Exploratory hypothesis" in markdown
    assert "Customer probe compatibility is unknown" in markdown
    assert "needs\\_source\\_review" in markdown
    assert "Review status does not determine scientific applicability" in markdown


def test_technical_export_shows_full_assessment_and_host_metadata():
    markdown = render_markdown(_report(), include_diagnostics=True)

    assert "Application-aware evidence assessment" in markdown
    assert "fingerprint-123" in markdown
    assert "FTIR measured API concentration in hot polymer" in markdown
    assert "matrix\\_state" in markdown
    assert "different" in markdown
    assert "Conflicts" in markdown
    assert "Evidence coverage" in markdown
    assert "Parent section not retrieved" in markdown
    assert "not deterministic verification" in markdown


def test_old_report_without_evidence_assessment_remains_renderable():
    report = _report().model_copy(update={"evidence_assessment": None})

    assert "Preliminary candidates" in render_markdown(report)
    assert "Application-aware evidence assessment" not in render_markdown(
        report, include_diagnostics=True
    )


def test_customer_export_reports_budget_omission_without_internal_evidence_ids():
    report = _report()
    report.evidence_assessment.diagnostics.dropped_evidence_ids = ["internal-ev-9"]

    customer = render_markdown(report)
    technical = render_markdown(report, include_diagnostics=True)

    assert "some retrieved evidence bundles could not be assessed" in customer
    assert "internal-ev-9" not in customer
    assert "internal-ev-9" in technical


def test_source_described_nir_geometry_does_not_claim_customer_compatibility_or_priority():
    report = _report()
    source_text = (
        "NIR diffuse reflection measured API concentration in hot polymer during extrusion."
    )
    reference = EvidenceReference(evidence_id="ev-1", quote=source_text)
    report.retrieved_evidence[0] = report.retrieved_evidence[0].model_copy(
        update={"excerpt": source_text}
    )
    report.evidence_assessment.observations[0] = report.evidence_assessment.observations[
        0
    ].model_copy(
        update={
            "method": "NIR",
            "geometry": "diffuse reflection",
            "source_statement": source_text,
            "references": [reference],
        }
    )
    report.evidence_assessment.support_assessments[0] = (
        report.evidence_assessment.support_assessments[0].model_copy(update={"method": "NIR"})
    )
    report.evidence_assessment.support_assessments[0].exploratory_bridge.source_premises = [
        reference
    ]
    report.candidate_techniques[0] = report.candidate_techniques[0].model_copy(
        update={"spectroscopy_method": "NIR", "measurement_geometry": "diffuse reflection"}
    )

    markdown = render_markdown(report, include_diagnostics=True)

    assert "Source-described geometry: diffuse reflection" in markdown
    assert "customer compatibility not established" in markdown
    assert "priority" not in markdown.casefold()
