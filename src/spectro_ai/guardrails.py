"""Deterministic intake checks and neutral abstention reports."""

from spectro_ai.models import (
    CUSTOMER_FACT_FIELDS,
    AssessmentReport,
    CustomerUseCase,
    FeasibilityPlan,
    ValidationDiagnostic,
)

CRITICAL_FIELDS = (
    "measurement_objective",
    "target_analyte_or_property",
    "sample_matrix",
)


def missing_fields(case: CustomerUseCase) -> list[str]:
    return [field for field in CUSTOMER_FACT_FIELDS if getattr(case, field) in (None, [])]


def prioritized_questions(case: CustomerUseCase) -> list[str]:
    questions = {
        "measurement_objective": "What decision must the measurement support?",
        "target_analyte_or_property": "Which analyte or property must be measured?",
        "sample_matrix": "What material or matrix will be measured?",
        "concentration_range": "What target range must the study cover?",
        "temperature": "What temperature range applies at the measurement point?",
        "pressure": "What pressure applies at the measurement point?",
        "measurement_frequency": "How often must measurements be available?",
        "installation_constraints": "What optical access and installation constraints apply?",
        "reference_measurement": "Which independent reference measurement is available?",
    }
    return [questions[field] for field in missing_fields(case) if field in questions]


def abstention_report(
    case: CustomerUseCase,
    reason: str,
    *,
    status: str = "outside_supported_scope",
    provider: str | None = None,
    evidence=None,
    evidence_version: str | None = None,
) -> AssessmentReport:
    return AssessmentReport(
        extracted_requirements=case,
        missing_information=missing_fields(case),
        suggested_technical_questions=prioritized_questions(case),
        candidate_techniques=[],
        recommendation_rationale=reason,
        proposed_feasibility_experiment=FeasibilityPlan(
            objective="Resolve requirements and evidence gaps before choosing a technique.",
            experimental_steps=[
                "Confirm the measurement objective, target, matrix, and operating conditions.",
                "Ask a spectroscopy expert to review relevant evidence before selecting a study.",
            ],
            expected_limitations=[
                "No method, configuration, compatibility, or customer performance is established."
            ],
            success_criteria=["Agree evidence requirements and customer-defined criteria."],
        ),
        assumptions_and_uncertainties=[reason],
        status=status,  # type: ignore[arg-type]
        assessment_method="evidence_abstention",
        assessment_provider=provider,  # type: ignore[arg-type]
        evidence_set_version=evidence_version,
        retrieved_evidence=evidence or [],
        report_version="grounded-abstention-v1",
        assessment_warnings=[reason],
        validation_diagnostics=[
            ValidationDiagnostic(
                level="not_checked",
                check="scientific_suitability",
                message="No candidate retained; expert review required.",
            )
        ],
    )
