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
_QUESTION_PRIORITY = (
    "measurement_objective",
    "target_analyte_or_property",
    "sample_matrix",
    "concentration_range",
    "required_accuracy",
    "detection_limit",
    "reference_measurement",
    "interference",
    "composition",
    "temperature",
    "pressure",
    "measurement_frequency",
    "response_time",
    "installation_constraints",
    "deployment_mode",
)


def missing_fields(case: CustomerUseCase) -> list[str]:
    return [field for field in CUSTOMER_FACT_FIELDS if getattr(case, field) in (None, [])]


def prioritized_missing_fields(case: CustomerUseCase) -> list[str]:
    missing = missing_fields(case)
    ranked = [field for field in _QUESTION_PRIORITY if field in missing]
    return [*ranked, *(field for field in missing if field not in _QUESTION_PRIORITY)]


def prioritized_questions(case: CustomerUseCase) -> list[str]:
    questions = {
        "measurement_objective": (
            "What decision must measurement support? This sets required output and validation."
        ),
        "target_analyte_or_property": (
            "Which analyte or property must be measured? This determines spectral target "
            "and technique selection."
        ),
        "sample_matrix": (
            "What material or matrix will be measured? This affects optical access, sample "
            "handling, and interference checks."
        ),
        "concentration_range": (
            "What target range must study cover? This sets calibration design and validation "
            "coverage."
        ),
        "required_accuracy": (
            "What accuracy must held-out results meet? This sets validation metrics and "
            "customer acceptance criteria."
        ),
        "detection_limit": (
            "What detection limit must be demonstrated? This affects technique selection, "
            "reference method, and validation range."
        ),
        "reference_measurement": (
            "Which independent reference can label study samples? This is needed for "
            "held-out validation."
        ),
        "interference": (
            "Which known interferents must representative samples cover? This affects study "
            "design and interpretation."
        ),
        "composition": (
            "Which composition or batch variation is expected? This defines representative "
            "variation."
        ),
        "temperature": (
            "What temperature range applies at measurement point? This affects experiment "
            "conditions and integration review."
        ),
        "pressure": (
            "What pressure applies at measurement point? This affects safe probe and "
            "installation review."
        ),
        "measurement_frequency": (
            "How often must results be available? This affects acquisition and integration design."
        ),
        "response_time": (
            "What response time is needed? This affects acquisition and process integration."
        ),
        "installation_constraints": (
            "What optical access and installation constraints apply? This affects probe "
            "feasibility and integration."
        ),
        "deployment_mode": (
            "Where and how will measurement run? This affects sampling and integration design."
        ),
    }
    priority = tuple(questions)
    missing = set(missing_fields(case))
    return [questions[field] for field in priority if field in missing]


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
        missing_information=prioritized_missing_fields(case),
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
