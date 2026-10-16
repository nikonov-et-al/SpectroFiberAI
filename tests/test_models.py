import pytest
from pydantic import ValidationError

from spectro_ai.models import (
    AssessmentReport,
    CustomerUseCase,
    Evidence,
    FeasibilityPlan,
    TechniqueCandidate,
)


@pytest.mark.parametrize("unknown", [None, "", "  ", "unknown", "TBD", "N/A", "not yet determined"])
def test_unknown_scalars_and_independent_collection_defaults(unknown: str | None) -> None:
    case = CustomerUseCase(required_accuracy=unknown, temperature=unknown)
    assert case.required_accuracy is None
    assert case.temperature is None
    case.unknown_or_missing_information.append("something")
    assert CustomerUseCase().unknown_or_missing_information == []


def test_customer_values_preserve_units_zero_and_original_text() -> None:
    case = CustomerUseCase(
        concentration_range="0-15 % w/w", pressure="0 bar gauge", source_text="  unknown\n  "
    )
    assert case.concentration_range == "0-15 % w/w"
    assert case.pressure == "0 bar gauge"
    assert case.source_text == "  unknown\n  "


def test_invalid_fields_and_wrong_scalar_types_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerUseCase(pressure=12)
    with pytest.raises(ValidationError):
        CustomerUseCase(instrument_performance="invented")
    with pytest.raises(ValidationError):
        TechniqueCandidate(spectroscopy_method="NIR", preliminary_suitability_assessment="certain")


def test_verified_evidence_requires_provenance() -> None:
    with pytest.raises(ValidationError, match="Verified sources require"):
        Evidence(kind="verified_source", statement="Unsupported source")
    assert Evidence(kind="assumption", statement="Response may be measurable").source_url is None


def test_unknown_plan_and_candidate_fields_and_json_round_trip() -> None:
    candidate = TechniqueCandidate(spectroscopy_method="Raman")
    assert candidate.measurement_geometry is None
    plan = FeasibilityPlan()
    assert plan.hypothesis is None
    report = AssessmentReport(
        extracted_requirements=CustomerUseCase(),
        proposed_feasibility_experiment=plan,
        status="insufficient_information",
    )
    assert AssessmentReport.model_validate_json(report.model_dump_json()) == report
