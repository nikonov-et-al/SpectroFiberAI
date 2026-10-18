from spectro_ai.guardrails import abstention_report, missing_fields
from spectro_ai.models import CustomerUseCase, FeasibilityPlan
from spectro_ai.providers import _apply_feasibility_safeguards


def test_abstention_is_neutral_and_missing_facts_are_explicit():
    case = CustomerUseCase(source_text="We need better monitoring.")
    report = abstention_report(case, "Insufficient customer facts.")
    assert report.candidate_techniques == []
    assert set(report.missing_information) == set(missing_fields(case))
    assert report.missing_information[:3] == [
        "measurement_objective",
        "target_analyte_or_property",
        "sample_matrix",
    ]
    assert report.assessment_method == "evidence_abstention"
    assert report.assessment_provider is None


def test_numeric_plan_specs_must_match_full_customer_quantity_and_unit():
    case = CustomerUseCase(pressure="10 bar")
    plan = FeasibilityPlan(
        experimental_steps=[
            "Challenge at 1 mm optical path.",
            "Record the stated pressure of 10 bar.",
            "Challenge at 10 mm optical path.",
        ]
    )
    safe, warnings = _apply_feasibility_safeguards(case, plan)
    assert "Record the stated pressure of 10 bar." in safe.experimental_steps
    assert not any("1 mm" in step or "10 mm" in step for step in safe.experimental_steps)
    assert any("did not exactly match" in warning for warning in warnings)


def test_reference_is_explicitly_tbd_when_customer_has_none():
    safe, _ = _apply_feasibility_safeguards(CustomerUseCase(), FeasibilityPlan())
    assert (
        safe.reference_measurement
        == "Independent reference measurement to be agreed before testing."
    )


def test_unverified_model_reference_details_are_replaced_by_customer_text():
    case = CustomerUseCase(reference_measurement="Laboratory checks")
    safe, _ = _apply_feasibility_safeguards(
        case, FeasibilityPlan(reference_measurement="Laboratory checks (current standard)")
    )
    assert safe.reference_measurement == "Laboratory checks"
