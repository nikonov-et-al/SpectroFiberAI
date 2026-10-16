import pytest

from spectro_ai.baseline import assess, extract_use_case, missing_fields
from spectro_ai.models import AssessmentReport, CustomerUseCase
from spectro_ai.reporting import render_markdown


def test_all_synthetic_scenarios_match_declared_rule_behavior(scenarios: list[dict]) -> None:
    assert len(scenarios) == 4
    for scenario in scenarios:
        report = assess(extract_use_case(scenario["problem"]))
        assert [c.spectroscopy_method for c in report.candidate_techniques] == scenario[
            "expected_methods"
        ]
        assert AssessmentReport.model_validate_json(report.model_dump_json()) == report
        assert report == assess(extract_use_case(scenario["problem"]))
        assert render_markdown(report) == render_markdown(
            assess(extract_use_case(scenario["problem"]))
        )
        assert report.proposed_feasibility_experiment.experimental_steps
        assert report.assumptions_and_uncertainties


def test_plain_text_food_problem_extracts_only_literal_facts() -> None:
    text = "We want to measure moisture in food powder at 22 C."
    case = extract_use_case(text)
    assert case.source_text == text
    assert case.target_analyte_or_property == "moisture"
    assert case.sample_matrix == "food powder"
    assert case.temperature is None  # Unlabeled quantities are deliberately not parsed.
    assert case.industry is None
    assert [c.spectroscopy_method for c in assess(case).candidate_techniques] == ["NIR"]


def test_labeled_values_override_phrase_extraction_including_unknown() -> None:
    case = extract_use_case(
        "We want to measure moisture in food powder.\nTarget: TBD; "
        "Pressure: 0 bar gauge; Accuracy: 1 %; Temperature: 20-30 C"
    )
    assert case.target_analyte_or_property is None
    assert case.pressure == "0 bar gauge"
    assert case.required_accuracy == "1 %"
    assert case.temperature == "20-30 C"
    assert assess(case).status == "insufficient_information"


@pytest.mark.parametrize(
    "missing", ["measurement_objective", "target_analyte_or_property", "sample_matrix"]
)
def test_each_missing_critical_requirement_prevents_candidates(missing: str) -> None:
    values = dict(
        measurement_objective="measure moisture",
        target_analyte_or_property="moisture",
        sample_matrix="flour",
    )
    values[missing] = None
    report = assess(CustomerUseCase(**values))
    assert report.status == "insufficient_information"
    assert missing in report.missing_information
    assert report.candidate_techniques == []
    assert report.proposed_feasibility_experiment.proposed_measurement_technique is None
    assert report.suggested_technical_questions


@pytest.mark.parametrize("text", ["", "   ", "We need industrial monitoring."])
def test_sparse_input_abstains(text: str) -> None:
    report = assess(extract_use_case(text))
    assert report.status == "insufficient_information"
    assert len(report.missing_information) == 11
    assert len(report.suggested_technical_questions) == 11
    assert not report.candidate_techniques


@pytest.mark.parametrize(
    "negative",
    [
        "We do not measure fluorescence in water.",
        "We measure color without fluorescence in water.",
        "There is no fluorescent dye in water.",
        "We cannot measure moisture in food powder.",
        "We measure non-fluorescent dye in water.",
    ],
)
def test_negated_clauses_do_not_create_facts_or_candidates(negative: str) -> None:
    case = extract_use_case(negative)
    assert case.measurement_objective is None
    assert case.target_analyte_or_property is None
    assert not assess(case).candidate_techniques


def test_negation_in_background_does_not_block_independent_positive_clause() -> None:
    case = extract_use_case("We do not measure fluorescence; we measure moisture in food powder.")
    assert [c.spectroscopy_method for c in assess(case).candidate_techniques] == ["NIR"]


def test_negated_labeled_target_does_not_trigger_rule() -> None:
    report = assess(
        extract_use_case("Objective: monitor response; Target: non-fluorescent dye; Matrix: water")
    )
    assert report.status == "no_rule_match"
    assert not report.candidate_techniques


def test_uv_vis_rule_and_explicit_no_match() -> None:
    report = assess(extract_use_case("We want to measure absorbance in aqueous solution."))
    assert [c.spectroscopy_method for c in report.candidate_techniques] == ["UV-Vis"]
    unmatched = assess(
        extract_use_case("Objective: inspect; Target: unspecified mechanical wear; Matrix: metal")
    )
    assert unmatched.status == "no_rule_match"
    assert not unmatched.candidate_techniques


def test_missing_fields_are_recomputed_and_no_performance_is_fabricated() -> None:
    report = assess(extract_use_case("We measure moisture in flour."))
    assert report.missing_information == missing_fields(report.extracted_requirements)
    assert (
        report.extracted_requirements.unknown_or_missing_information == report.missing_information
    )
    for candidate in report.candidate_techniques:
        assert candidate.measurement_geometry is None
        assert candidate.potential_probe_configuration is None
        assert candidate.preliminary_suitability_assessment == "provisional_candidate"
        assert {e.kind for e in candidate.supporting_evidence} == {
            "verified_source",
            "engineering_heuristic",
        }
    assert "to be agreed" in " ".join(report.proposed_feasibility_experiment.success_criteria)
