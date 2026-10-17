"""Paired cases exercise contextual screening without implying scientific validation."""

from spectro_ai.baseline import assess
from spectro_ai.models import CustomerUseCase


def case(**changes: str) -> CustomerUseCase:
    values = dict(
        measurement_objective="identify material",
        target_analyte_or_property="polymer identity",
        sample_matrix="polymer pellets",
    )
    return CustomerUseCase(**(values | changes))


def test_fluorescence_background_changes_raman_conditions_not_target_signal() -> None:
    plain = assess(case())
    background = assess(case(interference="strong fluorescence background"))
    assert [c.spectroscopy_method for c in background.candidate_techniques] == ["Raman", "MIR/FTIR"]
    assert "Reported background" not in plain.candidate_techniques[0].rule_rationale
    assert "retain only conditionally" in background.candidate_techniques[0].rule_rationale
    signal = assess(
        case(
            measurement_objective="measure fluorescence",
            target_analyte_or_property="fluorescent dye",
        )
    )
    assert [c.spectroscopy_method for c in signal.candidate_techniques] == ["Fluorescence"]
    objective_background = assess(
        case(measurement_objective="identify polymer with strong fluorescence background")
    )
    assert [c.spectroscopy_method for c in objective_background.candidate_techniques] == [
        "Raman",
        "MIR/FTIR",
    ]
    assert "obscure Raman" in objective_background.candidate_techniques[0].rule_rationale


def test_aqueous_matrix_changes_mir_conditions() -> None:
    dry = assess(case())
    wet = assess(case(sample_matrix="aqueous solution"))
    assert "water absorption" not in dry.candidate_techniques[1].rule_rationale
    assert "water absorption" in wet.candidate_techniques[1].rule_rationale
    assert any(
        "Water Affects" in (e.source_title or "")
        for e in wet.candidate_techniques[1].supporting_evidence
    )


def test_configuration_requires_explicit_state_or_access() -> None:
    unknown = assess(
        case(
            measurement_objective="measure moisture",
            target_analyte_or_property="moisture",
            sample_matrix="flour",
        )
    )
    powder = assess(
        case(
            measurement_objective="measure moisture",
            target_analyte_or_property="moisture",
            sample_matrix="flour",
            physical_state="powder",
        )
    )
    assert unknown.candidate_techniques[0].measurement_geometry is None
    assert "Diffuse reflectance" in powder.candidate_techniques[0].measurement_geometry
    forbidden = assess(case(installation_constraints="no contact"))
    assert "Non-contact" in forbidden.candidate_techniques[0].measurement_geometry
    assert "ATR study" not in forbidden.candidate_techniques[1].measurement_geometry


def test_sensitivity_and_experiment_preserve_customer_targets() -> None:
    report = assess(
        case(
            measurement_objective="monitor ethanol",
            target_analyte_or_property="ethanol",
            sample_matrix="aqueous fermentation broth",
            physical_state="liquid",
            detection_limit="5 mg/L",
            required_accuracy="+/- 2 mg/L",
            interference="cells and bubbles",
        )
    )
    assert [c.spectroscopy_method for c in report.candidate_techniques] == ["Raman", "MIR/FTIR"]
    assert all(
        "5 mg/L: capability unknown" in c.rule_rationale for c in report.candidate_techniques
    )
    plan = report.proposed_feasibility_experiment
    assert "ethanol" in plan.hypothesis
    assert any("cells and bubbles" in step for step in plan.experimental_steps)
    assert any("held-out batches" in step for step in plan.experimental_steps)
    assert any("5 mg/L" in criterion for criterion in plan.success_criteria)
    assert any("+/- 2 mg/L" in criterion for criterion in plan.success_criteria)
    assert plan.success_actions and plan.failure_actions
    assert plan.proposed_configuration


def test_identification_and_no_rule_have_different_experiments() -> None:
    identification = assess(case()).proposed_feasibility_experiment
    assert any("out-of-library rejection" in step for step in identification.experimental_steps)
    unsupported = assess(
        case(measurement_objective="inspect", target_analyte_or_property="mechanical wear")
    )
    assert unsupported.status == "no_rule_match"
    assert unsupported.proposed_feasibility_experiment.proposed_measurement_technique is None
    assert unsupported.proposed_feasibility_experiment.failure_actions
