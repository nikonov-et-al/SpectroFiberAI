import pytest

from spectro_ai.intake import correct_requirements
from spectro_ai.models import CustomerUseCase


def test_corrections_preserve_units_source_and_unmodified_facts():
    case = CustomerUseCase(
        source_text="Monitor ethanol in aqueous fermentation broth.\nPressure: 0 bar gauge",
        measurement_objective="monitor ethanol",
        target_analyte_or_property="ethanol",
        sample_matrix="aqueous fermentation broth",
        pressure="0 bar gauge",
        field_provenance={
            "measurement_objective": "customer quote",
            "target_analyte_or_property": "customer quote",
            "sample_matrix": "customer quote",
            "pressure": "customer quote",
        },
        field_quotes={
            "measurement_objective": "Monitor ethanol",
            "target_analyte_or_property": "ethanol",
            "sample_matrix": "aqueous fermentation broth",
            "pressure": "Pressure: 0 bar gauge",
        },
    )
    assert case.target_analyte_or_property == "ethanol"
    assert case.temperature is None
    corrected = correct_requirements(case, {"detection_limit": "0.1 % v/v", "pressure": "TBD"})
    assert corrected.source_text == case.source_text
    assert corrected.sample_matrix == case.sample_matrix
    assert corrected.detection_limit == "0.1 % v/v"
    assert corrected.pressure is None
    assert corrected.field_provenance["pressure"] == "expert edit"
    assert corrected.field_provenance["sample_matrix"] == "customer quote"
    assert corrected.target_analyte_or_property == "ethanol"
    with pytest.raises(ValueError):
        correct_requirements(case, {"source_text": "replace"})
