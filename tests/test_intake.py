from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from spectro_ai.baseline import assess, extract_use_case
from spectro_ai.intake import correct_requirements


def test_corrections_preserve_units_source_and_unmodified_facts():
    case = extract_use_case("Monitor ethanol in aqueous fermentation broth.\nPressure: 0 bar gauge")
    assert case.target_analyte_or_property == "ethanol"
    assert case.temperature is None
    corrected = correct_requirements(case, {"detection_limit": "0.1 % v/v", "pressure": "TBD"})
    assert corrected.source_text == case.source_text
    assert corrected.sample_matrix == case.sample_matrix
    assert corrected.detection_limit == "0.1 % v/v"
    assert corrected.pressure is None
    assert corrected.field_provenance["pressure"] == "expert edit"
    assert "customer literal" in corrected.field_provenance["sample_matrix"]
    assert assess(corrected).candidate_techniques
    with pytest.raises(ValueError):
        correct_requirements(case, {"source_text": "replace"})


def test_expert_followup_and_pending_export_hiding():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=15
    ).run()
    app.text_area[0].input("We need monitoring.").run()
    app.button[0].click().run()
    assert app.session_state["report"].status == "insufficient_information"
    app.text_input(key="review_measurement_objective").input("monitor ethanol").run()
    assert not app.get("download_button")
    assert not app.json
    app.text_input(key="review_target_analyte_or_property").input("ethanol").run()
    app.text_input(key="review_sample_matrix").input("aqueous fermentation broth").run()
    app.text_input(key="review_physical_state").input("liquid").run()
    app.text_input(key="review_deployment_mode").input("in-line").run()
    next(b for b in app.button if b.label == "Apply expert corrections").click().run()
    assert not app.exception
    assert app.session_state["report"].status == "provisional_candidates"
    assert app.get("download_button")
    assert app.session_state["report"].extracted_requirements.temperature is None
    app.text_input(key="review_interference").input("strong fluorescence; cells and bubbles").run()
    app.text_input(key="review_detection_limit").input("0.1 % v/v").run()
    assert not app.get("download_button")
    next(b for b in app.button if b.label == "Apply expert corrections").click().run()
    report = app.session_state["report"]
    assert [c.spectroscopy_method for c in report.candidate_techniques] == ["Raman", "MIR/FTIR"]
    assert "obscure Raman" in report.candidate_techniques[0].rule_rationale
    assert report.extracted_requirements.source_text == "We need monitoring."
    assert report.extracted_requirements.detection_limit == "0.1 % v/v"
    assert "0.1 % v/v" in app.markdown[-1].value
    assert app.get("download_button")
