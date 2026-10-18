import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from spectro_ai import providers
from spectro_ai.models import (
    AssessmentReport,
    CustomerUseCase,
    EvidenceClaim,
    FeasibilityPlan,
    TechniqueCandidate,
    ValidationDiagnostic,
)

APP = Path(__file__).resolve().parents[1] / "app.py"


def _abstention(text: str) -> AssessmentReport:
    return AssessmentReport(
        extracted_requirements=CustomerUseCase(source_text=text),
        proposed_feasibility_experiment=FeasibilityPlan(),
        status="insufficient_information",
        assessment_method="evidence_abstention",
        assessment_provider=None,
    )


def test_four_synthetic_examples_use_injected_assessment(scenarios, monkeypatch) -> None:
    requests = []

    def assess(text: str) -> AssessmentReport:
        requests.append(text)
        return _abstention(text)

    monkeypatch.setattr(providers, "assess_problem", assess)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value == "Spectro-AI"
    assert "Evidence-grounded" in app.caption[0].value
    for scenario in scenarios:
        app.selectbox[0].select(scenario["title"]).run()
        assert app.text_area[0].value == scenario["problem"]
        app.button[0].click().run()
        assert not app.exception
        assert app.json
        assert app.session_state["report"].candidate_techniques == []
        assert len(app.get("download_button")) == 2
    assert requests == [scenario["problem"] for scenario in scenarios]


def test_blank_input_and_stale_report_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "assess_problem", lambda text: _abstention(text))
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()
    assert app.error[0].value.startswith("Enter a customer problem")
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[0].click().run()
    assert app.json
    assert len(app.get("download_button")) == 2
    app.text_area[0].input("Changed problem.").run()
    assert not app.json
    assert not app.get("download_button")


def test_provider_error_clears_report_and_openai_warns_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "openai")
    calls = []

    def fail(text: str) -> None:
        calls.append(text)
        raise ValueError("Synthetic model unavailable")

    monkeypatch.setattr(providers, "assess_problem", fail)
    app = AppTest.from_file(str(APP)).run()
    assert any("sends the customer enquiry" in warning.value for warning in app.warning)
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[0].click().run()
    assert not app.exception
    assert "Synthetic model unavailable" in app.error[0].value
    assert not app.json
    assert not app.get("download_button")
    assert calls == ["We measure moisture in flour."]


def test_provider_configuration_change_clears_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "assess_problem", lambda text: _abstention(text))
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[0].click().run()
    assert app.json
    monkeypatch.setenv("OLLAMA_MODEL", "changed-model")
    app.run()
    assert not app.json
    assert not app.get("download_button")
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "changed-openai-model")
    app.run()
    assert not app.json
    assert not app.get("download_button")
    assert any("Retrieved source excerpts" in warning.value for warning in app.warning)


def test_failed_expert_correction_preserves_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "assess_problem", lambda text: _abstention(text))

    def fail(use_case: CustomerUseCase) -> AssessmentReport:
        assert use_case.target_analyte_or_property == "moisture"
        raise ValueError("Synthetic correction failure")

    monkeypatch.setattr(providers, "assess_use_case", fail)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[0].click().run()
    app.text_input(key="review_target_analyte_or_property").input("moisture").run()
    app.button[1].click().run()
    assert not app.exception
    assert any("Synthetic correction failure" in error.value for error in app.error)
    assert not app.json
    assert not app.get("download_button")


def test_json_export_and_diagnostics_are_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _abstention("We measure moisture in flour.")
    report.candidate_techniques = [
        TechniqueCandidate(
            spectroscopy_method="NIR",
            evidence_claims=[
                EvidenceClaim(
                    claim="The source describes powder measurement.",
                    evidence_id="source-1-page-3",
                    quote="A literal source excerpt.",
                )
            ],
        )
    ]
    if hasattr(report, "validation_diagnostics"):
        report.validation_diagnostics = [
            ValidationDiagnostic(
                level="not_checked",
                check="scientific_suitability",
                message="Expert review required.",
            )
        ]
    monkeypatch.setattr(providers, "assess_problem", lambda text: report)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.get("download_button")) == 2
    assert any("A literal source excerpt." in item.value for item in app.markdown)
    structured = json.loads(app.json[0].value)
    assert structured["validation_diagnostics"][0]["check"] == "scientific_suitability"
    if hasattr(report, "validation_diagnostics"):
        assert any("scientific_suitability" in item.value for item in app.info)
