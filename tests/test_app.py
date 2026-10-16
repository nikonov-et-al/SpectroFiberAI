from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_example_selection_assessment_and_structured_output(scenarios: list[dict]) -> None:
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value == "Spectro-AI"
    for scenario in scenarios:
        app.selectbox[0].select(scenario["title"]).run()
        assert app.text_area[0].value == scenario["problem"]
        app.button[0].click().run()
        assert not app.exception
        assert app.json
        report = app.session_state["report"]
        assert [c.spectroscopy_method for c in report.candidate_techniques] == scenario[
            "expected_methods"
        ]
        assert "## Extracted requirements" in app.markdown[-1].value
        assert len(app.get("download_button")) == 1


def test_blank_input_and_stale_report_behavior() -> None:
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()
    assert app.error[0].value.startswith("Enter a customer problem")
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[0].click().run()
    assert app.json
    app.text_area[0].input("Changed problem.").run()
    assert app.info[0].value.startswith("Input changed")
    assert not app.json
    assert not app.get("download_button")


def test_provider_error_visible_without_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "openai")
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[0].click().run()
    assert not app.exception
    assert "not implemented" in app.error[0].value
    assert not app.json
