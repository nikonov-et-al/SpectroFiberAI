import json
from pathlib import Path

from spectro_ai.baseline import assess, extract_use_case
from spectro_ai.models import AssessmentReport
from spectro_ai.reporting import render_markdown


def test_six_review_scenarios_and_exports():
    cases = json.loads(Path("examples/evaluation.json").read_text(encoding="utf-8-sig"))
    assert len(cases) == 6
    for case in cases:
        report = assess(extract_use_case(case["text"]))
        assert report.status == case["status"], case["id"]
        assert [c.spectroscopy_method for c in report.candidate_techniques] == case["methods"]
        assert set(case["unknowns"]) <= set(report.missing_information)
        assert AssessmentReport.model_validate_json(report.model_dump_json()) == report
        markdown = render_markdown(report)
        assert "expert" in markdown.lower()
        assert "Success Actions" in markdown and "Failure Actions" in markdown
        assert report.proposed_feasibility_experiment.success_actions
        assert report.proposed_feasibility_experiment.failure_actions
        if case["id"] == "fluorescence_background":
            assert "obscure Raman" in report.candidate_techniques[0].rule_rationale
            assert "Fluorescence" not in case["methods"]
        if case["id"] == "contamination":
            assert report.extracted_requirements.detection_limit == "1 ppm"
            assert "Which analyte" in report.suggested_technical_questions[0]
