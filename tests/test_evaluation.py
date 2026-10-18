import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from spectro_ai.guardrails import abstention_report
from spectro_ai.models import CustomerUseCase

ROOT = Path(__file__).resolve().parents[1]
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
SPEC = importlib.util.spec_from_file_location("spectro_ai_evaluate", ROOT / "scripts/evaluate.py")
assert SPEC and SPEC.loader
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def _write_docx_fixture(path: Path) -> None:
    def tag(name: str) -> str:
        return f"{{{WORD_NS}}}{name}"

    document = ET.Element(tag("document"))
    body = ET.SubElement(document, tag("body"))

    def paragraph(text: str) -> None:
        node = ET.SubElement(body, tag("p"))
        run = ET.SubElement(node, tag("r"))
        ET.SubElement(run, tag("t")).text = text

    # Actual source has a table-of-contents entry before the development section.
    paragraph("2. Worked examples D01 and D02")
    paragraph("1. Development scenarios D01-D08")
    for index in range(1, 9):
        case_id = f"D{index:02}"
        paragraph(f"{case_id} · Development case {index}")
        paragraph(f"Customer enquiry for {case_id}; API concentration changes are monitored.")
        paragraph("Facts stated  Test-only organizer notes.")
        paragraph("Useful sources  S01 p.5")
    paragraph("2. Worked examples D01 and D02")
    paragraph("D01 · Worked answer")
    paragraph("Decision status  Test-only answer content.")
    paragraph("Worked answer sentinel must not be extracted.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", ET.tostring(document, encoding="utf-8"))


def _write_holdout_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "H01 | Synthetic holdout one\n"
        "==========\nFirst holdout enquiry only.\n==========\n\n"
        "H02 | Synthetic holdout two\n"
        "==========\nSecond holdout enquiry only.\n==========\n",
        encoding="utf-8",
    )


def _write_dataset_root(path: Path) -> None:
    examples = path / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    (examples / "scenarios.json").write_text(
        (ROOT / "examples/scenarios.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_docx_fixture(path / "knowledge/data/materials/AFS_Examples_and_Reference.docx")
    _write_holdout_fixture(
        path / "knowledge/data/materials/customer_scenarios_for_final_evaluation.txt"
    )


def test_dataset_loaders_keep_only_customer_enquiries(tmp_path) -> None:
    docx_path = tmp_path / "AFS_Examples_and_Reference.docx"
    holdout_path = tmp_path / "customer_scenarios_for_final_evaluation.txt"
    _write_docx_fixture(docx_path)
    _write_holdout_fixture(holdout_path)
    development = evaluate._development_cases(docx_path)
    holdout = evaluate._holdout_cases(holdout_path)
    assert [case["id"] for case in development] == [f"D{i:02}" for i in range(1, 9)]
    assert [case["id"] for case in holdout] == ["H01", "H02"]
    for case in development + holdout:
        assert case["text"].strip()
        assert "Facts stated" not in case["text"]
        assert "Decision status" not in case["text"]
    assert "API concentration changes" in development[0]["text"]
    assert "Development case 1" not in development[0]["text"]
    assert "Worked answer sentinel" not in " ".join(case["text"] for case in development)


def test_named_datasets_and_all_have_unique_case_ids(tmp_path) -> None:
    _write_dataset_root(tmp_path)
    assert len(evaluate._dataset(ROOT, "paraphrase")) == 8
    assert len(evaluate._dataset(ROOT, "adversarial")) == 5
    for name in ("synthetic", "development", "paraphrase", "adversarial", "holdout", "all"):
        cases = evaluate._dataset(tmp_path, name)
        assert cases
        assert len({case["id"] for case in cases}) == len(cases)


def test_missing_dataset_files_error_without_fallback(tmp_path) -> None:
    with pytest.raises(ValueError, match="Cannot read development enquiries"):
        evaluate._dataset(tmp_path, "development")
    with pytest.raises(ValueError, match="Cannot read holdout enquiries"):
        evaluate._dataset(tmp_path, "holdout")
    with pytest.raises(ValueError, match="Cannot read evaluation cases"):
        evaluate._dataset(tmp_path, "synthetic")


def test_custom_json_cases_keep_cli_compatibility(tmp_path, monkeypatch, capsys) -> None:
    case_file = tmp_path / "cases.json"
    case_file.write_text(json.dumps([{"id": "example", "text": "An enquiry."}]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--cases", str(case_file), "--output", str(tmp_path / "out")],
    )
    seen = []

    def assess(text):
        seen.append(text)
        return abstention_report(CustomerUseCase(source_text=text), "No evidence in test.")

    monkeypatch.setattr(evaluate, "assess_problem", assess)
    evaluate.main()
    output = tmp_path / "out/custom"
    assert seen == ["An enquiry."]
    assert (output / "example.json").is_file()
    assert (output / "example.md").is_file()
    assert json.loads((output / "summary.json").read_text())["completed"] == 1
    assert "Completed: 1" in capsys.readouterr().out


def test_failed_case_recorded_and_later_cases_still_run(tmp_path) -> None:
    cases = [
        {"id": "bad", "text": "fails", "group": "test"},
        {"id": "good", "text": "works", "group": "test"},
    ]
    seen = []

    def assess(text):
        seen.append(text)
        if text == "fails":
            raise RuntimeError("synthetic provider failure")
        return abstention_report(CustomerUseCase(source_text=text), "No evidence in test.")

    summary = evaluate._run_cases(cases, tmp_path, assess)
    assert seen == ["fails", "works"]
    assert summary == {"completed": 1, "failed": 1}
    failure = json.loads((tmp_path / "bad.error.json").read_text(encoding="utf-8"))
    assert failure["error"] == "synthetic provider failure"
    assert failure["enquiry"] == "fails"
    assert (tmp_path / "good.json").is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["failed"] == 1


def test_reruns_remove_only_stale_artifacts_for_same_case(tmp_path) -> None:
    case = {"id": "case", "text": "enquiry", "group": "test"}
    report = abstention_report(CustomerUseCase(source_text="enquiry"), "No evidence in test.")
    evaluate._run_cases([case], tmp_path, lambda _: report)
    assert (tmp_path / "case.md").exists()
    assert (tmp_path / "case.json").exists()

    def fail(_):
        raise RuntimeError("provider failed")

    evaluate._run_cases([case], tmp_path, fail)
    assert not (tmp_path / "case.md").exists()
    assert not (tmp_path / "case.json").exists()
    assert (tmp_path / "case.error.json").exists()

    evaluate._run_cases([case], tmp_path, lambda _: report)
    assert (tmp_path / "case.md").exists()
    assert (tmp_path / "case.json").exists()
    assert not (tmp_path / "case.error.json").exists()


def test_invalid_custom_case_rejected(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([{"id": "empty", "text": "  "}]))
    with pytest.raises(ValueError, match="no enquiry text"):
        evaluate._json_cases(path, "custom")
