"""Reproduce six synthetic assessments and reviewable exports without providers."""

import json
from pathlib import Path

from spectro_ai.baseline import assess, extract_use_case
from spectro_ai.reporting import render_markdown


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "examples/evaluation.json").read_text(encoding="utf-8-sig"))
    output = root / ".cache/evaluation"
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        report = assess(extract_use_case(case["text"]))
        (output / f"{case['id']}.md").write_text(render_markdown(report), encoding="utf-8")
        (output / f"{case['id']}.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"{case['id']}: {report.status}; {len(report.candidate_techniques)} candidates")
    print(f"Synthetic review outputs: {output}")


if __name__ == "__main__":
    main()
