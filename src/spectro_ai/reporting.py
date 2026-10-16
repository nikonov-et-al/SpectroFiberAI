"""Stable Markdown export; no time, randomness, or network dependencies."""

import html
import re

from spectro_ai.models import AssessmentReport, CustomerUseCase


def _safe(value: object) -> str:
    text = html.escape(str(value), quote=False).replace("\n", " / ")
    return re.sub(r"([\\`*_{}\[\]()#+|])", r"\\\1", text)


def _bullets(values: list[str]) -> list[str]:
    return [f"- {_safe(value)}" for value in values] or ["- None recorded."]


def render_markdown(report: AssessmentReport) -> str:
    case = report.extracted_requirements
    lines = [
        "# Spectroscopy assessment",
        "",
        "**Provisional deterministic baseline. Expert verification required.**",
        "",
        f"Status: **{report.status.replace('_', ' ')}**",
        "",
        "## Extracted requirements",
        "",
        "| Field | Supplied or recognized value |",
        "| --- | --- |",
    ]
    for field in CustomerUseCase.model_fields:
        if field not in {"source_text", "unknown_or_missing_information"}:
            value = getattr(case, field)
            lines.append(f"| {field.replace('_', ' ').title()} | {_safe(value or 'Unknown')} |")
    lines += ["", "## Original customer text", ""]
    source = case.source_text or "No input supplied."
    # A longer fence safely contains arbitrary customer backticks.
    fence = "`" * max(3, max((len(m) + 1 for m in re.findall(r"`+", source)), default=0))
    lines += [fence + "text", source, fence]
    lines += ["", "## Missing information", ""] + _bullets(report.missing_information)
    lines += ["", "## Follow-up questions", ""] + _bullets(report.suggested_technical_questions)
    lines += ["", "## Preliminary candidates", "", _safe(report.recommendation_rationale), ""]
    for candidate in report.candidate_techniques:
        lines += [
            f"### {_safe(candidate.spectroscopy_method)}",
            "",
            f"Suitability: {_safe(candidate.preliminary_suitability_assessment)}",
            "",
            f"Rule: {_safe(candidate.rule_rationale)}",
            "",
            f"Spectral region: {_safe(candidate.proposed_spectral_region or 'Undetermined')}",
            "",
            f"Geometry: {_safe(candidate.measurement_geometry or 'Undetermined')}",
            "",
            f"Probe: {_safe(candidate.potential_probe_configuration or 'Undetermined')}",
        ]
        for heading, values in (
            ("Potential advantages", candidate.advantages),
            ("Limitations and sample considerations", candidate.limitations),
            ("Required assumptions", candidate.required_assumptions),
        ):
            lines += ["", f"**{heading}**", ""] + _bullets(values)
        lines += ["", "**Supporting evidence and provenance**", ""]
        for evidence in candidate.supporting_evidence:
            entry = f"- [{evidence.kind}] {_safe(evidence.statement)}"
            if evidence.source_url:
                entry += f" [{_safe(evidence.source_title)}]({evidence.source_url})"
                entry += f" (source checked {evidence.checked_on}; applicability unverified)."
            lines.append(entry)
    plan = report.proposed_feasibility_experiment
    lines += ["", "## Proposed feasibility experiment", ""]
    for field in (
        "objective",
        "hypothesis",
        "proposed_measurement_technique",
        "sample_preparation",
        "reference_measurement",
    ):
        label = field.replace("_", " ").title()
        value = _safe(getattr(plan, field) or "Undetermined")
        lines += [
            f"**{label}:** {value}",
            "",
        ]
    for field in (
        "experimental_steps",
        "evaluation_metrics",
        "expected_limitations",
        "success_criteria",
    ):
        lines += (
            [f"### {field.replace('_', ' ').title()}", ""] + _bullets(getattr(plan, field)) + [""]
        )
    lines += ["## Assumptions and uncertainties", ""] + _bullets(
        report.assumptions_and_uncertainties
    )
    return "\n".join(lines).rstrip() + "\n"
