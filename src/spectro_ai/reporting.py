"""Stable Markdown export; no time, randomness, or network dependencies."""

import html
import re

from spectro_ai.models import CUSTOMER_FACT_FIELDS, AssessmentReport


def _safe(value: object) -> str:
    text = html.escape(str(value), quote=False).replace("\n", " / ")
    return re.sub(r"([\\`*_{}\[\]()#+|])", r"\\\1", text)


def _bullets(values: list[str]) -> list[str]:
    return [f"- {_safe(value)}" for value in values] or ["- None recorded."]


def _location_text(locations: list) -> str:
    return ", ".join(location.locator for location in locations) or "location unavailable"


def _context_lines(item, *, indent: str = "  ") -> list[str]:
    lines = []
    for block in item.context_blocks:
        lines += [
            f"{indent}- Same-source context · block `{html.escape(block.block_id, quote=False)}` · "
            f"{_safe(_location_text(block.source_locations))} · "
            f"scope: {_safe(block.source_scope)} · review: {_safe(block.review_status)} · "
            f"version: {_safe(block.document_version)}",
            f"{indent}  > {_safe(block.text)}",
        ]
    return lines


def render_markdown(report: AssessmentReport, *, include_diagnostics: bool = False) -> str:
    """Render the customer report; technical detail is explicitly opt-in."""
    if not include_diagnostics:
        return _customer_markdown(report)
    case = report.extracted_requirements
    lines = [
        "# Spectroscopy assessment",
        "",
        (
            "**Assessment abstained because source evidence or required facts were insufficient.**"
            if report.assessment_method == "evidence_abstention"
            else "**LLM-generated assessment with deterministic checks; expert review required.**"
        ),
        "",
        f"Assessment method: **{_safe(report.assessment_method)}** · provider "
        f"**{_safe(report.assessment_provider or 'not run')}** · report "
        f"{_safe(report.report_version)}",
        f"Extraction mode: **{_safe(case.extraction_mode)}**",
        "",
        f"Status: **{report.status.replace('_', ' ')}**",
        "",
        "## Extracted requirements",
        "",
        "| Field | Supplied or recognized value |",
        "| --- | --- |",
    ]
    for field in (*CUSTOMER_FACT_FIELDS, "customer_preferences", "strict_requirements"):
        value = getattr(case, field)
        if isinstance(value, list):
            display = "; ".join(value) or "None stated"
        else:
            display = value or "Unknown"
        lines.append(f"| {field.replace('_', ' ').title()} | {_safe(display)} |")
    lines += ["", "## Field provenance", ""] + _bullets(
        [f"{field}: {origin}" for field, origin in case.field_provenance.items()]
    )
    lines += ["", "## Supporting customer quotes", ""] + _bullets(
        [f"{field}: {quote}" for field, quote in case.field_quotes.items()]
    )
    lines += ["", "## Extraction warnings", ""] + _bullets(case.extraction_warnings)
    lines += ["", "## Assessment warnings", ""] + _bullets(report.assessment_warnings)
    diagnostics = getattr(report, "validation_diagnostics", [])
    if diagnostics:
        lines += ["", "## Deterministic validation", ""]
        for item in diagnostics:
            references = f" · evidence: {', '.join(item.evidence_ids)}" if item.evidence_ids else ""
            lines.append(
                f"- **{_safe(item.level)} · {_safe(item.check)}:** "
                f"{_safe(item.message)}{_safe(references)}"
            )
    retrieval = getattr(report, "retrieval_diagnostics", [])
    if retrieval:
        lines += ["", "## Retrieval diagnostics", ""]
        for item in retrieval:
            detail = f"{item.status}; {item.candidate_count} candidates; {item.elapsed_ms:.1f} ms"
            if item.model_revision:
                detail += f"; model revision {item.model_revision}"
            if item.failure_reason:
                detail += f"; fallback reason: {item.failure_reason}"
            lines.append(f"- {_safe(detail)}")
    lines += ["", "## Original customer text", ""]
    source = case.source_text or "No input supplied."
    # A longer fence safely contains arbitrary customer backticks.
    fence = "`" * max(3, max((len(m) + 1 for m in re.findall(r"`+", source)), default=0))
    lines += [fence + "text", source, fence]
    lines += ["", "## Missing information", ""] + _bullets(report.missing_information)
    lines += ["", "## Follow-up questions", ""] + _bullets(report.suggested_technical_questions)
    rationale = report.recommendation_rationale or "No recommendation is made."
    lines += ["", "## Preliminary candidates", "", _safe(rationale), ""]
    evidence_by_id = {item.evidence_id: item for item in report.retrieved_evidence}
    for candidate in report.candidate_techniques:
        lines += [
            f"### {_safe(candidate.spectroscopy_method)}",
            "",
            f"Suitability: {_safe(candidate.preliminary_suitability_assessment)}",
            "",
            f"Spectral region: {_safe(candidate.proposed_spectral_region or 'Undetermined')}",
            "",
            f"Geometry: {_safe(candidate.measurement_geometry or 'Undetermined')}",
            "",
            f"Probe: {_safe(candidate.potential_probe_configuration or 'Undetermined')}",
        ]
        if candidate.rule_rationale:
            lines += ["", f"Rationale: {_safe(candidate.rule_rationale)}"]
        for heading, values in (
            ("Potential advantages", candidate.advantages),
            ("Limitations and sample considerations", candidate.limitations),
            ("Required assumptions", candidate.required_assumptions),
        ):
            lines += ["", f"**{heading}**", ""] + _bullets(values)
        lines += ["", "**Supporting evidence and provenance**", ""]
        for claim in candidate.evidence_claims:
            source = evidence_by_id.get(claim.evidence_id)
            source_label = f"{claim.evidence_id}"
            if source:
                context = next(
                    (
                        block
                        for block in source.context_blocks
                        if block.block_id == claim.context_block_id
                    ),
                    None,
                )
                page = (
                    f" {_location_text(context.source_locations)}"
                    if context
                    else f" {source.page_or_section}"
                    if source.page_or_section
                    else ""
                )
                source_label = f"{source.source_id}{page} · {claim.evidence_id}"
                if context:
                    source_label += f" · context block {context.block_id}"
            lines += [
                f"- Claim: {_safe(claim.claim)} · source: "
                f"`{html.escape(source_label, quote=False)}`",
                f"  > {_safe(claim.quote)}",
            ]
        for evidence_id in candidate.supporting_evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if item:
                page = f" {item.page_or_section}" if item.page_or_section else ""
                lines.append(
                    f"- {item.source_id}{page} · `{html.escape(item.evidence_id, quote=False)}` · "
                    f"scope: {_safe(item.source_scope)} · review: {_safe(item.review_status)} · "
                    f"version: {_safe(item.document_version or 'unspecified')}"
                )
                lines.append(f"  > {_safe(item.excerpt)}")
                lines.extend(_context_lines(item))
                if item.context_omissions:
                    lines.append("  - Context omitted: " + _safe("; ".join(item.context_omissions)))
        for evidence in candidate.supporting_evidence:
            entry = f"- [{evidence.kind}] {_safe(evidence.statement)}"
            if evidence.source_url:
                entry += f" [{_safe(evidence.source_title)}]({evidence.source_url})"
                entry += f" (source checked {evidence.checked_on}; applicability unverified)."
            lines.append(entry)
    if report.retrieved_evidence:
        lines += ["", "## Retrieved source excerpts", ""]
        for item in report.retrieved_evidence:
            page = f" {item.page_or_section}" if item.page_or_section else ""
            lines += [
                f"- {item.source_id}{page} · `{html.escape(item.evidence_id, quote=False)}` · "
                f"scope: {_safe(item.source_scope)} · review: {_safe(item.review_status)} · "
                f"version: {_safe(item.document_version or 'unspecified')}",
                f"  > {_safe(item.excerpt)}",
            ]
    plan = report.proposed_feasibility_experiment
    lines += ["", "## Proposed feasibility experiment", ""]
    for field in (
        "objective",
        "hypothesis",
        "proposed_measurement_technique",
        "sample_preparation",
        "reference_measurement",
        "proposed_configuration",
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
        "success_actions",
        "failure_actions",
    ):
        lines += (
            [f"### {field.replace('_', ' ').title()}", ""] + _bullets(getattr(plan, field)) + [""]
        )
    lines += ["## Assumptions and uncertainties", ""] + _bullets(
        report.assumptions_and_uncertainties
    )
    return "\n".join(lines).rstrip() + "\n"


def _customer_markdown(report: AssessmentReport) -> str:
    case = report.extracted_requirements
    lines = [
        "# Spectroscopy assessment",
        "",
        f"Status: **{report.status.replace('_', ' ')}**",
        "",
        "Exploratory assessment; customer-specific performance and production readiness "
        "are not established. Expert review required.",
        "",
        _safe(report.recommendation_rationale or "No recommendation is made."),
        "",
        "## Extracted requirements",
        "",
    ]
    for field in CUSTOMER_FACT_FIELDS:
        value = getattr(case, field)
        if value:
            lines.append(f"- {field.replace('_', ' ').title()}: {_safe(value)}")
    lines += ["", "## Missing information", ""] + _bullets(
        [field.replace("_", " ") for field in report.missing_information]
    )
    lines += ["", "## Follow-up questions", ""] + _bullets(report.suggested_technical_questions)
    lines += ["", "## Preliminary candidates", ""]
    sources = {item.evidence_id: item for item in report.retrieved_evidence}
    for candidate in report.candidate_techniques:
        lines += [f"### {_safe(candidate.spectroscopy_method)}", ""]
        for heading, values in (
            ("Source observations", candidate.advantages),
            ("Limitations", candidate.limitations),
            ("Study assumptions", candidate.required_assumptions),
        ):
            lines += [f"**{heading}**", ""] + _bullets(values) + [""]
        for claim in candidate.evidence_claims:
            source = sources.get(claim.evidence_id)
            if source:
                context = next(
                    (
                        block
                        for block in source.context_blocks
                        if block.block_id == claim.context_block_id
                    ),
                    None,
                )
                page = (
                    _location_text(context.source_locations)
                    if context
                    else re.sub(r"\s*\(chunk[^)]*\)", "", source.page_or_section or "")
                )
                lines += [
                    f"Source: {_safe(source.source_id)} {_safe(page)} · "
                    f"document version {_safe(source.document_version or 'unspecified')}",
                    *(
                        [f"Context block: `{html.escape(context.block_id, quote=False)}`"]
                        if context
                        else []
                    ),
                    "",
                    f"> {_safe(claim.quote)}",
                    "",
                ]
    if any(item.context_blocks for item in report.retrieved_evidence):
        lines += ["## Retrieved source context", ""]
        for item in report.retrieved_evidence:
            lines.append(
                f"- {item.source_id} {item.page_or_section or ''} · "
                f"`{html.escape(item.evidence_id, quote=False)}` · attached context; "
                "not independent corroboration"
            )
            lines.extend(_context_lines(item))
    plan = report.proposed_feasibility_experiment
    lines += ["## Proposed feasibility experiment", ""]
    for field in type(plan).model_fields:
        value = getattr(plan, field)
        if not value:
            continue
        lines += [f"**{field.replace('_', ' ').title()}**", ""]
        lines += _bullets(value) if isinstance(value, list) else [_safe(value)]
        lines.append("")
    lines += ["## Assumptions and uncertainties", ""] + _bullets(
        report.assumptions_and_uncertainties
    )
    return "\n".join(lines).rstrip() + "\n"
