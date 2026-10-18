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


def _support_for(report: AssessmentReport, support_id: str | None):
    assessment = report.evidence_assessment
    if assessment is None or support_id is None:
        return None
    return next(
        (
            item
            for item in assessment.support_assessments
            if item.support_assessment_id == support_id
        ),
        None,
    )


def _reference_lines(reference, evidence_by_id: dict) -> list[str]:
    item = evidence_by_id.get(reference.evidence_id)
    if item is None:
        return [f"  - Unresolved evidence reference `{_safe(reference.evidence_id)}`"]
    context = next(
        (block for block in item.context_blocks if block.block_id == reference.context_block_id),
        None,
    )
    location = (
        _location_text(context.source_locations)
        if context
        else _location_text(item.source_locations)
        if item.source_locations
        else item.page_or_section or "location unavailable"
    )
    return [
        f"  - {_safe(item.source_id)} {_safe(location)} · "
        f"version {_safe(item.document_version or 'unspecified')} · "
        f"review {_safe(context.review_status if context else item.review_status)}",
        f"    > {_safe(reference.quote)}",
    ]


def _evidence_assessment_lines(report: AssessmentReport) -> list[str]:
    assessment = report.evidence_assessment
    if assessment is None:
        return []
    evidence_by_id = {item.evidence_id: item for item in report.retrieved_evidence}
    lines = ["", "## Application-aware evidence assessment", ""]
    prompt_tokens = assessment.diagnostics.prompt_tokens
    completion_tokens = assessment.diagnostics.completion_tokens
    elapsed_ms = (
        assessment.diagnostics.elapsed_ms
        if assessment.diagnostics.elapsed_ms is not None
        else "unknown"
    )
    lines += [
        f"Fingerprint: `{_safe(assessment.input_fingerprint)}` · "
        f"assessor {_safe(assessment.assessor_version)} · "
        f"model {_safe(assessment.model_identity or 'unspecified')}",
        f"Diagnostics: {_safe(assessment.diagnostics.status)} · "
        f"{assessment.diagnostics.call_count} calls · "
        f"{elapsed_ms} ms · "
        f"{assessment.diagnostics.serialized_bytes or 'unknown'} serialized bytes · "
        f"{prompt_tokens if prompt_tokens is not None else 'unknown'} input tokens · "
        f"{completion_tokens if completion_tokens is not None else 'unknown'} output tokens",
        "Semantic entailment and transfer judgments are model-assessed, "
        "not deterministic verification.",
    ]
    if assessment.diagnostics.dropped_evidence_ids:
        lines.append(
            "Dropped evidence IDs: " + _safe(", ".join(assessment.diagnostics.dropped_evidence_ids))
        )
    for observation in assessment.observations:
        lines += [
            "",
            f"### Observation {_safe(observation.observation_id)} · "
            f"{_safe(observation.statement_kind)}",
            "",
            f"Statement: {_safe(observation.source_statement)}",
            f"Scenario: {_safe(observation.scenario_id)} · method "
            f"{_safe(observation.method or 'unspecified')} · geometry "
            f"{_safe(observation.geometry or 'unspecified')}",
            f"Entailment: {_safe(observation.entailment)} · "
            f"applicability: {_safe(observation.applicability)} · "
            f"context: {_safe(observation.context_availability)}",
            "Source references:",
        ]
        for reference in observation.references:
            lines.extend(_reference_lines(reference, evidence_by_id))
        lines += ["Condition comparisons:"]
        for comparison in observation.condition_comparisons:
            source_value = comparison.source_value or "unknown"
            customer_fields = ", ".join(comparison.customer_fields) or "none"
            lines.append(
                f"- {_safe(comparison.dimension)}: source {_safe(source_value)}; "
                f"customer fields {_safe(customer_fields)}; "
                f"comparison {_safe(comparison.comparison)}; {_safe(comparison.rationale)}"
            )
            for reference in comparison.supporting_references:
                lines.extend(_reference_lines(reference, evidence_by_id))
        lines += ["Transfer limitations:"] + _bullets(observation.transfer_limitations)
    lines += ["", "### Candidate consideration assessments", ""]
    for support in assessment.support_assessments:
        lines += [
            f"- {_safe(support.method)} · {_safe(support.basis)}: "
            f"{_safe(support.bounded_conclusion)}",
            f"  Observations: {_safe(', '.join(support.observation_ids) or 'none')}",
        ]
        if support.rejection_reason:
            lines.append(f"  Rejection reason: {_safe(support.rejection_reason)}")
        bridge = support.exploratory_bridge
        if bridge:
            lines.extend(
                [
                    f"  Engineering reasoning: {_safe(bridge.inference)}",
                    f"  Hypothesis: {_safe(bridge.hypothesis)}",
                    "  Source premises:",
                ]
            )
            for reference in bridge.source_premises:
                lines.extend(_reference_lines(reference, evidence_by_id))
            lines += ["  Application differences:"] + _bullets(bridge.application_differences)
            lines += ["  Unknown gaps:"] + _bullets(bridge.unknown_gaps)
            lines += ["  Possible failure mechanisms:"] + _bullets(
                bridge.possible_failure_mechanisms
            )
            lines += ["  Checks before application-specific study:"] + _bullets(bridge.checks)
    if assessment.conflicts:
        lines += ["", "### Conflicts", ""]
        for conflict in assessment.conflicts:
            lines.append(
                f"- {_safe(conflict.description)} · observations "
                f"{_safe(', '.join(conflict.observation_ids))}"
            )
            for reference in conflict.references:
                lines.extend(_reference_lines(reference, evidence_by_id))
    lines += ["", "### Evidence coverage", ""]
    for coverage in assessment.coverage:
        lines.append(
            f"- {_safe(coverage.evidence_id)}: {_safe(coverage.disposition)} · "
            f"context {_safe(coverage.context_availability)}"
        )
        lines.extend(
            f"  - Context unavailable/omitted: {_safe(item)}" for item in coverage.omissions
        )
    lines += ["", "### Assessment diagnostics", ""] + _bullets(assessment.diagnostics.warnings)
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
    if case.inferred_customer_context:
        lines += ["", "## Unverified inferred customer context", ""] + _bullets(
            [
                f"{item.field}: {item.value} — {item.rationale} "
                f"Supporting wording: {item.supporting_quote}"
                for item in case.inferred_customer_context
            ]
        )
    lines += ["", "## Supporting customer quotes", ""] + _bullets(
        [f"{field}: {quote}" for field, quote in case.field_quotes.items()]
    )
    lines += ["", "## Extraction warnings", ""] + _bullets(case.extraction_warnings)
    lines += ["", "## Assessment warnings", ""] + _bullets(report.assessment_warnings)
    if report.request_budget_diagnostics:
        lines += ["", "## Request budget diagnostics", ""]
        for item in report.request_budget_diagnostics:
            tokens = sum(item.component_estimated_tokens.values())
            lines.append(
                f"- {_safe(item.stage)} attempt {item.attempt} · "
                f"{_safe(item.provider or 'unknown')}:{_safe(item.model or 'unknown')} · "
                f"estimator {_safe(item.estimator_version)} · {item.fit_decision}"
            )
            lines.append(
                f"  - Estimated input {tokens} tokens; context {item.context_limit}; "
                f"output reserve {item.output_reserve}; margin {item.safety_margin}."
            )
            lines.append(
                "  - Components: "
                + "; ".join(
                    f"{_safe(name)} {item.component_bytes.get(name, 0)} bytes / "
                    f"{count} estimated tokens"
                    for name, count in item.component_estimated_tokens.items()
                )
            )
            lines.append(
                "  - Evidence IDs: selected "
                + _safe(", ".join(item.selected_evidence_ids) or "none")
                + "; retained "
                + _safe(", ".join(item.retained_evidence_ids) or "none")
                + "; dropped "
                + _safe(", ".join(item.dropped_evidence_ids) or "none")
            )
            prompt_usage = (
                item.provider_prompt_tokens
                if item.provider_prompt_tokens is not None
                else "unknown"
            )
            completion_usage = (
                item.provider_completion_tokens
                if item.provider_completion_tokens is not None
                else "unknown"
            )
            usage = f"{prompt_usage} prompt / {completion_usage} completion tokens"
            lines.append(f"  - Provider-reported usage: {usage}.")
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
    query = report.query_diagnostics
    if query is not None:
        lines += ["", "## Multi-query retrieval diagnostics", ""]
        lines.append(
            f"- Status: {_safe(query.status)}; {query.candidate_count} candidates; "
            f"{query.unique_document_count} documents; {query.selected_count} selected; "
            f"{query.elapsed_ms:.1f} ms"
        )
        if query.model_identity:
            lines.append(f"- Planner: {_safe(query.model_identity)}")
        if query.prompt_tokens is not None or query.completion_tokens is not None:
            input_tokens = query.prompt_tokens if query.prompt_tokens is not None else "unknown"
            output_tokens = (
                query.completion_tokens if query.completion_tokens is not None else "unknown"
            )
            lines.append(f"- Planner tokens: {input_tokens} input; {output_tokens} output")
        lines.append(f"- Planner calls: {query.planner_call_count}")
        if query.reason_codes:
            readable_reasons = ", ".join(code.replace("_", " ") for code in query.reason_codes)
            lines.append("- Reason codes: " + _safe(readable_reasons))
        if query.accepted_questions:
            lines += ["- Accepted questions:"]
            lines.extend(
                f"  - {_safe(item.question)} (anchors: {_safe(', '.join(item.anchor_fields))})"
                for item in query.accepted_questions
            )
        if query.question_evidence:
            lines += ["- Question evidence membership:"]
            lines.extend(
                f"  - {_safe(question)}: {_safe(', '.join(ids)) or 'no candidates'}"
                for question, ids in query.question_evidence.items()
            )
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
            f"Source-described geometry: "
            f"{_safe(candidate.measurement_geometry or 'Undetermined')} · "
            "customer compatibility not established.",
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
    lines.extend(_evidence_assessment_lines(report))
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
    if (
        report.evidence_assessment is not None
        and report.evidence_assessment.diagnostics.dropped_evidence_ids
    ):
        lines[6:6] = [
            "Evidence limitation: some retrieved evidence bundles could not be assessed "
            "within the available budget; conclusions may be incomplete.",
            "",
        ]
    for field in CUSTOMER_FACT_FIELDS:
        value = getattr(case, field)
        if value:
            lines.append(f"- {field.replace('_', ' ').title()}: {_safe(value)}")
    lines += ["", "## Missing information", ""] + _bullets(
        [field.replace("_", " ") for field in report.missing_information]
    )
    if case.inferred_customer_context:
        lines += ["", "## Unverified inferred context", ""] + _bullets(
            [
                f"{item.field}: {item.value} (unverified inference; "
                f"based on: {item.supporting_quote})"
                for item in case.inferred_customer_context
            ]
        )
    lines += ["", "## Follow-up questions", ""] + _bullets(report.suggested_technical_questions)
    lines += ["", "## Preliminary candidates", ""]
    sources = {item.evidence_id: item for item in report.retrieved_evidence}
    for candidate in report.candidate_techniques:
        lines += [f"### {_safe(candidate.spectroscopy_method)}", ""]
        support = _support_for(report, candidate.support_assessment_id)
        if candidate.consideration_basis:
            lines += [
                f"Consideration basis: **{_safe(candidate.consideration_basis)}**",
                "",
            ]
        if support:
            lines += [_safe(support.bounded_conclusion), ""]
            bridge = support.exploratory_bridge
            if bridge:
                lines += [
                    f"Engineering reasoning: {_safe(bridge.inference)}",
                    f"Exploratory hypothesis: {_safe(bridge.hypothesis)}",
                    "",
                    "Application differences and unknowns:",
                    *_bullets(bridge.application_differences + bridge.unknown_gaps),
                    "",
                    "Checks before application-specific study:",
                    *_bullets(bridge.checks),
                    "",
                ]
        if report.evidence_assessment is not None and support:
            observation_ids = set(support.observation_ids)
            observations = [
                item
                for item in report.evidence_assessment.observations
                if item.observation_id in observation_ids
            ]
            unreviewed = {
                reference.evidence_id
                for observation in observations
                for reference in observation.references
                if sources.get(reference.evidence_id)
                and sources[reference.evidence_id].review_status
                not in {"reviewed", "reviewed_for_phase2", "source_reviewed"}
            }
            if unreviewed:
                lines += [
                    "Source review warning: provisional source status ("
                    + ", ".join(
                        _safe(f"{item}: {sources[item].review_status}")
                        for item in sorted(unreviewed)
                    )
                    + "). Review status does not determine scientific applicability.",
                    "",
                ]
            coverage = [
                item
                for item in report.evidence_assessment.coverage
                if item.evidence_id
                in {
                    reference.evidence_id
                    for observation in observations
                    for reference in observation.references
                }
            ]
            if any(
                item.context_availability in {"child_only", "unavailable"} for item in observations
            ) or any(
                item.context_availability in {"child_only", "unavailable"} or item.omissions
                for item in coverage
            ):
                lines += [
                    "Context warning: some parent context or qualifications may be unavailable; "
                    "application matches cannot be inferred from missing context.",
                    "",
                ]
        for heading, values in (
            (
                "Application assessment"
                if candidate.consideration_basis
                else "Source observations",
                candidate.advantages,
            ),
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
