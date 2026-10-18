"""Deterministic validation for model-produced evidence assessments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from .models import (
    CUSTOMER_FACT_FIELDS,
    ApplicationSupportAssessment,
    CustomerUseCase,
    EvidenceAssessment,
    EvidenceConditionComparison,
    EvidenceConflict,
    EvidenceCoverage,
    EvidenceObservation,
    EvidenceReference,
    RetrievedEvidence,
)


def evidence_fingerprint(
    customer: CustomerUseCase,
    evidence: list[RetrievedEvidence],
    prompt_version: str = "application-evidence-assessment-v1",
) -> str:
    """Hash exact customer facts, source/context payloads, and prompt identity."""
    payload = {
        "customer": customer.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "prompt_version": prompt_version,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_assessment(
    assessment: EvidenceAssessment,
    customer: CustomerUseCase,
    evidence_by_id: Mapping[str, RetrievedEvidence],
    input_fingerprint: str,
    assessor_version: str,
    model_identity: str | None = None,
) -> EvidenceAssessment:
    """Validate model references and assign host-owned IDs and metadata.

    Invalid observations/support records are dropped with diagnostics. This checks
    structure and quote membership only; method and geometry attribution stays
    model-assessed because this validator does not verify semantic entailment.
    """
    warnings = list(assessment.diagnostics.warnings)
    observations: list[EvidenceObservation] = []
    observation_ids: dict[str, str] = {}

    for index, observation in enumerate(assessment.observations):
        if not observation.observation_id or observation.observation_id in observation_ids:
            warnings.append(f"Dropped observation {index}: missing or duplicate model key")
            continue
        if not observation.references or not all(
            _valid_reference(reference, evidence_by_id) for reference in observation.references
        ):
            warnings.append(f"Dropped observation {index}: invalid or unavailable source reference")
            continue
        source_keys = {
            (
                evidence_by_id[reference.evidence_id].source_id,
                evidence_by_id[reference.evidence_id].document_version,
                evidence_by_id[reference.evidence_id].source_locations[0].filename
                if evidence_by_id[reference.evidence_id].source_locations
                else None,
            )
            for reference in observation.references
        }
        if len(source_keys) != 1:
            warnings.append(f"Dropped observation {index}: references cross source scenarios")
            continue
        if not all(
            _valid_comparison(comparison, customer, evidence_by_id, observation.references)
            for comparison in observation.condition_comparisons
        ):
            warnings.append(f"Dropped observation {index}: invalid condition comparison")
            continue
        host_id = _host_id(input_fingerprint, "observation", index, observation.source_statement)
        observation_ids[observation.observation_id] = host_id
        context_state = _context_state(
            [evidence_by_id[ref.evidence_id] for ref in observation.references]
        )
        observations.append(
            observation.model_copy(
                update={"observation_id": host_id, "context_availability": context_state}
            )
        )

    supports: list[ApplicationSupportAssessment] = []
    support_ids: dict[str, str] = {}
    for index, support in enumerate(assessment.support_assessments):
        if not support.support_assessment_id or support.support_assessment_id in support_ids:
            warnings.append(f"Dropped support assessment {index}: missing or duplicate model key")
            continue
        mapped_ids = [
            observation_ids[item] for item in support.observation_ids if item in observation_ids
        ]
        if len(mapped_ids) != len(support.observation_ids):
            warnings.append(
                f"Dropped support assessment {index}: observation reference unavailable"
            )
            continue
        if not mapped_ids and support.basis != "unsupported":
            warnings.append(f"Dropped support assessment {index}: no valid observations")
            continue
        if support.basis != "unsupported" and not any(
            observation.method and observation.method.casefold() == support.method.casefold()
            for observation in observations
            if observation.observation_id in mapped_ids
        ):
            warnings.append(f"Dropped support assessment {index}: method identity mismatch")
            continue
        bridge = support.exploratory_bridge
        if support.basis == "application_supported" and not any(
            _supports_application(observation, support.method)
            for observation in observations
            if observation.observation_id in mapped_ids
        ):
            warnings.append(
                f"Dropped support assessment {index}: no supported application observation"
            )
            continue
        if support.basis == "reasoned_exploratory" and not _valid_bridge(
            bridge,
            customer,
            evidence_by_id,
            [
                observation
                for observation in observations
                if observation.observation_id in mapped_ids
            ],
            support.method,
        ):
            warnings.append(f"Dropped support assessment {index}: exploratory bridge incomplete")
            continue
        if support.basis == "unsupported" and not support.rejection_reason:
            warnings.append(f"Dropped support assessment {index}: unsupported basis lacks reason")
            continue
        host_id = _host_id(input_fingerprint, "support", index, support.method)
        support_ids[support.support_assessment_id] = host_id
        supports.append(
            support.model_copy(
                update={"support_assessment_id": host_id, "observation_ids": mapped_ids}
            )
        )

    conflicts: list[EvidenceConflict] = []
    conflict_blocked_ids: set[str] = set()
    for conflict in assessment.conflicts:
        mapped = [
            observation_ids[item] for item in conflict.observation_ids if item in observation_ids
        ]
        if (
            len(mapped) < 2
            or len(mapped) != len(conflict.observation_ids)
            or not all(
                _valid_reference(reference, evidence_by_id) for reference in conflict.references
            )
        ):
            conflict_blocked_ids.update(mapped)
            warnings.append("Dropped conflict: invalid observation or evidence reference")
            continue
        conflicts.append(conflict.model_copy(update={"observation_ids": mapped}))
        conflict_blocked_ids.update(mapped)

    if conflict_blocked_ids:
        supports = [
            support
            for support in supports
            if not conflict_blocked_ids.intersection(support.observation_ids)
        ]

    coverage: list[EvidenceCoverage] = []
    seen_coverage: set[str] = set()
    for item in assessment.coverage:
        if item.evidence_id not in evidence_by_id or item.evidence_id in seen_coverage:
            warnings.append(
                f"Dropped coverage entry for unknown or duplicate evidence {item.evidence_id}"
            )
            continue
        seen_coverage.add(item.evidence_id)
        source = evidence_by_id[item.evidence_id]
        coverage.append(
            item.model_copy(
                update={
                    "context_availability": _context_state([source]),
                    "omissions": list(source.context_omissions),
                }
            )
        )
    for evidence_id, item in evidence_by_id.items():
        if evidence_id not in seen_coverage:
            coverage.append(
                EvidenceCoverage(
                    evidence_id=evidence_id,
                    disposition="insufficient_context",
                    context_availability=_context_state([item]),
                    omissions=list(item.context_omissions)
                    or ["No valid coverage assessment supplied"],
                )
            )
            warnings.append(f"Evidence {evidence_id} lacks valid coverage assessment")

    diagnostics = assessment.diagnostics.model_copy(
        update={
            "status": "partial" if warnings else assessment.diagnostics.status,
            "warnings": warnings,
        }
    )
    return assessment.model_copy(
        update={
            "observations": observations,
            "support_assessments": supports,
            "conflicts": conflicts,
            "coverage": coverage,
            "input_fingerprint": input_fingerprint,
            "assessor_version": assessor_version,
            "model_identity": model_identity,
            "diagnostics": diagnostics,
        }
    )


def _valid_reference(
    reference: EvidenceReference, evidence_by_id: Mapping[str, RetrievedEvidence]
) -> bool:
    item = evidence_by_id.get(reference.evidence_id)
    if item is None or not reference.quote:
        return False
    if reference.context_block_id is not None:
        return any(
            block.block_id == reference.context_block_id and reference.quote in block.text
            for block in item.context_blocks
        )
    return reference.quote in item.excerpt


def _valid_comparison(
    comparison: EvidenceConditionComparison,
    customer: CustomerUseCase,
    evidence_by_id: Mapping[str, RetrievedEvidence],
    observation_references: list[EvidenceReference],
) -> bool:
    observation_ref_keys = {
        (reference.evidence_id, reference.context_block_id) for reference in observation_references
    }
    if any(field not in CUSTOMER_FACT_FIELDS for field in comparison.customer_fields):
        return False
    if not all(_valid_reference(item, evidence_by_id) for item in comparison.supporting_references):
        return False
    if any(
        (item.evidence_id, item.context_block_id) not in observation_ref_keys
        for item in comparison.supporting_references
    ):
        return False
    if comparison.source_value is not None:
        quoted_text = " ".join(item.quote for item in comparison.supporting_references).casefold()
        if comparison.source_value.casefold() not in quoted_text:
            return False
    if comparison.comparison in {"match", "different", "conflicting"}:
        if comparison.source_value is None or not comparison.customer_fields:
            return False
        if any(getattr(customer, field, None) is None for field in comparison.customer_fields):
            return False
    return True


def _supports_application(observation: EvidenceObservation, method: str) -> bool:
    comparisons = observation.condition_comparisons
    relevant = [
        item
        for item in comparisons
        if item.dimension in {"target_property", "matrix_state", "process_location"}
    ]
    return (
        observation.statement_kind in {"reported_result", "proposed_capability"}
        and observation.entailment == "supported"
        and observation.method is not None
        and observation.method.casefold() == method.casefold()
        and observation.applicability in {"direct_application", "related_application"}
        and any(item.comparison == "match" for item in relevant)
        and not any(item.comparison in {"different", "conflicting"} for item in comparisons)
    )


def _valid_bridge(
    bridge,
    customer: CustomerUseCase,
    evidence_by_id,
    observations: list[EvidenceObservation],
    method: str,
) -> bool:
    if (
        bridge is None
        or not bridge.source_premises
        or not bridge.hypothesis.strip()
        or not bridge.inference.strip()
    ):
        return False
    if not bridge.checks or not bridge.possible_failure_mechanisms or not bridge.unknown_gaps:
        return False
    if not bridge.customer_fields or any(
        field not in CUSTOMER_FACT_FIELDS or getattr(customer, field, None) is None
        for field in bridge.customer_fields
    ):
        return False
    supported_premise_refs = [
        reference
        for observation in observations
        if observation.method
        and observation.method.casefold() == method.casefold()
        and observation.entailment == "supported"
        and observation.statement_kind != "limitation"
        and not any(item.comparison == "conflicting" for item in observation.condition_comparisons)
        for reference in observation.references
        if reference.evidence_id in evidence_by_id
    ]
    if not any(
        observation.method
        and observation.method.casefold() == method.casefold()
        and observation.entailment == "supported"
        and observation.statement_kind != "limitation"
        and not any(item.comparison == "conflicting" for item in observation.condition_comparisons)
        for observation in observations
    ):
        return False
    return all(
        _valid_reference(item, evidence_by_id)
        and any(
            item.evidence_id == reference.evidence_id
            and item.context_block_id == reference.context_block_id
            and item.quote in reference.quote
            for reference in supported_premise_refs
        )
        for item in bridge.source_premises
    )


def _context_state(evidence: list[RetrievedEvidence]) -> str:
    if not evidence or not any(item.excerpt for item in evidence):
        return "unavailable"
    if any(item.context_blocks for item in evidence):
        return "attached_context"
    return "child_only"


def _host_id(fingerprint: str, kind: str, index: int, label: str) -> str:
    digest = hashlib.sha256(f"{fingerprint}:{kind}:{index}:{label}".encode()).hexdigest()[:16]
    return f"{kind}-{digest}"
