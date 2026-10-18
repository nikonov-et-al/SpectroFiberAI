"""Compare local retrieval variants against human-reviewed evidence labels."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from numbers import Real
from pathlib import Path
from typing import Any

from spectro_ai.evidence import (
    EvidenceRetriever,
    _PinnedEvidenceRetriever,
    _query_groups,
    _query_text,
    evidence_context_blocks_bytes,
)
from spectro_ai.models import CustomerUseCase, RetrievedEvidence
from spectro_ai.providers import RagSettings, rerank_scores

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "examples" / "retrieval_benchmark.json"
DEFAULT_INDEX = ROOT / ".cache" / "rag" / "index.sqlite3"
_FINAL_COUNT = 8
_CONTEXT_BYTES = 12000


def validate_labels(cases: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    """Reject unbound or stale judgments; return whether labels are reviewed."""
    human_reviewed = cases.get("status") == "human_reviewed"
    reviewed = human_reviewed and cases.get("review_complete", True) is True
    if not human_reviewed:
        if any(
            case.get("relevant")
            or any(need.get("relevant") for need in case.get("information_needs", []))
            for case in cases["cases"]
        ):
            raise ValueError("Unreviewed benchmark cases cannot contain relevance labels")
        return False
    identities = {
        (
            provenance["evidence_id"],
            provenance["source_id"],
            provenance["document_version"],
            chunk["content_sha256"],
        )
        for chunk in snapshot["chunks"]
        for provenance in chunk["provenance"]
    }
    for case in cases["cases"]:
        for label in [
            *case.get("relevant", []),
            *(
                label
                for need in case.get("information_needs", [])
                for label in need.get("relevant", [])
            ),
        ]:
            if (
                label.get("evidence_id"),
                label.get("source_id"),
                label.get("source_version"),
                label.get("content_sha256"),
            ) not in identities:
                raise ValueError(f"Stale or incomplete relevance label in {case['id']}")
    return reviewed


def rank_metrics(
    ranked_ids: list[str],
    relevant_ids: set[str],
    reviewed: bool,
    candidate_ids: list[str] | None = None,
) -> dict[str, float | None]:
    if not reviewed:
        return {
            key: None
            for key in (
                "precision_at_5",
                "recall_at_8",
                "recall_at_10",
                "mrr",
                "candidate_recall",
            )
        }
    if not relevant_ids:
        return {
            "precision_at_5": 0.0,
            "recall_at_8": None,
            "recall_at_10": None,
            "mrr": None,
            "candidate_recall": None,
        }
    hits5 = sum(item in relevant_ids for item in ranked_ids[:5])
    hits10 = sum(item in relevant_ids for item in ranked_ids[:10])
    reciprocal = next(
        (1 / index for index, item in enumerate(ranked_ids, 1) if item in relevant_ids), 0.0
    )
    candidate_hits = len(set(candidate_ids or ranked_ids) & relevant_ids)
    return {
        "precision_at_5": hits5 / 5,
        "recall_at_8": sum(item in relevant_ids for item in ranked_ids[:8]) / len(relevant_ids),
        "recall_at_10": hits10 / len(relevant_ids),
        "mrr": reciprocal,
        "candidate_recall": candidate_hits / len(relevant_ids),
    }


def _select(evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    selected, used = [], 0
    for item in evidence:
        size = len(item.excerpt.encode("utf-8"))
        if used + size > _CONTEXT_BYTES:
            continue
        selected.append(item)
        used += size
        if len(selected) == _FINAL_COUNT:
            break
    return selected


def _aggregate(results: list[dict[str, Any]], reviewed: bool) -> dict[str, Any]:
    modes = {}
    for mode in (
        "legacy",
        "grouped_baseline",
        "single_query",
        "grouped_pooled",
        "multi_query",
        "widened_rrf",
        "widened_reranked",
    ):
        keys = ("precision_at_5", "recall_at_8", "recall_at_10", "mrr", "candidate_recall")
        modes[mode] = {
            key: (
                statistics.mean(
                    (row["modes"].get(mode, {}).get("metrics") or {}).get(key)
                    for row in results
                    if (row["modes"].get(mode, {}).get("metrics") or {}).get(key) is not None
                )
                if reviewed
                and any(
                    (row["modes"].get(mode, {}).get("metrics") or {}).get(key) is not None
                    for row in results
                )
                else None
            )
            for key in keys
        }
        coverages = [
            row["modes"].get(mode, {}).get("information_need_coverage")
            for row in results
            if row["modes"].get(mode, {}).get("information_need_coverage") is not None
        ]
        modes[mode]["information_need_coverage"] = statistics.mean(coverages) if coverages else None
    return modes


def _paired_aggregate(
    results: list[dict[str, Any]], reviewed: bool
) -> tuple[list[str], dict[str, Any]]:
    eligible_statuses = {"available", "baseline_fallback"}
    paired = [
        row
        for row in results
        if isinstance(row["modes"].get("grouped_baseline", {}).get("metrics"), dict)
        and row["modes"].get("multi_query", {}).get("status") in eligible_statuses
        and isinstance(row["modes"].get("multi_query", {}).get("metrics"), dict)
    ]
    return [row["id"] for row in paired], _aggregate(paired, reviewed)


def _document_stats(items: list[RetrievedEvidence]) -> dict[str, float | int]:
    counts: dict[tuple[str, str], int] = {}
    ids: list[str] = []
    for item in items:
        source = item.source_locations[0] if item.source_locations else None
        document = (
            f"{source.filename}@{source.document_version}"
            if source
            else str(item.document_version or "")
        )
        key = (item.source_id, document)
        counts[key] = counts.get(key, 0) + 1
        ids.append(item.evidence_id)
    return {
        "unique_documents": len(counts),
        "largest_document_share": max(counts.values(), default=0) / len(items) if items else 0.0,
        "duplicate_rate": 1 - len(set(ids)) / len(ids) if ids else 0.0,
    }


def _context_byte_stats(items: list[RetrievedEvidence]) -> dict[str, int]:
    excerpt_bytes = sum(len(item.excerpt.encode("utf-8")) for item in items)
    context_block_bytes = sum(evidence_context_blocks_bytes(item.context_blocks) for item in items)
    return {
        "excerpt_bytes": excerpt_bytes,
        "evidence_context_blocks_bytes": context_block_bytes,
        "total_serialized_context_bytes": excerpt_bytes + context_block_bytes,
    }


def _query_overlap_rate(diagnostics: dict[str, Any]) -> float | None:
    membership = diagnostics.get("question_evidence")
    if not isinstance(membership, dict):
        return None
    ids = [item for values in membership.values() if isinstance(values, list) for item in values]
    return 1 - len(set(ids)) / len(ids) if ids else 0.0


def _need_coverage(ranked_ids: list[str], case: dict[str, Any], reviewed: bool) -> float | None:
    needs = case.get("information_needs", [])
    if not reviewed or not needs or case.get("needs_reviewed") is not True:
        return None
    labeled = [{label["evidence_id"] for label in need.get("relevant", [])} for need in needs]
    if any(not ids for ids in labeled):
        return None
    selected = set(ranked_ids)
    return sum(bool(selected & ids) for ids in labeled) / len(labeled)


def _peak_process_memory_bytes() -> int | None:
    """Read process peak RSS where stdlib exposes it; unsupported platforms return null."""
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1024 if sys.platform != "darwin" else 1))
    except ImportError:
        if os.name != "nt":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_Counters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            return int(counters.PeakWorkingSetSize) if ok else None
        except AttributeError, OSError:
            return None


def _final_select(retriever: Any, evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    if hasattr(retriever, "select_question_candidates"):
        try:
            return retriever.select_question_candidates(evidence, limit=_FINAL_COUNT)
        except AttributeError as exc:
            if "_select_question_results" not in str(exc):
                raise
    return _select(evidence)


def _warm_latency(
    run: Callable[[], tuple[dict[str, Any], str | None]], repeats: int
) -> dict[str, Any]:
    searches, reranks, timeouts = [], [], 0
    for _ in range(repeats):
        try:
            mode, _revision = run()
        except TimeoutError:
            timeouts += 1
            continue
        timing = mode.get("latency_ms", {})
        if mode.get("status") == "available":
            if timing.get("search") is not None:
                searches.append(timing["search"])
            if timing.get("rerank") is not None:
                reranks.append(timing["rerank"])
    return {
        "scope": "retrieval_only_reusing_plan",
        "repeats": repeats,
        "search_mean": statistics.mean(searches) if searches else None,
        "rerank_mean": statistics.mean(reranks) if reranks else None,
        "timeouts": timeouts,
    }


def _finish_mode_timings(mode: dict[str, Any]) -> None:
    original_timing = mode.get("latency_ms")
    if not isinstance(original_timing, dict):
        return
    timing = dict(original_timing)
    mode["latency_ms"] = timing
    generation = mode.get("generation", {}).get("latency_ms", 0.0)
    timing["end_to_end"] = sum(
        value for value in (timing.get("search"), timing.get("rerank"), generation) if value
    )
    original_warm = mode.get("warm_latency_ms")
    if isinstance(original_warm, dict):
        warm = dict(original_warm)
        mode["warm_latency_ms"] = warm
        warm["end_to_end_mean"] = sum(
            value
            for value in (
                warm.get("search_mean"),
                warm.get("rerank_mean"),
            )
            if value
        )


def _question_mode(
    retriever: Any,
    use_case: CustomerUseCase,
    query_text: str,
    query_texts: list[str],
    *,
    settings: RagSettings,
    scorer: Callable,
    timeout: float,
    include_anchor: bool,
    relevant: set[str],
    reviewed: bool,
    case: dict[str, Any],
    query_status: str = "available",
) -> tuple[dict[str, Any], str | None]:
    if not hasattr(retriever, "search_question_candidates"):
        return {"status": "unsupported_retriever_api", "metrics": None}, None
    started = time.perf_counter()
    try:
        found = retriever.search_question_candidates(
            use_case,
            query_texts,
            limit=settings.reranker_candidates,
            timeout=timeout,
            include_anchor=include_anchor,
        )
    except TimeoutError:
        return {"status": "deadline_exceeded", "metrics": None}, None
    except AttributeError as exc:
        if "_search_questions" not in str(exc):
            raise
        return {"status": "unsupported_retriever_api", "metrics": None}, None
    search_ms = (time.perf_counter() - started) * 1000
    rerank_ms = 0.0
    ordered = found
    resolved = None
    if settings.reranker == "cross_encoder" and found:
        started = time.perf_counter()
        ordered, resolved = _ordered(found, query_text, scorer)
        rerank_ms = (time.perf_counter() - started) * 1000
    selected = _final_select(retriever, ordered)
    ids = [item.evidence_id for item in selected]
    info = {
        "status": query_status,
        "question_count": len(query_texts) + int(include_anchor),
    }
    diagnostics = getattr(retriever, "last_query_diagnostics", {})
    info.update(diagnostics if isinstance(diagnostics, dict) else {})
    membership = info.get("question_evidence")
    retrieval_query_count = (
        len(membership) if isinstance(membership, dict) else len(query_texts) + int(include_anchor)
    )
    return (
        {
            "status": query_status,
            "evidence_ids": ids,
            "candidate_ids": [item.evidence_id for item in found],
            "metrics": rank_metrics(ids, relevant, reviewed, [item.evidence_id for item in found]),
            "information_need_coverage": _need_coverage(ids, case, reviewed),
            "document_stats": _document_stats(selected),
            "context_bytes": _context_byte_stats(selected),
            "diagnostics": info,
            "cross_query_duplicate_rate": _query_overlap_rate(info),
            "retrieval_query_count": retrieval_query_count,
            "candidate_count": len(found),
            "reranker_calls": int(settings.reranker == "cross_encoder" and bool(found)),
            "latency_ms": {"search": search_ms, "rerank": rerank_ms},
        },
        resolved,
    )


def _single_query_mode(
    retriever: Any,
    snapshot: dict[str, Any],
    use_case: CustomerUseCase,
    query_text: str,
    *,
    settings: RagSettings,
    scorer: Callable,
    relevant: set[str],
    reviewed: bool,
    case: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    engine = retriever.retriever if hasattr(retriever, "retriever") else retriever
    if not hasattr(engine, "_search_many"):
        return {"status": "unsupported_retriever_api", "metrics": None}, None
    started = time.perf_counter()
    candidates = engine._search_many(
        [use_case.source_text or query_text],
        settings.reranker_candidates,
        pinned_index=snapshot,
        candidate_mode=True,
    )
    search_ms = (time.perf_counter() - started) * 1000
    ordered = candidates
    resolved = None
    rerank_ms = 0.0
    if settings.reranker == "cross_encoder" and candidates:
        started = time.perf_counter()
        ordered, resolved = _ordered(candidates, query_text, scorer)
        rerank_ms = (time.perf_counter() - started) * 1000
    selected = _final_select(retriever, ordered)
    ids = [item.evidence_id for item in selected]
    return (
        {
            "status": "available",
            "evidence_ids": ids,
            "candidate_ids": [item.evidence_id for item in candidates],
            "metrics": rank_metrics(
                ids, relevant, reviewed, [item.evidence_id for item in candidates]
            ),
            "information_need_coverage": _need_coverage(ids, case, reviewed),
            "document_stats": _document_stats(selected),
            "context_bytes": _context_byte_stats(selected),
            "latency_ms": {"search": search_ms, "rerank": rerank_ms},
        },
        resolved,
    )


def _grouped_baseline_mode(
    retriever: Any,
    use_case: CustomerUseCase,
    *,
    relevant: set[str],
    reviewed: bool,
    case: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if not hasattr(retriever, "search"):
        return {"status": "unsupported_retriever_api", "metrics": None}, None
    started = time.perf_counter()
    selected = retriever.search(use_case, limit=_FINAL_COUNT)
    search_ms = (time.perf_counter() - started) * 1000
    ids = [item.evidence_id for item in selected]
    return (
        {
            "status": "available",
            "evidence_ids": ids,
            "metrics": rank_metrics(ids, relevant, reviewed),
            "information_need_coverage": _need_coverage(ids, case, reviewed),
            "document_stats": _document_stats(selected),
            "context_bytes": _context_byte_stats(selected),
            "latency_ms": {"search": search_ms, "rerank": None},
        },
        None,
    )


def _ordered(
    candidates: list[RetrievedEvidence], query: str, scorer: Callable
) -> tuple[list[RetrievedEvidence], str | None]:
    scores, revision = scorer(query, [item.excerpt for item in candidates])
    if len(scores) != len(candidates) or any(
        isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score)
        for score in scores
    ):
        raise ValueError("Scorer returned wrong number of finite scalar scores")
    return [
        item.model_copy(update={"rerank_score": score})
        for _, score, item in sorted(
            (
                (index, float(score), item)
                for index, (score, item) in enumerate(zip(scores, candidates, strict=True))
            ),
            key=lambda row: (-row[1], row[0]),
        )
    ], revision


def run_benchmark(
    cases: dict[str, Any],
    retriever: Any,
    *,
    baseline_retriever: Any | None = None,
    settings: RagSettings,
    scorer: Callable[[str, list[str]], tuple[list[float], str | None]] | None = None,
    repeats: int = 3,
    multi_query: bool = False,
    query_generator: Callable[[CustomerUseCase], Any] | None = None,
    query_validator: Callable[..., tuple[list[Any], list[str]]] | None = None,
    query_timeout: float = 30,
    generation_timeout: float | None = None,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    scorer = scorer or (lambda query, passages: rerank_scores(settings, query, passages))
    query_generator_injected = query_generator is not None
    baseline_retriever = baseline_retriever or retriever
    snapshot = retriever.index_metadata()
    if hasattr(retriever, "_search_many"):
        retriever = _PinnedEvidenceRetriever(retriever, snapshot)
    elif hasattr(retriever, "retriever"):
        retriever.metadata = snapshot
    if hasattr(baseline_retriever, "_search_many"):
        baseline_retriever = _PinnedEvidenceRetriever(baseline_retriever, snapshot)
    elif hasattr(baseline_retriever, "retriever"):
        baseline_retriever.metadata = snapshot
    reviewed = validate_labels(cases, snapshot)
    results = []
    process_cpu_start = time.process_time()
    memory_before = _peak_process_memory_bytes()
    cold_ms: float | None = None
    warm: dict[str, list[float]] = {
        "legacy": [],
        "widened_rrf": [],
        "widened_reranked": [],
    }
    resolved_revision: str | None = None
    for case in cases["cases"]:
        query = case["query"]
        use_case = CustomerUseCase.model_validate(
            {"source_text": query, **case.get("validated_requirements", {})}
        )
        rerank_query = _query_text(use_case)
        baseline = baseline_retriever.search(use_case, limit=_FINAL_COUNT)
        candidates = retriever.search_candidates(use_case, limit=settings.reranker_candidates)
        widened = _select(candidates)
        reranked = candidates
        if settings.reranker == "cross_encoder" and candidates:
            start = time.perf_counter()
            reranked, resolved_revision = _ordered(candidates, rerank_query, scorer)
            if cold_ms is None:
                cold_ms = (time.perf_counter() - start) * 1000
        reranked = _select(reranked)
        for _ in range(repeats):
            start = time.perf_counter()
            baseline_retriever.search(use_case, limit=_FINAL_COUNT)
            warm["legacy"].append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            warm_candidates = retriever.search_candidates(
                use_case, limit=settings.reranker_candidates
            )
            _select(warm_candidates)
            warm["widened_rrf"].append((time.perf_counter() - start) * 1000)

            if settings.reranker == "cross_encoder" and warm_candidates:
                start = time.perf_counter()
                reranked_candidates = retriever.search_candidates(
                    use_case, limit=settings.reranker_candidates
                )
                reranked, resolved_revision = _ordered(reranked_candidates, rerank_query, scorer)
                _select(reranked)
                warm["widened_reranked"].append((time.perf_counter() - start) * 1000)
        reranked = _select(reranked)
        relevant = {label["evidence_id"] for label in case.get("relevant", [])}
        modes = {
            "legacy": {
                "evidence_ids": [item.evidence_id for item in baseline],
                "metrics": rank_metrics(
                    [item.evidence_id for item in baseline], relevant, reviewed
                ),
                "document_stats": _document_stats(baseline),
                "context_bytes": _context_byte_stats(baseline),
                "information_need_coverage": _need_coverage(
                    [item.evidence_id for item in baseline], case, reviewed
                ),
            },
            "widened_rrf": {
                "evidence_ids": [item.evidence_id for item in widened],
                "candidate_ids": [item.evidence_id for item in candidates],
                "metrics": rank_metrics(
                    [item.evidence_id for item in widened],
                    relevant,
                    reviewed,
                    [item.evidence_id for item in candidates],
                ),
                "document_stats": _document_stats(widened),
                "context_bytes": _context_byte_stats(widened),
            },
            "widened_reranked": {
                "evidence_ids": [item.evidence_id for item in reranked],
                "candidate_ids": [item.evidence_id for item in candidates],
                "metrics": rank_metrics(
                    [item.evidence_id for item in reranked],
                    relevant,
                    reviewed,
                    [item.evidence_id for item in candidates],
                ),
                "document_stats": _document_stats(reranked),
                "context_bytes": _context_byte_stats(reranked),
            },
        }

        modes["grouped_baseline"], _resolved = _grouped_baseline_mode(
            retriever,
            use_case,
            relevant=relevant,
            reviewed=reviewed,
            case=case,
        )
        resolved_revision = _resolved or resolved_revision
        modes["grouped_baseline"]["warm_latency_ms"] = _warm_latency(
            partial(
                _grouped_baseline_mode,
                retriever,
                use_case,
                relevant=relevant,
                reviewed=reviewed,
                case=case,
            ),
            repeats,
        )

        # Broad control uses original single-query engine. New pool control uses
        # deterministic groups with same cross-question consolidation.
        modes["single_query"], resolved = _single_query_mode(
            retriever,
            snapshot,
            use_case,
            rerank_query,
            settings=settings,
            scorer=scorer,
            relevant=relevant,
            reviewed=reviewed,
            case=case,
        )
        resolved_revision = resolved or resolved_revision
        modes["single_query"]["warm_latency_ms"] = _warm_latency(
            partial(
                _single_query_mode,
                retriever,
                snapshot,
                use_case,
                rerank_query,
                settings=settings,
                scorer=scorer,
                relevant=relevant,
                reviewed=reviewed,
                case=case,
            ),
            repeats,
        )
        modes["grouped_pooled"], resolved = _question_mode(
            retriever,
            use_case,
            rerank_query,
            _query_groups(use_case),
            settings=settings,
            scorer=scorer,
            timeout=query_timeout,
            include_anchor=False,
            relevant=relevant,
            reviewed=reviewed,
            case=case,
        )
        resolved_revision = resolved or resolved_revision
        modes["grouped_pooled"]["warm_latency_ms"] = _warm_latency(
            partial(
                _question_mode,
                retriever,
                use_case,
                rerank_query,
                _query_groups(use_case),
                settings=settings,
                scorer=scorer,
                timeout=query_timeout,
                include_anchor=False,
                relevant=relevant,
                reviewed=reviewed,
                case=case,
            ),
            repeats,
        )
        if multi_query:
            if not all(
                getattr(use_case, field)
                for field in (
                    "measurement_objective",
                    "target_analyte_or_property",
                    "sample_matrix",
                )
            ):
                modes["multi_query"] = {
                    **modes["grouped_baseline"],
                    "status": "skipped_insufficient_information",
                    "retrieval_only": True,
                    "retrieval_only_baseline_metrics": modes["grouped_baseline"]["metrics"],
                    "metrics": None,
                    "information_need_coverage": None,
                    "generation": {"calls": 0, "latency_ms": 0.0},
                }
                results.append(
                    {
                        "id": case["id"],
                        "query": query,
                        "relevant_count": len(relevant),
                        "modes": modes,
                    }
                )
                continue
            generator_for_case = query_generator
            generation_usage: dict[str, int] = {}
            generation_model = settings.model
            if generator_for_case is None:
                from spectro_ai.providers import (
                    ExtractionSettings,
                    ProviderError,
                    generate_evidence_query_plan,
                )

                def generate_for_case(
                    uc: CustomerUseCase, usage: dict[str, int] = generation_usage
                ) -> Any:
                    nonlocal generation_model
                    generation_settings = ExtractionSettings.from_env()
                    if generation_settings.provider != "ollama":
                        raise ProviderError("generated benchmark requires local Ollama provider")
                    generation_settings = replace(
                        generation_settings,
                        model=settings.model,
                        ollama_url=settings.ollama_url,
                        timeout=int(generation_timeout or query_timeout),
                        output_tokens=768,
                    )
                    generation_model = generation_settings.model
                    return generate_evidence_query_plan(
                        uc,
                        generation_settings,
                        timeout=int(generation_timeout or query_timeout),
                        usage=usage,
                    )

                generator_for_case = generate_for_case

            started = time.perf_counter()
            try:
                plan = generator_for_case(use_case)
            except (TimeoutError, ValueError) as exc:
                generation_ms = (time.perf_counter() - started) * 1000
                modes["multi_query"] = {
                    **modes["grouped_baseline"],
                    "status": "baseline_fallback",
                    "generation": {
                        "latency_ms": generation_ms,
                        "calls": 1,
                        "fallback_reason": type(exc).__name__,
                        "plan_source": "injected"
                        if query_generator_injected
                        else "live_local_model",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                    },
                }
                results.append(
                    {
                        "id": case["id"],
                        "query": query,
                        "relevant_count": len(relevant),
                        "modes": modes,
                    }
                )
                continue
            generation_ms = (time.perf_counter() - started) * 1000
            if query_validator is None:
                from spectro_ai.providers import validate_evidence_query_plan

                query_validator = validate_evidence_query_plan
            generated_questions, rejected = query_validator(plan, use_case, max_questions=3)
            question_texts = [
                item.question if hasattr(item, "question") else str(item)
                for item in generated_questions
            ]
            if question_texts:
                modes["multi_query"], resolved = _question_mode(
                    retriever,
                    use_case,
                    rerank_query,
                    question_texts,
                    settings=settings,
                    scorer=scorer,
                    timeout=query_timeout,
                    include_anchor=True,
                    relevant=relevant,
                    reviewed=reviewed,
                    case=case,
                )
                resolved_revision = resolved or resolved_revision
                modes["multi_query"]["warm_latency_ms"] = _warm_latency(
                    partial(
                        _question_mode,
                        retriever,
                        use_case,
                        rerank_query,
                        question_texts,
                        settings=settings,
                        scorer=scorer,
                        timeout=query_timeout,
                        include_anchor=True,
                        relevant=relevant,
                        reviewed=reviewed,
                        case=case,
                    ),
                    repeats,
                )
                modes["multi_query"]["generation"] = {
                    "latency_ms": generation_ms,
                    "calls": 1,
                    "rejected_count": len(rejected),
                    "rejected_reasons": rejected,
                    "model": generation_model,
                    "plan_source": "injected" if query_generator_injected else "live_local_model",
                    "prompt_tokens": generation_usage.get("prompt_tokens"),
                    "completion_tokens": generation_usage.get("completion_tokens"),
                }
                if modes["multi_query"]["status"] == "deadline_exceeded":
                    modes["multi_query"] = {
                        **modes["grouped_baseline"],
                        "status": "baseline_fallback",
                        "generation": {
                            "latency_ms": generation_ms,
                            "calls": 1,
                            "fallback_reason": "search_deadline_exceeded",
                            "plan_source": "injected"
                            if query_generator_injected
                            else "live_local_model",
                            "prompt_tokens": None,
                            "completion_tokens": None,
                        },
                    }
            else:
                modes["multi_query"] = {
                    **modes["grouped_baseline"],
                    "status": "baseline_fallback",
                    "generation": {
                        "latency_ms": generation_ms,
                        "calls": 1,
                        "rejected_count": len(rejected),
                        "rejected_reasons": rejected,
                        "fallback_reason": "no_valid_questions",
                        "plan_source": "injected"
                        if query_generator_injected
                        else "live_local_model",
                        "prompt_tokens": generation_usage.get("prompt_tokens"),
                        "completion_tokens": generation_usage.get("completion_tokens"),
                    },
                }
        else:
            modes["multi_query"] = {"status": "disabled", "metrics": None}
        results.append(
            {
                "id": case["id"],
                "query": query,
                "relevant_count": len(relevant),
                "modes": modes,
            }
        )
    for row in results:
        for mode in row["modes"].values():
            _finish_mode_timings(mode)
    index_identity = {
        "revision": snapshot.get("revision"),
        "manifest_version": snapshot.get("manifest_version"),
        "embedding_identity": snapshot.get("embedding_identity"),
        "chunking_version": snapshot.get("chunking_version"),
    }
    paired_case_ids, paired_metrics = _paired_aggregate(results, reviewed)
    return {
        "schema_version": 1,
        "label_status": cases.get("status", "provisional_unreviewed"),
        "metrics_status": "reviewed" if reviewed else "provisional_metrics_unavailable",
        "index": index_identity,
        "reranker": {
            "model": settings.reranker_model,
            "revision": resolved_revision or settings.reranker_revision,
            "enabled": settings.reranker == "cross_encoder",
        },
        "latency_ms": {
            "cold_start_load_and_score": cold_ms,
            "warm_search_mean_by_mode": {
                mode: statistics.mean(samples) if samples else None
                for mode, samples in warm.items()
            },
            "warm_repeats_by_mode": {mode: len(samples) for mode, samples in warm.items()},
        },
        "aggregate_metrics": _aggregate(results, reviewed),
        "paired_case_ids": paired_case_ids,
        "paired_aggregate_metrics": paired_metrics,
        "resource_usage": {
            "process_cpu_seconds": time.process_time() - process_cpu_start,
            "whole_run_peak_process_memory_bytes": _peak_process_memory_bytes(),
            "whole_run_peak_process_memory_before_bytes": memory_before,
            "scope": "whole benchmark process high-water mark; not per-arm memory",
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--bm25", action="store_true", help="use lexical retrieval without Ollama")
    parser.add_argument("--output", type=Path, help="write report under .cache")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="skip local cross-encoder reranking for a faster smoke run",
    )
    parser.add_argument(
        "--multi-query",
        action="store_true",
        help="explicitly generate query plans with configured local Ollama model",
    )
    args = parser.parse_args()
    if args.output and not args.output.resolve().is_relative_to((ROOT / ".cache").resolve()):
        parser.error("--output must be inside .cache")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    settings = replace(
        RagSettings.from_env(),
        index=str(args.index),
        reranker="none" if args.no_reranker else "cross_encoder",
    )
    if args.multi_query and settings.provider != "ollama":
        parser.error(
            "--multi-query benchmark requires configured Ollama; paid providers are disabled"
        )
    if args.multi_query:
        from spectro_ai.providers import ExtractionSettings, ProviderError

        try:
            generation_settings = ExtractionSettings.from_env()
        except ProviderError as exc:
            parser.error(str(exc))
        if generation_settings.provider != "ollama":
            parser.error("--multi-query benchmark requires SPECTRO_AI_PROVIDER=ollama")
    retriever = EvidenceRetriever(
        args.index, settings=settings, mode="bm25" if args.bm25 else "hybrid"
    )
    baseline_settings = replace(settings, reranker="none")
    baseline_retriever = EvidenceRetriever(
        args.index, settings=baseline_settings, mode="bm25" if args.bm25 else "hybrid"
    )
    try:
        report = run_benchmark(
            cases,
            retriever,
            baseline_retriever=baseline_retriever,
            settings=settings,
            repeats=args.repeats,
            multi_query=args.multi_query,
            query_timeout=settings.multi_query_search_timeout,
            generation_timeout=settings.multi_query_timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
