"""Compare local retrieval variants against human-reviewed evidence labels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any

from spectro_ai.evidence import EvidenceRetriever, _PinnedEvidenceRetriever, _query_text
from spectro_ai.models import CustomerUseCase, RetrievedEvidence
from spectro_ai.providers import RagSettings, rerank_scores

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "examples" / "retrieval_benchmark.json"
DEFAULT_INDEX = ROOT / ".cache" / "rag" / "index.sqlite3"
_FINAL_COUNT = 8
_CONTEXT_BYTES = 12000


def validate_labels(cases: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    """Reject unbound or stale judgments; return whether labels are reviewed."""
    reviewed = cases.get("status") == "human_reviewed"
    if not reviewed:
        if any(case.get("relevant") for case in cases["cases"]):
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
        for label in case.get("relevant", []):
            if (
                label.get("evidence_id"),
                label.get("source_id"),
                label.get("source_version"),
                label.get("content_sha256"),
            ) not in identities:
                raise ValueError(f"Stale or incomplete relevance label in {case['id']}")
    return True


def rank_metrics(
    ranked_ids: list[str],
    relevant_ids: set[str],
    reviewed: bool,
    candidate_ids: list[str] | None = None,
) -> dict[str, float | None]:
    if not reviewed:
        return {key: None for key in ("precision_at_5", "recall_at_10", "mrr", "candidate_recall")}
    if not relevant_ids:
        return {"precision_at_5": 0.0, "recall_at_10": None, "mrr": None, "candidate_recall": None}
    hits5 = sum(item in relevant_ids for item in ranked_ids[:5])
    hits10 = sum(item in relevant_ids for item in ranked_ids[:10])
    reciprocal = next(
        (1 / index for index, item in enumerate(ranked_ids, 1) if item in relevant_ids), 0.0
    )
    candidate_hits = len(set(candidate_ids or ranked_ids) & relevant_ids)
    return {
        "precision_at_5": hits5 / 5,
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
    for mode in ("legacy", "widened_rrf", "widened_reranked"):
        keys = ("precision_at_5", "recall_at_10", "mrr", "candidate_recall")
        modes[mode] = {
            key: (
                statistics.mean(
                    row["modes"][mode]["metrics"][key]
                    for row in results
                    if row["modes"][mode]["metrics"][key] is not None
                )
                if reviewed
                and any(row["modes"][mode]["metrics"][key] is not None for row in results)
                else None
            )
            for key in keys
        }
    return modes


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
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    scorer = scorer or (lambda query, passages: rerank_scores(settings, query, passages))
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
    cold_ms: float | None = None
    warm: dict[str, list[float]] = {
        "legacy": [],
        "widened_rrf": [],
        "widened_reranked": [],
    }
    resolved_revision: str | None = None
    for case in cases["cases"]:
        query = case["query"]
        use_case = CustomerUseCase(source_text=query)
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
        results.append(
            {
                "id": case["id"],
                "query": query,
                "relevant_count": len(relevant),
                "modes": {
                    "legacy": {
                        "evidence_ids": [item.evidence_id for item in baseline],
                        "metrics": rank_metrics(
                            [item.evidence_id for item in baseline], relevant, reviewed
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
                    },
                },
            }
        )
    index_identity = {
        "revision": snapshot.get("revision"),
        "manifest_version": snapshot.get("manifest_version"),
        "embedding_identity": snapshot.get("embedding_identity"),
        "chunking_version": snapshot.get("chunking_version"),
    }
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
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--bm25", action="store_true", help="use lexical retrieval without Ollama")
    parser.add_argument("--output", type=Path, help="write report under .cache")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output and not args.output.resolve().is_relative_to((ROOT / ".cache").resolve()):
        parser.error("--output must be inside .cache")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    settings = replace(RagSettings.from_env(), index=str(args.index), reranker="cross_encoder")
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
