"""Compare BM25 retrieval with parent expansion off/on on one pinned index."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spectro_ai.evidence import (
    EvidenceRetriever,
    _PinnedEvidenceRetriever,
    evidence_context_blocks_bytes,
)
from spectro_ai.models import CustomerUseCase
from spectro_ai.providers import RagSettings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "knowledge" / "sources.yaml"
QUESTIONS = (
    (
        "table_specification",
        "What sample and geometry accompany the NIR transmission/transflection optical-path entry?",
    ),
    (
        "operating_conditions",
        "What makes NIR analysis possible and what analysis requirement accompanies its "
        "overlapping bands?",
    ),
    (
        "limitations",
        "Which sample constraints qualify the combined MIR/fluorescence probe's potential "
        "applications?",
    ),
)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected a model or mapping, got {type(value).__name__}")


def _evidence_record(item: Any) -> dict[str, Any]:
    data = _dump(item)
    locations = data.get("source_locations") or data.get("locations")
    if locations is None and data.get("page_or_section") is not None:
        locations = [{"locator": data["page_or_section"]}]
    contexts = [_dump(block) for block in (data.get("context_blocks") or [])]
    child_context = json.dumps(
        {"excerpt": data.get("excerpt", ""), "context_blocks": contexts},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    context_blocks = getattr(item, "context_blocks", None)
    context_field_bytes = (
        evidence_context_blocks_bytes(context_blocks)
        if context_blocks is not None
        else 0
        if not contexts
        else len(
            json.dumps(
                {"context_blocks": contexts},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    )
    return {
        "evidence_id": data.get("evidence_id"),
        "source_id": data.get("source_id"),
        "document_version": data.get("document_version"),
        "locations": locations or [],
        "excerpt": data.get("excerpt", ""),
        "context_blocks": contexts,
        "context_omissions": data.get("context_omissions") or [],
        "context_field_bytes": context_field_bytes,
        "child_plus_context_bytes": len(child_context.encode("utf-8")),
    }


def _source_hashes(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": item.get("source_id"),
            "filename": item.get("filename"),
            "sha256": item.get("document_version"),
        }
        for item in snapshot.get("files", [])
    ]


def _pin(retriever: Any, snapshot: dict[str, Any]) -> Any:
    if hasattr(retriever, "_search_many"):
        return _PinnedEvidenceRetriever(retriever, snapshot)
    pin = getattr(retriever, "pin_snapshot", None)
    return pin(snapshot) if callable(pin) else retriever


def run_review(
    index: Path,
    manifest: Path = DEFAULT_MANIFEST,
    repeats: int = 3,
    *,
    retriever_factory: Callable[..., Any] = EvidenceRetriever,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be a positive integer")
    retriever = retriever_factory(
        index_path=index,
        manifest_path=manifest,
        mode="bm25",
        settings=RagSettings(reranker="none"),
    )
    snapshot = retriever.index_metadata()
    pinned = _pin(retriever, snapshot)
    cases = []
    for case_id, question in QUESTIONS:
        use_case = CustomerUseCase(source_text=question)
        comparison: dict[str, Any] = {}
        for enabled in (False, True):
            retriever.parent_expansion = enabled
            pinned.search(use_case, limit=8)  # unmeasured warmup
            samples: list[float] = []
            records: list[dict[str, Any]] = []
            for _ in range(repeats):
                started = time.perf_counter()
                results = pinned.search(use_case, limit=8)
                samples.append((time.perf_counter() - started) * 1000)
                records = [_evidence_record(item) for item in results]
            comparison["on" if enabled else "off"] = {
                "ranked_children": records,
                "warm_latency_ms": {
                    "samples": samples,
                    "median": statistics.median(samples),
                },
                "serialized_context_bytes": sum(item["context_field_bytes"] for item in records),
                "child_plus_context_bytes": sum(
                    item["child_plus_context_bytes"] for item in records
                ),
            }
        cases.append({"id": case_id, "question": question, "comparison": comparison})

    return {
        "status": "comparison_only_unreviewed",
        "index": {
            "path": str(index),
            "revision": snapshot.get("revision"),
            "schema_version": snapshot.get("schema_version"),
            "manifest_version": snapshot.get("manifest_version"),
            "source_hashes": _source_hashes(snapshot),
            "parser_fingerprint": snapshot.get("extraction_fingerprint"),
            "chunking": {
                "version": snapshot.get("chunking_version"),
                "size": snapshot.get("chunk_size"),
                "overlap": snapshot.get("chunk_overlap"),
            },
            "retrieval_mode": "bm25",
            "reranker": None,
        },
        "repeats": repeats,
        "precision": None,
        "recall": None,
        "cases": cases,
    }


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    args = parser.parse_args(argv)
    report = run_review(args.index, args.manifest, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
