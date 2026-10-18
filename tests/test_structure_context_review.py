from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spectro_ai_review_structure_context", ROOT / "scripts" / "review_structure_context.py"
)
assert SPEC and SPEC.loader
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class _Retriever:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.metadata_calls = 0
        self.pin_calls = 0
        self.parent_expansion = False
        self.search_flags = []
        self.snapshot = {
            "revision": "revision-1",
            "schema_version": 3,
            "manifest_version": "manifest-hash",
            "extraction_fingerprint": "parser-hash",
            "chunking_version": 3,
            "chunk_size": 1200,
            "chunk_overlap": 180,
            "files": [
                {"source_id": "S01", "filename": "source.pdf", "document_version": "source-hash"}
            ],
        }

    def index_metadata(self):
        self.metadata_calls += 1
        return self.snapshot

    def pin_snapshot(self, metadata):
        self.pin_calls += 1
        assert metadata is self.snapshot
        return self

    def search(self, use_case, limit):
        self.search_flags.append(self.parent_expansion)
        return [
            {
                "evidence_id": "child-1",
                "source_id": "S01",
                "document_version": "source-hash",
                "page_or_section": "p.6",
                "excerpt": f"answer to {use_case.source_text}",
                "context_blocks": (
                    [{"block_id": "block-1", "text": "context Ω", "location": {"page": 6}}]
                    if self.parent_expansion
                    else []
                ),
                "context_omissions": ["table row did not fit"] if self.parent_expansion else [],
            }
        ][:limit]


def test_run_review_pins_once_and_compares_bm25_flags_with_warm_repeats():
    made = []

    def factory(**kwargs):
        made.append(_Retriever(**kwargs))
        return made[-1]

    report = review.run_review(
        Path("index.sqlite3"), Path("sources.yaml"), 2, retriever_factory=factory
    )
    retriever = made[0]
    assert retriever.metadata_calls == retriever.pin_calls == 1
    assert retriever.kwargs["mode"] == "bm25"
    assert retriever.kwargs["settings"].reranker == "none"
    assert retriever.search_flags == ([False] * 3 + [True] * 3) * 3
    assert len(report["cases"]) == 3
    assert report["index"]["revision"] == "revision-1"
    assert report["index"]["source_hashes"][0]["sha256"] == "source-hash"
    assert report["index"]["parser_fingerprint"] == "parser-hash"
    assert report["precision"] is report["recall"] is None

    first = report["cases"][0]["comparison"]
    assert first["off"]["serialized_context_bytes"] == 0
    assert first["on"]["ranked_children"][0]["locations"] == [{"locator": "p.6"}]
    assert first["on"]["ranked_children"][0]["context_blocks"][0]["block_id"] == "block-1"
    assert first["on"]["serialized_context_bytes"] == len(
        json.dumps(
            {
                "context_blocks": [
                    {"block_id": "block-1", "text": "context Ω", "location": {"page": 6}}
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert first["on"]["ranked_children"][0]["context_omissions"] == ["table row did not fit"]
    assert first["on"]["child_plus_context_bytes"] > first["on"]["serialized_context_bytes"]
    for mode in first.values():
        assert len(mode["warm_latency_ms"]["samples"]) == 2


def test_run_review_rejects_nonpositive_repeats():
    with pytest.raises(ValueError, match="positive"):
        review.run_review(Path("index.sqlite3"), repeats=0, retriever_factory=_Retriever)


def test_cli_writes_json_and_requires_positive_repeats(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(
        review,
        "run_review",
        lambda index, manifest, repeats: {
            "index": str(index),
            "manifest": str(manifest),
            "repeats": repeats,
        },
    )
    assert (
        review.main(["--index", "pinned.sqlite3", "--output", str(destination), "--repeats", "2"])
        == 0
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "index": "pinned.sqlite3",
        "manifest": str(review.DEFAULT_MANIFEST),
        "repeats": 2,
    }
    with pytest.raises(SystemExit):
        review.main(["--index", "x", "--output", str(destination), "--repeats", "0"])
