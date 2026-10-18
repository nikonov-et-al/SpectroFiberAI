"""Offline contracts for persistent assessment hybrid retrieval."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest

from spectro_ai import evidence, providers
from spectro_ai.evidence import EvidenceRetriever
from spectro_ai.models import CustomerUseCase
from spectro_ai.providers import ProviderError, RagSettings


def setup_store(tmp_path, monkeypatch, embed=None):
    materials = tmp_path / "materials"
    materials.mkdir()
    documents = tmp_path / "documents"
    documents.mkdir()
    manifest = tmp_path / "sources.yaml"
    manifest.write_text("schema_version: 1\nroot: materials\nsources: []\n", encoding="utf-8")
    index = tmp_path / "knowledge.sqlite3"
    monkeypatch.setattr(providers, "embedding_identity", lambda _: "model-digest")
    monkeypatch.setattr(
        providers, "embed_texts", embed or (lambda _, texts: [[1.0, 0.0] for _ in texts])
    )
    retriever = EvidenceRetriever(index, manifest, RagSettings(), documents)
    return retriever, documents, index


def test_hybrid_rrf_semantic_paraphrase_and_context_budget(tmp_path, monkeypatch):
    def embed(_, texts):
        return [
            [1.0, 0.0] if "particle" in text.lower() or "clarity" in text.lower() else [0.0, 1.0]
            for text in texts
        ]

    retriever, documents, _ = setup_store(tmp_path, monkeypatch, embed)
    (documents / "paraphrase.md").write_text(
        "# Optical response\n\n"
        "Suspended particles deflect the incoming beam and lower transmitted intensity.",
        encoding="utf-8",
    )
    (documents / "exact.txt").write_text(
        "Moisture measurement in a powder can use a reference calibration.", encoding="utf-8"
    )
    retriever.update_knowledge_base()
    result = retriever.search(
        CustomerUseCase(measurement_objective="liquid clarity", sample_matrix="suspension"), limit=4
    )
    assert any(
        "deflect" in item.excerpt for item in result
    )  # Semantic match has no shared keywords.
    assert all(item.evidence_id and item.source_id and item.document_version for item in result)

    constrained = EvidenceRetriever(
        retriever.index_path,
        retriever.manifest_path,
        RagSettings(),
        documents,
        context_budget=25,
    )
    assert len(constrained.search(CustomerUseCase(measurement_objective="liquid clarity"), 8)) <= 1


def test_incremental_add_modify_delete_and_vector_reuse(tmp_path, monkeypatch):
    calls = []

    def embed(_, texts):
        calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    retriever, documents, _ = setup_store(tmp_path, monkeypatch, embed)
    first = documents / "first.txt"
    first.write_text("First scientific statement.", encoding="utf-8")
    initial = retriever.update_knowledge_base()
    assert initial["documents"] == {"new": 1, "updated": 0, "removed": 0, "unchanged": 0}
    original_calls = len(calls)
    second = documents / "second.txt"
    second.write_text("Second source statement.", encoding="utf-8")
    added = retriever.update_knowledge_base()
    assert added["documents"]["new"] == 1
    assert len(calls) == original_calls + 1
    first.write_text("Modified scientific statement.", encoding="utf-8")
    second.unlink()
    changed = retriever.update_knowledge_base()
    assert changed["documents"]["updated"] == 1
    assert changed["documents"]["removed"] == 1
    assert changed["embeddings"]["new"] == 1
    assert changed["embeddings"]["reused"] == 0


@pytest.mark.parametrize("failure", ["embedding", "replace"])
def test_failed_update_keeps_previous_sqlite_revision(tmp_path, monkeypatch, failure):
    retriever, documents, index = setup_store(tmp_path, monkeypatch)
    source = documents / "source.txt"
    source.write_text("Committed source content.", encoding="utf-8")
    retriever.update_knowledge_base()
    prior = index.read_bytes()
    (documents / "new.txt").write_text("Uncommitted replacement content.", encoding="utf-8")
    if failure == "embedding":
        monkeypatch.setattr(
            providers, "embed_texts", lambda *_: (_ for _ in ()).throw(ProviderError("offline"))
        )
        with pytest.raises(ProviderError, match="offline"):
            retriever.update_knowledge_base()
    else:
        monkeypatch.setattr(
            evidence.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("replace"))
        )
        with pytest.raises(OSError, match="replace"):
            retriever.update_knowledge_base()
    assert index.read_bytes() == prior
    assert retriever.knowledge_status()["ready"]


def test_schema_chunking_compatibility_and_snapshot_revision(tmp_path, monkeypatch):
    retriever, documents, index = setup_store(tmp_path, monkeypatch)
    (documents / "original.txt").write_text("Stable source text.", encoding="utf-8")
    retriever.update_knowledge_base()
    snapshot = retriever.snapshot()
    old_revision = snapshot.index_metadata()["revision"]
    (documents / "added.txt").write_text("Added source text.", encoding="utf-8")
    retriever.update_knowledge_base()
    assert retriever.index_metadata()["revision"] != old_revision
    assert len(snapshot.index_metadata()["chunks"]) == 1
    assert len(retriever.index_metadata()["chunks"]) == 2

    monkeypatch.setenv("SPECTRO_RAG_CHUNK_SIZE", "1300")
    with pytest.raises(ValueError, match="chunking settings changed"):
        retriever.index_metadata()
    rebuilt = retriever.update_knowledge_base()
    assert rebuilt["embeddings"]["new"] == 0
    assert rebuilt["embeddings"]["reused"] == rebuilt["chunks"]
    with closing(sqlite3.connect(index)) as db:
        metadata = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
        metadata["schema_version"] = 99
        db.execute("UPDATE metadata SET value = ?", (json.dumps(metadata),))
        db.commit()
    assert not retriever.knowledge_status()["ready"]
    retriever.update_knowledge_base()
    assert retriever.knowledge_status()["ready"]


def test_v3_structure_records_validate_and_v2_index_remains_readable(tmp_path, monkeypatch):
    retriever, documents, index_path = setup_store(tmp_path, monkeypatch)
    (documents / "source.md").write_text("# Heading\n\nA cited statement.", encoding="utf-8")
    retriever.update_knowledge_base()
    current = retriever.index_metadata()
    assert current["schema_version"] == 3
    block = current["source_blocks"][0]
    section_index = next(
        index
        for index, item in enumerate(current["sections"])
        if item["section_id"] == block["section_id"]
    )
    section = current["sections"][section_index]
    assert block["block_id"] in section["block_ids"]
    assert block["section_id"] == section["section_id"]
    assert current["chunks"][0]["provenance"][0]["block_id"] == block["block_id"]
    with closing(sqlite3.connect(index_path)) as db:
        original_metadata = db.execute("SELECT value FROM metadata").fetchone()[0]
        broken = json.loads(original_metadata)
        broken["sections"][section_index]["block_ids"][0] = "dangling-block"
        db.execute("UPDATE metadata SET value = ?", (json.dumps(broken),))
        db.commit()
    with pytest.raises(ValueError, match="dangling section blocks"):
        retriever.index_metadata()
    with closing(sqlite3.connect(index_path)) as db:
        db.execute("UPDATE metadata SET value = ?", (original_metadata,))
        db.commit()

    # A v2 index lacks full embedding configuration, so migration rebuilds its vectors safely.
    with closing(sqlite3.connect(index_path)) as db, db:
        meta = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
        meta["schema_version"] = 2
        meta.pop("sections")
        meta.pop("source_blocks")
        for file_record in meta["files"]:
            file_record.pop("document_id")
        db.execute("UPDATE metadata SET value = ?", (json.dumps(meta),))
        rows = db.execute("SELECT id, chunk FROM chunks").fetchall()
        for chunk_id, payload in rows:
            chunk = json.loads(payload)
            chunk.pop("embedding_input_sha256", None)
            for provenance in chunk["provenance"]:
                provenance.pop("block_id", None)
                provenance.pop("section_id", None)
            db.execute("UPDATE chunks SET chunk = ? WHERE id = ?", (json.dumps(chunk), chunk_id))
    assert retriever.index_metadata()["schema_version"] == 2
    v2_snapshot = retriever.snapshot()
    migrated = retriever.update_knowledge_base()
    assert migrated["schema_version"] == 3
    assert migrated["embeddings"]["new"] == migrated["chunks"]
    assert v2_snapshot.index_metadata()["schema_version"] == 2
    assert len(v2_snapshot.index_metadata()["chunks"]) == 1
    assert retriever.knowledge_status()["ready"]


def test_parser_version_change_reuses_vectors_for_identical_embedding_input(tmp_path, monkeypatch):
    calls = []
    retriever, documents, _ = setup_store(
        tmp_path, monkeypatch, lambda _, texts: calls.extend(texts) or [[1.0, 0.0] for _ in texts]
    )
    (documents / "source.txt").write_text("Stable source text.", encoding="utf-8")
    retriever.update_knowledge_base()
    assert len(calls) == 1

    old_chunk_id = retriever.index_metadata()["chunks"][0]["chunk_id"]
    monkeypatch.setattr(evidence, "_CHUNKING_VERSION", evidence._CHUNKING_VERSION + 1)
    rebuilt = retriever.update_knowledge_base()
    assert rebuilt["embeddings"] == {"new": 0, "reused": 1}
    assert len(calls) == 1
    assert retriever.index_metadata()["chunks"][0]["chunk_id"] != old_chunk_id


def test_vector_reuse_rejects_changed_identity_and_invalid_dimension(tmp_path, monkeypatch):
    calls = []
    retriever, documents, index = setup_store(
        tmp_path, monkeypatch, lambda _, texts: calls.extend(texts) or [[1.0, 0.0] for _ in texts]
    )
    source = documents / "source.txt"
    source.write_text("Stable source text.", encoding="utf-8")
    retriever.update_knowledge_base()
    assert len(calls) == 1

    with closing(sqlite3.connect(index)) as db, db:
        db.execute("UPDATE chunks SET vector = '[1.0, 0.0, 1.0]'")
    refreshed = retriever.update_knowledge_base()
    assert refreshed["embeddings"] == {"new": 1, "reused": 0}
    assert len(calls) == 2

    monkeypatch.setattr(providers, "embedding_identity", lambda _: "other-model-digest")
    refreshed = retriever.update_knowledge_base()
    assert refreshed["embeddings"] == {"new": 1, "reused": 0}
    assert len(calls) == 3


def test_duplicate_chunk_keeps_provenance_without_duplicate_hit(tmp_path, monkeypatch):
    root = tmp_path
    materials = root / "materials"
    materials.mkdir()
    documents = root / "documents"
    documents.mkdir()
    (materials / "one.txt").write_text("Shared scientific excerpt.", encoding="utf-8")
    (materials / "two.txt").write_text("Shared scientific excerpt.", encoding="utf-8")
    manifest = root / "sources.yaml"
    manifest.write_text(
        "schema_version: 1\nroot: materials\nsources:\n"
        "  - {source_id: S01, filename: one.txt}\n"
        "  - {source_id: S02, filename: two.txt}\n",
        encoding="utf-8",
    )
    index = root / "knowledge.sqlite3"
    monkeypatch.setattr(providers, "embedding_identity", lambda _: "model-digest")
    monkeypatch.setattr(providers, "embed_texts", lambda _, texts: [[1.0, 0.0] for _ in texts])
    retriever = EvidenceRetriever(index, manifest, RagSettings(), documents)
    result = retriever.update_knowledge_base()
    assert result["chunks"] == 1
    chunk = retriever.index_metadata()["chunks"][0]
    assert {item["source_id"] for item in chunk["provenance"]} == {"S01", "S02"}
    hits = retriever.search(CustomerUseCase(source_text="Shared scientific excerpt"))
    assert len(hits) == 1
    assert hits[0].source_id == "S01"


def test_read_rejects_invalid_persisted_vectors(tmp_path, monkeypatch):
    retriever, documents, index = setup_store(tmp_path, monkeypatch)
    (documents / "reference.txt").write_text("A source excerpt.", encoding="utf-8")
    retriever.update_knowledge_base()
    with closing(sqlite3.connect(index)) as db:
        db.execute("UPDATE chunks SET vector = '[true, 0]'")
        db.commit()
    with pytest.raises(ValueError, match="invalid embeddings"):
        retriever.index_metadata()


def test_chunking_preserves_unbroken_words_and_drops_partial_overlap():
    unbroken = "x" * 260
    assert evidence._split_text(unbroken, maximum=200, overlap=20) == [unbroken]

    first = "prefix " + "z" * 48 + "."
    second = "y" * 154 + "."
    chunks = evidence._split_text(first + " " + second, maximum=200, overlap=100)
    assert chunks == [first, second]


def test_retrieval_limits_and_parameters_must_be_valid(tmp_path):
    retriever = EvidenceRetriever(
        tmp_path / "missing.sqlite3", mode="bm25", manifest_path=tmp_path / "missing.yaml"
    )
    with pytest.raises(ValueError, match="limit must be at least 1"):
        retriever._search_many([], 0)
    with pytest.raises(ValueError, match="Candidate counts"):
        EvidenceRetriever(mode="bm25", bm25_candidates=0)
    with pytest.raises(ValueError, match="Candidate counts"):
        EvidenceRetriever(mode="bm25", rrf_k=-1)


def test_hybrid_status_and_model_change_rebuild(tmp_path, monkeypatch):
    calls = []

    def embed(settings, texts):
        calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    retriever, documents, _ = setup_store(tmp_path, monkeypatch, embed)
    (documents / "reference.txt").write_text("Spectroscopy reference text.", encoding="utf-8")
    retriever.update_knowledge_base()
    assert retriever.knowledge_status()["ready"]
    before = len(calls)

    retriever.settings = replace(retriever.settings, embedding_model="changed-model")
    monkeypatch.setattr(
        providers, "embedding_identity", lambda settings: settings.embedding_model + "-digest"
    )
    status = retriever.knowledge_status()
    assert status["ready"] is False
    assert "Update Knowledge Base" in status["error"]
    refreshed = retriever.update_knowledge_base()
    assert refreshed["embeddings"]["new"] == refreshed["chunks"]
    assert len(calls) > before
    assert retriever.knowledge_status()["ready"]


def test_bm25_index_status_and_query_dimension_validation(tmp_path, monkeypatch):
    retriever, documents, index = setup_store(tmp_path, monkeypatch)
    (documents / "reference.txt").write_text("Spectroscopy reference text.", encoding="utf-8")
    # Create an explicit lexical index through compatibility entry point.
    from spectro_ai.evidence import reindex_sources

    reindex_sources(retriever.manifest_path, index)
    lexical = EvidenceRetriever(index, retriever.manifest_path, mode="bm25")
    assert lexical.knowledge_status()["ready"] is True
    assert lexical.knowledge_status()["mode"] == "bm25"

    retriever.update_knowledge_base()
    monkeypatch.setattr(providers, "embed_texts", lambda _, texts: [[1.0, 0.0, 0.0] for _ in texts])
    with pytest.raises(ValueError, match="dimension does not match"):
        retriever.search(CustomerUseCase(source_text="reference"))
