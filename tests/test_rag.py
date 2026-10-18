"""The legacy RAG CLI delegates to assessment's single SQLite knowledge base."""

import sqlite3
from contextlib import closing

from spectro_ai import providers, rag
from spectro_ai.providers import RagSettings


def test_index_compatibility_wrapper_uses_shared_schema(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "guide.md").write_text(
        "# Process control\n\nNIR spectra can estimate moisture in flowing material.",
        encoding="utf-8",
    )
    index = tmp_path / "knowledge.sqlite3"
    monkeypatch.setattr(providers, "embedding_identity", lambda _: "fake-embedding-digest")

    def embed(_, texts):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(providers, "embed_texts", embed)

    result = rag.index_documents(documents, index, RagSettings())
    assert result["embeddings"]["new"] == result["chunks"]
    with closing(sqlite3.connect(index)) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables == {"metadata", "chunks"}
    updated = rag.index_documents(documents, index, RagSettings())
    assert updated["embeddings"]["new"] == 0
    assert updated["embeddings"]["reused"] == result["chunks"]
