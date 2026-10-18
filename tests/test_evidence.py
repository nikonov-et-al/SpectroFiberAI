import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from types import ModuleType
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from spectro_ai.evidence import (
    EvidenceKnowledgeSource,
    EvidenceRetriever,
    _split_text,
    reindex_sources,
)
from spectro_ai.models import CustomerUseCase


def _read_index(path: Path) -> dict:
    with closing(sqlite3.connect(path)) as db:
        metadata = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
        metadata["chunks"] = [json.loads(row[0]) for row in db.execute("SELECT chunk FROM chunks")]
    return metadata


def _write_index(path: Path, index: dict) -> None:
    metadata = {key: value for key, value in index.items() if key != "chunks"}
    metadata["chunks"] = len(index["chunks"])
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE metadata SET value = ?", (json.dumps(metadata),))
        for chunk in index["chunks"]:
            db.execute(
                "UPDATE chunks SET chunk = ? WHERE id = ?", (json.dumps(chunk), chunk["chunk_id"])
            )
        db.commit()


def _manifest(root: Path, entries: list[dict]) -> Path:
    import yaml

    path = root / "sources.yaml"
    (root / "materials").mkdir(exist_ok=True)
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "root": "materials", "sources": entries}),
        encoding="utf-8",
    )
    return path


def _source(source_id: str, filename: str, **metadata: str) -> dict:
    return {
        "source_id": source_id,
        "filename": filename,
        "title": filename,
        "source_type": metadata.get("source_type", "technical_reference"),
        "scope": metadata.get("scope", "technical"),
        "review_status": metadata.get("review_status", "needs_source_review"),
    }


def _write_xlsx(path: Path) -> None:
    workbook = """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <sheets><sheet name="methods" sheetId="1" r:id="rId1"/></sheets></workbook>"""
    rels = """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"""
    sheet = """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Method</t></is></c>
      <c r="B1" t="inlineStr"><is><t>Geometry</t></is></c></row>
      <row r="2"><c r="A2" t="inlineStr"><is><t>NIR</t></is></c>
      <c r="B2" t="inlineStr"><is><t>diffuse reflection</t></is></c></row>
      </sheetData></worksheet>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def _write_docx(path: Path) -> None:
    xml = """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Methods</w:t></w:r></w:p>
      <w:p><w:r><w:t>MIR ATR tests use local source material.</w:t></w:r></w:p>
      <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Sample</w:t></w:r></w:p></w:tc>
      <w:tc><w:p><w:r><w:t>Oil</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
      <w:sectPr/></w:body></w:document>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)


def test_reindex_tracks_page_lines_and_worksheet_rows(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    materials = root / "materials"
    materials.mkdir(parents=True)
    (materials / "notes.txt").write_text("water in oil\nATR probe", encoding="utf-8")
    _write_xlsx(materials / "table.xlsx")
    manifest = _manifest(root, [_source("S01", "notes.txt"), _source("S02", "table.xlsx")])
    index_path = root / "index.json"

    result = reindex_sources(manifest, index_path)
    chunks = _read_index(index_path)["chunks"]
    locators = {p["page_or_section"] for chunk in chunks for p in chunk["provenance"]}

    assert result["files_indexed"] == 2
    assert "lines 1-2" in locators
    worksheet_provenance = [
        provenance
        for chunk in chunks
        for provenance in chunk["provenance"]
        if provenance["filename"] == "table.xlsx"
    ]
    assert "worksheet=methods rows=1-1; worksheet=methods rows=2-2" in locators
    assert {
        location["locator"]
        for provenance in worksheet_provenance
        for location in provenance["source_locations"]
    } == {"worksheet=methods rows=1-1", "worksheet=methods rows=2-2"}
    retrieved = EvidenceRetriever(index_path, manifest, mode="bm25").search(
        CustomerUseCase(source_text="near infrared ATR spectroscopy"), limit=3
    )
    assert retrieved
    assert retrieved[0].source_id in {"S01", "S02"}


def test_allowlisted_docx_indexes_sections_not_evaluation_material(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    materials = root / "materials"
    materials.mkdir(parents=True)
    _write_docx(materials / "reference.docx")
    manifest = _manifest(root, [_source("S04", "reference.docx")])
    index_path = root / "index.json"

    reindex_sources(manifest, index_path)
    index = _read_index(index_path)

    provenance = [p for c in index["chunks"] for p in c["provenance"]]
    assert {p["page_or_section"] for p in provenance} == {
        "paragraph=1; paragraph=2",
        "table=1 row=1",
    }
    assert all(p["source_locations"] for p in provenance)
    assert any("Sample | Oil" in chunk["text"] for chunk in index["chunks"])


def test_pdf_pages_cite_original_page_and_empty_pages_are_visible(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    (root / "materials").mkdir(parents=True)
    pdf = root / "materials" / "mock.pdf"
    pdf.write_bytes(b"fixture")

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    pypdf = ModuleType("pypdf")
    pypdf.PdfReader = lambda path: type(
        "Reader", (), {"pages": [FakePage("MIR ATR water"), FakePage("chart"), FakePage("")]}
    )()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", pypdf)
    manifest = _manifest(root, [_source("S01", "mock.pdf", review_status="needs_visual_review")])
    index_path = root / "index.json"

    result = reindex_sources(manifest, index_path)
    index = _read_index(index_path)

    assert result["empty_pdf_pages"] == 1
    assert result["poor_extraction_pages"] == 2
    assert index["files"][0]["page_count"] == 3
    assert index["files"][0]["empty_pages"] == ["p.3"]
    assert index["files"][0]["poor_extraction_pages"] == ["p.1", "p.2"]
    assert index["chunks"][0]["provenance"][0]["page_or_section"] == "p.1"
    assert index["chunks"][0]["provenance"][0]["review_status"] == (
        "needs_visual_review; poor_extraction_needs_manual_review"
    )
    retriever = EvidenceRetriever(index_path, manifest, mode="bm25")
    index["chunks"][0]["provenance"][0]["page_or_section"] = "p.0"
    _write_index(index_path, index)
    with pytest.raises(ValueError, match="outside source pages"):
        retriever.index_metadata()

    reindex_sources(manifest, index_path)
    index = _read_index(index_path)
    first_id = index["chunks"][0]["provenance"][0]["evidence_id"]
    index["chunks"][1]["provenance"][0]["evidence_id"] = first_id
    _write_index(index_path, index)
    with pytest.raises(ValueError, match="duplicated"):
        retriever.index_metadata()


def test_identical_content_deduplicates_but_keeps_both_provenances(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "materials").mkdir(parents=True)
    content = "MIR ATR petroleum water measurement"
    for name in ("a.txt", "b.txt"):
        (root / "materials" / name).write_text(content, encoding="utf-8")
    manifest = _manifest(root, [_source("S01", "a.txt"), _source("S02", "b.txt")])
    index_path = root / "index.json"

    reindex_sources(manifest, index_path)
    index = _read_index(index_path)

    assert len(index["chunks"]) == 1
    assert {item["source_id"] for item in index["chunks"][0]["provenance"]} == {"S01", "S02"}


def test_changed_file_replaces_version_and_missing_source_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "materials").mkdir(parents=True)
    source = root / "materials" / "source.txt"
    source.write_text("MIR ATR petroleum water", encoding="utf-8")
    manifest = _manifest(
        root,
        [_source("S01", "source.txt"), _source("S02", "not-yet-supplied.pdf")],
    )
    index_path = root / "index.json"

    first = reindex_sources(manifest, index_path)
    old_version = _read_index(index_path)["files"][0]["document_version"]
    source.write_text("NIR milk powder", encoding="utf-8")
    second = reindex_sources(manifest, index_path)
    index = _read_index(index_path)

    assert first["missing_sources"] == [
        {"source_id": "S02", "filename": "not-yet-supplied.pdf", "status": "missing"}
    ]
    assert second["chunks"] == 1
    assert index["files"][0]["document_version"] != old_version
    assert "petroleum" not in index["chunks"][0]["text"]


def test_failed_reindex_preserves_old_index(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "materials").mkdir(parents=True)
    (root / "materials" / "broken.pdf").write_bytes(b"not a pdf")
    manifest = _manifest(root, [_source("S01", "broken.pdf")])
    index_path = root / "index.json"
    index_path.write_bytes(b"previous index")

    pypdf = ModuleType("pypdf")

    def fail_reader(path):
        raise ValueError("invalid pdf")

    pypdf.PdfReader = fail_reader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", pypdf)
    with pytest.raises(ValueError, match="invalid pdf"):
        reindex_sources(manifest, index_path)

    assert index_path.read_bytes() == b"previous index"


def test_manifest_addition_works_and_fixture_or_evaluation_entries_are_excluded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    materials = root / "materials"
    materials.mkdir(parents=True)
    (materials / "new-reference.txt").write_text("new spectroscopy reference", encoding="utf-8")
    (materials / "renamed.txt").write_text("evaluation customer scenario", encoding="utf-8")
    entries = [
        _source("S01", "new-reference.txt"),
        _source("S02", "renamed.txt", scope="evaluation"),
        _source("S03", "AFS_Examples_and_Reference.docx"),
    ]
    manifest = _manifest(root, entries)
    index_path = root / "index.json"

    result = reindex_sources(manifest, index_path)
    index = _read_index(index_path)

    assert result["files_indexed"] == 1
    assert [item["filename"] for item in index["files"]] == ["new-reference.txt"]


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "materials").mkdir(parents=True)
    manifest = _manifest(root, [_source("S01", "..\\outside.txt")])

    with pytest.raises(ValueError, match="escapes manifest root"):
        reindex_sources(manifest, root / "index.json")


def test_search_rejects_stale_or_corrupt_index(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "materials").mkdir(parents=True)
    source = root / "materials" / "source.txt"
    source.write_text("MIR ATR water", encoding="utf-8")
    manifest = _manifest(root, [_source("S01", "source.txt")])
    index_path = root / "index.json"
    reindex_sources(manifest, index_path)
    retriever = EvidenceRetriever(index_path, manifest, mode="bm25")
    assert retriever.index_metadata()["files"][0]["document_version"]

    source.write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        retriever.search(CustomerUseCase(source_text="water"))

    index_path.write_bytes(b"not sqlite")
    with pytest.raises(ValueError, match="corrupt"):
        retriever.search(CustomerUseCase(source_text="water"))


def test_claim_search_finds_new_document_and_keeps_citation_provenance(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    materials = root / "materials"
    materials.mkdir(parents=True)
    (materials / "general.txt").write_text("spectroscopy process monitoring", encoding="utf-8")
    (materials / "raman.txt").write_text(
        "Raman measurements can be affected by fluorescence background", encoding="utf-8"
    )
    manifest = _manifest(
        root,
        [_source("S01", "general.txt"), _source("S02", "raman.txt")],
    )
    index_path = root / "index.json"
    reindex_sources(manifest, index_path)
    retriever = EvidenceRetriever(index_path, manifest, mode="bm25")
    use_case = CustomerUseCase(source_text="spectroscopy process monitoring")

    results = retriever.search_claim(use_case, "Raman fluorescence background")

    assert results
    result = next(item for item in results if item.source_id == "S02")
    assert result.document_version
    assert result.page_or_section == "lines 1-1"
    assert "fluorescence" in result.excerpt


def test_claim_search_rejects_invalid_citation_provenance(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    materials = root / "materials"
    materials.mkdir(parents=True)
    (materials / "source.txt").write_text("NIR spectra support process screening", encoding="utf-8")
    manifest = _manifest(root, [_source("S01", "source.txt")])
    index_path = root / "index.json"
    reindex_sources(manifest, index_path)
    index = _read_index(index_path)
    index["chunks"][0]["provenance"][0]["document_version"] = "unknown-version"
    _write_index(index_path, index)

    with pytest.raises(ValueError, match="unknown source version"):
        EvidenceRetriever(index_path, manifest).search_claim(
            CustomerUseCase(source_text="NIR process screening"), "NIR spectra"
        )


def test_source_adapter_does_not_turn_raw_text_into_curated_techniques() -> None:
    assert EvidenceKnowledgeSource().get_techniques() == []


def test_hybrid_update_reuses_embeddings_and_retrieves_paraphrase(tmp_path, monkeypatch):
    from spectro_ai import providers
    from spectro_ai.providers import RagSettings

    root = tmp_path / "repo"
    (root / "materials").mkdir(parents=True)
    documents = root / "documents"
    documents.mkdir()
    (documents / "reference.md").write_text(
        "# Optical interference\n\n"
        "Suspended particles deflect the incoming beam, lowering transmitted intensity.",
        encoding="utf-8",
    )
    manifest = _manifest(root, [])
    index = root / "index.sqlite3"
    monkeypatch.setattr(providers, "embedding_identity", lambda settings: "test-model-digest")

    monkeypatch.setattr(
        providers, "embed_texts", lambda settings, texts: [[1.0, 0.0] for _ in texts]
    )
    retriever = EvidenceRetriever(index, manifest, settings=RagSettings(), documents_path=documents)
    first = retriever.update_knowledge_base()
    assert first["documents"] == {"new": 1, "updated": 0, "removed": 0, "unchanged": 0}
    assert first["embeddings"]["new"] >= 1
    assert retriever.knowledge_status()["ready"]

    snapshot = retriever.snapshot()
    second = retriever.update_knowledge_base()
    assert second["embeddings"]["new"] == 0
    assert second["embeddings"]["reused"] == first["chunks"]
    hits = retriever.search(CustomerUseCase(measurement_objective="assess liquid clarity"))
    assert hits and hits[0].source_id.startswith("D")
    assert hits[0].page_or_section.startswith("section=Optical interference")
    assert snapshot.index_metadata()["revision"] != retriever.index_metadata()["revision"]


def test_hybrid_update_failure_preserves_committed_index(tmp_path, monkeypatch):
    from spectro_ai import providers
    from spectro_ai.providers import ProviderError, RagSettings

    root = tmp_path / "repo"
    (root / "materials").mkdir(parents=True)
    documents = root / "documents"
    documents.mkdir()
    (documents / "one.txt").write_text("stable spectroscopy reference", encoding="utf-8")
    manifest = _manifest(root, [])
    index = root / "index.sqlite3"
    monkeypatch.setattr(providers, "embedding_identity", lambda settings: "test-model-digest")
    monkeypatch.setattr(
        providers, "embed_texts", lambda settings, texts: [[1.0, 0.0] for _ in texts]
    )
    retriever = EvidenceRetriever(index, manifest, RagSettings(), documents)
    retriever.update_knowledge_base()
    previous = index.read_bytes()
    (documents / "two.txt").write_text("new spectroscopy reference", encoding="utf-8")
    monkeypatch.setattr(
        providers,
        "embed_texts",
        lambda settings, texts: (_ for _ in ()).throw(ProviderError("offline")),
    )
    with pytest.raises(ProviderError, match="offline"):
        retriever.update_knowledge_base()
    assert index.read_bytes() == previous


def test_chunking_preserves_sentence_and_word_boundaries():
    text = "First scientific sentence. " + ("word " * 55) + "Second sentence follows."
    chunks = _split_text(text, maximum=200, overlap=30)
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert all(not chunk.startswith("ord ") for chunk in chunks)
    assert all(not chunk.endswith("wor") for chunk in chunks)
    assert any(chunk.endswith(".") for chunk in chunks[:-1])
