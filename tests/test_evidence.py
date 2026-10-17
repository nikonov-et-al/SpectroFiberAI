import json
import sys
from pathlib import Path
from types import ModuleType
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from spectro_ai.evidence import EvidenceKnowledgeSource, EvidenceRetriever, reindex_sources
from spectro_ai.models import CustomerUseCase


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
    chunks = json.loads(index_path.read_text(encoding="utf-8"))["chunks"]
    locators = {p["page_or_section"] for chunk in chunks for p in chunk["provenance"]}

    assert result["files_indexed"] == 2
    assert "lines 1-2" in locators
    assert "worksheet=methods rows=1-2" in locators
    retrieved = EvidenceRetriever(index_path, manifest).search(
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
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert {p["page_or_section"] for c in index["chunks"] for p in c["provenance"]} == {
        "section=Methods"
    }
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
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert result["empty_pdf_pages"] == 1
    assert result["poor_extraction_pages"] == 2
    assert index["files"][0]["page_count"] == 3
    assert index["files"][0]["empty_pages"] == ["p.3"]
    assert index["files"][0]["poor_extraction_pages"] == ["p.1", "p.2"]
    assert index["chunks"][0]["provenance"][0]["page_or_section"] == "p.1"
    assert index["chunks"][0]["provenance"][0]["review_status"] == (
        "needs_visual_review; poor_extraction_needs_manual_review"
    )
    retriever = EvidenceRetriever(index_path, manifest)
    index["chunks"][0]["provenance"][0]["page_or_section"] = "p.0"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="outside source pages"):
        retriever.index_metadata()

    reindex_sources(manifest, index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    first_id = index["chunks"][0]["provenance"][0]["evidence_id"]
    index["chunks"][1]["provenance"][0]["evidence_id"] = first_id
    index_path.write_text(json.dumps(index), encoding="utf-8")
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
    index = json.loads(index_path.read_text(encoding="utf-8"))

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
    old_version = json.loads(index_path.read_text(encoding="utf-8"))["files"][0]["document_version"]
    source.write_text("NIR milk powder", encoding="utf-8")
    second = reindex_sources(manifest, index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))

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
    index = json.loads(index_path.read_text(encoding="utf-8"))

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
    retriever = EvidenceRetriever(index_path, manifest)
    assert retriever.index_metadata()["files"][0]["document_version"]

    source.write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        retriever.search(CustomerUseCase(source_text="water"))

    index_path.write_text('{"schema_version": 99}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        retriever.search(CustomerUseCase(source_text="water"))


def test_source_adapter_does_not_turn_raw_text_into_curated_techniques() -> None:
    assert EvidenceKnowledgeSource().get_techniques() == []
