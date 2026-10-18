from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from spectro_ai import evidence, providers
from spectro_ai.evidence import EvidenceRetriever, canonical_context_block
from spectro_ai.providers import RagSettings
from spectro_ai.structure import docx_source, pack_source, txt_source, xlsx_source

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _write_docx(path: Path) -> None:
    xml = f"""<w:document xmlns:w="{WORD_NS}"><w:body>
      <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
        <w:r><w:t>Measurement</w:t></w:r></w:p>
      <w:p><w:r><w:t>Operating condition</w:t><w:tab/><w:t>20 °C</w:t>
        <w:br/><w:t>under dry nitrogen</w:t></w:r></w:p>
      <w:p><w:pPr><w:pStyle w:val="Heading 2"/></w:pPr>
        <w:r><w:t>Calibration</w:t></w:r></w:p>
      <w:p><w:pPr><w:pStyle w:val="Caption"/></w:pPr>
        <w:r><w:t>Calibration standards</w:t></w:r></w:p>
      <w:tbl><w:tr><w:trPr><w:tblHeader w:val="1"/></w:trPr>
        <w:tc><w:p><w:r><w:t>Sample</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Wavelength</w:t></w:r></w:p></w:tc></w:tr>
        <w:tr><w:tc><w:p/></w:tc><w:tc><w:p/></w:tc></w:tr>
        <w:tr><w:tc><w:p><w:r><w:t>Oil</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>4.2 μm</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
      <w:sectPr/></w:body></w:document>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)


def _write_xlsx(path: Path) -> None:
    workbook = f"""<workbook xmlns="{XLSX_NS}" xmlns:r="{DOC_REL_NS}">
      <sheets><sheet name="calibration" sheetId="1" r:id="rId1"/></sheets></workbook>"""
    workbook_rels = f"""<Relationships xmlns="{PACKAGE_REL_NS}">
      <Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"""
    worksheet = f"""<worksheet xmlns="{XLSX_NS}" xmlns:r="{DOC_REL_NS}">
      <sheetData>
        <row r="2"><c r="A2" t="inlineStr"><is><t>Parameter</t></is></c>
          <c r="C2" t="inlineStr"><is><t>Source value</t></is></c></row>
        <row r="3"><c r="A3"/><c r="C3"/></row>
        <row r="4"><c r="A4" t="inlineStr"><is><t>Range</t></is></c>
          <c r="C4" t="inlineStr"><is><t>400–700 nm</t></is></c></row>
        <row r="7"><c r="A7" t="inlineStr"><is><t>Unmarked first row</t></is></c>
          <c r="B7" t="inlineStr"><is><t>Data, not a declared header</t></is></c></row>
      </sheetData><tableParts count="1"><tablePart r:id="rId2"/></tableParts>
      </worksheet>"""
    worksheet_rels = f"""<Relationships xmlns="{PACKAGE_REL_NS}">
      <Relationship Id="rId2" Target="../tables/table1.xml"/></Relationships>"""
    table = f"""<table xmlns="{XLSX_NS}" id="1" name="Calibration" displayName="Calibration"
      ref="A2:C4" headerRowCount="1"><autoFilter ref="A2:C4"/>
      <tableColumns count="3"><tableColumn id="1" name="Parameter"/>
        <tableColumn id="2" name="Column2"/><tableColumn id="3" name="Source value"/>
      </tableColumns></table>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/worksheets/_rels/sheet1.xml.rels", worksheet_rels)
        archive.writestr("xl/tables/table1.xml", table)


def _write_xlsx_side_by_side(path: Path) -> None:
    workbook = f"""<workbook xmlns="{XLSX_NS}" xmlns:r="{DOC_REL_NS}">
      <sheets><sheet name="paired" sheetId="1" r:id="rId1"/></sheets></workbook>"""
    workbook_rels = f"""<Relationships xmlns="{PACKAGE_REL_NS}">
      <Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>"""
    worksheet = f"""<worksheet xmlns="{XLSX_NS}" xmlns:r="{DOC_REL_NS}">
      <sheetData>
        <row r="1"><c r="A1" t="inlineStr"><is><t>Reading</t></is></c>
          <c r="B1" t="inlineStr"><is><t>Unit</t></is></c>
          <c r="D1" t="inlineStr"><is><t>Range</t></is></c>
          <c r="E1" t="inlineStr"><is><t>Value</t></is></c></row>
        <row r="2"><c r="A2" t="inlineStr"><is><t>Wavelength</t></is></c>
          <c r="B2" t="inlineStr"><is><t>nm</t></is></c>
          <c r="D2" t="inlineStr"><is><t>Operating</t></is></c>
          <c r="E2" t="inlineStr"><is><t>20 °C</t></is></c></row>
      </sheetData><tableParts count="2">
        <tablePart r:id="rId2"/><tablePart r:id="rId3"/></tableParts></worksheet>"""
    worksheet_rels = f"""<Relationships xmlns="{PACKAGE_REL_NS}">
      <Relationship Id="rId2" Target="../tables/left.xml"/>
      <Relationship Id="rId3" Target="../tables/right.xml"/></Relationships>"""
    left = f"""<table xmlns="{XLSX_NS}" id="1" name="Left" displayName="Left"
      ref="A1:B2" headerRowCount="1"/>"""
    right = f"""<table xmlns="{XLSX_NS}" id="2" name="Right" displayName="Right"
      ref="D1:E2" headerRowCount="1"/>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/worksheets/_rels/sheet1.xml.rels", worksheet_rels)
        archive.writestr("xl/tables/left.xml", left)
        archive.writestr("xl/tables/right.xml", right)


def test_txt_paragraphs_keep_original_lines_and_source_units() -> None:
    source = txt_source("Operating at 20 °C\nwith dry nitrogen.\n\nLimit: 4.2 μm.\n")

    assert [unit.text for unit in source.units] == [
        "Operating at 20 °C\nwith dry nitrogen.",
        "Limit: 4.2 μm.",
    ]
    assert [(unit.line_start, unit.line_end) for unit in source.units] == [(1, 2), (4, 4)]
    assert [unit.locator for unit in source.units] == ["lines 1-2", "lines 4-4"]
    assert not source.sections[source.units[0].section_key].heading_path


def test_docx_preserves_heading_paths_runs_cell_positions_and_explicit_headers(tmp_path) -> None:
    path = tmp_path / "reference.docx"
    _write_docx(path)

    source = docx_source(path)
    operating = next(unit for unit in source.units if unit.text.startswith("Operating"))
    heading = next(unit for unit in source.units if unit.text == "Calibration")
    table_rows = [unit for unit in source.units if unit.block_type == "table_row"]
    caption = next(unit for unit in source.units if unit.block_type == "table_caption")

    assert source.sections[heading.section_key].heading_path == ("Measurement", "Calibration")
    assert operating.text == "Operating condition\t20 °C\nunder dry nitrogen"
    assert caption.table_id == table_rows[0].table_id
    assert table_rows[0].is_table_header
    assert not table_rows[1].is_table_header
    assert table_rows[1].table_header_indexes == (source.units.index(table_rows[0]),)
    assert table_rows[1].text == "Oil | 4.2 μm"
    assert table_rows[1].table_row_index == 2
    assert all("row=2" not in unit.locator for unit in table_rows)


def test_docx_heading_level_gaps_are_retained_as_invalid_not_repaired(tmp_path) -> None:
    path = tmp_path / "ambiguous.docx"
    xml = f"""<w:document xmlns:w="{WORD_NS}"><w:body>
      <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Method</w:t></w:r></w:p>
      <w:p><w:pPr><w:pStyle w:val="Heading3"/></w:pPr><w:r><w:t>Calibration</w:t></w:r></w:p>
      </w:body></w:document>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)

    source = docx_source(path)

    assert source.diagnostics == ["Heading level jump 1 to 3"]
    assert source.sections["docx-root"].hierarchy_valid is False
    assert [unit.detected_heading_level for unit in source.units] == [1, 3]
    assert all(not unit.heading_path for unit in source.units)


def test_xlsx_uses_only_declared_table_headers_and_keeps_sparse_row_positions(tmp_path) -> None:
    path = tmp_path / "values.xlsx"
    _write_xlsx(path)

    source = xlsx_source(path)
    table_rows = [unit for unit in source.units if unit.block_type == "table_row"]
    unmarked = next(unit for unit in source.units if "Unmarked first row" in unit.text)

    assert [unit.is_table_header for unit in table_rows] == [True, False]
    assert table_rows[1].table_header_indexes == (source.units.index(table_rows[0]),)
    assert table_rows[1].text == "Range |  | 400–700 nm"
    assert table_rows[1].locator == "worksheet=calibration rows=4-4"
    assert all("rows=3-3" not in unit.locator for unit in source.units)
    assert unmarked.block_type == "paragraph"
    assert unmarked.table_id is None
    assert not unmarked.is_table_header

    passages = pack_source(source, 36, 0, lambda text, _maximum, _overlap: [text])
    assert all(
        "Parameter |  | Source value" in passage.text
        for passage in passages
        if "Range |  | 400–700 nm" in passage.text
    )


def test_xlsx_splits_disjoint_side_by_side_tables_without_losing_headers(tmp_path) -> None:
    path = tmp_path / "side-by-side.xlsx"
    _write_xlsx_side_by_side(path)

    source = xlsx_source(path)
    table_rows = [unit for unit in source.units if unit.block_type == "table_row"]
    data_rows = [unit for unit in table_rows if not unit.is_table_header]

    assert len(table_rows) == 4
    assert len({unit.table_id for unit in table_rows}) == 2
    assert all(len(unit.table_header_indexes) == 1 for unit in data_rows)
    assert [unit.text for unit in data_rows] == ["Wavelength | nm", "Operating | 20 °C"]
    assert not source.diagnostics


def test_structured_txt_index_has_exact_context_provenance(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(providers, "embedding_identity", lambda _settings: "model-digest")
    monkeypatch.setattr(
        providers, "embed_texts", lambda _settings, texts: [[1.0, 0.0] for _ in texts]
    )
    materials = tmp_path / "materials"
    documents = tmp_path / "documents"
    materials.mkdir()
    documents.mkdir()
    source_file = documents / "source.txt"
    source_file.write_text(
        "The sensor operates at 20 °C.\nIts range is 400–700 nm.\n\n"
        "The range excludes humid samples.\n",
        encoding="utf-8",
    )
    manifest = materials / "sources.yaml"
    manifest.write_text(
        "schema_version: 1\nroot: .\nsources:\n"
        "  - source_id: S01\n    filename: source.txt\n    title: Source\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.sqlite3"
    retriever = EvidenceRetriever(
        index_path=index_path,
        manifest_path=manifest,
        settings=RagSettings(),
        documents_path=documents,
        mode="bm25",
    )
    retriever.update_knowledge_base()
    index = retriever.index_metadata()
    blocks = index["source_blocks"]

    assert [block["location"]["locator"] for block in blocks] == ["lines 1-2", "lines 4-4"]
    assert canonical_context_block(index, blocks[1]["block_id"]).text == (
        "The range excludes humid samples."
    )


def test_vector_reuse_survives_structural_parser_fingerprint_change(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def embed(_settings, texts: list[str]) -> list[list[float]]:
        calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(providers, "embedding_identity", lambda _settings: "model-digest")
    monkeypatch.setattr(providers, "embed_texts", embed)
    monkeypatch.setattr(evidence, "_extraction_fingerprint", lambda *_args: "parser-v1")
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "reference.txt").write_text(
        "A source block with a customer unit, 4.2 μm, and an explicit limitation.",
        encoding="utf-8",
    )
    materials = tmp_path / "materials"
    materials.mkdir()
    manifest = tmp_path / "sources.yaml"
    manifest.write_text("schema_version: 1\nroot: materials\nsources: []\n", encoding="utf-8")
    retriever = EvidenceRetriever(
        index_path=tmp_path / "index.sqlite3",
        manifest_path=manifest,
        settings=RagSettings(),
        documents_path=documents,
    )
    retriever.update_knowledge_base()
    monkeypatch.setattr(evidence, "_extraction_fingerprint", lambda *_args: "parser-v2")

    updated = retriever.update_knowledge_base()

    assert len(calls) == 1
    assert updated["embeddings"]["reused"] == 1


def test_changed_passage_gets_new_id_but_unchanged_passage_keeps_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(providers, "embedding_identity", lambda _settings: "model-digest")
    monkeypatch.setattr(
        providers, "embed_texts", lambda _settings, texts: [[1.0, 0.0] for _ in texts]
    )
    monkeypatch.setattr(evidence, "_extraction_fingerprint", lambda *_args: "txt-parser-v1")
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "changed.txt").write_text(
        "Original paragraph with a measured unit of 4.2 μm.", encoding="utf-8"
    )
    (documents / "stable.txt").write_text(
        "Unchanged paragraph with a measured unit of 700 nm.", encoding="utf-8"
    )
    materials = tmp_path / "materials"
    materials.mkdir()
    manifest = tmp_path / "sources.yaml"
    manifest.write_text("schema_version: 1\nroot: materials\nsources: []\n", encoding="utf-8")
    retriever = EvidenceRetriever(
        index_path=tmp_path / "index.sqlite3",
        manifest_path=manifest,
        settings=RagSettings(),
        documents_path=documents,
    )
    retriever.update_knowledge_base()
    previous = {
        provenance["filename"]: provenance["evidence_id"]
        for chunk in retriever.index_metadata()["chunks"]
        for provenance in chunk["provenance"]
    }
    original_parser = evidence.txt_source

    def changed_parser(text: str):
        source = original_parser(text)
        if text.startswith("Original paragraph"):
            source.units[0] = replace(source.units[0], text=source.units[0].text + " Revised.")
        return source

    monkeypatch.setattr(evidence, "txt_source", changed_parser)
    monkeypatch.setattr(evidence, "_extraction_fingerprint", lambda *_args: "txt-parser-v2")
    retriever.update_knowledge_base()
    current = {
        provenance["filename"]: provenance["evidence_id"]
        for chunk in retriever.index_metadata()["chunks"]
        for provenance in chunk["provenance"]
    }

    assert current["changed.txt"] != previous["changed.txt"]
    assert current["stable.txt"] == previous["stable.txt"]
