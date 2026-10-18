from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectro_ai import evidence
from spectro_ai.structure import markdown_source, pack_source


def _split(text: str, maximum: int | None, overlap: int | None) -> list[str]:
    assert maximum is not None
    return [
        text[index : index + maximum] for index in range(0, len(text), maximum - (overlap or 0))
    ]


def test_markdown_keeps_heading_tree_and_source_blocks() -> None:
    source = markdown_source(
        "# Method\n\nIntro paragraph.\n\n## Conditions\n\n"
        "- first item\n- second item\n\n"
        "| Property | Value |\n| --- | --- |\n| range | 400–700 nm |\n"
    )

    assert source.sections["markdown:5:2"].parent_key == "markdown:1:1"
    assert source.sections["markdown:5:2"].heading_path == ("Method", "Conditions")
    assert [unit.block_type for unit in source.units] == [
        "heading",
        "paragraph",
        "heading",
        "list",
        "table_row",
        "table_row",
    ]
    assert source.units[-1].table_header_indexes == (4,)
    assert source.units[-1].line_start == 12
    assert source.units[-1].locator.startswith("section=Method > Conditions; lines 12-")
    assert "section=" not in source.units[1].text


def test_packer_repeats_markdown_headers_and_keeps_oversized_rows() -> None:
    source = markdown_source(
        "# Values\n\n| Property | Value |\n| --- | --- |\n"
        "| A | " + "a" * 150 + " |\n"
        "| B | " + "b" * 220 + " |\n"
        "| C | " + "c" * 150 + " |\n"
    )
    passages = pack_source(source, 200, 20, _split)

    assert len(passages) >= 2
    table_passages = [passage for passage in passages if "Property | Value" in passage.text]
    assert len(table_passages) >= 2
    assert all(passage.text.count("Property | Value") == 1 for passage in table_passages)
    assert any(passage.oversized and "b" * 220 in passage.text for passage in table_passages)
    assert all(
        source.units[span.unit_index].block_type in {"table_row", "heading"}
        for passage in table_passages
        for span in passage.spans
    )


def test_packer_overlaps_only_split_prose_and_keeps_fenced_blocks_whole() -> None:
    prose = "This is a sentence. " * 20
    code = "```text\n" + "x" * 240 + "\n```"
    source = markdown_source(f"# Section\n\n{prose}\n\n{code}\n")
    passages = pack_source(source, 200, 40, _split)

    code_passage = next(passage for passage in passages if "```text" in passage.text)
    assert code_passage.oversized
    assert len(code_passage.spans) == 1
    assert any(passage.oversized for passage in passages if "This is a sentence" in passage.text)


def test_docling_invalid_heading_nesting_is_flattened_with_levels_preserved(monkeypatch) -> None:
    def item(label: str, text: str, page: int, level: int | None = None):
        return SimpleNamespace(
            label=SimpleNamespace(value=label),
            text=text,
            level=level,
            prov=[SimpleNamespace(page_no=page)],
            self_ref=f"#/texts/{text}",
        )

    entries = [
        item("section_header", "1 Method", 1, 1),
        item("text", "Body text with source detail.", 1),
        item("section_header", "1.1.1 Results", 1, 3),
        item("text", "Result body.", 1),
        item("section_header", "Unclassified heading", 1, 0),
    ]
    doc = SimpleNamespace(
        pages={"1": object()},
        iterate_items=lambda **_kwargs: iter((entry, None) for entry in entries),
    )
    monkeypatch.setattr(evidence, "_docling_pdf_document", lambda *_args: doc)

    source = evidence._docling_pdf_source(evidence.Path("fake.pdf"), evidence.Path("models"))

    assert not source.sections["pdf-page:1"].hierarchy_valid
    assert any("jump" in diagnostic for diagnostic in source.diagnostics)
    result_heading = next(unit for unit in source.units if unit.text == "1.1.1 Results")
    assert result_heading.detected_heading_level == 3
    assert result_heading.heading_path == ()
    zero_level = next(unit for unit in source.units if unit.text == "Unclassified heading")
    assert zero_level.detected_heading_level == 0
    assert all(unit.section_key == "pdf-page:1" for unit in source.units)


def test_docling_table_keeps_rows_and_explicit_header_associations(monkeypatch) -> None:
    def cell(text: str, row: int, col: int, *, header: bool = False):
        return SimpleNamespace(
            text=text,
            start_row_offset_idx=row,
            end_row_offset_idx=row + 1,
            start_col_offset_idx=col,
            column_header=header,
        )

    table = SimpleNamespace(
        label=SimpleNamespace(value="table"),
        self_ref="#/tables/0",
        prov=[SimpleNamespace(page_no=2)],
        captions=[],
        data=SimpleNamespace(
            num_rows=2,
            num_cols=2,
            table_cells=[
                cell("Property", 0, 0, header=True),
                cell("Value", 0, 1, header=True),
                cell("Range", 1, 0),
                cell("400-700 nm", 1, 1),
            ],
        ),
        caption_text=lambda _document: "",
    )
    doc = SimpleNamespace(
        pages={"2": object()},
        iterate_items=lambda **_kwargs: iter([(table, None)]),
    )
    monkeypatch.setattr(evidence, "_docling_pdf_document", lambda *_args: doc)

    source = evidence._docling_pdf_source(evidence.Path("fake.pdf"), evidence.Path("models"))
    passages = pack_source(source, 200, 10, _split)

    assert [unit.text for unit in source.units] == ["Property | Value", "Range | 400-700 nm"]
    assert source.units[1].table_header_indexes == (0,)
    assert "Property | Value" in passages[0].text
    assert "Range | 400-700 nm" in passages[0].text


def test_docling_partial_conversion_is_rejected() -> None:
    result = SimpleNamespace(
        status=SimpleNamespace(value="partial_success"),
        document=object(),
        errors=[],
    )

    with pytest.raises(ValueError, match="conversion failed"):
        evidence._checked_docling_document(result, evidence.Path("fake.pdf"))


@pytest.mark.parametrize(
    ("provenance_pages", "message"),
    [([1, 2], "spans pages"), ([3], "unknown PDF page")],
)
def test_docling_rejects_ambiguous_or_out_of_bounds_page_provenance(
    monkeypatch, provenance_pages: list[int], message: str
) -> None:
    table = SimpleNamespace(
        label=SimpleNamespace(value="table"),
        self_ref="#/tables/ambiguous",
        prov=[SimpleNamespace(page_no=page) for page in provenance_pages],
        captions=[],
        data=SimpleNamespace(num_rows=0, num_cols=0, table_cells=[]),
        caption_text=lambda _document: "",
    )
    doc = SimpleNamespace(
        pages={"1": object(), "2": object()},
        iterate_items=lambda **_kwargs: iter([(table, None)]),
    )
    monkeypatch.setattr(evidence, "_docling_pdf_document", lambda *_args: doc)

    with pytest.raises(ValueError, match=message):
        evidence._docling_pdf_source(evidence.Path("fake.pdf"), evidence.Path("models"))


def test_docling_preserves_all_known_pages_for_spanning_text(monkeypatch) -> None:
    text = SimpleNamespace(
        label=SimpleNamespace(value="text"),
        text="A paragraph that crosses a PDF page boundary.",
        self_ref="#/texts/spanning",
        prov=[SimpleNamespace(page_no=1), SimpleNamespace(page_no=2)],
    )
    doc = SimpleNamespace(
        pages={"1": object(), "2": object()},
        iterate_items=lambda **_kwargs: iter([(text, None)]),
    )
    monkeypatch.setattr(evidence, "_docling_pdf_document", lambda *_args: doc)

    source = evidence._docling_pdf_source(evidence.Path("fake.pdf"), evidence.Path("models"))

    assert source.units[0].page_numbers == (1, 2)
    assert source.units[0].page_number == 1
