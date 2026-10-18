"""Small source-order models and packers for structure-aware ingestion."""

from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZipFile


@dataclass(frozen=True)
class SourceUnit:
    text: str
    locator: str
    section_key: str
    heading_path: tuple[str, ...] = ()
    block_type: str = "paragraph"
    page_number: int | None = None
    page_numbers: tuple[int, ...] = ()
    line_start: int | None = None
    line_end: int | None = None
    detected_heading_level: int | None = None
    table_id: str | None = None
    table_row_index: int | None = None
    is_table_header: bool = False
    table_header_indexes: tuple[int, ...] = ()


@dataclass
class SectionSpec:
    key: str
    heading_path: tuple[str, ...] = ()
    parent_key: str | None = None
    detected_heading_level: int | None = None
    hierarchy_valid: bool = True
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class StructuredSource:
    units: list[SourceUnit] = field(default_factory=list)
    sections: dict[str, SectionSpec] = field(default_factory=dict)
    empty_pages: list[str] = field(default_factory=list)
    poor_pages: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BlockSpan:
    unit_index: int
    start: int
    end: int


@dataclass(frozen=True)
class PackedPassage:
    text: str
    section_key: str
    spans: tuple[BlockSpan, ...]
    oversized: bool = False


_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_LIST = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
_TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def txt_source(text: str) -> StructuredSource:
    """Keep TXT paragraphs as blocks with their original one-based line ranges."""
    source = StructuredSource()
    root_key = "txt-root"
    source.sections[root_key] = SectionSpec(root_key)
    lines = text.splitlines()
    start: int | None = None
    paragraph: list[str] = []

    def flush(end: int) -> None:
        nonlocal start, paragraph
        if start is not None:
            source.units.append(
                SourceUnit(
                    text="\n".join(paragraph),
                    locator=f"lines {start}-{end}",
                    section_key=root_key,
                    line_start=start,
                    line_end=end,
                )
            )
        start, paragraph = None, []

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            if start is not None:
                flush(line_number - 1)
            continue
        if start is None:
            start = line_number
        paragraph.append(line)
    if start is not None:
        flush(len(lines))
    return source


def _word_tag(name: str) -> str:
    return f"{{{_WORD_NS}}}{name}"


def _word_value(node: ElementTree.Element | None, name: str) -> str | None:
    return node.get(_word_tag(name)) if node is not None else None


def _word_text(node: ElementTree.Element) -> str:
    parts: list[str] = []
    for child in node.iter():
        if child.tag == _word_tag("t"):
            parts.append(child.text or "")
        elif child.tag == _word_tag("tab"):
            parts.append("\t")
        elif child.tag in {_word_tag("br"), _word_tag("cr")}:
            parts.append("\n")
    return "".join(parts)


def _word_paragraph_text(paragraph: ElementTree.Element) -> str:
    return _word_text(paragraph).strip()


def _word_cell_text(cell: ElementTree.Element) -> str:
    return "\n".join(
        text
        for paragraph in cell.findall(f".//{_word_tag('p')}")
        if (text := _word_paragraph_text(paragraph))
    )


def _word_heading_level(paragraph: ElementTree.Element) -> tuple[int | None, str | None]:
    properties = paragraph.find(_word_tag("pPr"))
    style = _word_value(
        properties.find(_word_tag("pStyle")) if properties is not None else None, "val"
    )
    if style:
        match = re.fullmatch(r"heading\s*(\d+)", style.strip(), re.IGNORECASE)
        if match:
            return int(match.group(1)), style
    outline = properties.find(_word_tag("outlineLvl")) if properties is not None else None
    raw_level = _word_value(outline, "val")
    if raw_level and raw_level.isdigit():
        if int(raw_level) == 9:
            return None, style
        return int(raw_level) + 1, style
    return None, style


def docx_source(path: Path) -> StructuredSource:
    """Extract DOCX body order, heading styles, explicit table headers and cell positions."""
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{{{_WORD_NS}}}body")
    source = StructuredSource()
    if body is None:
        return source

    items: list[tuple[str, ElementTree.Element, int]] = []
    heading_levels: list[tuple[int, int]] = []
    table_number = 0
    for body_index, element in enumerate(body, 1):
        if element.tag == _word_tag("p"):
            level, _style = _word_heading_level(element)
            if level is not None:
                heading_levels.append((body_index, level))
            items.append(("paragraph", element, body_index))
        elif element.tag == _word_tag("tbl"):
            table_number += 1
            items.append(("table", element, table_number))

    diagnostics: list[str] = []
    previous_level = 0
    for _body_index, level in heading_levels:
        if level < 1 or level > 9:
            diagnostics.append(f"Heading style level {level} is outside the supported range")
        if previous_level and level > previous_level + 1:
            diagnostics.append(f"Heading level jump {previous_level} to {level}")
        previous_level = level
    hierarchy_valid = not diagnostics
    source.diagnostics.extend(diagnostics)
    root_key = "docx-root"
    source.sections[root_key] = SectionSpec(
        root_key, hierarchy_valid=hierarchy_valid, diagnostics=list(diagnostics)
    )
    current_key = root_key
    heading_stack: list[tuple[int, str, str]] = []
    for item_index, (kind, element, ordinal) in enumerate(items):
        if kind == "paragraph":
            text = _word_paragraph_text(element)
            if not text:
                continue
            level, style = _word_heading_level(element)
            if level is not None:
                if hierarchy_valid:
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    parent_key = heading_stack[-1][2] if heading_stack else root_key
                    heading_path = tuple(entry[1] for entry in heading_stack) + (text,)
                    current_key = f"docx-heading:{ordinal}:{level}"
                    source.sections[current_key] = SectionSpec(
                        current_key,
                        heading_path=heading_path,
                        parent_key=parent_key,
                        detected_heading_level=level,
                    )
                    heading_stack.append((level, text, current_key))
                else:
                    heading_path = ()
                    current_key = root_key
                source.units.append(
                    SourceUnit(
                        text=text,
                        locator=f"paragraph={ordinal}",
                        section_key=current_key,
                        heading_path=heading_path,
                        block_type="heading",
                        detected_heading_level=level,
                    )
                )
                continue
            properties = element.find(_word_tag("pPr"))
            if style and style.casefold().startswith("caption"):
                next_item = items[item_index + 1] if item_index + 1 < len(items) else None
                if next_item and next_item[0] == "table":
                    caption_id = f"docx-table:{next_item[2]}"
                    caption_locator = f"table={next_item[2]} caption"
                    caption_type = "table_caption"
                else:
                    caption_id = None
                    caption_locator = f"paragraph={ordinal}"
                    caption_type = "caption"
                source.units.append(
                    SourceUnit(
                        text=text,
                        locator=caption_locator,
                        section_key=current_key,
                        heading_path=source.sections[current_key].heading_path,
                        block_type=caption_type,
                        table_id=caption_id,
                    )
                )
                continue
            numbering = properties.find(_word_tag("numPr")) if properties is not None else None
            source.units.append(
                SourceUnit(
                    text=text,
                    locator=f"paragraph={ordinal}",
                    section_key=current_key,
                    heading_path=source.sections[current_key].heading_path,
                    block_type="list" if numbering is not None else "paragraph",
                )
            )
        else:
            table_id = f"docx-table:{ordinal}"
            rows = element.findall(_word_tag("tr"))
            row_texts: list[tuple[int, str, bool]] = []
            for row_number, row in enumerate(rows, 1):
                row_properties = row.find(_word_tag("trPr"))
                header_marker = (
                    row_properties.find(_word_tag("tblHeader"))
                    if row_properties is not None
                    else None
                )
                header = header_marker is not None and _word_value(header_marker, "val") not in {
                    "0",
                    "false",
                    "off",
                }
                cells = row.findall(_word_tag("tc"))
                values: list[str] = []
                for cell in cells:
                    text = _word_cell_text(cell)
                    cell_properties = cell.find(_word_tag("tcPr"))
                    span_node = (
                        cell_properties.find(_word_tag("gridSpan"))
                        if cell_properties is not None
                        else None
                    )
                    span_value = _word_value(span_node, "val")
                    span = int(span_value) if span_value and span_value.isdigit() else 1
                    values.append(text)
                    values.extend([""] * (span - 1))
                if any(value.strip() for value in values):
                    row_texts.append((row_number, " | ".join(values), header))
            header_positions = {row_number for row_number, _text, header in row_texts if header}
            first_unit = len(source.units)
            for row_number, text, header in row_texts:
                source.units.append(
                    SourceUnit(
                        text=text,
                        locator=f"table={ordinal} row={row_number}",
                        section_key=current_key,
                        heading_path=source.sections[current_key].heading_path,
                        block_type="table_row",
                        table_id=table_id,
                        table_row_index=row_number - 1,
                        is_table_header=header,
                        table_header_indexes=tuple(
                            first_unit + offset
                            for offset, (candidate_row, _value, _header) in enumerate(row_texts)
                            if candidate_row in header_positions
                        )
                        if not header
                        else (),
                    )
                )
    return source


def _column_index(cell_reference: str) -> int | None:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if letters is None:
        return None
    value = 0
    for letter in letters.group():
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _cell_reference(cell_reference: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\$?([A-Z]+)\$?(\d+)", cell_reference.upper())
    if not match:
        return None
    column = _column_index(match.group(1))
    return (int(match.group(2)), column) if column is not None else None


def _range_bounds(reference: str) -> tuple[int, int, int, int] | None:
    parts = reference.split(":", maxsplit=1)
    start = _cell_reference(parts[0])
    end = _cell_reference(parts[-1])
    if start is None or end is None:
        return None
    return start[0], start[1], end[0], end[1]


def _xlsx_cell_text(cell: ElementTree.Element, shared: list[str]) -> str:
    kind = cell.get("t")
    if kind == "inlineStr":
        inline = cell.find(f"{{{_XLSX_NS}}}is")
        return "".join(inline.itertext()) if inline is not None else ""
    raw = cell.findtext(f"{{{_XLSX_NS}}}v", default="")
    if kind == "s" and raw:
        return shared[int(raw)]
    return raw


def _relationship_targets(xml: ElementTree.Element) -> dict[str, str]:
    return {
        str(relation.get("Id")): str(relation.get("Target", ""))
        for relation in xml.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
    }


def xlsx_source(path: Path) -> StructuredSource:
    """Extract worksheet rows while recognizing headers only from explicit tables."""
    ns = {
        "main": _XLSX_NS,
        "doc": _DOC_REL_NS,
        "package": _PACKAGE_REL_NS,
    }
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(item.itertext()) for item in shared_root.findall("main:si", ns)]
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        workbook_rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        workbook_targets = _relationship_targets(workbook_rels)
        source = StructuredSource()
        sheets = workbook.findall("main:sheets/main:sheet", ns)
        for sheet_number, sheet in enumerate(sheets, 1):
            title = str(sheet.get("name", f"sheet{sheet_number}"))
            relationship_id = sheet.get(f"{{{_DOC_REL_NS}}}id")
            target = workbook_targets.get(relationship_id or "")
            if not target:
                raise ValueError(f"Missing worksheet relationship for {title}")
            worksheet_path = posixpath.normpath(
                target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
            )
            if worksheet_path not in names:
                raise ValueError(f"Missing worksheet XML for {title}")
            worksheet = ElementTree.fromstring(archive.read(worksheet_path))
            section_key = f"xlsx-sheet:{sheet_number}"
            source.sections[section_key] = SectionSpec(section_key, heading_path=(title,))
            worksheet_dir, worksheet_name = posixpath.split(worksheet_path)
            relation_path = posixpath.join(worksheet_dir, "_rels", worksheet_name + ".rels")
            table_parts: dict[str, str] = {}
            if relation_path in names:
                worksheet_rels = ElementTree.fromstring(archive.read(relation_path))
                table_parts = _relationship_targets(worksheet_rels)
            tables: list[tuple[str, tuple[int, int, int, int], int]] = []
            for table_part in worksheet.findall("main:tableParts/main:tablePart", ns):
                part_id = table_part.get(f"{{{_DOC_REL_NS}}}id", "")
                part_target = table_parts.get(part_id)
                if not part_target:
                    continue
                table_path = posixpath.normpath(
                    part_target.lstrip("/")
                    if part_target.startswith("/")
                    else posixpath.join(worksheet_dir, part_target)
                )
                if table_path not in names:
                    raise ValueError(f"Missing XLSX table definition {table_path}")
                table_xml = ElementTree.fromstring(archive.read(table_path))
                bounds = _range_bounds(str(table_xml.get("ref", "")))
                if bounds is None:
                    continue
                header_rows = int(table_xml.get("headerRowCount", "1"))
                if header_rows < 0:
                    raise ValueError(f"XLSX table {table_xml.get('name')} has invalid header count")
                table_name = str(
                    table_xml.get("displayName") or table_xml.get("name") or table_xml.get("id")
                )
                tables.append((table_name, bounds, header_rows))

            worksheet_data = worksheet.find("main:sheetData", ns)
            if worksheet_data is None:
                continue
            rows = worksheet_data.findall("main:row", ns)
            parsed_rows: list[tuple[int, dict[int, str]]] = []
            max_column = max((bounds[3] for _name, bounds, _headers in tables), default=-1)
            for row in rows:
                row_number = int(row.get("r", "0"))
                cells_by_column: dict[int, str] = {}
                for cell in row.findall("main:c", ns):
                    reference = _cell_reference(str(cell.get("r", "")))
                    if reference is None:
                        continue
                    _actual_row, column = reference
                    cells_by_column[column] = _xlsx_cell_text(cell, shared)
                    max_column = max(max_column, column)
                parsed_rows.append((row_number, cells_by_column))

            tables.sort(key=lambda table: (table[1][0], table[1][1], table[0]))
            invalid_tables: set[str] = set()
            for table_index, (name, bounds, _header_count) in enumerate(tables):
                for other_name, other_bounds, _other_header_count in tables[table_index + 1 :]:
                    rows_overlap = bounds[0] <= other_bounds[2] and other_bounds[0] <= bounds[2]
                    columns_overlap = bounds[1] <= other_bounds[3] and other_bounds[1] <= bounds[3]
                    if rows_overlap and columns_overlap:
                        invalid_tables.update((name, other_name))
                        source.diagnostics.append(
                            f"Overlapping XLSX table ranges on {title}: {name}, {other_name}; "
                            "their rows are retained without table associations."
                        )

            for row_number, cells_by_column in parsed_rows:
                memberships = sorted(
                    (
                        (name, bounds, header_count)
                        for name, bounds, header_count in tables
                        if name not in invalid_tables and bounds[0] <= row_number <= bounds[2]
                    ),
                    key=lambda table: table[1][1],
                )
                occupied = {
                    column
                    for _name, bounds, _header_count in memberships
                    for column in range(bounds[1], bounds[3] + 1)
                }
                row_units: list[tuple[int, SourceUnit]] = []
                for name, bounds, header_count in memberships:
                    values = [
                        cells_by_column.get(column, "")
                        for column in range(bounds[1], bounds[3] + 1)
                    ]
                    if not any(value.strip() for value in values):
                        continue
                    table_id = f"xlsx-table:{sheet_number}:{name}"
                    row_index = row_number - bounds[0]
                    header = row_index < header_count
                    columns = (
                        ""
                        if bounds[1] == 0 and bounds[3] == max_column
                        else f" columns={bounds[1]}-{bounds[3]}"
                    )
                    header_indexes = tuple(
                        index
                        for index, unit in enumerate(source.units)
                        if unit.table_id == table_id and unit.is_table_header
                    )
                    row_units.append(
                        (
                            bounds[1],
                            SourceUnit(
                                text=" | ".join(values),
                                locator=(
                                    f"worksheet={title} rows={row_number}-{row_number}{columns}"
                                ),
                                section_key=section_key,
                                heading_path=(title,),
                                block_type="table_row",
                                table_id=table_id,
                                table_row_index=row_index,
                                is_table_header=header,
                                table_header_indexes=header_indexes if not header else (),
                            ),
                        )
                    )
                residual = {
                    column: value
                    for column, value in cells_by_column.items()
                    if column not in occupied and value.strip()
                }
                if residual:
                    values = [residual.get(column, "") for column in range(max_column + 1)]
                    columns = (
                        ""
                        if all(column in residual for column in range(max_column + 1))
                        else f" columns={min(residual)}-{max(residual)}"
                    )
                    row_units.append(
                        (
                            min(residual),
                            SourceUnit(
                                text=" | ".join(values),
                                locator=(
                                    f"worksheet={title} rows={row_number}-{row_number}{columns}"
                                ),
                                section_key=section_key,
                                heading_path=(title,),
                            ),
                        )
                    )
                row_units.sort(key=lambda item: item[0])
                for _column, unit in row_units:
                    source.units.append(unit)
    return source


def markdown_source(text: str) -> StructuredSource:
    """Parse Markdown blocks with exact line locations and explicit heading ancestry."""
    lines = text.splitlines()
    source = StructuredSource()
    heading_stack: list[tuple[int, str, str]] = []
    root_key = "markdown-root"
    source.sections[root_key] = SectionSpec(root_key)
    current_key = root_key
    current_path: tuple[str, ...] = ()
    index = 0

    def add_block(start: int, end: int, block_text: str, kind: str, **extra: object) -> None:
        line_locator = f"lines {start}-{end}"
        locator = (
            f"section={' > '.join(current_path)}; {line_locator}" if current_path else line_locator
        )
        source.units.append(
            SourceUnit(
                text=block_text,
                locator=locator,
                section_key=current_key,
                heading_path=current_path,
                block_type=kind,
                line_start=start,
                line_end=end,
                **extra,
            )
        )

    while index < len(lines):
        line = lines[index]
        number = index + 1
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            parent_key = heading_stack[-1][2] if heading_stack else root_key
            current_path = tuple(entry[1] for entry in heading_stack) + (title,)
            current_key = f"markdown:{number}:{level}"
            source.sections[current_key] = SectionSpec(
                key=current_key,
                heading_path=current_path,
                parent_key=parent_key,
                detected_heading_level=level,
            )
            add_block(number, number, title, "heading", detected_heading_level=level)
            heading_stack.append((level, title, current_key))
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            start = index
            index += 1
            while index < len(lines):
                closing = re.match(
                    r"^\s{0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$",
                    lines[index],
                )
                index += 1
                if closing:
                    break
            add_block(start + 1, index, "\n".join(lines[start:index]), "code")
            continue
        if index + 1 < len(lines) and "|" in line and _TABLE_DELIMITER.match(lines[index + 1]):
            table_start = index
            table_id = f"markdown-table:{number}"
            table_rows: list[tuple[int, str]] = []
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                raw = lines[index].strip().strip("|")
                table_rows.append((index + 1, " | ".join(cell.strip() for cell in raw.split("|"))))
                index += 1
            # Markdown separator row describes formatting, not source evidence.
            data_rows = [row for row in table_rows if not _TABLE_DELIMITER.match(lines[row[0] - 1])]
            header_indexes = tuple(range(len(source.units), len(source.units) + 1))
            for row_index, (line_number, row_text) in enumerate(data_rows):
                source.units.append(
                    SourceUnit(
                        text=row_text,
                        locator=(
                            f"section={' > '.join(current_path)}; lines {line_number}-{line_number}"
                            if current_path
                            else f"lines {line_number}-{line_number}"
                        ),
                        section_key=current_key,
                        heading_path=current_path,
                        block_type="table_row",
                        line_start=line_number,
                        line_end=line_number,
                        table_id=table_id,
                        table_row_index=row_index,
                        is_table_header=row_index == 0,
                        table_header_indexes=header_indexes if row_index else (),
                    )
                )
            if index == table_start:
                index += 1
            continue
        if _LIST.match(line):
            start = index
            while (
                index < len(lines)
                and lines[index].strip()
                and (_LIST.match(lines[index]) or lines[index][:1].isspace())
            ):
                index += 1
            add_block(start + 1, index, "\n".join(lines[start:index]), "list")
            continue
        start = index
        paragraphs = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip():
            if re.match(r"^\s{0,3}#{1,6}\s+", lines[index]) or _FENCE.match(lines[index]):
                break
            if _LIST.match(lines[index]):
                break
            if (
                index + 1 < len(lines)
                and "|" in lines[index]
                and _TABLE_DELIMITER.match(lines[index + 1])
            ):
                break
            paragraphs.append(lines[index].strip())
            index += 1
        add_block(start + 1, index, " ".join(paragraphs), "paragraph")
    return source


def pack_source(
    source: StructuredSource,
    maximum: int,
    overlap: int,
    split_text: Callable[[str, int | None, int | None], list[str]],
) -> list[PackedPassage]:
    """Pack complete blocks by contiguous section; repeat table header rows on splits."""
    output: list[PackedPassage] = []
    pending: list[BlockSpan] = []
    pending_text = ""
    pending_section: str | None = None

    def flush() -> None:
        nonlocal pending, pending_text, pending_section
        if pending_text and pending_section is not None:
            output.append(PackedPassage(pending_text, pending_section, tuple(pending)))
        pending, pending_text, pending_section = [], "", None

    index = 0
    while index < len(source.units):
        unit = source.units[index]
        if pending_section != unit.section_key:
            flush()
        if unit.block_type in {"table_row", "table_caption"} and unit.table_id:
            flush()
            table_rows: list[int] = []
            while (
                index < len(source.units)
                and source.units[index].block_type in {"table_row", "table_caption"}
                and source.units[index].table_id == unit.table_id
                and source.units[index].section_key == unit.section_key
            ):
                table_rows.append(index)
                index += 1
            headers = sorted(
                {header for i in table_rows for header in source.units[i].table_header_indexes}
                | {
                    i
                    for i in table_rows
                    if source.units[i].is_table_header
                    or source.units[i].block_type == "table_caption"
                }
            )
            data_rows = [i for i in table_rows if i not in headers]
            header_text = "\n".join(source.units[i].text for i in headers)
            group: list[int] = []
            group_text = header_text
            for row_index in data_rows:
                row = source.units[row_index]
                candidate = "\n".join(value for value in (group_text, row.text) if value)
                if group and len(candidate) > maximum:
                    spans = [BlockSpan(i, 0, len(source.units[i].text)) for i in headers]
                    spans.extend(BlockSpan(i, 0, len(source.units[i].text)) for i in group)
                    output.append(
                        PackedPassage(
                            "\n".join(source.units[i].text for i in headers + group),
                            unit.section_key,
                            tuple(spans),
                            any(len(source.units[i].text) > maximum for i in group),
                        )
                    )
                    group, group_text = [], header_text
                    candidate = "\n".join(value for value in (group_text, row.text) if value)
                group.append(row_index)
                group_text = candidate
            if group:
                spans = [BlockSpan(i, 0, len(source.units[i].text)) for i in headers]
                spans.extend(BlockSpan(i, 0, len(source.units[i].text)) for i in group)
                oversized = any(len(source.units[i].text) > maximum for i in group)
                output.append(
                    PackedPassage(
                        "\n".join(source.units[i].text for i in headers + group),
                        unit.section_key,
                        tuple(spans),
                        oversized,
                    )
                )
            elif headers:
                output.append(
                    PackedPassage(
                        header_text,
                        unit.section_key,
                        tuple(BlockSpan(i, 0, len(source.units[i].text)) for i in headers),
                    )
                )
            continue
        text = unit.text
        if len(text) > maximum:
            flush()
            if unit.block_type in {"code", "list"}:
                output.append(
                    PackedPassage(
                        text,
                        unit.section_key,
                        (BlockSpan(index, 0, len(text)),),
                        True,
                    )
                )
                index += 1
                continue
            pieces = split_text(text, maximum, overlap if unit.block_type == "paragraph" else 0)
            cursor = 0
            for piece in pieces:
                start = text.find(piece, cursor)
                if start < 0:
                    start = text.find(piece)
                end = start + len(piece)
                output.append(
                    PackedPassage(piece, unit.section_key, (BlockSpan(index, start, end),), True)
                )
                cursor = max(start + 1, end - overlap if unit.block_type == "paragraph" else end)
            index += 1
            continue
        candidate = "\n\n".join(value for value in (pending_text, text) if value)
        if pending and len(candidate) > maximum:
            flush()
            candidate = text
        if not pending:
            pending_section = unit.section_key
        pending_text = candidate
        pending.append(BlockSpan(index, 0, len(text)))
        index += 1
    flush()
    return output
