"""Small source-order models and packers for structure-aware ingestion."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


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
