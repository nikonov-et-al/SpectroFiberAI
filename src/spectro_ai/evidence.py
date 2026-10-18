"""Allowlisted AFS source indexing and lexical evidence retrieval."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import posixpath
import re
import sqlite3
import threading
import time
import uuid
import zipfile
from collections import Counter
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import yaml

from spectro_ai.knowledge import KnowledgeSource
from spectro_ai.models import (
    CustomerUseCase,
    EvidenceContextBlock,
    RetrievedEvidence,
    SourceBlock,
    SourceBlockSpan,
    SourceLocation,
    SourceSection,
    TechniqueKnowledge,
)
from spectro_ai.structure import (
    BlockSpan,
    PackedPassage,
    SectionSpec,
    SourceUnit,
    StructuredSource,
    markdown_source,
    pack_source,
)

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "knowledge" / "sources.yaml"
_INDEX = _ROOT / ".cache" / "rag" / "index.sqlite3"
_DOCUMENTS = _ROOT / "documents"
_CHUNKING_VERSION = 3
_INDEX_SCHEMA_VERSION = 3
_SUPPORTED_CHUNKING_VERSIONS = {3}
_DOCLING_ENV_LOCK = threading.Lock()
_CHUNK_SIZE = 1200
_CHUNK_OVERLAP = 180
_BANNED_FILES = {
    "afs_examples_and_reference.docx",
    "customer_scenarios_for_final_evaluation.txt",
}
_BANNED_TAGS = {"evaluation", "fixture", "test_fixture", "scenario"}
_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_STOP = set(
    "a an and are as at be by for from have in into is it of on".split()
    + "or our the this to want we with".split()
)
_SYNONYMS = {
    "api": {"active pharmaceutical ingredient"},
    "nir": {"near infrared"},
    "near": {"nir"},
    "mir": {"mid infrared"},
    "mid": {"mir"},
    "atr": {"attenuated total reflection"},
    "attenuated": {"atr"},
    "fluorimetry": {"fluorescence"},
    "fluorescence": {"fluorimetry"},
    "water": {"moisture"},
    "moisture": {"water"},
    "whey": {"dairy"},
    "dairy": {"whey"},
    "petroleum": {"oil"},
    "oil": {"petroleum"},
    "blend": {"powder"},
    "powder": {"blend"},
    "fermentation": {"ferment"},
    "ferment": {"fermentation"},
    "biomass": {"cell density"},
    "heterogeneous": {"inhomogeneous"},
    "inhomogeneous": {"heterogeneous"},
    "probe": {"fiber optic"},
    "fiber": {"probe"},
    "calibration": {"chemometric model"},
    "chemometric": {"calibration"},
    "remote": {"distance", "cable"},
    "outside": {"remote", "cable"},
    "cable": {"fiber optic"},
    "location": {"area", "point"},
    "sequential": {"simultaneous"},
    "simultaneous": {"sequential"},
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _embedding_config(settings: Any, identity: str, dimension: int | None = None) -> dict[str, Any]:
    return {
        "provider": settings.provider,
        "endpoint": settings.ollama_url,
        "model": settings.embedding_model,
        "revision": identity,
        "input_encoding": "utf-8",
        "input_preprocessing": "exact-text-v1",
        "truncate": False,
        "dimension": dimension,
    }


def _pdf_backend() -> str:
    backend = os.getenv("SPECTRO_PDF_BACKEND", "pypdf").casefold()
    if backend not in {"pypdf", "docling"}:
        raise ValueError("SPECTRO_PDF_BACKEND must be 'pypdf' or 'docling'")
    return backend


def _parent_expansion_enabled(value: bool | None) -> bool:
    if value is not None:
        if not isinstance(value, bool):
            raise ValueError("parent_expansion must be a boolean or None")
        return value
    setting = os.getenv("SPECTRO_RAG_PARENT_EXPANSION", "false").strip().casefold()
    if setting in {"1", "true", "yes", "on"}:
        return True
    if setting in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("SPECTRO_RAG_PARENT_EXPANSION must be a boolean value")


@lru_cache(maxsize=32)
def _file_digest(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns  # Cache key invalidates the digest when the file changes.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _docling_models_fingerprint(models_path: Path) -> str:
    assets = sorted(
        path
        for path in models_path.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".safetensors", ".json", ".yaml"}
    )
    if not assets:
        return "missing-local-assets"
    return _sha256(
        json.dumps(
            [
                (
                    path.relative_to(models_path).as_posix(),
                    _file_digest(
                        str(path.resolve()),
                        path.stat().st_size,
                        path.stat().st_mtime_ns,
                    ),
                )
                for path in assets
            ],
            sort_keys=True,
        ).encode()
    )


def _extraction_fingerprint(chunk_size: int, chunk_overlap: int, pdf_backend: str) -> str:
    package = "docling" if pdf_backend == "docling" else "pypdf"
    try:
        parser_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        parser_version = "not-installed"
    settings = {
        "extractors": {
            "pdf": {"backend": pdf_backend, "version": parser_version},
            "markdown": "markdown-structure-v1",
            "txt": "txt-v1",
            "docx": "docx-v1",
            "xlsx": "xlsx-v1",
        },
        "docling": (
            {
                "device": "cpu",
                "table_structure": True,
                "ocr": False,
                "remote_services": False,
                "heading_hierarchy": {
                    "enabled": True,
                    "use_bookmarks": True,
                    "use_numbering": True,
                    "use_style": True,
                    "use_font_style": True,
                },
            }
            if pdf_backend == "docling"
            else None
        ),
        "pdf_backend": pdf_backend,
        "docling_models": (
            _docling_models_fingerprint(_ROOT / ".cache" / "docling" / "models")
            if pdf_backend == "docling"
            else None
        ),
        "chunking_version": _CHUNKING_VERSION,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    return _sha256(json.dumps(settings, sort_keys=True).encode())


def _load_manifest(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load source manifest {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported or invalid source manifest")
    root = (path.parent / str(document.get("root", ""))).resolve()
    sources = document.get("sources")
    if not root.is_dir() or not isinstance(sources, list):
        raise ValueError("Manifest root must exist and sources must be a list")
    excluded = {str(name).casefold() for name in document.get("excluded_files", [])}
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source manifest entry must be a mapping")
        filename = str(source.get("filename", ""))
        source_id = str(source.get("source_id", ""))
        tags = {str(source.get(key, "")).casefold() for key in ("scope", "source_type")}
        if Path(filename).name.casefold() in excluded | _BANNED_FILES or any(
            banned in tag for tag in tags for banned in _BANNED_TAGS
        ):
            continue
        if not re.fullmatch(r"S\d{2}", source_id):
            raise ValueError(f"Invalid AFS source ID: {source_id!r}")
        if not filename or Path(filename).is_absolute():
            raise ValueError(f"Source path must be relative: {filename!r}")
        identity = (source_id, filename.casefold())
        if identity in seen:
            raise ValueError(f"Duplicate source entry: {source_id} {filename}")
        seen.add(identity)
        accepted.append(source)
    return root, accepted


def _safe_source_path(root: Path, filename: str) -> Path:
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Source escapes manifest root: {filename}") from exc
    return path


def _split_text(text: str, maximum: int | None = None, overlap: int | None = None) -> list[str]:
    maximum = maximum or int(os.getenv("SPECTRO_RAG_CHUNK_SIZE", _CHUNK_SIZE))
    overlap = (
        overlap
        if overlap is not None
        else int(os.getenv("SPECTRO_RAG_CHUNK_OVERLAP", _CHUNK_OVERLAP))
    )
    if maximum < 200 or overlap < 0 or overlap >= maximum:
        raise ValueError("Chunk size must be >= 200 and overlap smaller than chunk size")
    clean = " ".join(text.split())
    if len(clean) <= maximum:
        return [clean] if clean else []
    chunks: list[str] = []
    pending = ""
    for sentence in re.split(r"(?<=[.!?])\s+|(?<=;)\s+", clean):
        if not sentence:
            continue
        if len(sentence) > maximum:
            if pending:
                chunks.append(pending)
                pending = ""
            start = 0
            while start < len(sentence):
                end = min(start + maximum, len(sentence))
                if end < len(sentence):
                    boundary = sentence.rfind(" ", start + maximum // 2, end)
                    if boundary > start:
                        end = boundary
                    else:
                        boundary = sentence.find(" ", end)
                        if boundary >= 0:
                            end = boundary
                        else:
                            end = len(sentence)
                chunks.append(sentence[start:end].strip())
                if end == len(sentence):
                    break
                start = max(start + 1, end - overlap)
                boundary = sentence.find(" ", start)
                if boundary >= 0:
                    start = boundary + 1
            continue
        combined = f"{pending} {sentence}".strip()
        if pending and len(combined) > maximum:
            chunks.append(pending)
            available = max(0, maximum - len(sentence) - 1)
            tail_words: list[str] = []
            tail_length = 0
            for word in reversed(pending.split()):
                added_length = len(word) + bool(tail_words)
                if tail_length + added_length > min(overlap, available):
                    break
                tail_words.insert(0, word)
                tail_length += added_length
            tail = " ".join(tail_words)
            pending = f"{tail} {sentence}".strip()
        else:
            pending = combined
    if pending:
        chunks.append(pending)
    return chunks


def _pdf_blocks(path: Path) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF indexing needs the project 'rag' extra (pypdf)") from exc
    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        raise ValueError(f"Encrypted PDF cannot be indexed: {path.name}")
    blocks: list[tuple[str, str]] = []
    empty_pages: list[str] = []
    poor_pages: list[str] = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if not text.strip():
            empty_pages.append(f"p.{number}")
            continue
        if len(text.strip()) < 50:
            poor_pages.append(f"p.{number}")
        parts = _split_text(text)
        for number_in_page, part in enumerate(parts, 1):
            locator = f"p.{number}" + (
                f" (chunk {number_in_page}/{len(parts)})" if len(parts) > 1 else ""
            )
            blocks.append((locator, part))
    if not blocks:
        raise ValueError(
            f"Scanned or empty PDF cannot be indexed: {path.name}; OCR is not supported"
        )
    return blocks, empty_pages, poor_pages


def _legacy_structured(blocks: list[tuple[str, str]]) -> StructuredSource:
    source = StructuredSource()
    for locator, text in blocks:
        section_match = re.search(r"section=([^;]+)", locator)
        page_match = re.match(r"p\.(\d+)", locator)
        lines_match = re.search(r"lines (\d+)-(\d+)", locator)
        key = (
            section_match.group(1)
            if section_match
            else (f"page:{page_match.group(1)}" if page_match else locator)
        )
        heading_path = (section_match.group(1).strip(),) if section_match else ()
        source.sections.setdefault(key, SectionSpec(key, heading_path))
        source.units.append(
            SourceUnit(
                text=text,
                locator=locator,
                section_key=key,
                heading_path=heading_path,
                block_type="paragraph",
                page_number=int(page_match.group(1)) if page_match else None,
                line_start=int(lines_match.group(1)) if lines_match else None,
                line_end=int(lines_match.group(2)) if lines_match else None,
            )
        )
    return source


def _docling_pdf_document(path: Path, models_path: Path) -> Any:
    """Convert PDF with explicit local CPU pipeline; never permit remote model fetches."""
    required = (
        models_path / "docling-project--docling-layout-heron" / "model.safetensors",
        models_path
        / "docling-project--docling-models"
        / "model_artifacts"
        / "tableformer"
        / "accurate"
        / "tableformer_accurate.safetensors",
    )
    missing = [str(asset) for asset in required if not asset.is_file()]
    if missing:
        raise RuntimeError(
            "Docling local models are missing; run `.tools/Scripts/uv.exe run --extra rag "
            f"python scripts/setup_docling.py`. Missing: {', '.join(missing)}"
        )
    try:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError("PDF backend 'docling' requires the project 'rag' extra") from exc

    options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device="cpu"),
        artifacts_path=models_path,
        enable_remote_services=False,
        allow_external_plugins=False,
        do_ocr=False,
        do_table_structure=True,
        do_picture_classification=False,
        do_picture_description=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        generate_parsed_pages=True,
        heading_hierarchy_options=HeadingHierarchyOptions(
            enabled=True,
            use_bookmarks=True,
            use_numbering=True,
            use_style=True,
            use_font_style=True,
        ),
    )
    local_environment = {
        "HF_HOME": str(models_path.parent),
        "HF_HUB_OFFLINE": "1",
        "DOCLING_ARTIFACTS_PATH": str(models_path),
    }
    with _DOCLING_ENV_LOCK:
        previous_environment = {key: os.environ.get(key) for key in local_environment}
        os.environ.update(local_environment)
        try:
            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
            )
            result = converter.convert(path)
        finally:
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return _checked_docling_document(result, path)


def _checked_docling_document(result: Any, path: Path) -> Any:
    status = getattr(result.status, "value", str(result.status)).casefold()
    if status != "success" or result.document is None:
        raise ValueError(f"Docling PDF conversion failed for {path.name}: {result.status}")
    if getattr(result, "errors", None):
        raise ValueError(f"Docling PDF conversion reported errors for {path.name}: {result.errors}")
    return result.document


def _docling_pdf_source(path: Path, models_path: Path) -> StructuredSource:
    document = _docling_pdf_document(path, models_path)
    entries: list[tuple[Any, int | None, tuple[int, ...]]] = []
    page_ids = {int(key) for key in getattr(document, "pages", {}) if str(key).isdigit()}
    caption_refs: set[str] = set()
    for item, _ in document.iterate_items(with_groups=False, traverse_pictures=False):
        for reference in getattr(item, "captions", []):
            caption_refs.add(reference.cref)
    for item, _ in document.iterate_items(with_groups=False, traverse_pictures=False):
        prov = getattr(item, "prov", []) or []
        item_pages = sorted(
            {int(value.page_no) for value in prov if getattr(value, "page_no", None)}
        )
        label = getattr(getattr(item, "label", None), "value", "")
        if len(item_pages) > 1 and label in {"table", "document_index"}:
            raise ValueError(
                f"Docling table {getattr(item, 'self_ref', '<unknown>')} spans pages "
                f"{item_pages}; exact row/page provenance is ambiguous"
            )
        if item_pages and page_ids and any(page not in page_ids for page in item_pages):
            raise ValueError(
                f"Docling item {getattr(item, 'self_ref', '<unknown>')} refers to "
                f"unknown PDF page(s) {item_pages}"
            )
        page = item_pages[0] if item_pages else None
        entries.append((item, page, tuple(item_pages)))

    headings: list[tuple[int, int | None, str]] = []
    previous_page = 0
    diagnostics: list[str] = []
    for item, page, _page_numbers in entries:
        if page is not None:
            if page < previous_page:
                diagnostics.append("Docling item order moves backwards across PDF pages")
            previous_page = page
        if getattr(getattr(item, "label", None), "value", "") == "section_header":
            level = getattr(item, "level", None)
            title = str(getattr(item, "text", "")).strip()
            if not isinstance(level, int):
                diagnostics.append(f"Invalid detected heading level {level!r}: {title}")
                headings.append((0, page, title))
                continue
            if not 1 <= level <= 6:
                diagnostics.append(f"Invalid detected heading level {level!r}: {title}")
            headings.append((level, page, title))
    if not headings:
        diagnostics.append("No PDF headings detected; using flat page-local sections")
    previous_level = 0
    number_stack: dict[int, tuple[int, ...]] = {}
    siblings: dict[tuple[int, tuple[int, ...]], int] = {}
    for level, _page, title in headings:
        if not 1 <= level <= 6:
            continue
        if previous_level and level > previous_level + 1:
            diagnostics.append(f"Detected heading level jump {previous_level} to {level}: {title}")
        previous_level = level
        number = re.match(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+", title)
        if number:
            parts = tuple(int(part) for part in number.group(1).split("."))
            if len(parts) != level:
                diagnostics.append(
                    f"Heading numbering depth conflicts with detected level: {title}"
                )
            parent_number = number_stack.get(level - 1)
            if level > 1 and parent_number and parts[:-1] != parent_number:
                diagnostics.append(f"Heading numbering conflicts with detected parent: {title}")
            sibling_key = (level, parts[:-1])
            if sibling_key in siblings and parts[-1] <= siblings[sibling_key]:
                diagnostics.append(f"Heading numbering is not increasing: {title}")
            siblings[sibling_key] = parts[-1]
            number_stack[level] = parts
            for deeper in [candidate for candidate in number_stack if candidate > level]:
                number_stack.pop(deeper)

    hierarchy_valid = not diagnostics and bool(headings)
    source = StructuredSource(diagnostics=diagnostics)
    pages = sorted({int(key) for key in getattr(document, "pages", {}) if str(key).isdigit()})
    source_pages: set[int] = set()
    heading_stack: list[tuple[int, str, str]] = []
    root_key = "pdf-root"
    source.sections[root_key] = SectionSpec(root_key, hierarchy_valid=hierarchy_valid)
    current_key = root_key
    current_path: tuple[str, ...] = ()
    item_number = 0
    for item, page, page_numbers in entries:
        item_number += 1
        label = getattr(getattr(item, "label", None), "value", "")
        if label in {"page_header", "page_footer"}:
            continue
        if not hierarchy_valid:
            page_key = f"pdf-page:{page}" if page is not None else f"pdf-order:{item_number}"
            source.sections.setdefault(
                page_key,
                SectionSpec(
                    page_key,
                    hierarchy_valid=False,
                    diagnostics=list(diagnostics),
                ),
            )
            current_key, current_path = page_key, ()
        elif label == "section_header":
            level = int(getattr(item, "level", 1))
            title = str(getattr(item, "text", "")).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            parent_key = heading_stack[-1][2] if heading_stack else root_key
            current_path = tuple(entry[1] for entry in heading_stack) + (title,)
            current_key = f"pdf-heading:{item_number}:{level}"
            source.sections[current_key] = SectionSpec(
                key=current_key,
                heading_path=current_path,
                parent_key=parent_key,
                detected_heading_level=level,
            )
            heading_stack.append((level, title, current_key))
        if label in {"table", "document_index"}:
            table = item
            table_id = f"docling:{getattr(item, 'self_ref', item_number)}"
            caption = str(table.caption_text(document) or "").strip()
            local_units: list[SourceUnit] = []
            if caption:
                local_units.append(
                    SourceUnit(
                        text=caption,
                        locator=f"p.{page}" if page else f"item {item_number}",
                        section_key=current_key,
                        heading_path=current_path,
                        block_type="table_caption",
                        page_number=page,
                        page_numbers=page_numbers,
                        table_id=table_id,
                    )
                )
            data = table.data
            row_count, col_count = int(data.num_rows), int(data.num_cols)
            grid = [[""] * col_count for _ in range(row_count)]
            header_rows: set[int] = set()
            for cell in data.table_cells:
                row_start = int(cell.start_row_offset_idx)
                row_end = min(int(cell.end_row_offset_idx), row_count)
                col_start = int(cell.start_col_offset_idx)
                if getattr(cell, "column_header", False):
                    header_rows.update(range(row_start, row_end))
                if 0 <= row_start < row_count and 0 <= col_start < col_count:
                    for row_index in range(row_start, row_end):
                        grid[row_index][col_start] = str(cell.text)
            row_units = [
                SourceUnit(
                    text=" | ".join(row),
                    locator=f"p.{page}" if page else f"item {item_number}",
                    section_key=current_key,
                    heading_path=current_path,
                    block_type="table_row",
                    page_number=page,
                    page_numbers=page_numbers,
                    table_id=table_id,
                    table_row_index=row_index,
                    is_table_header=row_index in header_rows,
                )
                for row_index, row in enumerate(grid)
                if any(row)
            ]
            first_row_index = len(source.units) + len(local_units)
            header_indexes = tuple(
                first_row_index + row_offset
                for row_offset, row in enumerate(row_units)
                if row.is_table_header
            )
            local_units.extend(
                SourceUnit(
                    **{
                        **row.__dict__,
                        "table_header_indexes": header_indexes,
                    }
                )
                for row in row_units
            )
            source.units.extend(local_units)
            if page is not None and local_units:
                source_pages.update(page_numbers or (page,))
            continue
        if label == "section_header":
            block_text = str(getattr(item, "text", "")).strip()
            block_type = "heading"
        else:
            if getattr(item, "self_ref", "") in caption_refs:
                continue
            block_text = str(getattr(item, "text", getattr(item, "orig", ""))).strip()
            block_type = (
                "list"
                if label == "list_item"
                else ("caption" if label == "caption" else "paragraph")
            )
            if label == "list_item":
                marker = str(getattr(item, "marker", "-")).strip()
                block_text = f"{marker} {block_text}".strip()
        if not block_text:
            continue
        if page is not None:
            source_pages.update(page_numbers or (page,))
        locator = f"p.{page}" if page is not None else f"item {item_number}"
        source.units.append(
            SourceUnit(
                text=block_text,
                locator=locator,
                section_key=current_key,
                heading_path=current_path,
                block_type=block_type,
                page_number=page,
                page_numbers=page_numbers,
                detected_heading_level=(
                    item.level
                    if label == "section_header" and isinstance(getattr(item, "level", None), int)
                    else None
                ),
            )
        )
    if pages:
        source.empty_pages = [f"p.{page}" for page in pages if page not in source_pages]
    if not source.units:
        raise ValueError(f"Scanned or empty PDF cannot be indexed: {path.name}; OCR is disabled")
    for page in sorted(source_pages):
        page_length = sum(
            len(unit.text)
            for unit in source.units
            if page in (unit.page_numbers or ((unit.page_number,) if unit.page_number else ()))
        )
        if page_length < 50:
            source.poor_pages.append(f"p.{page}")
    return source


def _txt_blocks(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[str, str]] = []
    size, overlap = 80, 10
    for start in range(0, len(lines), size - overlap):
        stop = min(start + size, len(lines))
        text = "\n".join(
            f"{number}: {line}" for number, line in enumerate(lines[start:stop], start + 1)
        )
        for part in _split_text(text):
            blocks.append((f"lines {start + 1}-{stop}", part))
        if stop == len(lines):
            break
    return blocks


def _markdown_blocks(path: Path) -> list[tuple[str, str]]:
    heading = "document"
    paragraphs: list[str] = []
    blocks: list[tuple[str, str]] = []
    first_line = 1
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    def flush(last_line: int) -> None:
        text = "\n".join(paragraphs).strip()
        if text:
            for part in _split_text(f"{heading}\n{text}"):
                blocks.append((f"section={heading}; lines {first_line}-{last_line}", part))
        paragraphs.clear()

    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            flush(number - 1)
            heading = line.lstrip("# ").strip() or "document"
            first_line = number
        elif line.strip():
            if not paragraphs:
                first_line = number
            paragraphs.append(line.strip())
        elif paragraphs:
            flush(number - 1)
    flush(len(lines))
    return blocks


def _xml_text(node: ElementTree.Element) -> str:
    return "".join(node.itertext()).strip()


def _docx_blocks(path: Path) -> list[tuple[str, str]]:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    body = root.find("w:body", namespace)
    if body is None:
        return []
    blocks: list[tuple[str, str]] = []
    section = "document"
    for element in body:
        if element.tag.endswith("}p"):
            text = _xml_text(element)
            style = element.find("w:pPr/w:pStyle", namespace)
            if style is not None and str(style.get(f"{{{namespace['w']}}}val", "")).startswith(
                "Heading"
            ):
                if text:
                    section = text
                continue
        elif element.tag.endswith("}tbl"):
            text = "\n".join(
                " | ".join(_xml_text(cell) for cell in row.findall("w:tc", namespace))
                for row in element.findall("w:tr", namespace)
            )
        else:
            continue
        if text.strip():
            for part in _split_text(text):
                blocks.append((f"section={section}", part))
    return blocks


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if letters is None:
        return 0
    value = 0
    for letter in letters.group():
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _xlsx_blocks(path: Path) -> list[tuple[str, str]]:
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [_xml_text(item) for item in shared_root.findall("m:si", ns)]
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relations = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation_paths = {}
        for relation in relations.findall("rel:Relationship", ns):
            target = str(relation.get("Target", "")).lstrip("/")
            relation_paths[relation.get("Id")] = posixpath.normpath(
                target if target.startswith("xl/") else posixpath.join("xl", target)
            )
        blocks: list[tuple[str, str]] = []
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            title = str(sheet.get("name", "sheet"))
            rel_id = sheet.get(f"{{{ns['r']}}}id")
            worksheet_path = relation_paths.get(rel_id)
            if worksheet_path not in names:
                raise ValueError(f"Missing worksheet XML for {title}")
            worksheet = ElementTree.fromstring(archive.read(worksheet_path))
            rows: list[tuple[int, str]] = []
            for row in worksheet.findall("m:sheetData/m:row", ns):
                row_number = int(row.get("r", "0"))
                cells: list[str] = []
                for cell in row.findall("m:c", ns):
                    reference = str(cell.get("r", ""))
                    index = _column_index(reference)
                    while len(cells) <= index:
                        cells.append("")
                    kind = cell.get("t")
                    if kind == "inlineStr":
                        value = (
                            _xml_text(cell.find("m:is", ns))
                            if cell.find("m:is", ns) is not None
                            else ""
                        )
                    else:
                        raw = cell.findtext("m:v", default="", namespaces=ns)
                        value = shared[int(raw)] if kind == "s" and raw else raw
                    cells[index] = value
                if any(cells):
                    rows.append((row_number, " | ".join(cells)))
            size, overlap = 24, 2
            for offset in range(0, len(rows), size - overlap):
                group = rows[offset : offset + size]
                text = "\n".join(f"row {number}: {value}" for number, value in group)
                for part in _split_text(text):
                    blocks.append((f"worksheet={title} rows={group[0][0]}-{group[-1][0]}", part))
                if offset + size >= len(rows):
                    break
    return blocks


def _extract(path: Path) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return _pdf_blocks(path)
    if suffix == ".txt":
        return _txt_blocks(path), [], []
    if suffix == ".docx":
        return _docx_blocks(path), [], []
    if suffix == ".xlsx":
        return _xlsx_blocks(path), [], []
    raise ValueError(f"Unsupported source type: {path.suffix}")


def reindex_sources(
    manifest_path: Path | None = None, index_path: Path | None = None
) -> dict[str, Any]:
    """Index manifest sources in the same SQLite format used by assessment retrieval."""
    return _update_index(
        manifest_path=manifest_path or _MANIFEST,
        index_path=index_path or _INDEX,
        documents_path=None,
        semantic=False,
    )


def _load_old_index(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}, {}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            metadata = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
            rows = db.execute("SELECT id, chunk, vector FROM chunks").fetchall()
        return metadata, {
            row[0]: {"chunk": json.loads(row[1]), "vector": json.loads(row[2])} for row in rows
        }
    except sqlite3.Error, ValueError, TypeError, IndexError, json.JSONDecodeError:
        raise ValueError("Knowledge index is corrupt or incompatible; rebuild it") from None


def canonical_context_block(index: dict[str, Any], block_id: str) -> EvidenceContextBlock:
    """Resolve one context block only from its pinned, validated index snapshot."""
    raw = next(
        (block for block in index.get("source_blocks", []) if block.get("block_id") == block_id),
        None,
    )
    if raw is None:
        raise ValueError("Context block is not present in the pinned knowledge index")
    try:
        block = SourceBlock.model_validate(raw)
    except Exception as exc:
        raise ValueError("Pinned context block is invalid") from exc
    section = next(
        (item for item in index.get("sections", []) if item.get("section_id") == block.section_id),
        None,
    )
    file_record = next(
        (
            item
            for item in index.get("files", [])
            if item.get("source_id") == block.source_id
            and item.get("filename") == block.filename
            and item.get("document_version") == block.document_version
        ),
        None,
    )
    if (
        section is None
        or block.block_id not in section.get("block_ids", [])
        or section.get("document_id") != block.document_id
        or section.get("source_id") != block.source_id
        or section.get("filename") != block.filename
        or section.get("document_version") != block.document_version
        or file_record is None
        or file_record.get("document_id") != block.document_id
    ):
        raise ValueError("Pinned context block has inconsistent source relationships")
    if not section.get("hierarchy_valid", True):
        raise ValueError("Pinned context block belongs to an unvalidated section hierarchy")
    locations = block.locations or [block.location]
    if not locations or any(
        location.source_id != block.source_id
        or location.filename != block.filename
        or location.document_version != block.document_version
        for location in locations
    ):
        raise ValueError("Pinned context block has invalid source locations")
    source_scope = raw.get("source_scope") or "; ".join(
        value for value in (file_record.get("source_type"), file_record.get("scope")) if value
    )
    review_status = raw.get("review_status") or file_record.get(
        "review_status", "needs_source_review"
    )
    return EvidenceContextBlock(
        block_id=block.block_id,
        document_id=block.document_id,
        section_id=block.section_id,
        source_id=block.source_id,
        filename=block.filename,
        document_version=block.document_version,
        text=block.text,
        source_locations=locations,
        review_status=review_status,
        source_scope=source_scope,
    )


def evidence_context_block_bytes(block: EvidenceContextBlock) -> int:
    """Return the UTF-8 cost used by retrieval/provider context budgeting."""
    return evidence_context_blocks_bytes([block])


def evidence_context_blocks_bytes(blocks: list[EvidenceContextBlock]) -> int:
    """Charge a child's complete serialized context field, including JSON overhead."""
    if not blocks:
        return 0
    serialized = json.dumps(
        {"context_blocks": [block.model_dump(mode="json") for block in blocks]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return len(serialized.encode("utf-8"))


def _review_status_for_locations(
    locations: list[SourceLocation],
    file_record: dict[str, Any],
    overrides: dict[str, str],
) -> str:
    statuses: list[str] = []
    poor_pages = set(file_record.get("poor_extraction_pages", []))
    needs_manual_review = False
    for location in locations:
        page_number = location.page_number
        if page_number is None and (match := re.match(r"p\.(\d+)", location.locator)):
            page_number = int(match.group(1))
        if page_number is None:
            statuses.append(str(file_record.get("review_status", "needs_source_review")))
            continue
        page_key = str(page_number)
        statuses.append(
            str(overrides.get(page_key, file_record.get("review_status", "needs_source_review")))
        )
        needs_manual_review |= f"p.{page_number}" in poor_pages
    status = "; ".join(dict.fromkeys(statuses)) or "needs_source_review"
    if needs_manual_review:
        status = f"{status}; poor_extraction_needs_manual_review"
    return status


def _update_index(
    *,
    manifest_path: Path,
    index_path: Path,
    documents_path: Path | None,
    semantic: bool,
    settings: Any = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    from spectro_ai import providers

    root, sources = _load_manifest(manifest_path)
    if documents_path and documents_path.is_dir():
        for path in sorted(documents_path.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in {".pdf", ".md", ".txt"}:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(documents_path.resolve()):
                raise ValueError("Document links outside the documents directory are not supported")
            relative = path.relative_to(documents_path).as_posix()
            if path.name.casefold() in _BANNED_FILES or any(
                tag in relative.casefold() for tag in _BANNED_TAGS
            ):
                continue
            source_id = "D" + _sha256(relative.encode())[:7].upper()
            sources.append(
                {
                    "source_id": source_id,
                    "filename": relative,
                    "_root": documents_path,
                    "title": path.stem,
                    "source_type": "customer_document",
                    "scope": "unreviewed_document",
                    "review_status": "needs_source_review",
                }
            )
    try:
        old_meta, old_chunks = _load_old_index(index_path)
    except ValueError:
        old_meta, old_chunks = {}, {}
    if force_rebuild:
        old_meta, old_chunks = {}, {}
    if old_meta.get("schema_version") not in {2, _INDEX_SCHEMA_VERSION}:
        old_chunks = {}
    settings = settings or providers.RagSettings.from_env()
    chunk_size = int(os.getenv("SPECTRO_RAG_CHUNK_SIZE", _CHUNK_SIZE))
    chunk_overlap = int(os.getenv("SPECTRO_RAG_CHUNK_OVERLAP", _CHUNK_OVERLAP))
    _split_text("config validation", chunk_size, chunk_overlap)
    pdf_backend = _pdf_backend()
    identity = providers.embedding_identity(settings) if semantic else "bm25-only"
    current_embedding_config = _embedding_config(settings, identity)
    reusable_vectors: dict[str, list[float]] = {}
    if (
        semantic
        and old_meta.get("embedding_model") == settings.embedding_model
        and old_meta.get("embedding_identity") == identity
        and old_meta.get("schema_version") == _INDEX_SCHEMA_VERSION
        and {
            key: value
            for key, value in old_meta.get("embedding_configuration", {}).items()
            if key != "dimension"
        }
        == {key: value for key, value in current_embedding_config.items() if key != "dimension"}
    ):
        previous_dimension = old_meta.get("embedding_dimension")
        previous_config = old_meta.get("embedding_configuration")
        if (
            isinstance(previous_dimension, int)
            and previous_dimension > 0
            and isinstance(previous_config, dict)
            and previous_config.get("dimension") == previous_dimension
        ):
            for old in old_chunks.values():
                vector = old.get("vector")
                chunk = old.get("chunk", {})
                if (
                    isinstance(vector, list)
                    and len(vector) == previous_dimension
                    and vector
                    and (
                        chunk.get("embedding_input_sha256")
                        == _sha256(str(chunk.get("text", "")).encode("utf-8"))
                    )
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        for value in vector
                    )
                    and math.hypot(*vector)
                ):
                    reusable_vectors.setdefault(
                        _sha256(str(chunk.get("text", "")).encode("utf-8")), vector
                    )
    files: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    chunks: dict[str, dict[str, Any]] = {}
    source_sections: dict[str, dict[str, Any]] = {}
    source_blocks: dict[str, dict[str, Any]] = {}
    for source in sources:
        filename = str(source["filename"])
        source_root = Path(source.get("_root", root))
        path = _safe_source_path(source_root, filename)
        source_id = str(source["source_id"])
        if not path.is_file():
            missing.append({"source_id": source_id, "filename": filename, "status": "missing"})
            continue
        try:
            content = path.read_bytes()
            version = _sha256(content)
            if path.suffix.casefold() == ".md":
                structured = markdown_source(path.read_text(encoding="utf-8-sig"))
                passages = pack_source(structured, chunk_size, chunk_overlap, _split_text)
                empty_pages, poor_pages = structured.empty_pages, structured.poor_pages
            elif path.suffix.casefold() == ".pdf" and pdf_backend == "docling":
                structured = _docling_pdf_source(path, _ROOT / ".cache" / "docling" / "models")
                passages = pack_source(structured, chunk_size, chunk_overlap, _split_text)
                empty_pages, poor_pages = structured.empty_pages, structured.poor_pages
            else:
                blocks, empty_pages, poor_pages = _extract(path)
                structured = _legacy_structured(blocks)
                passages = [
                    PackedPassage(
                        text=unit.text,
                        section_key=unit.section_key,
                        spans=(BlockSpan(index, 0, len(unit.text)),),
                    )
                    for index, unit in enumerate(structured.units)
                ]
        except Exception as exc:
            raise ValueError(f"Cannot extract source {filename}: {exc}") from exc
        page_numbers = [
            page
            for unit in structured.units
            for page in (unit.page_numbers or ((unit.page_number,) if unit.page_number else ()))
        ] + [int(value.split(".", 1)[1]) for value in empty_pages]
        file_record = {
            "source_id": source_id,
            "filename": filename,
            "title": str(source.get("title", filename)),
            "source_type": str(source.get("source_type", "technical_reference")),
            "scope": str(source.get("scope", "technical")),
            "review_status": str(source.get("review_status", "needs_source_review")),
            "related_source": source.get("related_source"),
            "citation_format": source.get("citation_format"),
            "document_version": version,
            "document_id": _sha256(f"{source_id}:{filename}:{version}".encode()),
            "page_count": max(page_numbers, default=0),
            "empty_pages": empty_pages,
            "poor_extraction_pages": poor_pages,
            "chunk_count": len(passages),
            "pdf_backend": pdf_backend if path.suffix.casefold() == ".pdf" else None,
            "heading_hierarchy_valid": (
                all(section.hierarchy_valid for section in structured.sections.values())
                if path.suffix.casefold() == ".pdf" and pdf_backend == "docling"
                else None
            ),
            "extraction_diagnostics": list(structured.diagnostics),
        }
        files.append(file_record)
        review_overrides = source.get("page_review", {})
        document_id = file_record["document_id"]
        unit_block_ids: dict[int, str] = {}
        unit_locations: dict[int, list[SourceLocation]] = {}
        section_ids = {
            key: _sha256(f"{document_id}:{key}:{json.dumps(spec.heading_path)}".encode())
            for key, spec in structured.sections.items()
        }
        for section_key, section_spec in structured.sections.items():
            section_id = section_ids[section_key]
            source_sections[section_id] = SourceSection(
                section_id=section_id,
                document_id=document_id,
                source_id=source_id,
                filename=filename,
                document_version=version,
                heading_path=list(section_spec.heading_path),
                parent_section_id=(
                    section_ids.get(section_spec.parent_key) if section_spec.parent_key else None
                ),
                detected_heading_level=section_spec.detected_heading_level,
                hierarchy_valid=section_spec.hierarchy_valid,
                diagnostics=list(section_spec.diagnostics),
            ).model_dump()
        for unit_index, unit in enumerate(structured.units):
            section_spec = structured.sections[unit.section_key]
            section_id = section_ids[unit.section_key]
            block_id = _sha256(
                f"{document_id}:{unit_index}:{unit.locator}:{_sha256(unit.text.encode())}".encode()
            )
            unit_block_ids[unit_index] = block_id
            locations_for_unit = (
                [
                    SourceLocation(
                        source_id=source_id,
                        filename=filename,
                        document_version=version,
                        locator=f"p.{page}",
                        page_number=page,
                    )
                    for page in unit.page_numbers
                ]
                if unit.page_numbers
                else [
                    SourceLocation(
                        source_id=source_id,
                        filename=filename,
                        document_version=version,
                        locator=unit.locator,
                        page_number=unit.page_number,
                        line_start=unit.line_start,
                        line_end=unit.line_end,
                    )
                ]
            )
            unit_locations[unit_index] = locations_for_unit
            location = locations_for_unit[0]
            source_sections[section_id]["block_ids"].append(block_id)
            source_blocks[block_id] = SourceBlock(
                block_id=block_id,
                document_id=document_id,
                section_id=section_id,
                source_id=source_id,
                filename=filename,
                document_version=version,
                ordinal=unit_index,
                text=unit.text,
                location=location,
                locations=locations_for_unit,
                block_type=unit.block_type,
                detected_heading_level=unit.detected_heading_level,
                table_id=unit.table_id,
                table_row_index=unit.table_row_index,
                is_table_header=unit.is_table_header,
                oversized=len(unit.text) > chunk_size,
                review_status=_review_status_for_locations(
                    locations_for_unit, file_record, review_overrides
                ),
                source_scope="; ".join((file_record["source_type"], file_record["scope"])),
            ).model_dump()
        for unit_index, unit in enumerate(structured.units):
            if unit.table_header_indexes:
                source_blocks[unit_block_ids[unit_index]]["table_header_block_ids"] = [
                    unit_block_ids[header_index] for header_index in unit.table_header_indexes
                ]
        for number, passage in enumerate(passages, 1):
            excerpt = passage.text
            content_hash = _sha256(excerpt.encode())
            chunk_id = _sha256(
                f"{content_hash}:{_CHUNKING_VERSION}:{chunk_size}:{chunk_overlap}".encode()
            )
            item = chunks.setdefault(
                content_hash,
                {
                    "content_sha256": content_hash,
                    "text": excerpt,
                    "embedding_input_sha256": _sha256(excerpt.encode("utf-8")),
                    "provenance": [],
                    "chunk_id": chunk_id,
                },
            )
            evidence_id = f"{source_id}-{path.stem}-{version[:12]}-{number:04d}"
            span_by_unit: dict[int, tuple[int, int]] = {}
            for span in passage.spans:
                previous = span_by_unit.get(span.unit_index)
                if previous:
                    span_by_unit[span.unit_index] = (
                        min(previous[0], span.start),
                        max(previous[1], span.end),
                    )
                else:
                    span_by_unit[span.unit_index] = (span.start, span.end)
            contributing_indices = list(span_by_unit)
            block_ids = [unit_block_ids[index] for index in contributing_indices]
            locations_by_identity = {
                (
                    location.source_id,
                    location.filename,
                    location.document_version,
                    location.locator,
                ): location
                for index in contributing_indices
                for location in unit_locations[index]
            }
            locations = list(locations_by_identity.values())
            section_id = section_ids[passage.section_key]
            locator = "; ".join(dict.fromkeys(location.locator for location in locations))
            page_matches = [re.match(r"p\.(\d+)", location.locator) for location in locations]
            review = (
                "; ".join(
                    dict.fromkeys(
                        review_overrides.get(page.group(1), file_record["review_status"])
                        for page in page_matches
                        if page
                    )
                )
                or file_record["review_status"]
            )
            if any(page and f"p.{page.group(1)}" in poor_pages for page in page_matches):
                review = f"{review}; poor_extraction_needs_manual_review"
            provenance = {
                "evidence_id": evidence_id,
                "source_id": source_id,
                "document_id": document_id,
                "document_version": version,
                "page_or_section": locator,
                "review_status": str(review),
                "source_scope": "; ".join((file_record["source_type"], file_record["scope"])),
                "title": file_record["title"],
                "filename": filename,
                "related_source": source.get("related_source"),
                "citation_format": source.get("citation_format"),
                "block_id": block_ids[0],
                "block_ids": block_ids,
                "block_spans": [
                    SourceBlockSpan(
                        block_id=unit_block_ids[index], start=start, end=end
                    ).model_dump()
                    for index, (start, end) in span_by_unit.items()
                ],
                "source_locations": [location.model_dump() for location in locations],
                "section_id": section_id,
                "heading_path": list(structured.sections[passage.section_key].heading_path),
                "oversized": passage.oversized,
            }
            if provenance not in item["provenance"]:
                item["provenance"].append(provenance)
    pending = [
        chunk
        for chunk in chunks.values()
        if not semantic or chunk["embedding_input_sha256"] not in reusable_vectors
    ]
    vectors: dict[str, list[float]] = {}
    if semantic:
        for chunk in chunks.values():
            vector = reusable_vectors.get(chunk["embedding_input_sha256"])
            if vector is not None:
                vectors[chunk["chunk_id"]] = vector
    if semantic:
        for start in range(0, len(pending), 16):
            batch = pending[start : start + 16]
            embedded = providers.embed_texts(settings, [c["text"] for c in batch])
            if len(embedded) != len(batch) or any(
                not v
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(float(x))
                    for x in v
                )
                or not math.hypot(*v)
                for v in embedded
            ):
                raise ValueError("Ollama returned invalid chunk embeddings")
            expected_dimension = len(next(iter(vectors.values()))) if vectors else len(embedded[0])
            if any(len(vector) != expected_dimension for vector in embedded):
                raise ValueError("Ollama returned inconsistent embedding dimensions")
            vectors.update(
                (chunk["chunk_id"], vector) for chunk, vector in zip(batch, embedded, strict=True)
            )
    else:
        vectors = {chunk["chunk_id"]: [] for chunk in chunks.values()}
    manifest_version = _sha256(manifest_path.read_bytes())
    metadata = {
        "schema_version": _INDEX_SCHEMA_VERSION,
        "manifest_version": manifest_version,
        "files": files,
        "missing_sources": missing,
        "embedding_model": settings.embedding_model if semantic else None,
        "embedding_identity": identity,
        "embedding_dimension": len(next(iter(vectors.values()))) if vectors and semantic else 0,
        "embedding_configuration": (
            _embedding_config(
                settings,
                identity,
                len(next(iter(vectors.values()))) if vectors else 0,
            )
            if semantic
            else None
        ),
        "extraction_fingerprint": _extraction_fingerprint(chunk_size, chunk_overlap, pdf_backend),
        "chunking_version": _CHUNKING_VERSION,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "revision": uuid.uuid4().hex,
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunks": len(chunks),
        "documents": len(files),
        "sections": list(source_sections.values()),
        "source_blocks": list(source_blocks.values()),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    staging = index_path.with_name(index_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with closing(sqlite3.connect(staging)) as db, db:
            db.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
            db.execute(
                "CREATE TABLE chunks (id TEXT PRIMARY KEY, chunk TEXT NOT NULL, "
                "vector TEXT NOT NULL)"
            )
            db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?)",
                [
                    (c["chunk_id"], json.dumps(c), json.dumps(vectors[c["chunk_id"]]))
                    for c in chunks.values()
                ],
            )
            db.commit()
        _validate_database(staging)
        os.replace(staging, index_path)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return {
        "index_path": str(index_path),
        "files_indexed": len(files),
        "chunks": len(chunks),
        "new_embeddings": len(pending) if semantic else 0,
        "reused_embeddings": len(chunks) - len(pending) if semantic else 0,
        "missing_sources": missing,
        "empty_pdf_pages": sum(len(f["empty_pages"]) for f in files),
        "poor_extraction_pages": sum(len(f["poor_extraction_pages"]) for f in files),
        **metadata,
    }


def _validate_database(
    path: Path, loaded: tuple[dict[str, Any], dict[str, dict[str, Any]]] | None = None
) -> None:
    meta, chunks = loaded or _load_old_index(path)
    if meta.get("schema_version") not in {2, _INDEX_SCHEMA_VERSION} or meta.get("chunks") != len(
        chunks
    ):
        raise ValueError("Staged knowledge index failed validation")
    dimension = meta.get("embedding_dimension")
    for chunk in chunks.values():
        vector = chunk["vector"]
        if meta.get("embedding_model") is None:
            if vector:
                raise ValueError("Staged BM25 index contains unexpected vectors")
            continue
        if (
            not isinstance(dimension, int)
            or len(vector) != dimension
            or not vector
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in vector
            )
            or math.hypot(*vector) == 0
        ):
            raise ValueError("Staged knowledge index contains invalid embeddings")
    if meta.get("schema_version") == _INDEX_SCHEMA_VERSION:
        configuration = meta.get("embedding_configuration")
        if meta.get("embedding_model") is None:
            if configuration is not None:
                raise ValueError("Staged knowledge index has invalid embedding configuration")
        elif (
            not isinstance(configuration, dict)
            or configuration.get("dimension") != dimension
            or configuration.get("revision") != meta.get("embedding_identity")
            or configuration.get("model") != meta.get("embedding_model")
        ):
            raise ValueError("Staged knowledge index has invalid embedding configuration")
        sections = meta.get("sections")
        blocks = meta.get("source_blocks")
        if not isinstance(sections, list) or not isinstance(blocks, list):
            raise ValueError("Staged knowledge index has invalid structure records")
        try:
            parsed_sections = [SourceSection.model_validate(item) for item in sections]
            parsed_blocks = [SourceBlock.model_validate(item) for item in blocks]
        except Exception as exc:
            raise ValueError("Staged knowledge index has invalid structure records") from exc
        section_map = {item.section_id: item for item in parsed_sections}
        block_map = {item.block_id: item for item in parsed_blocks}
        if len(section_map) != len(parsed_sections) or len(block_map) != len(parsed_blocks):
            raise ValueError("Staged knowledge index has duplicate structure IDs")
        block_memberships = Counter(
            block_id for section in parsed_sections for block_id in section.block_ids
        )
        files_by_identity = {
            (item.get("source_id"), item.get("filename"), item.get("document_version")): item
            for item in meta.get("files", [])
            if isinstance(item, dict)
        }
        for section in parsed_sections:
            file_record = files_by_identity.get(
                (section.source_id, section.filename, section.document_version)
            )
            section_ordinals = [
                block_map[block_id].ordinal
                for block_id in section.block_ids
                if block_id in block_map
            ]
            parent = section_map.get(section.parent_section_id)
            if (
                file_record is None
                or file_record.get("document_id") != section.document_id
                or section.document_id
                != _sha256(
                    f"{section.source_id}:{section.filename}:{section.document_version}".encode()
                )
                or len(section.block_ids) != len(set(section.block_ids))
                or any(block_id not in block_map for block_id in section.block_ids)
                or section_ordinals != sorted(set(section_ordinals))
                or (
                    section.parent_section_id is not None
                    and (
                        parent is None
                        or parent.document_id != section.document_id
                        or parent.source_id != section.source_id
                        or parent.filename != section.filename
                        or parent.document_version != section.document_version
                        or section.heading_path[: len(parent.heading_path)] != parent.heading_path
                    )
                )
            ):
                raise ValueError("Staged knowledge index has dangling section blocks")
            parent_ids = {section.section_id}
            ancestor = parent
            while ancestor is not None:
                if ancestor.section_id in parent_ids:
                    raise ValueError("Staged knowledge index has cyclic section ancestry")
                parent_ids.add(ancestor.section_id)
                ancestor = section_map.get(ancestor.parent_section_id)
        for block in parsed_blocks:
            section = section_map.get(block.section_id)
            provenance_locations = [
                location
                for chunk in chunks.values()
                for provenance in chunk["chunk"].get("provenance", [])
                if block.block_id in provenance.get("block_ids", [provenance.get("block_id")])
                for location in provenance.get("source_locations", [])
                if location.get("source_id") == block.source_id
                and location.get("filename") == block.filename
                and location.get("document_version") == block.document_version
                and location.get("locator") == block.location.locator
            ]
            legacy_provenance = any(
                provenance.get("block_id") == block.block_id
                and provenance.get("page_or_section") == block.location.locator
                for chunk in chunks.values()
                for provenance in chunk["chunk"].get("provenance", [])
            )
            if (
                block_memberships.get(block.block_id) != 1
                or section is None
                or section.document_id != block.document_id
                or block.block_id not in section.block_ids
                or section.source_id != block.source_id
                or section.filename != block.filename
                or section.document_version != block.document_version
                or block.location.source_id != block.source_id
                or block.location.filename != block.filename
                or block.location.document_version != block.document_version
                or not provenance_locations
                and not legacy_provenance
                or len(block.table_header_block_ids) != len(set(block.table_header_block_ids))
                or any(header not in block_map for header in block.table_header_block_ids)
                or any(
                    block_map[header].section_id != block.section_id
                    or block_map[header].table_id != block.table_id
                    or not block_map[header].is_table_header
                    for header in block.table_header_block_ids
                )
            ):
                raise ValueError(
                    "Evidence citation points outside source pages or disagrees with its block"
                )
        for chunk in chunks.values():
            text = chunk["chunk"].get("text")
            for provenance in chunk["chunk"].get("provenance", []):
                block_ids = provenance.get("block_ids") or [provenance.get("block_id")]
                spans = provenance.get("block_spans")
                if (
                    not isinstance(block_ids, list)
                    or not block_ids
                    or any(block_id not in block_map for block_id in block_ids)
                ):
                    raise ValueError("Staged knowledge index has inconsistent chunk provenance")
                referenced = [block_map[block_id] for block_id in block_ids]
                if any(
                    provenance.get("document_version") != block.document_version
                    or provenance.get("section_id") != block.section_id
                    or provenance.get("source_id") != block.source_id
                    or provenance.get("document_id", block.document_id) != block.document_id
                    or provenance.get("filename") != block.filename
                    for block in referenced
                ):
                    raise ValueError("Evidence citation points to an unknown source version")
                if spans is None:
                    # Schema-v3 indexes created before structure-aware packing store one
                    # complete source block per passage and remain readable.
                    if len(referenced) != 1 or referenced[0].text != text:
                        raise ValueError("Staged knowledge index has inconsistent chunk provenance")
                    continue
                try:
                    parsed_spans = [SourceBlockSpan.model_validate(span) for span in spans]
                except Exception as exc:
                    raise ValueError("Staged knowledge index has invalid block spans") from exc
                if [span.block_id for span in parsed_spans] != block_ids:
                    raise ValueError("Staged knowledge index block spans do not match block IDs")
                expected_location_map = {
                    (
                        location.source_id,
                        location.filename,
                        location.document_version,
                        location.locator,
                    ): location.model_dump()
                    for block in referenced
                    for location in (block.locations or [block.location])
                }
                expected_locations = list(expected_location_map.values())
                if provenance.get("source_locations") != expected_locations or provenance.get(
                    "page_or_section"
                ) != "; ".join(
                    dict.fromkeys(location["locator"] for location in expected_locations)
                ):
                    raise ValueError(
                        "Staged knowledge index has inconsistent source locations; "
                        "citation may point outside source pages"
                    )
                if any(
                    span.end > len(block_map[span.block_id].text)
                    or block_map[span.block_id].text[span.start : span.end] not in text
                    for span in parsed_spans
                ):
                    raise ValueError("Staged knowledge index has inconsistent source spans")


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text) if token.casefold() not in _STOP]


def _query_text(use_case: CustomerUseCase) -> str:
    values: list[str] = []
    for name in CustomerUseCase.FACT_FIELDS:
        value = getattr(use_case, name)
        if value:
            values.append(f"{name.replace('_', ' ')}: {value}")
    for name in ("source_text", "customer_preferences", "strict_requirements"):
        value = getattr(use_case, name)
        if isinstance(value, str) and value:
            values.append(f"{name.replace('_', ' ')}: {value}")
        elif isinstance(value, list):
            values.extend(f"{name.replace('_', ' ')}: {item}" for item in value)
    return " ".join(values)


def _query_groups(use_case: CustomerUseCase, claim: str | None = None) -> list[str]:
    buckets = (
        (
            "application and sample",
            ("industry", "process", "sample_matrix", "composition", "physical_state"),
        ),
        (
            "measurement objective",
            ("measurement_objective", "target_analyte_or_property", "concentration_range"),
        ),
        (
            "measurement setup",
            ("measurement_location", "deployment_mode", "measurement_frequency", "desired_stage"),
        ),
    )
    grouped = {name for _, fields in buckets for name in fields}
    buckets += (
        (
            "technical limitations",
            tuple(name for name in CustomerUseCase.FACT_FIELDS if name not in grouped),
        ),
    )
    queries = [use_case.source_text] if use_case.source_text else []
    for label, fields in buckets:
        values = [
            f"{name.replace('_', ' ')}: {getattr(use_case, name)}"
            for name in fields
            if getattr(use_case, name)
        ]
        if values:
            queries.append(f"{label}: " + "; ".join(values))
    for name in ("customer_preferences", "strict_requirements"):
        values = getattr(use_case, name)
        if values:
            queries.append(f"{name.replace('_', ' ')}: " + "; ".join(values))
    if claim:
        queries.append(f"assessment claim: {claim}")
    return list(dict.fromkeys(query for query in queries if query and query.strip()))


def _expanded_query(text: str) -> list[str]:
    tokens = _tokens(text)
    expanded = list(tokens)
    for token in set(tokens):
        for synonym in _SYNONYMS.get(token, ()):
            expanded.extend(_tokens(synonym))
    return expanded


class EvidenceRetriever:
    """Persistent SQLite BM25 and cosine retrieval with source provenance."""

    def __init__(
        self,
        index_path: Path | None = None,
        manifest_path: Path | None = None,
        settings: Any = None,
        documents_path: Path | None = None,
        mode: str = "hybrid",
        bm25_candidates: int = 30,
        semantic_candidates: int = 30,
        context_budget: int = 12000,
        rrf_k: int = 60,
        parent_expansion: bool | None = None,
    ) -> None:
        from spectro_ai.providers import RagSettings

        self.settings = settings or RagSettings.from_env()
        self.index_path = index_path or Path(self.settings.index)
        self.manifest_path = manifest_path or _MANIFEST
        self.documents_path = documents_path if documents_path is not None else _DOCUMENTS
        self.mode = mode
        self.bm25_candidates, self.semantic_candidates = bm25_candidates, semantic_candidates
        self.context_budget, self.rrf_k = context_budget, rrf_k
        self.parent_expansion = _parent_expansion_enabled(parent_expansion)
        if mode not in {"hybrid", "bm25"}:
            raise ValueError("mode must be 'hybrid' or 'bm25'")
        if bm25_candidates < 1 or semantic_candidates < 1 or context_budget < 1 or rrf_k < 0:
            raise ValueError(
                "Candidate counts and context budget must be positive; rrf_k nonnegative"
            )

    def index_metadata(self) -> dict[str, Any]:
        """Return validated source/version/page metadata for citation checks."""
        try:
            index, rows = _load_old_index(self.index_path)
            _validate_database(self.index_path, (index, rows))
            index["chunks"] = [row["chunk"] for row in rows.values()]
            index["vectors"] = {key: row["vector"] for key, row in rows.items()}
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read evidence index {self.index_path}: {exc}") from exc
        if index.get("schema_version") not in {2, _INDEX_SCHEMA_VERSION} or index.get(
            "chunking_version"
        ) not in {*_SUPPORTED_CHUNKING_VERSIONS, _CHUNKING_VERSION}:
            raise ValueError(
                "Knowledge index schema is unsupported or stale; run Update Knowledge Base"
            )
        if index.get("chunk_size") != int(
            os.getenv("SPECTRO_RAG_CHUNK_SIZE", _CHUNK_SIZE)
        ) or index.get("chunk_overlap") != int(
            os.getenv("SPECTRO_RAG_CHUNK_OVERLAP", _CHUNK_OVERLAP)
        ):
            raise ValueError("Knowledge index chunking settings changed; run Update Knowledge Base")
        if index.get("schema_version") == _INDEX_SCHEMA_VERSION and index.get(
            "extraction_fingerprint"
        ) != _extraction_fingerprint(
            int(index["chunk_size"]),
            int(index["chunk_overlap"]),
            _pdf_backend(),
        ):
            raise ValueError(
                "Knowledge index extraction settings changed; run Update Knowledge Base"
            )
        if not all(
            isinstance(index.get(key), list) for key in ("files", "missing_sources", "chunks")
        ):
            raise ValueError("Knowledge index collections are corrupt; run Update Knowledge Base")
        try:
            manifest_version = _sha256(self.manifest_path.read_bytes())
        except OSError as exc:
            raise ValueError(
                f"Cannot validate source manifest {self.manifest_path}: {exc}"
            ) from exc
        if index.get("manifest_version") != manifest_version:
            raise ValueError(
                "Knowledge index is stale for the current manifest; run Update Knowledge Base"
            )
        try:
            root, sources = _load_manifest(self.manifest_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot validate evidence manifest: {exc}") from exc
        if not all(
            isinstance(item, dict)
            and all(
                isinstance(item.get(key), str)
                for key in ("source_id", "filename", "document_version", "source_type", "scope")
            )
            and isinstance(item.get("page_count"), int)
            and item["page_count"] >= 0
            for item in index["files"]
        ):
            raise ValueError("Evidence index source records are corrupt; run reindex")
        records = {
            (item.get("source_id"), item.get("filename")): item
            for item in index["files"]
            if isinstance(item, dict)
        }
        gaps = {
            (item.get("source_id"), item.get("filename"))
            for item in index["missing_sources"]
            if isinstance(item, dict)
        }
        for source in sources:
            identity = (source["source_id"], source["filename"])
            path = _safe_source_path(root, str(source["filename"]))
            record = records.get(identity)
            if not path.is_file():
                if identity not in gaps or record:
                    raise ValueError("Evidence index source gaps are stale; run reindex")
                continue
            if (
                identity in gaps
                or not record
                or record.get("document_version") != _sha256(path.read_bytes())
            ):
                raise ValueError(f"Evidence index is stale for {source['filename']}; run reindex")
        file_lookup = {(item["source_id"], item["filename"]): item for item in index["files"]}
        evidence_ids: set[str] = set()
        for chunk in index["chunks"]:
            if (
                not isinstance(chunk, dict)
                or not isinstance(chunk.get("text"), str)
                or not chunk["text"].strip()
                or chunk.get("content_sha256") != _sha256(chunk["text"].encode("utf-8"))
                or not isinstance(chunk.get("provenance"), list)
                or not chunk["provenance"]
            ):
                raise ValueError("Evidence index chunk is corrupt; run reindex")
            for provenance in chunk["provenance"]:
                if not isinstance(provenance, dict) or not all(
                    isinstance(provenance.get(key), str)
                    for key in (
                        "evidence_id",
                        "source_id",
                        "filename",
                        "document_version",
                        "page_or_section",
                        "review_status",
                        "source_scope",
                    )
                ):
                    raise ValueError("Evidence citation metadata is corrupt; run reindex")
                source_record = file_lookup.get((provenance["source_id"], provenance["filename"]))
                if (
                    not source_record
                    or source_record["document_version"] != provenance["document_version"]
                ):
                    raise ValueError(
                        "Evidence citation points to an unknown source version; run reindex"
                    )
                if provenance["evidence_id"] in evidence_ids:
                    raise ValueError("Evidence citation ID is duplicated; run reindex")
                evidence_ids.add(provenance["evidence_id"])
                locations = provenance.get("source_locations", [])
                page_locators = (
                    [location.get("locator", "") for location in locations]
                    if locations
                    else [provenance["page_or_section"]]
                )
                for locator in page_locators:
                    page = re.match(r"p\.(\d+)", locator)
                    if page and not 1 <= int(page.group(1)) <= source_record["page_count"]:
                        raise ValueError(
                            "Evidence citation points outside source pages; run reindex"
                        )
        return index

    def search(self, use_case: CustomerUseCase, limit: int = 8) -> list[RetrievedEvidence]:
        self.retrieval_diagnostics: list[dict[str, Any]] = []
        return self._search_many(
            _query_groups(use_case),
            limit,
            rerank_query=_query_text(use_case),
            diagnostics=self.retrieval_diagnostics,
        )

    def search_claim(
        self, use_case: CustomerUseCase, claim: str, limit: int = 4
    ) -> list[RetrievedEvidence]:
        """Retrieve source excerpts relevant to one assessment claim."""
        self.retrieval_diagnostics = []
        return self._search_many(
            _query_groups(use_case, claim),
            limit,
            rerank_query=f"{claim}\n{_query_text(use_case)}",
            diagnostics=self.retrieval_diagnostics,
        )

    def search_candidates(
        self, use_case: CustomerUseCase, limit: int = 30
    ) -> list[RetrievedEvidence]:
        """Return fused unique candidates before final excerpt and byte limits."""
        return self._search_many(_query_groups(use_case), limit, candidate_mode=True)

    def search_claim_candidates(
        self, use_case: CustomerUseCase, claim: str, limit: int = 30
    ) -> list[RetrievedEvidence]:
        return self._search_many(_query_groups(use_case, claim), limit, candidate_mode=True)

    def _search_many(
        self,
        queries: list[str],
        limit: int,
        pinned_index: dict[str, Any] | None = None,
        *,
        candidate_mode: bool = False,
        rerank_query: str | None = None,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> list[RetrievedEvidence]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        index = pinned_index or self.index_metadata()
        if self.mode == "hybrid":
            from spectro_ai import providers

            if index.get("embedding_model") != self.settings.embedding_model:
                raise ValueError("Embedding model changed; run Update Knowledge Base")
            if providers.embedding_identity(self.settings) != index.get("embedding_identity"):
                raise ValueError("Embedding model revision changed; run Update Knowledge Base")
            if index.get("schema_version") == _INDEX_SCHEMA_VERSION and index.get(
                "embedding_configuration"
            ) != _embedding_config(
                self.settings,
                index["embedding_identity"],
                index["embedding_dimension"],
            ):
                raise ValueError("Embedding configuration changed; run Update Knowledge Base")
        found: dict[str, tuple[float, RetrievedEvidence]] = {}
        reranking = self.settings.reranker == "cross_encoder" and not candidate_mode
        candidate_limit = (
            limit
            if candidate_mode
            else (
                self.settings.reranker_candidates
                if reranking
                else max(self.bm25_candidates, self.semantic_candidates, limit)
            )
        )
        for query in queries:
            for rank, hit in enumerate(
                self._search(
                    query,
                    candidate_limit,
                    index,
                    apply_context_budget=not (reranking or candidate_mode or self.parent_expansion),
                ),
                1,
            ):
                score, _ = found.get(hit.evidence_id, (0.0, hit))
                found[hit.evidence_id] = (score + 1 / (self.rrf_k + rank), hit)
        ranked = sorted(found.values(), key=lambda item: (-item[0], item[1].evidence_id))
        if candidate_mode:
            return [
                hit.model_copy(update={"retrieval_score": score}) for score, hit in ranked[:limit]
            ]
        if reranking:
            from time import perf_counter

            from spectro_ai.providers import ProviderError, rerank_scores

            pool = ranked[:candidate_limit]
            started = perf_counter()
            if not pool:
                if diagnostics is not None:
                    diagnostics.append({"status": "skipped", "candidate_count": 0})
                return []
            try:
                query_text = rerank_query or "\n".join(queries)
                scores, revision = rerank_scores(
                    self.settings, query_text, [hit.excerpt for _, hit in pool]
                )
                if len(scores) != len(pool):
                    raise ValueError("reranker returned wrong score count")
                ranked = sorted(
                    (
                        (
                            pool[i][0],
                            hit.model_copy(
                                update={"retrieval_score": pool[i][0], "rerank_score": scores[i]}
                            ),
                        )
                        for i, (_, hit) in enumerate(pool)
                    ),
                    key=lambda item: (-float(item[1].rerank_score), -item[0]),
                )
                diagnostic = {
                    "status": "applied",
                    "model_revision": revision,
                    "candidate_count": len(pool),
                    "elapsed_ms": (perf_counter() - started) * 1000,
                }
            except (ProviderError, ValueError, TypeError) as exc:
                diagnostic = {
                    "status": "fallback",
                    "model_revision": self.settings.reranker_revision,
                    "candidate_count": len(pool),
                    "elapsed_ms": (perf_counter() - started) * 1000,
                    "failure_reason": str(exc),
                }
                ranked = pool
            if diagnostics is not None:
                diagnostics.append(diagnostic)
            ranked = [(score, hit) for score, hit in ranked]
        else:
            if diagnostics is not None:
                diagnostics.append({"status": "disabled"})
        if self.parent_expansion:
            return self._select_with_parent_context(ranked, limit, index)
        selected: list[RetrievedEvidence] = []
        used_bytes = 0
        for score, hit in ranked:
            cost = len(hit.excerpt.encode("utf-8"))
            if used_bytes + cost > self.context_budget:
                continue
            selected.append(hit.model_copy(update={"retrieval_score": score}))
            used_bytes += cost
            if len(selected) == limit:
                break
        return selected

    def _select_with_parent_context(
        self,
        ranked: list[tuple[float, RetrievedEvidence]],
        limit: int,
        index: dict[str, Any],
    ) -> list[RetrievedEvidence]:
        child_capacity = int(self.context_budget * 0.7)
        context_capacity = self.context_budget - child_capacity
        child_used = 0
        selected: list[tuple[float, RetrievedEvidence]] = []
        selected_ids: set[str] = set()
        for score, hit in ranked:
            cost = len(hit.excerpt.encode("utf-8"))
            if child_used + cost > child_capacity:
                continue
            selected.append((score, hit))
            selected_ids.add(hit.evidence_id)
            child_used += cost
            if len(selected) >= limit:
                break

        provenance_by_id = {
            provenance.get("evidence_id"): provenance
            for chunk in index.get("chunks", [])
            for provenance in chunk.get("provenance", [])
            if provenance.get("evidence_id")
        }
        block_map = {
            item.get("block_id"): item
            for item in index.get("source_blocks", [])
            if isinstance(item, dict) and isinstance(item.get("block_id"), str)
        }
        section_map = {
            item.get("section_id"): item
            for item in index.get("sections", [])
            if isinstance(item, dict) and isinstance(item.get("section_id"), str)
        }
        child_block_ids_by_evidence = {
            hit.evidence_id: set(
                (provenance_by_id.get(hit.evidence_id) or {}).get("block_ids")
                or [(provenance_by_id.get(hit.evidence_id) or {}).get("block_id")]
            )
            - {None}
            for _score, hit in ranked
        }
        fully_covered_by_evidence: dict[str, set[str]] = {}
        for _score, hit in ranked:
            provenance = provenance_by_id.get(hit.evidence_id) or {}
            spans = provenance.get("block_spans")
            if isinstance(spans, list):
                fully_covered_by_evidence[hit.evidence_id] = {
                    span["block_id"]
                    for span in spans
                    if isinstance(span, dict)
                    and isinstance(span.get("block_id"), str)
                    and isinstance(block_map.get(span["block_id"]), dict)
                    and span.get("start") == 0
                    and span.get("end") == len(block_map[span["block_id"]].get("text", ""))
                }
            else:
                # Older schema-v3 passages had one complete source block per child.
                fully_covered_by_evidence[hit.evidence_id] = {
                    block_id
                    for block_id in child_block_ids_by_evidence[hit.evidence_id]
                    if isinstance(block_map.get(block_id), dict)
                    and block_map[block_id].get("text") == hit.excerpt
                }
        canonical_blocks: dict[str, EvidenceContextBlock] = {}

        def canonical(block_id: str) -> EvidenceContextBlock:
            if block_id not in canonical_blocks:
                canonical_blocks[block_id] = canonical_context_block(index, block_id)
            return canonical_blocks[block_id]

        context_used = 0
        covered_child_block_ids: set[str] = set().union(
            *(fully_covered_by_evidence.get(hit.evidence_id, set()) for _score, hit in selected)
        )
        attached_block_ids = set(covered_child_block_ids)
        expanded: list[RetrievedEvidence] = []
        for score, hit in selected:
            provenance = provenance_by_id.get(hit.evidence_id)
            block_ids = (provenance or {}).get("block_ids") or [(provenance or {}).get("block_id")]
            section = section_map.get((provenance or {}).get("section_id"))
            omissions: list[str] = []
            attached: list[EvidenceContextBlock] = []
            if not section or any(not isinstance(block_id, str) for block_id in block_ids):
                omissions.append("No validated source section is available for expansion.")
            elif not section.get("hierarchy_valid", True):
                omissions.append("Section hierarchy is invalid; context expansion was suppressed.")
            else:
                section_blocks = [
                    block_id for block_id in section.get("block_ids", []) if block_id in block_map
                ]
                child_positions = [
                    section_blocks.index(block_id)
                    for block_id in block_ids
                    if block_id in section_blocks
                ]
                if not child_positions:
                    omissions.append("Retrieved child blocks are outside the validated section.")
                else:
                    child_block_ids = set(block_ids)
                    fully_covered_child_blocks = fully_covered_by_evidence.get(
                        hit.evidence_id, set()
                    )
                    available_ids = [
                        block_id
                        for block_id in section_blocks
                        if block_id not in fully_covered_child_blocks
                        and block_id not in attached_block_ids
                    ]
                    full_section = [canonical(block_id) for block_id in available_ids]
                    full_cost = evidence_context_blocks_bytes(full_section)
                    remaining = context_capacity - context_used
                    if full_cost <= remaining:
                        attached = full_section
                    else:
                        attached, omitted = self._context_window(
                            index=index,
                            section_blocks=section_blocks,
                            child_positions=child_positions,
                            anchor_block_ids=child_block_ids - fully_covered_child_blocks,
                            child_block_ids=fully_covered_child_blocks,
                            attached_block_ids=attached_block_ids,
                            remaining=remaining,
                            block_map=block_map,
                            canonical=canonical,
                        )
                        if omitted:
                            omissions.append(omitted)
                    context_used += evidence_context_blocks_bytes(attached)
                    attached_block_ids.update(block.block_id for block in attached)
                    if not attached and not omissions:
                        omissions.append(
                            "No surrounding source blocks were available in this section."
                        )
            expanded.append(
                hit.model_copy(
                    update={
                        "retrieval_score": score,
                        "context_blocks": attached,
                        "context_omissions": omissions,
                    }
                )
            )

        # Use unspent reserved capacity for additional ranked children after context
        # has been allocated in rank order. These results carry no parent expansion.
        total_used = child_used + context_used
        if len(expanded) < limit:
            for score, hit in ranked:
                if hit.evidence_id in selected_ids:
                    continue
                cost = len(hit.excerpt.encode("utf-8"))
                if total_used + cost > self.context_budget:
                    continue
                child_blocks = fully_covered_by_evidence.get(hit.evidence_id, set())
                if child_blocks & attached_block_ids:
                    continue
                expanded.append(
                    hit.model_copy(
                        update={
                            "retrieval_score": score,
                            "context_omissions": [
                                "Context capacity was allocated to higher-ranked children first."
                            ],
                        }
                    )
                )
                selected_ids.add(hit.evidence_id)
                attached_block_ids.update(child_blocks)
                total_used += cost
                if len(expanded) >= limit:
                    break
        return expanded

    def _context_window(
        self,
        *,
        index: dict[str, Any],
        section_blocks: list[str],
        child_positions: list[int],
        anchor_block_ids: set[str],
        child_block_ids: set[str],
        attached_block_ids: set[str],
        remaining: int,
        block_map: dict[str, dict[str, Any]],
        canonical: Any,
    ) -> tuple[list[EvidenceContextBlock], str | None]:
        left_edge, right_edge = min(child_positions), max(child_positions)
        selected: dict[str, EvidenceContextBlock] = {}
        used = 0
        left_open = right_open = True

        def bundle(candidate_id: str) -> list[str]:
            candidate = block_map[candidate_id]
            required: list[str] = [candidate_id]
            table_id = candidate.get("table_id")
            if table_id:
                required.extend(candidate.get("table_header_block_ids", []))
                required.extend(
                    block_id
                    for block_id in section_blocks
                    if block_map[block_id].get("table_id") == table_id
                    and block_map[block_id].get("block_type") == "table_caption"
                )
            return list(dict.fromkeys(required))

        max_distance = max(left_edge, len(section_blocks) - right_edge - 1)

        # A packed child can cover only part of its source block. Prefer the
        # complete original block first so nearby qualifications are recoverable.
        for anchor_id in section_blocks:
            if anchor_id not in anchor_block_ids:
                continue
            required_ids = bundle(anchor_id)
            if any(block_id not in section_blocks for block_id in required_ids):
                continue
            new_ids = [
                block_id
                for block_id in required_ids
                if block_id not in child_block_ids
                and block_id not in attached_block_ids
                and block_id not in selected
            ]
            new_blocks = [canonical(block_id) for block_id in new_ids]
            cost = evidence_context_blocks_bytes(
                [*selected.values(), *new_blocks]
            ) - evidence_context_blocks_bytes(list(selected.values()))
            if used + cost <= remaining:
                selected.update((block.block_id, block) for block in new_blocks)
                used += cost

        for distance in range(1, max_distance + 1):
            candidates = []
            if left_open and left_edge - distance >= 0:
                candidates.append(("left", left_edge - distance))
            if right_open and right_edge + distance < len(section_blocks):
                candidates.append(("right", right_edge + distance))
            for side, position in candidates:
                candidate_id = section_blocks[position]
                if candidate_id in child_block_ids or candidate_id in attached_block_ids:
                    continue
                required_ids = bundle(candidate_id)
                if any(block_id not in section_blocks for block_id in required_ids):
                    if side == "left":
                        left_open = False
                    else:
                        right_open = False
                    continue
                new_ids = [
                    block_id
                    for block_id in required_ids
                    if block_id not in child_block_ids
                    and block_id not in attached_block_ids
                    and block_id not in selected
                ]
                new_blocks = [canonical(block_id) for block_id in new_ids]
                cost = evidence_context_blocks_bytes(
                    [*selected.values(), *new_blocks]
                ) - evidence_context_blocks_bytes(list(selected.values()))
                if used + cost <= remaining:
                    selected.update((block.block_id, block) for block in new_blocks)
                    used += cost
                elif side == "left":
                    left_open = False
                else:
                    right_open = False
        omission = None
        if not selected and remaining > 0:
            omission = "No complete surrounding source block fit the reserved context budget."
        elif anchor_block_ids and any(
            block_id not in selected and block_id not in attached_block_ids
            for block_id in anchor_block_ids
        ):
            omission = (
                "The complete retrieved source block did not fit the reserved context budget."
            )
        return [selected[block_id] for block_id in section_blocks if block_id in selected], omission

    def _search(
        self,
        query_text: str,
        limit: int,
        pinned_index: dict[str, Any] | None = None,
        *,
        apply_context_budget: bool = True,
    ) -> list[RetrievedEvidence]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        index = pinned_index or self.index_metadata()
        chunks = index.get("chunks", [])
        query = Counter(_expanded_query(query_text))
        if not chunks:
            return []
        corpus = [Counter(_tokens(chunk["text"])) for chunk in chunks]
        lengths = [sum(counts.values()) for counts in corpus]
        average_length = sum(lengths) / max(len(lengths), 1)
        document_frequency: Counter[str] = Counter()
        for terms in corpus:
            document_frequency.update(terms.keys())
        ranked: list[tuple[float, int]] = []
        for index_number, terms in enumerate(corpus):
            score = 0.0
            for term, query_frequency in query.items():
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse = math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * lengths[index_number] / (average_length or 1)
                )
                score += inverse * frequency * 2.5 / denominator * (1 + math.log(query_frequency))
            if score > 0:
                ranked.append((score, index_number))
        ranked.sort(key=lambda item: (-item[0], chunks[item[1]]["content_sha256"]))
        bm25 = [number for _, number in ranked[: self.bm25_candidates]]
        semantic: list[int] = []
        if self.mode == "hybrid":
            from spectro_ai import providers

            query_vector = providers.embed_texts(self.settings, [query_text])[0]
            valid_query = bool(query_vector) and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in query_vector
            )
            query_norm = math.hypot(*query_vector) if valid_query else 0
            if not query_norm:
                raise ValueError("Ollama returned invalid query embedding")
            if len(query_vector) != index.get("embedding_dimension"):
                raise ValueError("Query embedding dimension does not match knowledge index")
            similarities = []
            for number, chunk in enumerate(chunks):
                vector = index["vectors"].get(chunk["chunk_id"], [])
                if len(vector) != len(query_vector) or not vector:
                    raise ValueError("Knowledge index embedding dimensions are incompatible")
                similarity = sum(a * b for a, b in zip(query_vector, vector, strict=True)) / (
                    query_norm * math.hypot(*vector)
                )
                similarities.append((similarity, number))
            similarities.sort(key=lambda item: (-item[0], chunks[item[1]]["content_sha256"]))
            semantic = [number for _, number in similarities[: self.semantic_candidates]]
        fused: dict[int, float] = {}
        for rank, number in enumerate(bm25, 1):
            fused[number] = fused.get(number, 0) + 1 / (self.rrf_k + rank)
        for rank, number in enumerate(semantic, 1):
            fused[number] = fused.get(number, 0) + 1 / (self.rrf_k + rank)
        ordered = sorted(fused, key=lambda n: (-fused[n], chunks[n]["content_sha256"]))
        output: list[RetrievedEvidence] = []
        used_bytes = 0
        for chunk_number in ordered:
            excerpt = chunks[chunk_number]["text"]
            cost = len(excerpt.encode("utf-8"))
            if apply_context_budget and used_bytes + cost > self.context_budget:
                continue
            for provenance in chunks[chunk_number]["provenance"][:1]:
                output.append(
                    RetrievedEvidence(
                        evidence_id=provenance["evidence_id"],
                        source_id=provenance["source_id"],
                        document_version=provenance["document_version"],
                        page_or_section=provenance["page_or_section"],
                        excerpt=excerpt,
                        review_status=provenance["review_status"],
                        retrieval_score=fused[chunk_number],
                        source_scope=provenance["source_scope"],
                        source_locations=[
                            SourceLocation.model_validate(location)
                            for location in provenance.get("source_locations", [])
                        ],
                    )
                )
                used_bytes += cost
                break
            if len(output) >= limit:
                return output
        return output

    def snapshot(self) -> _PinnedEvidenceRetriever:
        return _PinnedEvidenceRetriever(self, self.index_metadata())

    def update_knowledge_base(self, force_rebuild: bool = False) -> dict[str, Any]:
        try:
            before, _ = _load_old_index(self.index_path)
        except ValueError:
            before = {}
        result = _update_index(
            manifest_path=self.manifest_path,
            index_path=self.index_path,
            documents_path=self.documents_path,
            semantic=True,
            settings=self.settings,
            force_rebuild=force_rebuild,
        )
        previous = {item["filename"]: item for item in before.get("files", [])}
        current = {item["filename"]: item for item in result["files"]}
        added = sum(name not in previous for name in current)
        changed = sum(
            name in previous and previous[name]["document_version"] != item["document_version"]
            for name, item in current.items()
        )
        unchanged = sum(
            name in previous and previous[name]["document_version"] == item["document_version"]
            for name, item in current.items()
        )
        removed = sum(name not in current for name in previous)
        result["documents"] = {
            "new": added,
            "updated": changed,
            "removed": removed,
            "unchanged": unchanged,
        }
        result["embeddings"] = {
            "new": result["new_embeddings"],
            "reused": result["reused_embeddings"],
        }
        return result

    def knowledge_status(self) -> dict[str, Any]:
        try:
            meta = self.index_metadata()
            mode = "hybrid" if meta["embedding_model"] else "bm25"
            if self.mode == "hybrid" and mode != "hybrid":
                return {
                    "ready": False,
                    "documents": meta["documents"],
                    "chunks": len(meta["chunks"]),
                    "embedding_model": meta["embedding_model"],
                    "mode": mode,
                    "indexed_at": meta["indexed_at"],
                    "revision": meta["revision"],
                    "error": (
                        "Knowledge base has no embeddings. Click Update Knowledge Base "
                        "to enable hybrid retrieval."
                    ),
                }
            if self.mode == "hybrid" and meta["embedding_model"] != self.settings.embedding_model:
                return {
                    "ready": False,
                    "documents": meta["documents"],
                    "chunks": len(meta["chunks"]),
                    "embedding_model": meta["embedding_model"],
                    "mode": mode,
                    "indexed_at": meta["indexed_at"],
                    "revision": meta["revision"],
                    "error": (
                        "Embedding model changed. Click Update Knowledge Base "
                        "to rebuild embeddings."
                    ),
                }
            return {
                "ready": True,
                "documents": meta["documents"],
                "chunks": len(meta["chunks"]),
                "embedding_model": meta["embedding_model"],
                "mode": mode,
                "indexed_at": meta["indexed_at"],
                "revision": meta["revision"],
                "error": None,
            }
        except (OSError, ValueError) as exc:
            return {
                "ready": False,
                "documents": 0,
                "chunks": 0,
                "embedding_model": self.settings.embedding_model,
                "indexed_at": None,
                "revision": None,
                "error": str(exc),
            }


class _PinnedEvidenceRetriever:
    def __init__(self, retriever: EvidenceRetriever, metadata: dict[str, Any]) -> None:
        self.retriever, self.metadata = retriever, metadata
        self.retrieval_diagnostics: list[dict[str, Any]] = []

    def index_metadata(self) -> dict[str, Any]:
        return self.metadata

    def search(self, use_case: CustomerUseCase, limit: int = 8) -> list[RetrievedEvidence]:
        self.retrieval_diagnostics = []
        return self.retriever._search_many(
            _query_groups(use_case),
            limit,
            self.metadata,
            rerank_query=_query_text(use_case),
            diagnostics=self.retrieval_diagnostics,
        )

    def search_claim(
        self, use_case: CustomerUseCase, claim: str, limit: int = 4
    ) -> list[RetrievedEvidence]:
        return self.retriever._search_many(
            _query_groups(use_case, claim),
            limit,
            self.metadata,
            rerank_query=f"{claim}\n{_query_text(use_case)}",
            diagnostics=self.retrieval_diagnostics,
        )

    def search_candidates(
        self, use_case: CustomerUseCase, limit: int = 30
    ) -> list[RetrievedEvidence]:
        return self.retriever._search_many(
            _query_groups(use_case), limit, self.metadata, candidate_mode=True
        )

    def search_claim_candidates(
        self, use_case: CustomerUseCase, claim: str, limit: int = 30
    ) -> list[RetrievedEvidence]:
        return self.retriever._search_many(
            _query_groups(use_case, claim), limit, self.metadata, candidate_mode=True
        )


class EvidenceKnowledgeSource(KnowledgeSource):
    """Provides citations only; excerpts are not curated technique profiles."""

    def __init__(self, retriever: EvidenceRetriever | None = None) -> None:
        self.retriever = retriever or EvidenceRetriever()

    def get_techniques(self) -> list[TechniqueKnowledge]:
        """Raw source text cannot satisfy validated, expert-reviewed technique claims."""
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reindex_parser = subparsers.add_parser("reindex", help="atomically rebuild local source index")
    reindex_parser.add_argument("--manifest", type=Path, default=_MANIFEST)
    reindex_parser.add_argument("--index", type=Path, default=_INDEX)
    reindex_parser.add_argument("--documents", type=Path, default=_DOCUMENTS)
    reindex_parser.add_argument("--bm25", action="store_true", help="explicit lexical-only index")
    args = parser.parse_args()
    if args.command == "reindex":
        if args.bm25:
            result = reindex_sources(args.manifest, args.index)
        else:
            result = EvidenceRetriever(
                args.index, args.manifest, documents_path=args.documents
            ).update_knowledge_base()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
