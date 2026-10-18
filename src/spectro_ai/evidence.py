"""Allowlisted AFS source indexing and lexical evidence retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import posixpath
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import yaml

from spectro_ai.knowledge import KnowledgeSource
from spectro_ai.models import CustomerUseCase, RetrievedEvidence, TechniqueKnowledge

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "knowledge" / "sources.yaml"
_INDEX = _ROOT / ".cache" / "evidence" / "index.json"
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


def _split_text(text: str, maximum: int = 1500, overlap: int = 180) -> list[str]:
    clean = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + maximum, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start + maximum // 2, end)
            if boundary > start:
                end = boundary
        excerpt = clean[start:end].strip()
        if excerpt:
            chunks.append(excerpt)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _pdf_blocks(path: Path) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF indexing needs the project 'rag' extra (pypdf)") from exc
    reader = PdfReader(str(path))
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
    return blocks, empty_pages, poor_pages


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
    """Rebuild full index in memory, then atomically replace prior index."""
    manifest_path = manifest_path or _MANIFEST
    index_path = index_path or _INDEX
    root, sources = _load_manifest(manifest_path)
    files: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    chunks: dict[str, dict[str, Any]] = {}
    extraction_cache: dict[tuple[str, str], tuple[list[tuple[str, str]], list[str], list[str]]] = {}
    for source in sources:
        filename = str(source["filename"])
        path = _safe_source_path(root, filename)
        source_id = str(source["source_id"])
        if not path.is_file():
            missing.append({"source_id": source_id, "filename": filename, "status": "missing"})
            continue
        content = path.read_bytes()
        version = _sha256(content)
        cache_key = (version, path.suffix.casefold())
        if cache_key not in extraction_cache:
            extraction_cache[cache_key] = _extract(path)
        blocks, empty_pages, poor_pages = extraction_cache[cache_key]
        page_numbers = [
            int(match.group(1))
            for locator, _ in blocks
            if (match := re.match(r"p\.(\d+)", locator))
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
            "page_count": max(page_numbers, default=0),
            "empty_pages": empty_pages,
            "poor_extraction_pages": poor_pages,
            "chunk_count": len(blocks),
        }
        files.append(file_record)
        overrides = source.get("page_review", {})
        for chunk_number, (locator, excerpt) in enumerate(blocks, 1):
            page_match = re.match(r"p\.(\d+)", locator)
            review = (
                overrides.get(page_match.group(1), file_record["review_status"])
                if page_match
                else file_record["review_status"]
            )
            if page_match and f"p.{page_match.group(1)}" in poor_pages:
                review = f"{review}; poor_extraction_needs_manual_review"
            content_hash = _sha256(excerpt.encode("utf-8"))
            item = chunks.setdefault(
                content_hash, {"content_sha256": content_hash, "text": excerpt, "provenance": []}
            )
            evidence_id = f"{source_id}-{path.stem}-{version[:12]}-{chunk_number:04d}"
            provenance = {
                "evidence_id": evidence_id,
                "source_id": source_id,
                "document_version": version,
                "page_or_section": locator,
                "review_status": str(review),
                "source_scope": "; ".join(
                    dict.fromkeys(
                        (
                            str(source.get("source_type", "technical_reference")),
                            str(source.get("scope", "technical")),
                        )
                    )
                ),
                "title": file_record["title"],
                "filename": filename,
                "related_source": source.get("related_source"),
                "citation_format": source.get("citation_format"),
            }
            if provenance not in item["provenance"]:
                item["provenance"].append(provenance)

    result = {
        "schema_version": 1,
        "manifest_version": _sha256(manifest_path.read_bytes()),
        "files": files,
        "missing_sources": missing,
        "chunks": list(chunks.values()),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=index_path.parent, prefix=f"{index_path.name}.", delete=False
        ) as temp:
            temp.write(encoded)
            temp.flush()
            os.fsync(temp.fileno())
            temp_name = temp.name
        os.replace(temp_name, index_path)
    except Exception:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
        raise
    return {
        "index_path": str(index_path),
        "files_indexed": len(files),
        "chunks": len(chunks),
        "missing_sources": missing,
        "empty_pdf_pages": sum(len(item["empty_pages"]) for item in files),
        "poor_extraction_pages": sum(len(item["poor_extraction_pages"]) for item in files),
    }


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text) if token.casefold() not in _STOP]


def _query_text(use_case: CustomerUseCase) -> str:
    values: list[str] = []
    for name in CustomerUseCase.FACT_FIELDS:
        value = getattr(use_case, name)
        if value:
            values.append(value)
    for name in ("source_text", "customer_preferences", "strict_requirements"):
        value = getattr(use_case, name)
        if isinstance(value, str) and value:
            values.append(value)
        elif isinstance(value, list):
            values.extend(value)
    return " ".join(values)


def _expanded_query(text: str) -> list[str]:
    tokens = _tokens(text)
    expanded = list(tokens)
    for token in set(tokens):
        for synonym in _SYNONYMS.get(token, ()):
            expanded.extend(_tokens(synonym))
    return expanded


class EvidenceRetriever:
    """Local lexical BM25 search over indexed, source-cited excerpts."""

    def __init__(self, index_path: Path | None = None, manifest_path: Path | None = None) -> None:
        self.index_path = index_path or _INDEX
        self.manifest_path = manifest_path or _MANIFEST

    def index_metadata(self) -> dict[str, Any]:
        """Return validated source/version/page metadata for citation checks."""
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read evidence index {self.index_path}: {exc}") from exc
        if not isinstance(index, dict) or index.get("schema_version") != 1:
            raise ValueError("Evidence index schema is unsupported or corrupt; run reindex")
        if not all(
            isinstance(index.get(key), list) for key in ("files", "missing_sources", "chunks")
        ):
            raise ValueError("Evidence index collections are corrupt; run reindex")
        try:
            manifest_version = _sha256(self.manifest_path.read_bytes())
        except OSError as exc:
            raise ValueError(
                f"Cannot validate source manifest {self.manifest_path}: {exc}"
            ) from exc
        if index.get("manifest_version") != manifest_version:
            raise ValueError("Evidence index is stale for the current manifest; run reindex")
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
                page = re.match(r"p\.(\d+)", provenance["page_or_section"])
                if page and not 1 <= int(page.group(1)) <= source_record["page_count"]:
                    raise ValueError("Evidence citation points outside source pages; run reindex")
        return index

    def search(self, use_case: CustomerUseCase, limit: int = 8) -> list[RetrievedEvidence]:
        return self._search(_query_text(use_case), limit)

    def search_claim(
        self, use_case: CustomerUseCase, claim: str, limit: int = 4
    ) -> list[RetrievedEvidence]:
        """Retrieve source excerpts relevant to one assessment claim."""
        return self._search(f"{_query_text(use_case)} {claim}", limit)

    def _search(self, query_text: str, limit: int) -> list[RetrievedEvidence]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        index = self.index_metadata()
        chunks = index.get("chunks", [])
        query = Counter(_expanded_query(query_text))
        if not query or not chunks:
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
        output: list[RetrievedEvidence] = []
        for score, chunk_number in ranked:
            for provenance in chunks[chunk_number]["provenance"]:
                output.append(
                    RetrievedEvidence(
                        evidence_id=provenance["evidence_id"],
                        source_id=provenance["source_id"],
                        document_version=provenance["document_version"],
                        page_or_section=provenance["page_or_section"],
                        excerpt=chunks[chunk_number]["text"],
                        review_status=provenance["review_status"],
                        retrieval_score=score,
                        source_scope=provenance["source_scope"],
                    )
                )
                if len(output) >= limit:
                    return output
        return output


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
    args = parser.parse_args()
    if args.command == "reindex":
        print(json.dumps(reindex_sources(args.manifest, args.index), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
