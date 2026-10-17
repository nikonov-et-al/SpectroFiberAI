"""Local document ingestion, atomic indexing, retrieval and cited Q&A."""

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import uuid
from contextlib import closing
from pathlib import Path

from spectro_ai import providers
from spectro_ai.providers import ProviderError, RagSettings
from spectro_ai.rag_models import DocumentAnswer, DocumentChunk, RagResult, RetrievalHit

CHUNK_SIZE = 1200
OVERLAP = 200


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_chunks(path: Path, root: Path) -> list[DocumentChunk]:
    name = path.relative_to(root).as_posix()
    try:
        data = path.read_bytes()
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                raise ProviderError("Install document support: uv sync --extra rag") from None
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ProviderError(f"Encrypted PDF: {name}")
            sections = [(n, page.extract_text() or "") for n, page in enumerate(reader.pages, 1)]
            if any(not text.strip() for _, text in sections) or not sections:
                raise ProviderError(f"Scanned or empty PDF page: {name}; OCR is not supported")
        else:
            sections = [(None, data.decode("utf-8-sig"))]
    except ProviderError:
        raise
    except Exception:
        raise ProviderError(f"Cannot read document: {name}") from None
    chunks = []
    for page, text in sections:
        for start in range(0, len(text), CHUNK_SIZE - OVERLAP):
            end = min(start + CHUNK_SIZE, len(text))
            excerpt = text[start:end]
            if excerpt.strip():
                identifier = digest(json.dumps([name, page, start, excerpt]).encode())
                chunks.append(
                    DocumentChunk(
                        chunk_id=identifier,
                        document=name,
                        file_hash=digest(data),
                        text=excerpt,
                        page=page,
                        line_start=text.count("\n", 0, start) + 1 if page is None else None,
                        line_end=text.count("\n", 0, end - 1) + 1 if page is None else None,
                    )
                )
            if end == len(text):
                break
    return chunks


def validate_vectors(vectors: list, count: int, dimension: int | None = None) -> int:
    if len(vectors) != count:
        raise ProviderError("Invalid embedding count")
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise ProviderError("Invalid empty embedding")
        dimension = dimension or len(vector)
        if (
            len(vector) != dimension
            or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in vector
            )
            or not math.isfinite(math.hypot(*vector))
            or math.hypot(*vector) == 0
        ):
            raise ProviderError("Invalid embedding dimension, values or norm")
    return dimension or 0


def load_index(path: Path) -> tuple[dict, list[tuple[DocumentChunk, list[float]]]]:
    if not path.is_file():
        raise ProviderError(
            "Index missing. Run python -m spectro_ai.rag index --documents documents"
        )
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            meta = json.loads(connection.execute("SELECT value FROM metadata").fetchone()[0])
            rows = [
                (DocumentChunk.model_validate_json(c), json.loads(v))
                for c, v in connection.execute("SELECT chunk, vector FROM chunks ORDER BY id")
            ]
        if meta["schema"] != 1 or meta["chunk_size"] != CHUNK_SIZE or meta["overlap"] != OVERLAP:
            raise ValueError
        for key in ("embedding_model", "digest", "root", "revision"):
            if not isinstance(meta[key], str) or not meta[key]:
                raise ValueError
        if meta["chunks"] != len(rows) or meta["documents"] != len({c.document for c, _ in rows}):
            raise ValueError
        if not isinstance(meta["dimension"], int) or meta["dimension"] < bool(rows):
            raise ValueError
        validate_vectors([v for _, v in rows], len(rows), meta["dimension"])
        return meta, rows
    except sqlite3.Error, ValueError, KeyError, TypeError, IndexError:
        raise ProviderError("Invalid or incompatible index. Rebuild with index --rebuild") from None


def index_documents(
    documents: Path,
    index: Path,
    settings: RagSettings,
    rebuild: bool = False,
) -> dict:
    if not documents.is_dir():
        raise ProviderError("Document directory does not exist; create it and add PDF/MD/TXT files")
    if index.suffix.lower() in {".txt", ".md", ".pdf"}:
        raise ProviderError("Index path must not use a source-document extension; use .sqlite3")
    identity = providers.embedding_identity(settings)
    meta = {
        "schema": 1,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "embedding_model": settings.embedding_model,
        "digest": identity,
        "root": str(documents.resolve()),
        "dimension": 0,
    }
    old = {}
    if index.exists() and not rebuild:
        previous, rows = load_index(index)
        if any(previous[k] != meta[k] for k in meta if k != "dimension"):
            raise ProviderError("Embedding settings or document root changed. Use index --rebuild")
        meta["dimension"] = previous["dimension"]
        old = {chunk.chunk_id: vector for chunk, vector in rows}
    chunks = []
    for path in sorted(documents.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
            if path.is_symlink() or not path.resolve().is_relative_to(documents.resolve()):
                raise ProviderError("Document links outside the source directory are not supported")
            chunks.extend(extract_chunks(path, documents))
    pending = [c for c in chunks if c.chunk_id not in old]
    for start in range(0, len(pending), 16):
        batch = pending[start : start + 16]
        vectors = providers.embed_texts(settings, [c.text for c in batch])
        meta["dimension"] = validate_vectors(vectors, len(batch), meta["dimension"] or None)
        old.update(zip((c.chunk_id for c in batch), vectors, strict=True))
    meta.update(
        revision=uuid.uuid4().hex, documents=len({c.document for c in chunks}), chunks=len(chunks)
    )
    index.parent.mkdir(parents=True, exist_ok=True)
    staging = index.with_name(index.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with closing(sqlite3.connect(staging)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, chunk TEXT, vector TEXT)"
            )
            connection.execute("DELETE FROM metadata")
            connection.execute("DELETE FROM chunks")
            connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(meta),))
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?)",
                [(c.chunk_id, c.model_dump_json(), json.dumps(old[c.chunk_id])) for c in chunks],
            )
        staging.replace(index)
    except sqlite3.Error, OSError:
        raise ProviderError("Cannot update index; previous committed contents preserved") from None
    finally:
        staging.unlink(missing_ok=True)
    return {**meta, "embedded_chunks": len(pending)}


def search(question: str, index: Path, settings: RagSettings) -> list[RetrievalHit]:
    if not question.strip():
        raise ProviderError("Enter a document question")
    meta, rows = load_index(index)
    if not rows:
        return []
    if meta["embedding_model"] != settings.embedding_model or (
        meta["digest"] != providers.embedding_identity(settings)
    ):
        raise ProviderError("Embedding model changed. Rebuild with index --rebuild")
    query = providers.embed_texts(settings, [question])
    validate_vectors(query, 1, meta["dimension"])
    try:
        import numpy as np
    except ImportError:
        raise ProviderError("Install retrieval support: uv sync --extra rag") from None
    # ponytail: exact O(chunks * dimensions) search; replace only after measured corpus growth.
    matrix = np.array([v for _, v in rows], dtype=float)
    vector = np.array(query[0], dtype=float)
    matrix /= np.array([math.hypot(*v) for _, v in rows])[:, None]
    vector /= math.hypot(*query[0])
    scores = matrix @ vector
    order = sorted(range(len(rows)), key=lambda n: (-float(scores[n]), rows[n][0].chunk_id))[:4]
    return [RetrievalHit(chunk=rows[n][0], score=float(scores[n])) for n in order]


def ask(question: str, index: Path, settings: RagSettings) -> RagResult:
    hits = search(question, index, settings)
    if not hits:
        answer = DocumentAnswer(status="insufficient_evidence")
    else:
        prompt = Path(__file__).resolve().parents[2] / "prompts" / "rag.md"
        if not prompt.exists():
            prompt = Path(__file__).parent / "data" / "prompts" / "rag.md"
        system = prompt.read_text(encoding="utf-8")
        # Bound prompt by UTF-8 bytes (conservative token upper bound), leaving output space.
        while True:
            user = json.dumps(
                {
                    "question": question,
                    "excerpts": [
                        {"chunk_id": h.chunk.chunk_id, "text": h.chunk.text} for h in hits
                    ],
                },
                ensure_ascii=False,
            )
            if len((system + user).encode()) + 768 <= settings.context or not hits:
                break
            hits.pop()
        if not hits:
            raise ProviderError("Context budget exceeded. Increase SPECTRO_RAG_CONTEXT.")
        answer = providers.generate_document_answer(settings, system, user)
        ids = {h.chunk.chunk_id for h in hits}
        if any(cid not in ids for s in answer.statements for cid in s.chunk_ids):
            raise ProviderError("Model returned a citation outside retrieved excerpts")
    return RagResult(question=question, provider=settings.provider, answer=answer, sources=hits)


def render_answer(result: RagResult) -> str:
    import re
    from html import escape

    def safe(value: str) -> str:
        return re.sub(r"([\\`*_{}\[\]()#+|!>])", r"\\\1", escape(value))

    lines = [
        "# Document answer",
        "",
        safe(result.question),
        "",
        "Document retrieval only; scientific validity and customer suitability unverified.",
        "",
    ]
    if result.answer.status == "insufficient_evidence":
        lines.append("Insufficient evidence in the retrieved documents.")
    for statement in result.answer.statements:
        lines.append(safe(statement.text))
        lines.append("Sources: " + ", ".join(statement.chunk_ids))
        lines.append("")
    lines.extend(["", "## Retrieved excerpts", ""])
    for hit in result.sources:
        c = hit.chunk
        location = f"page {c.page}" if c.page else f"lines {c.line_start}-{c.line_end}"
        lines.extend(
            [f"### {safe(c.document)} — {location}", f"Chunk: {c.chunk_id}", "", safe(c.text), ""]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "search", "ask"):
        command = sub.add_parser(name)
        command.add_argument("--index", type=Path)
        if name == "index":
            command.add_argument("--documents", type=Path, default=Path("documents"))
            command.add_argument("--rebuild", action="store_true")
        else:
            command.add_argument("question")
    args = parser.parse_args()
    try:
        settings = RagSettings.from_env()
        index = args.index or Path(settings.index)
        if args.command == "index":
            print(json.dumps(index_documents(args.documents, index, settings, args.rebuild)))
        elif args.command == "search":
            print(
                json.dumps(
                    [h.model_dump(mode="json") for h in search(args.question, index, settings)],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(render_answer(ask(args.question, index, settings)))
        return 0
    except (ProviderError, OSError) as exc:
        print(f"RAG error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
