"""Command-line wrappers for the assessment's unified knowledge index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectro_ai.evidence import EvidenceRetriever
from spectro_ai.models import CustomerUseCase
from spectro_ai.providers import RagSettings


def index_documents(
    documents: Path, index: Path, settings: RagSettings, rebuild: bool = False
) -> dict:
    """Compatibility wrapper; all indexing uses assessment SQLite schema."""
    retriever = EvidenceRetriever(index, settings=settings, documents_path=documents)
    return retriever.update_knowledge_base(force_rebuild=rebuild)


def search(question: str, index: Path, settings: RagSettings) -> list[dict]:
    """Search same source-cited hybrid index used by assessment."""
    retriever = EvidenceRetriever(index, settings=settings)
    return [
        item.model_dump(mode="json")
        for item in retriever.search(CustomerUseCase(source_text=question))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index", help="update assessment knowledge base")
    index_parser.add_argument("--documents", type=Path, default=Path("documents"))
    index_parser.add_argument("--index", type=Path)
    index_parser.add_argument("--rebuild", action="store_true")
    search_parser = commands.add_parser("search", help="search assessment knowledge base")
    search_parser.add_argument("question")
    search_parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    settings = RagSettings.from_env()
    index = args.index or Path(settings.index)
    if args.command == "index":
        result = index_documents(args.documents, index, settings, args.rebuild)
    else:
        result = search(args.question, index, settings)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
