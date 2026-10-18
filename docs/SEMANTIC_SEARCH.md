# Unified assessment knowledge base

## Experimental complementary queries

The optional multi-query path extends this same engine; it does not replace grouped
retrieval or create another index. One generation request follows validated facts
and corrections, with an original-case anchor plus up to three complementary
questions. One snapshot spans retrieval and verification. Pooling prevents repeated
query hits from accumulating votes; final reranking, parent-context expansion,
source review and citation validation use shared code. Default remains disabled.

Query plans are technical diagnostics only, never customer facts or evidence.
Invalid/no-useful generation and optional deadline exhaustion visibly restore
grouped retrieval; incompatible indexes and provider failures remain errors.
See [multi-query protocol](MULTI_QUERY_RETRIEVAL.md) for evaluation and demo gates.

Implemented 2026-10-18. Spectroscopy assessment now retrieves from one SQLite
knowledge base using BM25 and persisted Ollama embeddings. Standalone Document Q&A
has been removed.

## Operation

Place authorized PDF, Markdown and TXT additions in `documents/`. Existing curated
AFS sources remain governed by `knowledge/sources.yaml`; their page review overrides,
source scope, document hashes and citation locators are preserved. Both inputs use
the same extraction/chunking/indexing pipeline and database. Evaluation/reference
answer fixtures remain excluded.

Click **Update Knowledge Base**. The app scans sources, detects document changes,
reuses unchanged chunk vectors, embeds new chunks, removes deleted entries, validates
a staging SQLite database and atomically replaces the committed index. Errors preserve
the previous database. The panel reports document changes, chunk counts, new/reused
embeddings, model and last successful update. Opening the app and submitting an
assessment never initiate document indexing.

SQLite stores chunks, embeddings and provenance. BM25 statistics are derived from
those same committed chunks; no independent lexical index exists. Requirements form
generic application, objective, setup and limitation queries. Query embeddings are
compared by exact cosine scan; rank fusion combines semantic and BM25 rankings.
Whole excerpts are deduplicated and bounded by the context budget. All duplicate
source provenance stays in SQLite; retrieval selects one source representative.

Each assessment pins one committed snapshot, including claim-specific searches and
citation checks. Reports record `knowledge_index_revision`; subsequent assessments
load the latest committed revision. The UI clears stale reports and exports when
knowledge revision or retrieval configuration changes. Claims still need valid
excerpt membership; a source quotation does not prove applicability or correctness.

Defaults and setup are documented in [README](../README.md). Schema/chunking/model
changes require an explicit update. Earlier independent indexes are rebuilt in place
only after successful staging; source files and historical reports are preserved.
Encrypted and scanned PDFs need external preprocessing. PDF extraction quality still
requires source review; local source PDFs emitted optional fontTools warnings during
the smoke run, and one page retained a poor-extraction review flag.

## Changed files

- Created: `tests/test_hybrid_knowledge.py`, `tests/test_knowledge_assessment.py`,
  and this implementation report.
- Modified: `app.py`; `src/spectro_ai/evidence.py`, `models.py`, `providers.py`,
  `rag.py`; `tests/test_app.py`, `test_evidence.py`, `test_rag.py`;
  `.env.example`, `README.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`,
  `docs/HACKATHON_PLAN.md`, `HISTORY.md`.
- Removed: `prompts/rag.md`, `src/spectro_ai/rag_models.py`,
  `tests/test_app_rag.py`; unused Q&A generation adapter and CLI workflow.

Pre-existing assessment/validation edits were preserved. Verification covers the
combined working tree; task commits deliberately leave those initial edits uncommitted.
Scientific acceptance tasks in `PLAN.md` remain open.

## Verification

`uv run pytest`: **121 passed**. `uv run ruff check .`: passed.
`uv run ruff format --check .`: passed (49 Python files).

An isolated staged checkout passed 110 tests and retained three failures already
present in untouched HEAD: unknown-citation status classification, a diagnostic
export assertion, and a numeric-plan expectation. Each was rerun against HEAD and
failed identically. Existing uncommitted edits address them in the combined workspace;
they were deliberately excluded from this task commit. The task commit therefore
must be used with those pending edits for the fully passing workspace result.

Network-free tests cover SQLite creation, incremental add/change/delete, vector
reuse, atomic embedding/replace failures, schema/model/chunk compatibility, snapshot
freshness, BM25/semantic fusion, paraphrase retrieval, context bounds, deduplication,
provenance corruption, supported-versus-unsupported claims, actual SQLite-to-assessor
integration, exports, and Streamlit update/rerun/failure behavior. Ollama is mocked.

Real local Ollama smoke used `qwen3-embedding:0.6b` and
`gemma4:12b-it-q4_K_M`, with an isolated database under ignored
`.cache/semantic-smoke/`. Seven authorized documents produced 197 chunks with
1,024-dimensional embeddings. A repeat update reused all 197 vectors and generated
zero new embeddings; an assertion prohibited embedding generation during that update.

Real D01 ran through extraction, hybrid retrieval, generation, validation and export.
The final run retained a provisional NIR/Raman comparison, removed unsupported
geometry/configuration and an invalid separate candidate, and retained five cited
excerpts from S01/S02. Six single-query embedding calls occurred during assessment;
document vectors were not regenerated. Results varied across smoke runs; no scientific
acceptance or performance guarantee follows. JSON/Markdown smoke artifacts remain in
ignored `.cache/semantic-smoke/`.

Streamlit AppTest and local HTTP startup were exercised outside the Windows sandbox,
which otherwise blocks asyncio's loopback socketpair. HTTP health returned 200/ok
and the temporary server stopped. Real browser clicking with live Ollama was not
separately exercised; AppTest covers the button with mocked indexing, and real Ollama
indexing/assessment were verified through the same Python services.
