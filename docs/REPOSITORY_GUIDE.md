# Repository guide for a new ChatGPT

## Purpose and current status

Spectro-AI is a provisional decision-support demo for ArtFiberSystems (AFS) spectroscopy experts. It takes a customer enquiry, extracts stated requirements with a configured LLM, searches an allowlisted local evidence index, drafts a technical assessment, applies deterministic checks, and presents an expert-reviewable report.

Treat this as a software prototype and assessment aid, not an autonomous recommender or validated scientific instrument. Source citations prove where text came from; they do not prove that the text applies to a customer's sample, instrument, process, or required performance. Hardware details, representative customer data, achievable limits, organizer confirmation, and expert acceptance remain unconfirmed. Preserve abstention when facts or evidence are inadequate.

## Repository map

| Path | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI, session state, example selection, correction controls, assessment execution, knowledge update, diagnostics and exports. Presentation layer only. |
| `src/spectro_ai/models.py` | Pydantic domain contracts: customer facts, provenance, source locations/blocks, retrieved evidence, candidates, feasibility plan, diagnostics and report. No provider/UI imports. |
| `src/spectro_ai/providers.py` | Ollama and OpenAI adapters; extraction, optional evidence-query planning, embeddings, assessment orchestration, context fitting, validation, feasibility safeguards and neutral abstention. Provider errors do not silently become recommendations. |
| `src/spectro_ai/evidence.py` | Authorized-source ingestion and SQLite index; source metadata, passage/chunk and block/section records, embeddings, hybrid retrieval, rank fusion, optional reranking and bounded same-source context. Also CLI reindex entry point. |
| `src/spectro_ai/structure.py` | Format-aware source units and passage packing. Preserves source order and structural locations for Markdown, TXT, DOCX, XLSX, and PDF paths. |
| `src/spectro_ai/guardrails.py` | Deterministic missing-field prioritization, technical questions and evidence-insufficiency abstention. No technique recommender. |
| `src/spectro_ai/intake.py` | Validates expert edits and tracks edited-field provenance; clears stale quote for changed value. |
| `src/spectro_ai/knowledge.py` | `KnowledgeSource` contract and validation/loading of curated technique YAML. Profiles are background knowledge, not executable recommendation rules. |
| `src/spectro_ai/reporting.py` | Stable Markdown rendering. Customer report is default; diagnostics are opt-in. JSON export comes from validated Pydantic report. |
| `src/spectro_ai/rag.py` | Compatibility CLI for the unified index; not a separate question-answering product or store. |
| `prompts/` | Shared extraction, assessment, experiment-planning, and retrieval-query prompts, included in package data. |
| `knowledge/techniques.yaml` | Curated technique profiles and evidence provenance. All require expert review; do not turn generic profiles into customer specifications. |
| `knowledge/sources.yaml` | Source allowlist, citation templates, scope/review metadata, exclusions and page-level review flags. |
| `knowledge/data/materials/` | Local source materials. Many are ignored by Git; inspect actual local files before assuming a clean checkout has them. |
| `docs_schubKI/` | Challenge/event and source material provided for local work; not application runtime code. |
| `examples/scenarios.json` | Four synthetic enquiries for UI/demo. These still use configured LLM and evidence pipeline. |
| `examples/evaluation.json` | Evaluation cases for development, paraphrase, adversarial and holdout datasets. |
| `examples/retrieval_benchmark.json` | Retrieval benchmark fixture. Relevance labels are provisional until explicitly human-reviewed and bound to exact index evidence identities. |
| `scripts/` | Evaluation runner, retrieval benchmark, setup/smoke and structure-review utilities. |
| `tests/` | Offline tests with injected provider doubles. Tests do not establish scientific suitability or replace expert/source review. |
| `docs/` | Architecture, decisions, hackathon/demo strategy, source/retrieval reviews, evaluation protocols and known blockers. |
| `PLAN.md` | Executable backlog and gates. Read it before implementation; do not silently treat blocked acceptance gates as completed. |
| `HISTORY.md` | Accepted work history. Planned-task acceptance requires exact original task text, date, implementation, verification and deviations recorded here. |
| `AGENTS.md` | Repository-specific engineering, delegation, verification, planning and commit rules. Follow it alongside the user's explicit task. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python 3.14 project/dependency configuration and locked environment. `.tools` bootstraps uv; `.venv` is project environment. |
| `.github/workflows/checks.yml` | CI checks. `.streamlit/config.toml` | Streamlit runtime configuration. `.env.example` | Documented environment variables; `.env` is not auto-loaded. |

## Runtime data flow

1. User enters enquiry or selects one of four synthetic examples in Streamlit.
2. `assess_problem` in `providers.py` runs schema-constrained extraction using the explicitly selected provider. Each extracted fact should retain a literal customer quote; expert corrections have separate provenance.
3. `assess_use_case` uses a pinned snapshot of the SQLite index. Requirement-grounded query groups search BM25 and (when indexed/configured) local Ollama embeddings. Reciprocal Rank Fusion combines ranks; raw semantic/BM25 scores are not added and none is confidence.
4. Optional candidate reranking or multi-query planning is opt-in. Multi-query is disabled by default and remains an unproven retrieval experiment. Parent context expansion is also disabled by default.
5. The selected excerpts and full enquiry are sent to the same configured provider for structured technical assessment. Whole optional excerpts/context blocks may be dropped to fit provider budget; the complete customer enquiry is retained. Structural output repair is bounded to one attempt.
6. Deterministic validators check fact/quote membership, citation IDs and exact quotation membership, source scope/review metadata, claim support and feasibility safeguards. These checks are screening controls, not semantic proof.
7. Unsupported individual claims can be removed; remaining output may be provisional. If critical facts or suitable evidence are missing, report is explicit abstention with questions/next study actions.
8. User reviews/corrects requirements, candidate evidence, limitations, diagnostics and plan. Markdown is customer-facing; JSON and technical diagnostics expose audit details. Input/config/index changes hide stale reports and exports.

Index refresh is explicit only (button or CLI), never an app rerun or assessment side effect. It stages and validates a new SQLite DB before atomic replacement. Failed updates preserve prior index; an assessment pins one snapshot and records revision. Adding/removing a file does not change retrieval until successful update.

## Key domain rules

- Unknown scalar values are `None`; missing collections are empty lists. Do not fill gaps with plausible-sounding values.
- Keep customer-provided units and ranges as text. Never convert or infer absent units/specifications.
- Separate source-backed statements, engineering heuristics and assumptions. A verified source URL still does not validate a customer recommendation.
- A source demonstration is bounded to its studied sample/application. Do not transfer its performance or geometry to a new customer scenario without explicit evidence and expert review.
- Acceptance thresholds must come from customer; otherwise TBD. Plans should use independent reference assays and representative/held-out validation. No confidence number is invented.
- Operational failure (provider/index/invalid output) is an error. Scientific insufficiency is a neutral abstention. There is no offline extraction/recommender and no fallback provider.
- `production_readiness` intentionally remains `not_assessed` or `not_established`.

## Data and evidence lifecycle

`knowledge/sources.yaml` lists manifest sources and provenance. Authorized extra PDFs, Markdown and TXT can be placed in configured `documents/`; source files and document evaluations are not model-generated corpus. Ingestion stores document versions/hashes, source scope/review state, precise locations and structured blocks; schema 3 also records blocks/sections and table relationships while older schema 2 remains readable. PDF defaults to `pypdf`; optional Docling runs only when explicitly selected and local assets are available. It never silently falls back. Scanned PDFs need OCR outside this default path.

Default index is `.cache/rag/index.sqlite3` (override with `SPECTRO_RAG_INDEX`). Semantic retrieval uses local Ollama embeddings even if OpenAI is selected for assessment. Exact cosine scan is intended for this small corpus. Similarity is a retrieval score, not confidence. Indexing requires explicit `Update Knowledge Base` or `python -m spectro_ai.evidence reindex`.

## Configuration and setup

See README for full Windows commands. Important choices:

- `SPECTRO_AI_PROVIDER=ollama|openai` selects extraction and assessment; Ollama is default. No automatic provider switch.
- Ollama defaults: `OLLAMA_BASE_URL=http://127.0.0.1:11434`, model `gemma4:12b-it-q4_K_M`, embedding model `qwen3-embedding:0.6b`.
- OpenAI requires explicit `SPECTRO_AI_PROVIDER=openai`, model and API key. Customer text, extracted facts and selected excerpts are sent to configured endpoint. A key being present does not authorize a live paid request.
- `SPECTRO_AI_CONTEXT`, `SPECTRO_AI_TIMEOUT`, `SPECTRO_AI_OUTPUT_TOKENS` set assessment budgets (defaults in README).
- `SPECTRO_RAG_*` configures index path, retrieval mode/count, reranker, parent expansion and experimental multi-query. Keep multi-query, reranker and parent expansion off unless task explicitly concerns their evaluation.
- Dependencies are managed with uv and `uv.lock`; optional extras are `rag`, `llm`, and `rerank`.

## Tests and interpretation

Tests live under `tests/`; `tests/conftest.py` removes provider selection/credential variables and fixtures inject doubles, so ordinary suite should need no model/network credentials. AppTest coverage lives in `test_app.py`. Useful checks: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run python scripts/smoke_check.py`. Live provider and evaluation commands are separate, explicit operations and may require local model/index or paid endpoint configuration.

Read results carefully: a green test suite means tested software contracts passed. It does not establish retrieval relevance, scientific correctness, AFS acceptance or customer performance. Evaluation reports are under `docs/`; review the relevant gate docs before citing results.

## Current known blockers and boundaries

- Phase 3 semantic acceptance is open. Prior D01/D02 outputs transferred application evidence, overstated fusion benefits, or treated source dimensions as customer choices; focused repairs did not close expert review. See `PLAN.md`, `docs/LLM_EVALUATION.md`, and `docs/ASSESSMENT_REVIEW.md`.
- Docling did not pass real-source inspection: S01 p.6 table value association and inferred heading nesting remain incorrect/ambiguous after two focused attempts. Keep pypdf default; Docling is explicit experimental option. See `docs/STRUCTURE_CONTEXT_REVIEW.md` and backlog.
- Parent-context expansion stays disabled. Reproducible real-source BM25 comparison missed expected operating-condition and probe-limitation passages; default provider fitting retained no attached context in those cases. No usefulness/default-promotion claim.
- Multi-query retrieval stays disabled pending human-reviewed index-bound relevance labels, expert usefulness review, fair comparisons and no safety/provenance regression. Runtime completion alone proves no quality gain.
- OpenAI live smoke is not run unless model/credentials are explicitly supplied and request authorized. Demo rehearsal and scientific acceptance remain human gates.
- Challenge documentation is provisional; organizer confirmation, actual hardware, measured customer data, and achievable performance are not established.

## Safe starting point for another ChatGPT

First read `AGENTS.md`, `PLAN.md`, this guide, then relevant architecture/decision/source-review docs. Inspect git status before edits and preserve unrelated changes. Make small scoped changes, avoid inventing domain facts, keep provider calls out of tests, and update `README.md`, `docs/DECISIONS.md`, or `docs/HACKATHON_PLAN.md` when their stated conditions apply. For a planned milestone, record accepted exact task text in `HISTORY.md` before removing it from `PLAN.md`. Never mark human/scientific gates complete from software checks alone.

