# Spectro-AI

Provisional spectroscopy decision support for AFS experts. Customer enquiry → LLM
requirements extraction → local evidence retrieval → LLM technical assessment →
deterministic validation → expert-reviewable report. Scientific suitability requires
expert and sample validation; citations and software tests do not establish it.

## Setup (Windows PowerShell)

Use CPython 3.14 and uv. `.tools` holds uv; `.venv` holds project dependencies.

```powershell
py -3.14 -m venv .tools
.\.tools\Scripts\python.exe -m pip install uv==0.13.1
$env:Path = "$PWD\.tools\Scripts;$env:Path"
$env:UV_CACHE_DIR = Join-Path $PWD '.cache\uv'
$env:UV_PYTHON_DOWNLOADS = 'never'
uv sync --locked --extra rag
ollama pull gemma4:12b-it-q4_K_M
ollama pull qwen3-embedding:0.6b
uv run python -m spectro_ai.evidence reindex
uv run streamlit run app.py
```

Before indexing, place authorized AFS files listed in `knowledge/sources.yaml` under
`knowledge/data/materials/`. These ignored source files are not bundled in Git.
Add additional authorized PDF, Markdown or TXT files under ignored `documents/`.
Both inputs feed one SQLite knowledge base. Missing prerequisites are reported explicitly.

Ollama must be running at `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`).
`SPECTRO_AI_PROVIDER` defaults to `ollama`; `OLLAMA_MODEL` selects installed model.
The four synthetic examples populate customer text; they require the same LLM pipeline
as custom input. No offline parser, deterministic recommender, or provider fallback
exists. Unavailable models, invalid output, and missing indexes produce visible errors.
Missing facts or technical evidence produce explicit abstention.

## OpenAI assessment

```powershell
uv sync --locked --extra rag --extra llm
$env:SPECTRO_AI_PROVIDER = 'openai'
$env:OPENAI_MODEL = '<structured-output-capable-model>'
# Set OPENAI_API_KEY securely in current process. Never commit credentials.
uv run streamlit run app.py
```

`OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`. Assessment requires Chat
Completions structured JSON-schema output support. Customer text, extracted facts,
and selected source excerpts go to configured endpoint. Model, credentials, and SDK
are required explicitly. No automatic switch to another provider or paid service.
See [official structured-output documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).
`.env.example` lists settings; `.env` is not automatically loaded.
`SPECTRO_AI_CONTEXT`, `SPECTRO_AI_TIMEOUT`, and `SPECTRO_AI_OUTPUT_TOKENS` control
assessment budgets (defaults: 16,384 context, 240 seconds per request, 4,096 output tokens). Complete enquiry is retained; evidence may be dropped only as
whole excerpts to fit context. Structural repair is bounded to one attempt.

## Review and export

The **Prompts and queries** tab shows the latest run's actual outgoing LLM
messages, retries, schemas, generation settings, retrieval queries, and source
excerpts sent after context fitting. Failed attempts remain inspectable. Traces
stay in browser-session memory, outside Markdown/JSON report exports; new runs
replace them and changed inputs, settings, or knowledge revisions clear them.
Prompt text includes the customer enquiry and source excerpts. Credentials and
embedding requests are excluded.

Enter natural-language enquiry or choose synthetic example. Inspect facts and exact
customer quotes, missing facts, candidates, evidence, validation diagnostics, and
feasibility plan. Correct facts in expert-review panel and apply corrections to rerun
retrieval/assessment without re-extraction. Download Markdown and JSON. Changed input,
configuration, or pending corrections hide stale exports.

Unknown scalars stay `None`; missing collections stay empty. Customer units remain
text. Source demonstrations are not customer performance specifications. Acceptance
criteria come from customer or remain TBD. Independent reference assays and held-out
validation remain necessary. No numerical confidence is invented.

## Architecture

- `models.py`: typed Pydantic domain contracts, provenance, validation diagnostics.
- `providers.py`: Ollama/OpenAI extraction and technical assessment adapters and pipeline.
- `evidence.py`: shared ingestion, SQLite storage, BM25 + semantic retrieval and
  rank fusion, retaining pages, excerpts, hashes, scope and review flags.
- `rag.py`: compatibility CLI for indexing/searching the same knowledge base.
- `intake.py`: validated expert corrections. `reporting.py`: stable Markdown export.
- `prompts/`: shared versioned prompts. `knowledge/techniques.yaml`: curated profiles;
  profiles do not execute recommendation rules.
- `app.py`: Streamlit presentation; tests use injected provider doubles without network.

Legacy `baseline` provider, parser APIs and `allow_offline_fallback` argument were
intentionally removed. Existing historical reports are records, not new pipeline
outputs. No data migration or deletion is performed.

## Checks and evaluation

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python scripts/smoke_check.py
uv run python scripts/evaluate.py --help
```

Run actual model evaluations explicitly after configuring provider:

```powershell
uv run python scripts/evaluate.py --dataset development
uv run python scripts/evaluate.py --dataset paraphrase
uv run python scripts/evaluate.py --dataset adversarial
uv run python scripts/evaluate.py --dataset holdout
```

`--dataset all` includes four original synthetic inputs, D01–D08, P01–P08,
A01–A05 and H01–H06. Official documents must exist under ignored
`knowledge/data/materials/`; missing datasets fail clearly. `--cases PATH` accepts
custom JSON cases. `--output PATH` selects output root; dataset gets its own folder.
Per-case failures are recorded and batch continues; nonzero exit signals any failure.

Evaluation uses same assessment service as UI and saves generated JSON/Markdown
without manual answer editing. Tests are network-free; live evaluation requires
configured model and evidence index. D01/D02 scientific acceptance remains open:
prior runs overstated fusion benefits, transferred application evidence, and treated
source pathlength as customer configuration. New guardrails do not replace expert
review. H01–H06 must use identical pipeline without per-case prompt/rule tuning.

## Unified knowledge base

Spectroscopy assessment is the only question-to-report workflow. Standalone Document
Q&A, answer models and prompt are removed. Local Ollama embeddings remain independent
of the configured Ollama/OpenAI assessment generator.

Use **Update Knowledge Base** in Streamlit after adding, changing or removing files.
The panel shows readiness, document/chunk counts, embedding model and last successful
update. Indexing occurs only on explicit update, never on app rerun or assessment.
New/changed chunks receive embeddings; unchanged chunks reuse stored vectors. Updates
validate a staging SQLite database before atomic replacement. Failures preserve the
last committed index. Reports record the exact index revision; changes hide stale
reports and exports. Existing assessments use a pinned snapshot through validation.

```powershell
ollama pull qwen3-embedding:0.6b
uv run python -m spectro_ai.evidence reindex
uv run python -m spectro_ai.rag search "Which limitations are documented?"
```

Default database: `.cache/rag/index.sqlite3` (`SPECTRO_RAG_INDEX`). Old independent
JSON/Document Q&A indexes are incompatible; rebuild with the update button or CLI.
Source files and historical reports are preserved. No vector database or extra
framework is needed. Exact cosine scan suits the small local corpus; similarity
is a retrieval score, never scientific confidence. Scanned PDFs need external OCR.

Requirements produce targeted queries for application, objective, setup and limitations,
without scenario-specific technique rules. BM25 and semantic rankings combine with
Reciprocal Rank Fusion; raw scores are never added. Whole excerpts fit the retrieval
budget, then the existing provider context fitter bounds the complete assessment.
A retrieved source still requires claim membership checks and expert scientific review.

Initial retrieval defaults: 30 BM25 candidates, 30 semantic candidates, 8 final
excerpts, RRF constant 60, and 12,000 UTF-8 bytes of excerpt context. Candidate counts,
RRF and context budget can be supplied to `EvidenceRetriever`; final assessment count
uses `SPECTRO_RAG_EVIDENCE_COUNT`. Chunking defaults to 1,200 characters with 180
characters of overlap (`SPECTRO_RAG_CHUNK_SIZE`, `SPECTRO_RAG_CHUNK_OVERLAP`). Changing
chunking requires an explicit knowledge-base update. New updates write index schema
3 with source block/section relationships; existing schema 2 indexes remain readable.
Updates validate a staged database before atomic replacement, so failures preserve
the previous usable index and running assessments retain their pinned snapshot.
Parser/chunking changes trigger extraction again, but vectors can be reused when
the exact embedding input and embedding configuration are unchanged. Model/config
changes or invalid cached vectors prevent unsafe reuse.
These defaults are engineering choices, not measured scientific optima.

### Structure-aware ingestion

Markdown ingestion preserves heading ancestry, paragraphs, lists, fenced blocks,
table rows and explicit headers. Index schema 3 records source blocks separately
from packed retrieval passages, including exact source spans and locations.
Titles and heading paths remain metadata; embedding input is still passage text.

TXT uses paragraphs with original line ranges. DOCX preserves heading ancestry,
paragraphs, table/cell positions and explicitly marked repeating headers. XLSX
preserves worksheet row numbers and empty cell positions; only declared table
headers are repeated. Cached cell values are read without formula evaluation or
number-format rendering. Unit strings in source values remain text; missing units
and header meanings are never inferred. Curated YAML profiles stay structured.
Normal updates retain existing evidence IDs only for matching passage text and
citation provenance; changed passages receive content/location-bound IDs.

The PDF default remains `pypdf`. Docling 2.140+ is available explicitly through
the `rag` extra, with CPU layout/table models and validated heading inference:

```powershell
uv sync --extra rag
uv run --extra rag python scripts/setup_docling.py
$env:SPECTRO_PDF_BACKEND = "docling"
```

Then explicitly update the knowledge base. Normal conversion uses local model
assets, with OCR and remote services disabled; missing assets or conversion
failures preserve the prior index and do not switch parsers silently. Set
`SPECTRO_PDF_BACKEND=pypdf` and update again to select the existing parser.

Docling has **not passed the real AFS source gate**: S01 p.6 has an incorrect
table-cell association after two configuration attempts. Detected heading nesting
also needs validation; invalid nesting uses flat page-local sections. Do not
treat extracted specifications as approved. See
[source-review findings](docs/STRUCTURE_CONTEXT_REVIEW.md).

Parent expansion is disabled by default. To try it after an explicit knowledge
base update, set `SPECTRO_RAG_PARENT_EXPANSION=true`; set it to `false` to retain
existing child-only selection. Expansion reserves 30% of retrieval byte capacity
before choosing children, then attaches complete source blocks from the same
validated section. Oversized sections use source-order windows; unsafe hierarchy
suppresses expansion. Unused reserve can admit more ranked children.

Context stays nested under its child evidence and never counts as another
corroborating source. Quoting a context block requires its exact block ID and
source locations. Retrieval charges serialized UTF-8 context; provider fitting
drops whole optional context blocks before children. Diagnostics show omissions.
Real AFS usefulness and any default-promotion decision remain pending.

Run a comparison on an explicitly prepared index without generation requests:

```powershell
uv run python scripts/review_structure_context.py --index .cache/rag/index.sqlite3 --output .cache/structure-context/review.json
```

This pins one snapshot, compares BM25 expansion off/on, and records source hashes,
child/context locations, context volume and warm latency. Relevance metrics remain
unset: agent inspection is not human-reviewed retrieval labeling.

### Optional local reranker (English trial)

Default retrieval is unchanged. To try CPU cross-encoder reranking, install the
optional extra and explicitly download the model before starting the app:

```powershell
uv sync --extra rag --extra rerank
uv run python -m spectro_ai.providers setup-reranker
$env:SPECTRO_RAG_RERANKER = "cross_encoder"
uv run streamlit run app.py
```

Local uv bootstrap: replace `uv` with `.tools/Scripts/uv.exe` if needed. In restricted
environments, set `$env:UV_CACHE_DIR = Join-Path $PWD '.cache/uv'` first.
Assessment loads cached model files only, with remote code disabled. Model setup
downloads public weights; no paid inference service is used. The trial model is
`cross-encoder/ms-marco-MiniLM-L6-v2`, an English MS MARCO passage-ranking model,
not a validated spectroscopy model. Python 3.14 remains required.

Configure `SPECTRO_RAG_RERANKER_MODEL`, `SPECTRO_RAG_RERANKER_REVISION` (default `main`)
and `SPECTRO_RAG_RERANK_CANDIDATES` (default 30) as needed; setup records the resolved
model revision. Use an immutable revision to reproduce trials. Set
`SPECTRO_RAG_RERANKER=none` to restore default retrieval. No index rebuild is needed.

Enabled retrieval collects candidates before context truncation, combines existing
query rankings, then reranks the top 30 distinct passages once. Final selection
retains whole excerpts within existing count and byte budgets. Model inputs use
at most 512 tokens, including at most 128 query tokens; citations retain original
full excerpts. `retrieval_score` remains the RRF score; nullable `rerank_score`
records relevance only. Neither score is scientific confidence.

If the optional model cannot load or score candidates, retrieval uses that candidate
pool's RRF order and records the reason in Technical diagnostics and JSON. Index,
embedding and assessment failures remain errors. Customer Markdown omits ranking
diagnostics. Configuration changes invalidate stale reports and exports.

Retrieval benchmark questions live outside the indexed corpus. Human review must
bind relevance labels to exact evidence IDs and document versions; unreviewed
fixtures are provisional and cannot establish quality gains. The benchmark compares
original retrieval, widened RRF and reranking on one pinned index, with Precision@5,
Recall@10, MRR, candidate recall and separate model-load/warm-search timings.
Default promotion and scientific acceptance require separate review.

After model setup, run the cached-model benchmark without generation requests:

```powershell
uv run python scripts/benchmark_retrieval.py --output .cache/retrieval-benchmark.json
```

Add `--bm25` to avoid Ollama embeddings and compare lexical retrieval variants;
this still runs the local reranker. Default warm repetitions: 3. JSON reports
the first model-load-and-score duration separately from warm timings for each mode.
No relevance labels ship with `examples/retrieval_benchmark.json`. After reviewing
every question, set its top-level `status` to `human_reviewed` and populate each
case's `relevant` list with records containing `evidence_id`, `source_id`,
`source_version` (indexed `document_version`) and `content_sha256`, copied from the
pinned index. An empty reviewed list means the reviewer confirmed no relevant
evidence. Stale or incomplete identities are rejected. Unreviewed precision, recall
and MRR remain `null`; runtime measurements alone establish no retrieval benefit.

See [architecture](docs/ARCHITECTURE.md), [decisions](docs/DECISIONS.md),
[hackathon plan](docs/HACKATHON_PLAN.md), and [pending backlog](PLAN.md).

### Experimental complementary evidence questions

Multi-query retrieval is disabled by default. It makes one additional structured
generation request after requirements extraction or expert corrections, then searches
the existing knowledge index with the original-case anchor and up to three focused
evidence questions. No additional dependency or index rebuild is needed.

```powershell
$env:SPECTRO_RAG_MULTI_QUERY = "true"
$env:SPECTRO_RAG_MULTI_QUERY_MAX_QUERIES = "4"  # includes original-case anchor; 2–4
$env:SPECTRO_RAG_MULTI_QUERY_TIMEOUT = "30"
$env:SPECTRO_RAG_MULTI_QUERY_SEARCH_TIMEOUT = "30"
uv run streamlit run app.py
```

Generation uses the explicitly selected assessment provider/model. Ollama works
locally; OpenAI requires explicit configuration and sends the same customer-case
data for this additional request. No provider switching or structural retry occurs.
Set `SPECTRO_RAG_MULTI_QUERY=false` to restore unchanged grouped retrieval.

Generated questions are exploratory search hypotheses, never customer requirements
or scientific evidence. Invalid, duplicate-only or empty plans fall back visibly to
existing grouped retrieval. Optional search deadline exhaustion also discards the
partial pool before baseline retrieval; embedding/index/provider failures stay errors.
Request budgets do not impose a hard total assessment deadline: baseline fallback,
CPU reranking and assessment have their own runtimes.

Candidate pooling avoids repeated-hit vote inflation, caps the consolidated pool
using `SPECTRO_RAG_RERANK_CANDIDATES` (30 by default), and reranks once when enabled.
Existing evidence-count, byte, parent-context and provider-fitting budgets remain.
Technical diagnostics and JSON retain accepted questions and retrieval membership;
customer Markdown omits query planning. Configuration changes invalidate reports.

Quality improvement is unproven. Keep this mode out of the demonstration until
human-reviewed, index-bound comparisons show coverage/Recall@8 gains without safety,
Precision@5 or MRR regressions. See [evaluation protocol](docs/MULTI_QUERY_RETRIEVAL.md).

Run the comparative benchmark with explicit local query generation:

```powershell
$env:SPECTRO_AI_PROVIDER = "ollama"
uv run python scripts/benchmark_retrieval.py --multi-query --output .cache/multi-query-benchmark.json
```

Use `--bm25 --no-reranker --repeats 1` for a lighter runtime smoke. `--multi-query`
still makes local generation requests; omitting it makes none. The benchmark CLI
rejects cloud generation. Review `validated_requirements` in synthetic fixtures and
label exact indexed passages before interpreting quality metrics. These manually
specified facts evaluate retrieval/planning, not extraction accuracy.

Current local run: [31-case LLM evaluation](docs/LLM_EVALUATION.md).

Assessment presents a customer report and a separate Technical diagnostics tab.
Markdown downloads contain the customer report; structured JSON retains the audit
trail. Python callers can use `render_markdown(report, include_diagnostics=True)`
for the full technical export. Source observations retain document/page/version
references; internal chunk IDs and extraction logs stay in diagnostics.

Claim validation removes unsupported statements individually. Retained source
observations justify an exploratory comparison only, not customer performance or
production readiness. Missing specifications become prioritized study questions.
Claim-specific searches use the same allowlisted, version-checked evidence index;
additional folder documents enter through Update Knowledge Base with unreviewed metadata;
curated AFS provenance remains in the manifest.

### Application-aware evidence assessment

Assessment now evaluates source observations and their application conditions before
technical synthesis. Reports distinguish application-supported consideration,
scientifically reasoned exploration, and unsupported suggestions. Novel applications
can be explored through an explicit source-backed mechanism, customer-linked
hypothesis, transfer gaps and next checks; exact application precedent is not required.
Exploratory consideration does not automatically select a measurement technique for
the experiment. Ranking and experiment-selection improvements remain Step 2.

The evidence pass uses the selected Ollama/OpenAI provider and existing context,
output and timeout settings. It adds one request plus at most one structural retry;
optional claim repair can add one augmented evidence assessment. Whole source bundles
retain available qualifications. Parent expansion stays off by default: missing
surrounding context remains unavailable, never a confirmed application match.
Successful augmented assessment rebuilds synthesis once against its new IDs; repair
does not recurse. This is additional local runtime cost, not a quality guarantee.

Exact quotes and linked observation IDs establish traceability, not scientific truth.
Semantic entailment and transfer reasoning are model-assessed and need expert review.
Unreviewed technical sources remain usable provisionally with visible warnings;
source review status is independent of application relevance. Old report JSON remains
readable without retrospective evidence assessment; new reports include the assessment
audit trail. Index schema and retrieval interfaces are unchanged; no reindex is needed.
See [Step 1 plan and acceptance](PLAN_EVIDENCE_ASSESSMENT.md).

Custom Python assessors must accept the keyword-only `evidence_assessment` argument
on `assess(use_case, evidence, *, evidence_assessment)`. Tests may inject an
`evidence_assessor(use_case, evidence, *, allow_retry=True)` callable into
`assess_use_case`; its output still passes host reference and permission checks.
There is no legacy lexical bypass when that pass fails. Review protocol and source
boundaries are in [evidence assessment](docs/EVIDENCE_ASSESSMENT.md).

Scientific acceptance remains blocked: the local S02 smoke still missed the polymer
qualification. The generic permission guard withheld the resulting support record
and left the experiment technique unset. Passing software tests does not establish
correct application interpretation; further model tuning stopped after two focused
repairs. Details and runtime evidence are in the review protocol above.
