# Spectro-AI

A small SchubKI hackathon foundation for spectroscopy use-case assessment. Enter a
customer problem, inspect requirements and gaps, explore provisional techniques,
and export a feasibility outline as Markdown.

**Rules-and-templates demo, not validated scientific analysis.** The supplied brief
lists the challenge and judging criteria, and AFS sources plus development and holdout
enquiries are present locally. Organizer confirmation, hardware specs, measured
customer data, and achievable performance remain unconfirmed. No API key is needed
for the default assessment; baseline makes no network requests. Streamlit usage
telemetry is disabled.

## Quick start — Windows PowerShell

Requirements: installed CPython 3.14 and internet for initial setup. Python 3.14
replaces the initial 3.12 target. Initialize Git if working from a source archive.
From this checkout:

```powershell
Set-Location <repository-directory>
New-Item -ItemType Directory -Force .cache\tmp | Out-Null
$env:TEMP = Join-Path $PWD '.cache\tmp'
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $PWD '.cache\uv'
$env:UV_PYTHON_DOWNLOADS = 'never'
py -3.14 -m venv .tools
.\.tools\Scripts\python.exe -m pip install uv==0.13.1
$env:Path = "$PWD\.tools\Scripts;$env:Path"
uv sync --locked
uv run streamlit run app.py
```

Open [the local demo](http://127.0.0.1:8501). Stop with Ctrl+C. No environment
activation, global package installation, or execution-policy change needed. `.tools`
contains uv; `.venv` contains the project. A fresh terminal needs the environment
assignments above again, or use `.\.tools\Scripts\uv.exe` directly. Other platforms
with Python 3.14 and uv installed use the same `uv` commands from the repo root.

## Demo walkthrough

1. Select **Synthetic: Food moisture**, then **Assess use case**.
2. Inspect requirements, gaps, NIR screening rationale, evidence, and experiment.
3. Download Markdown; the **Structured report** tab shows validated JSON.
4. Try polymer identification, fluorescent dye, then the underspecified case:
   it must abstain and ask questions.
5. Choose **Custom input** for a use case. Editing hides stale output until
   another assessment is run.
6. Expand **Expert review and follow-up answers**. Edit facts or answer gaps, then
   **Apply expert corrections**. Pending edits hide exports; applied edits recompute
   questions, candidate conditions, and experiments while preserving original text.

See [DEMO_PLAN.md](docs/DEMO_PLAN.md) for the incomplete-input → ethanol monitoring →
fluorescence-background sequence, recovery, and offline fallback. Six separate
synthetic evaluations and review limits are in
[EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md); reproduce full exports with
`uv run python scripts/evaluate.py`. [REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md)
records capability evidence and remaining external dependencies.

Plain text recognizes only literal phrases listed in `baseline.py`. For reliable
entry use `Label: value`, one per line or separated with semicolons:

```text
Industry: food processing
Objective: measure moisture
Target: moisture
Matrix: food powder
Concentration range: unknown
Temperature: 20-30 C
Pressure: ambient
State: powder
Accuracy: TBD
Frequency: once per batch
Constraints: laboratory screening
```

Additional labels: `Process`, `Composition`, `Detection limit`, `Deployment mode`,
`Interference`. Field provenance distinguishes customer labels, literal phrases and
expert edits. The bounded phrase set includes `monitor ethanol` in
`aqueous fermentation broth`; quantities require labels or expert input.

Objective, target, and matrix are required for candidates. Other unknowns stay visible;
questions prioritize consequential gaps. Quantities retain supplied units; no conversion or unlabeled
numeric extraction occurs. Blank values, `unknown`, `TBD`, and `N/A` become null;
zero stays a valid supplied value. Negated clauses are excluded from phrase matching,
which can deliberately miss positive facts in those clauses. Repeated labels use the
last value. Explicit labels, including unknowns, override phrase matches.

Screening is unranked and conditional: matrix, reported fluorescence background,
requested detection limit, explicit physical state and access constraints change
conditions and study geometry. Experiments include controls, held-out validation,
thresholds to agree, and both outcome branches. No AFS hardware or customer performance
is validated; a domain expert must review rules and actual samples.

## Architecture and repository

```text
app.py                         Streamlit presentation and session state
src/spectro_ai/
  models.py                    Pydantic domain types and evidence provenance
  baseline.py                  Literal extraction, gaps, rules, experiment templates
  knowledge.py                 KnowledgeSource protocol and validated YAML loader
  providers.py                 Baseline selection and document model adapters
  reporting.py                 Deterministic Markdown export
knowledge/techniques.yaml      Five preliminary profiles with checked sources
examples/                     Four labeled synthetic scenarios
prompts/                      Packaged prompts; baseline does not execute them
tests/                        Domain, rules, knowledge, providers, export, UI checks
docs/                         Challenge, architecture, decisions, evaluation, plan
.github/workflows/checks.yml   Locked install, tests, lint, format checks
AGENTS.md                      Guidance for future engineering sessions
pyproject.toml / uv.lock       Dependencies and reproducible resolution
```

Text → `extract_use_case` → `CustomerUseCase` → `assess` → `AssessmentReport` →
Streamlit/`render_markdown`. Core modules do not import Streamlit or an LLM SDK.
YAML knowledge also ships as package data in built wheels.

Bulk-property phrases map to NIR; molecular identification to Raman/MIR; absorption
or color to UV-Vis; explicit fluorescence to fluorescence. Candidates are unranked
hypotheses. Hardware, performance, detection limits, geometry, probe configuration,
and exact wavelengths remain unvalidated/undetermined.

## Checks

```powershell
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python scripts/smoke_check.py
```

Run after relevant changes. `uv run ruff format .` applies formatting. Tests include
Streamlit AppTest; the smoke script starts a temporary local server, checks its HTTP
health endpoint, and stops it. If sandbox temp
restrictions affect pytest, set TEMP/TMP above and use
`uv run pytest --basetemp=.cache/pytest-tmp`. GitHub Actions targets Windows and
Python 3.14; remote workflow execution requires pushing to GitHub.

Original baseline verified on Windows / Python 3.14.4: 44 pytest cases, Ruff lint/format, local server
HTTP 200/ok, optional SDK installation/import, and installed wheel knowledge/assessment.
The minimal environment was restored after the optional SDK check. Restricted
execution sandboxes may require local socket access for AppTest and server checks;
they use Windows asyncio loopback sockets even without external requests.

## Optional local extraction

Set `SPECTRO_AI_PROVIDER=ollama` to extract structured customer facts with the local
Ollama model configured by `OLLAMA_MODEL`. Each returned fact needs an exact customer
quote; missing or nonliteral quotes clear the field and add a warning. Numeric values
must remain exact text from their quoted span. Quote membership does not prove correct
interpretation. If Ollama is unavailable or returns invalid extraction output, the app
uses the conservative offline extractor and marks `extraction_mode=offline` with a
warning. Ollama mode then retrieves allowlisted, source-versioned AFS excerpts and
requests a provisional assessment tied to those excerpt IDs. The service checks IDs
and source membership, rejects business-only support, and applies a conservative
lexical relevance screen. This screen does not establish entailment or scientific
suitability. Missing, unrelated, or unreviewed evidence remains visible and may cause
abstention or a limited deterministic fallback. Expert review is required. Applying
expert corrections reruns the same provider and retrieval service using the corrected
case without re-extraction. Set `SPECTRO_AI_CONTEXT`, `SPECTRO_AI_TIMEOUT`, and
`SPECTRO_AI_OUTPUT_TOKENS` to adjust local generation; retrieved excerpts are dropped
whole if needed to fit context. Grounded experiment plans use only customer-stated
acceptance criteria; if none were supplied, criteria stay TBD. Deterministic planning
guardrails require independent references and held-out runs, and are labeled as process
safeguards rather than scientific validation.

The grounded path is experimental and has not passed its Phase 3 live review. Two
focused repairs still left semantic errors in D01/D02 runs: source-level fusion claims
were overstated, FTIR guidance from a different application was transferred to the
powder feed-frame case, and a typical source pathlength was placed into a customer
configuration. Do not present this path as validated; the
offline baseline remains the default. H01–H06 holdouts and Phases 4–5 have not been run.

## Optional API integration

`.env.example` documents configuration; `.env` is **not automatically loaded**.
Set process environment variables explicitly. Default: `SPECTRO_AI_PROVIDER=baseline`.

```powershell
# Optional API SDK for document Q&A only; not needed for assessment extraction.
uv sync --locked --extra llm
# Restore minimal environment:
uv sync --locked
```

`SPECTRO_AI_PROVIDER=openai` returns an explicit unimplemented-adapter error. The
optional API SDK is used only for document Q&A. Never commit credentials.

## Extending during the hackathon

- Add YAML profiles with provenance, then explicit rules and tests when justified.
  New knowledge entries do not automatically become recommendation rules.
- Supplied AFS sources are allowlisted in `knowledge/sources.yaml` and indexed locally;
  source review flags remain visible. Actual instrument specifications and measured
  datasets are still unavailable. Retrieval does not enrich technique knowledge.
- See [architecture](docs/ARCHITECTURE.md), [decisions](docs/DECISIONS.md),
  [evaluation](docs/EVALUATION.md), and [hackathon plan](docs/HACKATHON_PLAN.md).
  Confirm criteria first, validate knowledge second, implement one specific workflow third.

## Local document RAG

Document Q&A is separate from spectroscopy assessment. It retrieves local excerpts,
then asks Ollama or an OpenAI-compatible endpoint for a cited answer. Citations show
where text came from; they do not validate scientific claims or customer suitability.
The four synthetic assessment scenarios and their export remain available offline.

Install optional document dependencies, start Ollama, and ensure these models exist:

```powershell
uv sync --locked --extra rag
ollama pull gemma4:12b-it-q4_K_M
ollama pull qwen3-embedding:0.6b
New-Item -ItemType Directory -Force documents | Out-Null
```

Put authorized UTF-8 `.txt`/`.md` files or text-based `.pdf` files in `documents/`.
Customer documents and `.cache/` indexes are ignored by Git. PDF citations use pages;
text citations use line ranges. Scanned/encrypted/unreadable PDFs report errors;
OCR and complex table reconstruction are outside this version.

```powershell
uv run python -m spectro_ai.rag index --documents documents
uv run python -m spectro_ai.rag search "Which measurement limitations are documented?"
uv run python -m spectro_ai.rag ask "Which measurement limitations are documented?"
uv run streamlit run app.py
```

Use the separate **Document Q&A** panel after indexing. `search` shows retrieved
excerpts without generation; `ask` returns cited statements or insufficient evidence.
Both need Ollama for the question embedding. The default index is
`.cache/rag/index.sqlite3`; `--index PATH` selects another index. Repeat indexing
reuses unchanged embeddings and removes deleted documents from the index only.
Failures preserve the previous index. After changing embedding model/settings, run:

```powershell
uv run python -m spectro_ai.rag index --documents documents --rebuild
```

Changing only the generation model does not require rebuilding. Defaults target an
RTX 4070 with 12 GB VRAM: Gemma generation, Qwen3 Embedding 0.6B, 4,096-token context,
sequential embedding/generation. A synthetic local rehearsal passed; single-run measurements below are not a
performance guarantee.

## Scientific AFS evidence retrieval

This local lexical index serves the assessment evidence layer. It uses only files
allowlisted in `knowledge/sources.yaml`, under `knowledge/data/materials/`; the worked
development examples and final-evaluation enquiries are explicitly excluded. It has no
LLM, embedding, or network dependency. PDF pages, text line spans, document sections,
and spreadsheet worksheet rows retain source locations and content hashes. The S01
page-6 workbook is identified as a pending-review transcription linked to S01 p.6.
S02 citations remain marked for visual review, including slides with short extraction.

After installing the `rag` extra (for `pypdf`), rebuild the ignored local index:

```powershell
uv sync --locked --extra rag
uv run python -m spectro_ai.evidence reindex
```

`EvidenceRetriever.search(CustomerUseCase, limit=8)` returns typed, source-cited
excerpts using deterministic lexical BM25. Retrieval scores rank text only; they are
not scientific confidence. Raw passages never become validated technique profiles.
The baseline assessment does not call this retrieval module yet.

`.env.example` lists configuration; set process environment variables explicitly:

| Variable | Default / purpose |
| --- | --- |
| `SPECTRO_RAG_PROVIDER` | `ollama`; alternatively `openai` |
| `SPECTRO_RAG_INDEX` | `.cache/rag/index.sqlite3` |
| `SPECTRO_RAG_CONTEXT` | `4096` generation context |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `gemma4:12b-it-q4_K_M` |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` |
| `OPENAI_BASE_URL` | OpenAI-compatible API base URL |
| `OPENAI_MODEL`, `OPENAI_API_KEY` | Explicit model and credentials for API generation |

For API generation, install both extras and configure the endpoint:

```powershell
uv sync --locked --extra rag --extra llm
$env:SPECTRO_RAG_PROVIDER = 'openai'
$env:OPENAI_BASE_URL = 'https://your-endpoint.example/v1'
$env:OPENAI_MODEL = '<your-model>'
# Set OPENAI_API_KEY securely in the current process.
```

The endpoint must support Chat Completions JSON-object mode. This sends the question
and retrieved excerpts to that endpoint; embeddings remain local to Ollama. There
is no automatic provider fallback. Unavailable models, timeouts, invalid outputs,
unknown citations and incompatible indexes report errors. Provider selection here
is independent of `SPECTRO_AI_PROVIDER`: assessment's OpenAI adapter remains a stub.

The local index contains document text and vectors. Treat it with the same care as
its source files. Retrieval scores are similarity diagnostics, never confidence.
Document text is untrusted prompt content; citation membership checks cannot prove
that a model's statement is correct. Inspect source passages before using an answer.

For full RAG regression coverage on a fresh environment, install both optional extras:

```powershell
uv sync --locked --extra rag --extra llm
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build --wheel
```

Baseline-only environments skip optional PDF tests when pypdf is absent. Latest
implementation verification: 68 tests passed, Ruff lint/format passed (30 files),
and wheel build passed. Real Ollama/GPU rehearsal also passed; no live API call
was made. API behavior is tested with mocks.

Generation uses a conservative UTF-8-byte context guard and removes lowest-ranked
whole excerpts until the request fits. The default context can therefore include
fewer than four retrieved chunks. Inspect the passages actually used with the answer.

### Local GPU rehearsal — 2026-10-16

Installed Ollama 0.40.5 was found outside PATH. If `ollama` is not recognized, use
the installed executable directly, for example:

```powershell
& 'C:\Users\nikon\AppData\Local\Programs\Ollama\ollama.exe' list
# Start its server if it is not already running:
& 'C:\Users\nikon\AppData\Local\Programs\Ollama\ollama.exe' serve
```

One synthetic TXT chunk was indexed and searched. Gemma returned the fictional
BLUE-17 inventory label with the correct citation and abstained on an unsupported
detection-limit question. One cold-ish run: indexing 29.560 s, search 4.985 s,
answer 55.337 s (query embedding/model loading included), abstention 7.034 s.
`ollama ps` reported Gemma 8.1 GB, 100% GPU, context 4096. Afterward, total GPU usage
was 9,792 of 12,282 MiB, including desktop/other allocations; not per-model or peak
memory. These observations are not benchmarks or guarantees for larger documents.
PDF behavior is covered by offline fixtures; real customer content requires review.
