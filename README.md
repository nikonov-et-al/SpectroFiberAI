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
uv run python -m spectro_ai.evidence reindex
uv run streamlit run app.py
```

Before indexing, place authorized AFS files listed in `knowledge/sources.yaml` under
`knowledge/data/materials/`. These ignored source files are not bundled in Git.
Missing source/index prerequisites are reported explicitly.

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
- `evidence.py`: allowlisted AFS document ingestion and local BM25 retrieval with pages,
  excerpts, content hashes, scope and review flags. Ingestion parsers are retained.
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

## Separate Document Q&A

Document Q&A retains independent provider/index settings. Install `rag` extra; use
Ollama embedding model `qwen3-embedding:0.6b`. Put authorized TXT/Markdown or text PDFs
under ignored `documents/`, then:

```powershell
ollama pull qwen3-embedding:0.6b
uv run python -m spectro_ai.rag index --documents documents
uv run python -m spectro_ai.rag search "Which limitations are documented?"
uv run python -m spectro_ai.rag ask "Which limitations are documented?"
```

`SPECTRO_RAG_PROVIDER=ollama|openai` controls answer generation; query embeddings still
need Ollama. Default index `.cache/rag/index.sqlite3`. Rebuild with `--rebuild` after
embedding model changes. Scanned PDFs/OCR remain unsupported. Document Q&A does not
replace assessment evidence retrieval or scientifically validate answers.

See [architecture](docs/ARCHITECTURE.md), [decisions](docs/DECISIONS.md),
[hackathon plan](docs/HACKATHON_PLAN.md), and [pending backlog](PLAN.md).

Current local run: [31-case LLM evaluation](docs/LLM_EVALUATION.md).
