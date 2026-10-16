# Spectro-AI

A small SchubKI hackathon foundation for spectroscopy use-case assessment. Enter a
customer problem, inspect requirements and gaps, explore provisional techniques,
and export a feasibility outline as Markdown.

**Rules-and-templates demo, not validated scientific analysis.** Challenge, datasets,
instruments, and judging criteria remain unconfirmed. All customer examples are
synthetic. No API key needed; assessment makes no network requests. Streamlit usage
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

Objective, target, and matrix are required for candidates. Other unknowns stay visible
and generate questions. Quantities retain supplied units; no conversion or unlabeled
numeric extraction occurs. Blank values, `unknown`, `TBD`, and `N/A` become null;
zero stays a valid supplied value. Negated clauses are excluded from phrase matching,
which can deliberately miss positive facts in those clauses. Repeated labels use the
last value. Explicit labels, including unknowns, override phrase matches.

## Architecture and repository

```text
app.py                         Streamlit presentation and session state
src/spectro_ai/
  models.py                    Pydantic domain types and evidence provenance
  baseline.py                  Literal extraction, gaps, rules, experiment templates
  knowledge.py                 KnowledgeSource protocol and validated YAML loader
  providers.py                 Baseline/future-provider selection
  reporting.py                 Deterministic Markdown export
knowledge/techniques.yaml      Five preliminary profiles with checked sources
examples/                     Four labeled synthetic scenarios
prompts/                      Reserved prompts; baseline does not execute them
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

Verified on Windows / Python 3.14.4: 44 pytest cases, Ruff lint/format, local server
HTTP 200/ok, optional SDK installation/import, and installed wheel knowledge/assessment.
The minimal environment was restored after the optional SDK check. Restricted
execution sandboxes may require local socket access for AppTest and server checks;
they use Windows asyncio loopback sockets even without external requests.

## Optional AI integration

`.env.example` documents configuration; `.env` is **not automatically loaded**.
Set process environment variables explicitly. Default: `SPECTRO_AI_PROVIDER=baseline`.

```powershell
# Optional SDK only; not needed for the demo.
uv sync --locked --extra llm
# Restore minimal environment:
uv sync --locked
```

`SPECTRO_AI_PROVIDER=openai` returns an explicit unimplemented-adapter error whether
or not `OPENAI_MODEL` and `OPENAI_API_KEY` exist. It never sends a request or silently
falls back. Future adapters must validate the same domain models and keep scientific
rules separate. Never commit credentials.

## Extending during the hackathon

- Add YAML profiles with provenance, then explicit rules and tests when justified.
  New knowledge entries do not automatically become recommendation rules.
- New sources implement `KnowledgeSource.get_techniques()`. Official AFS documents,
  instrument specifications, datasets, and PDFs await access and expert review.
  No document ingestion or vector store is implemented.
- See [architecture](docs/ARCHITECTURE.md), [decisions](docs/DECISIONS.md),
  [evaluation](docs/EVALUATION.md), and [hackathon plan](docs/HACKATHON_PLAN.md).
  Confirm criteria first, validate knowledge second, implement one specific workflow third.
