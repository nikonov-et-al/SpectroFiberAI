# Implementation history

## Completed — 2026-10-16 — Milestone 1

### Original milestone text

### 1. Implement document ingestion and persistent indexing

- Default input directory: `documents/`. Default index: `.cache/rag/index.sqlite3`. Ignore customer documents and generated indexes in Git.
- Support UTF-8 TXT/Markdown and text-based PDF. Preserve PDF page boundaries; split text deterministically into approximately 1,200-character chunks with 200-character overlap.
- Record relative document path, file hash, chunk text, location, and deterministic chunk ID.
- Store vectors alongside chunk metadata. Record index schema, chunking settings, embedding model identity/digest, and vector dimension.
- Reuse embeddings for unchanged chunks. Update changed files and remove deleted files from the index only; never modify source documents.
- Commit index updates atomically after successful extraction and embedding. Failures preserve the previous usable index. Report unreadable, encrypted, and scanned PDFs explicitly.
- Reject incompatible embedding/index settings with a rebuild instruction. Changing the generation model does not require reindexing.

**Acceptance:** repeat indexing makes no embedding requests for unchanged chunks; interrupted updates preserve the previous index.

### Implementation and verification

Implemented deterministic text/PDF ingestion with provenance, cached embeddings, index compatibility checks, staged SQLite storage and atomic replacement. Failed updates preserve the prior usable index; explicit rebuild can replace a corrupt index.

Verification accepted by orchestrator: `uv run pytest` — 68 passed; `uv run ruff check .` — passed; `uv run ruff format --check .` — passed (30 files); `uv build --wheel` — passed.

Deviation: Automated indexing/provider tests use mocks. Real Ollama indexing rehearsal was subsequently completed in milestone 4 on 2026-10-16.

## Completed — 2026-10-16 — Milestone 2

### Original milestone text

### 2. Implement model connections and retrieval

- Default Ollama endpoint: `http://127.0.0.1:11434`.
- Default generation model: `gemma4:12b-it-q4_K_M`.
- Default embedding model: `qwen3-embedding:0.6b`.
- Start generation context at 4,096 tokens and one request at a time. Make context size configurable; validate actual GPU residency and context capacity during hardware testing.
- Use Ollama `/api/embed` for documents and questions, with silent input truncation disabled. Validate vector counts, dimensions, finite values, and nonzero norms.
- Release the embedding model after each indexing batch or query embedding before generation. This avoids requiring both models to remain resident.
- Retrieve the top four chunks using normalized cosine similarity and stable ordering for ties. Similarity scores are retrieval diagnostics, never confidence.
- Add independent RAG settings: `SPECTRO_RAG_PROVIDER=ollama|openai`, Ollama endpoint/model settings, and `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`.
- External generation uses Chat Completions with JSON-object output and Pydantic validation; JSON-mode support is required for the chosen endpoint. Ollama uses a JSON schema. JSON validity alone does not establish schema validity. [OpenAI structured-output guidance](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).
- Apply finite timeouts, explicit errors, and no automatic provider fallback. Never log credentials or raw provider error bodies.

**Acceptance:** mocked Ollama and API responses produce the same validated answer type; provider failures never become fabricated answers.

### Implementation and verification

Implemented independent Ollama embedding and Ollama/OpenAI-compatible generation adapters, validated vectors and answers, local cosine retrieval, finite errors/timeouts and no provider fallback.

Verification accepted by orchestrator: `uv run pytest` — 68 passed; `uv run ruff check .` — passed; `uv run ruff format --check .` — passed (30 files); `uv build --wheel` — passed.

Deviation: GPU rehearsal was initially deferred: Ollama was outside PATH and its endpoint was closed. The installation was subsequently found and started; real GPU/context rehearsal completed in milestone 4. No live API call was made.

## Completed — 2026-10-16 — Milestone 3

### Original milestone text

### 3. Implement grounded Q&A, CLI, and Streamlit panel

Expose:

```powershell
uv run python -m spectro_ai.rag index --documents documents
uv run python -m spectro_ai.rag search "Which measurement limitations are documented?"
uv run python -m spectro_ai.rag ask "Which measurement limitations are documented?"
```

- `search` returns excerpts and locations without generation. `ask` uses the configured generation provider.
- Instruct models to treat excerpts as untrusted data, answer only from supplied material, and abstain when support is absent.
- Require each substantive answer statement to reference retrieved chunk IDs. Reject unknown citations and uncited statements. Citation validation confirms reference membership, not scientific correctness.
- Return `insufficient_evidence` for an empty corpus or model abstention. Missing indexes, unavailable models, invalid output, and context overflow produce actionable errors.
- Do not invent a universal similarity threshold. Test retrieval and abstention on labeled synthetic cases.
- Add a separate document Q&A panel showing provider, index summary, answer, excerpts, and locations. External-provider selection clearly states that questions and retrieved excerpts leave the machine.
- Keep current assessment controls and Markdown export working. Add a separate Markdown download for document answers.
- Hide stale answers after question, provider, or index changes.

**Acceptance:** after indexing, a user can ask a document question in Streamlit and inspect the supporting passages.

### Implementation and verification

Implemented CLI indexing/search/answering and separate Streamlit document Q&A with citation validation, excerpt inspection, Markdown export, explicit API data transfer and stale-result invalidation. A conservative UTF-8-byte context guard drops lowest-ranked whole excerpts until the request fits; 4096 context may include fewer than four retrieved chunks.

Verification accepted by orchestrator: `uv run pytest` — 68 passed; `uv run ruff check .` — passed; `uv run ruff format --check .` — passed (30 files); `uv build --wheel` — passed.

Deviation: The context guard is deliberately conservative rather than a model-specific tokenizer; it may supply fewer than four excerpts. Real Ollama end-to-end rehearsal subsequently completed in milestone 4 on 2026-10-16.

## Completed — 2026-10-16 — Milestone 4

### Original milestone text

### 4. Verify, document, and hand off

- Add labeled synthetic PDF/Markdown/TXT fixtures and questions with expected source chunks, including an unsupported question.
- Test ingestion locations, stable IDs, incremental updates, deletion handling, failed-update recovery, model mismatch, and malformed vectors.
- Test grounded responses, abstention, invalid citations, malformed JSON, prompt-injection text, timeouts, and missing models. Default tests use mocked providers and require no network.
- Extend Streamlit AppTest for Q&A, errors, stale results, and export. Preserve all four original scenarios and baseline no-network tests.
- Run:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

- Separately rehearse real Ollama indexing and answering on the RTX 4070. Record GPU residency, memory use, indexing time, and answer latency; claim no performance until measured.
- Document installation, model downloads, environment settings, indexing, rebuilds, API data handling, and limitations. Update README, architecture/decision records, and hackathon plan.
- After accepting each milestone, copy its exact original text into `HISTORY.md` with date, implementation, verification, and deviations; then remove it from `Plan.md`.

### Implementation and verification

Implemented synthetic fixtures and regression coverage, operational documentation,
configuration examples, and real Ollama rehearsal. Verification: `uv run pytest` —
68 passed; `uv run ruff check .` — passed; `uv run ruff format --check .` — passed
(30 files); `uv build --wheel` — passed. Fresh full-feature setup:
`uv sync --locked --extra rag --extra llm`.

Real rehearsal used installed Ollama 0.40.5, discovered outside PATH at
`C:\Users\nikon\AppData\Local\Programs\Ollama\ollama.exe`. A synthetic TXT document
was indexed and searched; Gemma answered BLUE-17 with the correct chunk citation
and abstained on an unsupported detection-limit question. Local run records:
`.cache/rag_rehearsal.py` and `.cache/rag-rehearsal-result.json` (ignored artifacts).

One cold-ish run measured indexing 29.560 s, search 4.985 s, answer 55.337 s
(including query embedding and model loading), and abstention 7.034 s. `ollama ps`
reported Gemma 8.1 GB, 100% GPU, context 4096. `nvidia-smi` after the run reported
9,792 MiB used of 12,282 MiB; this includes desktop/other allocations and is neither
per-model usage nor peak memory. These are single-run observations, not benchmarks
or performance guarantees.

Deviations: hardware rehearsal covered one synthetic TXT chunk; PDFs were covered
by offline fixtures. External API behavior was validated with mocks, without a live
API call. No real customer documents or scientific suitability were validated.


## Completed — 2026-10-16 — Readiness milestone 1

### Exact original milestone text

### 1. P0 — Complete the evidence-based repository audit

- Inspect remaining source, providers, retrieval, prompts, schemas, tests, examples,
  export, `.python-version`, lockfile, CI, Git configuration/ignore rules, and safe
  environment configuration. Never print secrets or inspect unrelated customer data.
- Establish baseline: locked dependency consistency, tests, lint/format, startup
  smoke, and one synthetic input-to-export assessment. Record exact results/skips;
  distinguish historical RAG rehearsal from new verification.
- Create `docs/REPOSITORY_AUDIT.md`, covering attachment sections A–I and developer
  experience, with **Requirement | Current State | Evidence | Priority |
  Recommended Action**. Reference actual files, functions, and tests.
- Use states **Complete, Partial, Missing, Not verified, Not needed for MVP**.
  Explain each gap, its impact, and smallest useful fix. Use P0 for essential demo
  gaps, P1 for credibility/usability, P2 for optional enhancements, P3 for future work.
- Check accessible official sources; update `docs/CHALLENGE.md` and
  `docs/HACKATHON_PLAN.md` with source/date, supplied facts, and unresolved questions.
  Missing proprietary information must not block safe offline improvements.

**Acceptance:** every requested capability has an evidence-backed status; baseline
results and blockers are reproducible; revise later scope against actual P0 gaps.


### Implementation and verification

Evidence-based audit of retained A-I capabilities and developer experience; official AFS/SchubKI checks, initial-state matrix and final closure in docs/REPOSITORY_AUDIT.md and docs/CHALLENGE.md.

Verification: locked minimal sync and pip check passed (47 packages); minimal pytest 82 passed, 1 optional PDF skipped. Locked rag/llm sync and pip check passed (55 packages); final optional pytest 83 passed. Ruff lint passed; format check passed (39 files). Startup smoke HTTP 200/ok with server stopped; evaluation script generated all six JSON/Markdown reports. Checks used README local cache/temp; completed AppTest/check run outside sandbox after stalled sandbox attempts. No new live paid calls.

Deviations / remaining external work: Original attachment unavailable; mapped PLAN-retained capabilities explicitly. Untouched initial whole-suite benchmark unavailable because first sandbox run stalled while work proceeded. Post-change minimal/optional checks replace it; no scientific validation inferred.


## Completed — 2026-10-16 — Readiness milestone 2

### Exact original milestone text

### 2. P0 — Close intake, follow-up, and expert correction gaps

- Add compatible optional requirement fields only where needed for process,
  composition, detection limit, monitoring/deployment mode, and installation limits.
  Trace facts to customer text or expert edits; distinguish assumptions explicitly.
- Support the chosen natural-language demo with bounded, transparent extraction
  and manual correction. Do not claim unrestricted understanding from more keywords.
  Keep structured labels as the offline fallback.
- Prioritize questions by recommendation impact: critical facts first, then relevant
  sensitivity, matrix/interference, access, and operating conditions. Avoid asking
  every missing field regardless of relevance.
- Add structured expert review and follow-up answers in Streamlit. Validate edits,
  preserve unmodified facts and units, recompute gaps/candidates, and hide stale exports.

**Likely files:** `models.py`, `baseline.py`, `app.py`, `reporting.py`, related tests.

**Acceptance:** incomplete input produces useful questions; expert answers update
requirements and consequential assessment behavior; unknowns remain visible.
Domain and AppTest regressions cover corrections and stale results.


### Implementation and verification

Added optional requirements, literal/label provenance, validated expert correction helper, prioritized consequential questions and structured Streamlit answers. Pending corrections and changed raw input hide exports. Full demo AppTest checks corrections, preserved text/units and changed screening.

Verification: locked minimal sync and pip check passed (47 packages); minimal pytest 82 passed, 1 optional PDF skipped. Locked rag/llm sync and pip check passed (55 packages); final optional pytest 83 passed. Ruff lint passed; format check passed (39 files). Startup smoke HTTP 200/ok with server stopped; evaluation script generated all six JSON/Markdown reports. Checks used README local cache/temp; completed AppTest/check run outside sandbox after stalled sandbox attempts. No new live paid calls.

Deviations / remaining external work: Extraction deliberately bounded; labels/expert edits handle quantities and unsupported language. No unrestricted language understanding or LLM dependency added.


## Completed — 2026-10-16 — Readiness milestone 3

### Exact original milestone text

### 3. P0 — Make screening contextual and explainable

- Keep five technique profiles and a small deterministic engine. Add explicit,
  justified rules with tests, considering relevant matrix/state, target response,
  fluorescence background, water interference, sensitivity requests, and integration
  constraints. Unknown equipment capability requires confirmation, not guessed limits.
- Distinguish fluorescence as a target signal from fluorescence interfering with
  Raman. Explain conditional candidates, prerequisites, limitations, and why new
  facts change the outcome. Preserve insufficient-information and no-rule behavior.
- Explain mechanism, relevant customer facts, advantages, assumptions, evidence,
  conditions that change selection, and laboratory validation for each candidate.
- Keep scientific provenance in `knowledge/techniques.yaml`; verify supporting
  passages. General principles and retrieved citations do not validate AFS suitability.

**Likely files:** `baseline.py`, technique YAML, models/reporting as needed, rule tests.

**Acceptance:** paired cases with changed interference or matrix produce explained
changes; unsupported cases abstain or remain conditional; no fabricated performance,
confidence, citations, or AFS capabilities. Record expert review as pending until obtained.


### Implementation and verification

Contextual rules/conditions cover aqueous ethanol, reaction progress, matrix water, powder presentation, Raman fluorescence interference, sensitivity and integration dependencies. Five profiles retained; verified general principle provenance and heuristic rationale remain distinct. Paired regressions cover changes.

Verification: locked minimal sync and pip check passed (47 packages); minimal pytest 82 passed, 1 optional PDF skipped. Locked rag/llm sync and pip check passed (55 packages); final optional pytest 83 passed. Ruff lint passed; format check passed (39 files). Startup smoke HTTP 200/ok with server stopped; evaluation script generated all six JSON/Markdown reports. Checks used README local cache/temp; completed AppTest/check run outside sandbox after stalled sandbox attempts. No new live paid calls.

Deviations / remaining external work: Candidates remain unranked conditional laboratory hypotheses, including Raman under reported background; expert validation and real performance remain pending.


## Completed — 2026-10-16 — Readiness milestone 4

### Exact original milestone text

### 4. P0/P1 — Tailor experiments and conditional configurations

- P0: generate objective-specific hypothesis, representative samples, blanks,
  calibration/reference samples, preparation, proposed configuration, steps,
  independent reference, interference checks, metrics, acceptance criteria, and
  success/failure next actions. Include held-out validation where relevant.
- Use supplied acceptance targets or mark them for agreement. Do not invent validated
  performance, reference methods, operating limits, or sample counts.
- P1: suggest immersion, transmission, reflection/diffuse reflectance, ATR, or
  non-contact options only when justified. Explain relevant contact, optical path,
  fiber transmission, material compatibility, fouling, cleaning, temperature, and
  pressure dependencies. Keep unsupported numerical ranges and AFS selections unknown.
- Keep comparison, rationale, evidence, assumptions, and full experiments consistent
  across Streamlit, JSON, and Markdown export.

**Likely files:** `models.py`, `baseline.py`, `reporting.py`, YAML, UI and tests.

**Acceptance:** expert can refine a practical study with clear controls, reference,
decision criteria, and both outcome branches; unavailable hardware specs remain explicit.


### Implementation and verification

Tailored feasibility plans include target/matrix/process/composition, representative variability, blanks/references, interference, held-out validation, metrics, agreed criteria and outcome branches. Explicit state/contact facts justify conditional study geometry; Markdown, JSON and UI use the same report.

Verification: locked minimal sync and pip check passed (47 packages); minimal pytest 82 passed, 1 optional PDF skipped. Locked rag/llm sync and pip check passed (55 packages); final optional pytest 83 passed. Ruff lint passed; format check passed (39 files). Startup smoke HTTP 200/ok with server stopped; evaluation script generated all six JSON/Markdown reports. Checks used README local cache/temp; completed AppTest/check run outside sandbox after stalled sandbox attempts. No new live paid calls.

Deviations / remaining external work: No sample counts, numerical hardware limits, independent reference method or AFS probe model invented. Unsupported geometry stays unknown.


## Completed — 2026-10-16 — Readiness milestone 5

### Exact original milestone text

### 5. P1 — Audit provider and knowledge boundaries

- Verify structured-output validation, prompts/settings, timeouts, malformed outputs,
  missing models, citation checks, credentials, and untrusted customer/document text.
- Keep the assessment stub labeled. Add LLM extraction only if the audited workflow
  needs it: adapters in `providers.py`, prompts in `prompts/`, validated domain output,
  and an independent offline path. No live paid calls in automated tests.
- Reuse local retrieval only for demonstrated value. Retrieved content remains
  reviewable evidence, never automatically validated technique knowledge. New data
  sources follow the existing `KnowledgeSource` boundary where applicable.
- Test failures with mocks; never fabricate answers or silently switch providers.
  Keep external question/excerpt transfer explicit.

**Likely files:** providers/RAG modules, prompts, tests, README.

**Acceptance:** implemented adapters pass validation/error checks; placeholders and
data transfer are clear; assessment and document Q&A contracts remain independent.
Larger retrieval-to-assessment integration stays P2 unless the audit justifies it.


### Implementation and verification

Audited provider/retrieval separation, typed outputs, prompts, budgets, citations and explicit transfer. Rejected unknown direct provider settings and added mocked malformed response/timeout/truncation/configuration/credential failures.

Verification: locked minimal sync and pip check passed (47 packages); minimal pytest 82 passed, 1 optional PDF skipped. Locked rag/llm sync and pip check passed (55 packages); final optional pytest 83 passed. Ruff lint passed; format check passed (39 files). Startup smoke HTTP 200/ok with server stopped; evaluation script generated all six JSON/Markdown reports. Checks used README local cache/temp; completed AppTest/check run outside sandbox after stalled sandbox attempts. No new live paid calls.

Deviations / remaining external work: No new LLM extraction or retrieval-to-assessment integration justified; assessment adapter remains labeled stub. No new live paid calls; historical real RAG rehearsal remains separate.


## Completed — 2026-10-16 — Readiness milestone 6

### Exact original milestone text

### 6. P0/P1 — Evaluate all six scenarios and verify the workflow

- Preserve the four original examples. Add separate evaluation fixtures if needed:
  current baseline tests explicitly assert four originals.
- Cover aqueous fermentation ethanol, powder moisture, reactor progress, liquid
  contamination, strong fluorescence interfering with Raman, and insufficient input.
  Declare supplied facts, expected unknowns, question priorities, conditional behavior,
  and prohibited claims. Correct abstention is acceptable; no forced positive match.
- Add meaningful checks for extraction/updates, validation, contextual screening,
  uncertainty/evidence, explanations, experiments, export, provider failures, and UI.
- Apply `docs/EVALUATION.md` to measurable completeness, plausibility, uncertainty,
  and practical usefulness. Record reviewer notes and open scientific questions;
  automated behavior checks cannot replace expert plausibility review.
- Run checks below plus a synthetic end-to-end export. Verify baseline without keys
  or model services; test optional integrations separately and report skips honestly.

**Likely files:** evaluation fixtures/tests, `docs/EVALUATION.md`, existing smoke script.

**Acceptance:** all six cases have reviewable outputs; required checks pass or have
explicit environment blockers; original baseline, export, AppTest, and RAG regressions
remain intact. Scientific readiness requires actual expert review.


### Implementation and verification

Preserved four originals; added six separate scenario fixtures, reproducible Markdown/JSON evaluation script, export/status/unknown checks, full expert-edit AppTest and paired contextual regressions. EVALUATION_RESULTS.md applies structural rubric scores, per-case questions and unreviewed scientific dimensions.

Verification: locked minimal sync and pip check passed (47 packages); minimal pytest 82 passed, 1 optional PDF skipped. Locked rag/llm sync and pip check passed (55 packages); final optional pytest 83 passed. Ruff lint passed; format check passed (39 files). Startup smoke HTTP 200/ok with server stopped; evaluation script generated all six JSON/Markdown reports. Checks used README local cache/temp; completed AppTest/check run outside sandbox after stalled sandbox attempts. No new live paid calls.

Deviations / remaining external work: Actual spectroscopy plausibility, evidence applicability and experiment scoring remain Not reviewed pending expert. Structural scores are not scientific readiness or confidence.


## Completed — 2026-10-16 — Readiness milestone 7

### Exact original milestone text

### 7. P0 — Prepare demo and handoff

- Create `docs/DEMO_PLAN.md`: ready-to-use synthetic input, follow-up answers,
  expected assessment changes, candidate comparison, uncertainty, experiment, export,
  presenter steps, recovery, and a credential-free offline fallback.
- Rehearse incomplete input → expert answers → changed recommendation → export
  on the demo machine. Use implemented behavior, not aspirational capabilities.
- Update README and `docs/HACKATHON_PLAN.md`; update `docs/DECISIONS.md` and
  architecture docs when setup/architecture changes. Review `AGENTS.md` for necessary
  evaluation/source-ingestion/validation guidance without duplicating this plan.
- Report repository health, files/features/tests changed, exact verification,
  failures/limits, readiness, AFS dependencies, top three actions, and architecture
  recommendations.

**Acceptance:** coherent full demo works without inaccessible services; software
verification and unresolved scientific/hardware claims are clearly distinguished.


### Implementation and verification

Added docs/DEMO_PLAN.md with input, expert answers, expected conditional changes, comparison, experiment, export, recovery and offline path. Automated full sequence rehearsed in AppTest on development machine; updated README, architecture, decisions and hackathon plan. AGENTS reviewed; existing validation/source guidance sufficient.

Verification: locked minimal sync and pip check passed (47 packages); minimal pytest 82 passed, 1 optional PDF skipped. Locked rag/llm sync and pip check passed (55 packages); final optional pytest 83 passed. Ruff lint passed; format check passed (39 files). Startup smoke HTTP 200/ok with server stopped; evaluation script generated all six JSON/Markdown reports. Checks used README local cache/temp; completed AppTest/check run outside sandbox after stalled sandbox attempts. No new live paid calls.

Deviations / remaining external work: Human presenter rehearsal on event setup, organizer facts and scientific/hardware acceptance remain explicit pending follow-ups in PLAN.md. No commit requested or made.


## Completed — 2026-10-17 — Implementation Phase 0

### Exact original task text

### Phase 0 — Audit and baseline (first)

- Inspect the repo and run existing tests.
- Write `docs/gap-analysis.md` and list missing AFS source files.
- Freeze/record current behavior for one known case and one insufficient-info case.

**Gate:** existing application still runs offline; current tests are understood.

### Implementation and verification

Audited retained contracts, baseline, providers, RAG, correction service, exports, tests and source inventory. Added docs/gap-analysis.md with keep/change/add/defer and reproducible baseline snapshots. All listed AFS files found under knowledge/data/materials; brief under docs_schubKI. D01 currently abstains from extraction gaps; known synthetic moisture case and insufficient-input behavior recorded.

Verification: uv run pytest — 83 passed (3.12s, local socket access outside sandbox); uv run ruff check . — passed; uv run ruff format --check . — passed (39 files). Source/code review by Sol accepted.

Deviations: actual materials directory is knowledge/data/materials rather than knowledge/materials. No application source changes in audit; no missing AFS filenames. Scientific suitability remains unverified.


## Completed — 2026-10-17 — Implementation Phase 1

### Exact original task text

### Phase 1 — Contracts + extraction (P0)

- Unify Pydantic contracts; support customer-text evidence spans and unknowns.
- Implement working structured LLM extraction, conservative fallback, and tests.
- Allow expert correction at the application-service level even if UI comes later.

**Gate:** D01–D08 can be extracted without inventing missing client values; provider errors are handled.

### Implementation and verification

Extended existing flat Pydantic contracts with stable request IDs, customer process/access/reference fields, exact quote provenance, preferences/requirements and ambiguity diagnostics; added compatible evidence/report fields. Added offline/Ollama extraction protocol implementations, constrained schema also supplied in prompt, one structural retry, numeric/field checks, explicit fallback and context/output guards. Expert correction accepts scalar facts and removes stale quotes. Updated exports, packaged extraction prompt, setup and architecture notes.

Verification: uv run pytest — 101 passed (2.74s); uv run ruff check . — passed; uv run ruff format --check . — passed (43 files). Sol reviewed repaired live D01–D08 records and essential customer facts/unknown quantities. Final D02 smoke with context16384/output2048 passed in 11.12s. Live records remain ignored under .cache/phase1-live.

Deviations/limits: initial live attempts omitted quotes or timed out; fixed by explicit prompt schema, lean transport and disabling optional thinking. D02 semantic misclassifications were cleared via generic field guidance/checks. Live extraction remains conservative: some process/state/preferences missed, D05 ambiguity missing, D08 target unknown; warnings and expert correction expose these gaps. Quote checks do not prove semantics. Baseline default and legacy statuses preserved; no scientific or hardware validation claimed.


## Completed — 2026-10-17 — Implementation Phase 2

### Exact original task text

### Phase 2 — Scientific retrieval (P0)

- Build source manifest, page-aware extraction, chunk store/index, and retrieval API.
- Index scientific AFS materials; exclude evaluation answers from retrieval.
- Check S01 references, image-heavy slides, and coverage by development case.

**Gate:** every development case retrieves relevant source passages with valid page references, or reports genuine corpus gaps.

### Implementation and verification

Added knowledge/sources.yaml allowlist, page/section/worksheet-aware PDF/TXT/DOCX/XLSX extraction, content/hash deduplication with retained provenance, atomic local JSON reindex CLI, and reusable BM25 EvidenceRetriever. Registry tracks scope/type/review/citation formats and actual source hashes; stale/corrupt metadata/pages/IDs rejected. Evaluation fixtures/worked answers excluded. EvidenceKnowledgeSource deliberately returns no curated technique profiles from raw excerpts. Updated README, architecture, decisions, hackathon plan and source coverage map.

Verification: uv run pytest --basetemp .cache/pytest-phase2-final — 111 passed (2.79s, local socket access outside sandbox); targeted evidence tests — 10 passed; uv run ruff check . — passed; uv run ruff format --check . — passed. Actual reindex: 7 files, 160 deduplicated chunks, zero missing/empty PDF pages, one short page (S02 p20) flagged. Sol independently checked index freshness and D01/D02 source/page/scope results; all eight development queries surface relevant primary pages in top8.

Deviations/limits: canonical materials path differs from plan as previously recorded. D06 fiber/remote pages S02 pp6–7 and S01 pp9–10 miss top8; D08 same-location S01p8 can rank below top8. These are documented ranking misses, not absent-corpus gaps. S02 image-heavy pages and research papers retain explicit review requirements; workbook transcription retains worksheet locators and its own hash, not invented PDF pages. No scientific suitability or instrument performance validated.


## Attempted — 2026-10-17 — Implementation Phase 3 (not accepted)

### Exact original task text

### Phase 3 — Grounded assessment (P0)

- Add structured LLM assessor, conditional method comparison, questions, experiment plan and evidence IDs.
- Keep deterministic checks and fallback behavior.

**Gate:** outputs on D01/D02 are defensible against worked examples without matching canned text; unfamiliar inputs can abstain.

### Implementation and verification

Implemented provider-neutral TechnicalAssessor returning AssessmentReport, actual Ollama structured generation, immutable source-ID/page/hash membership checks, per-component lexical relevance screens, complete-request context fitting by dropping whole excerpts, and the shared corrected-case assessment service. Trusted facts/unknowns survive correction and fallback; empty evidence abstains. Streamlit corrections rerun retrieval/assessment. JSON/Markdown share source excerpts/scope/review/provenance. Deterministic engineering safeguards preserve customer criteria or TBD, independent references/held-out validation and expert-review next steps; these are visibly labeled and do not validate scientific claims.

Verification: Sol integration uv run pytest --basetemp .cache/pytest-phase3-root — 127 passed (3.27s); uv run ruff format --check . — passed (46 files); startup smoke — HTTP 200/ok, verification server stopped. Initial final lint found two long lines and import order; worker applied formatting-only fixes and reran Ruff lint/format successfully, with unchanged string contents. No paid calls. Real Ollama D01/D02 assessments and insufficient-input abstention captured with raw requests/responses and generated reports under ignored .cache/phase3-live*, without manual output edits.

### Failed gate and deviations

Phase 3 is NOT completed: original task retained in PLAN.md. Initial live D01 assumed fusion superiority and omitted independent validation; D02 invented R² > 0.90 and premature production actions. Repair 1 strengthened prompt/schema and generic planning safeguards, but context fit exposed too little source evidence and semantic transfer remained weak. Repair 2 removed redundant payload/schema data, preserved five whole excerpts for D01/D02 at default context, and required evidence for all named technique components. Final D01 still includes unqualified fusion accuracy advantage, empty candidate limitations, and under-supported FTIR/ATR transfer from extrusion into powder feed-frame use. Sol rejected the live gate. No additional prompt tuning after two failed repairs, per AGENTS.md.

Default deterministic baseline, original examples, abstention, exports and AppTest remain green. Optional grounded mode is experimental and not accepted for the demonstration. Scientific/expert review, semantic reliability, Phase 4 evaluation/holdout execution and Phase 5 presentation remain pending; no H outputs generated or modified for this request. No commit requested or made.

Final D02 v3 retains five excerpts, including S01 p.17 reflection and p.18 transmission context, and no invented acceptance threshold survives. It still promotes source-typical 1–10 mm pathlength into a proposed probe configuration without an explicit customer-setup qualification. D01 and D02 therefore require further semantic/expert review; the incomplete case correctly abstains. Final v3 reports/raw captures remain untouched in `.cache/phase3-live-v3`.

## Completed — 2026-10-18 — Replace offline baseline with grounded LLM pipeline (software implementation)

Accepted software change from user-approved plan. Scientific acceptance is **not**
completed; original phase gates and 2026-10-17 blocker remain in PLAN.md.

### Exact original implementation task text

1. **Remove legacy path; preserve useful checks**
   - Move missing-field detection, clarification helpers, and neutral abstention construction out of `baseline.py`; then delete baseline module and parser-specific tests.
   - Remove phrase dictionaries, label parser, deterministic method rankings/configurations, baseline provider branches, and `allow_offline_fallback`.
   - Keep deterministic schema, provenance, citation, constraint, and feasibility checks. Preserve document ingestion parsers; those serve retrieval.

2. **Complete both provider adapters**
   - Keep adapters in `providers.py`; reuse existing Ollama extraction/assessment and OpenAI configuration conventions.
   - Use `SPECTRO_AI_PROVIDER=ollama|openai` for both stages; default Ollama. Reject `baseline` with clear migration message.
   - Implement OpenAI structured extraction and technical assessment using existing optional SDK dependency. Require explicit model and credentials; lazy-load SDK.
   - Both providers return same validated domain models and use shared prompts, evidence selection, context handling, and validators.
   - No provider switching on failure. Missing configuration, timeout, refusal, malformed output, or exhausted context produces actionable error. Allow one structural repair attempt; never silently truncate enquiry.

3. **Strengthen extraction and grounded assessment**
   - Extraction preserves original text, literal supporting quotations, customer units, unknown scalars as `None`, and missing collections as `[]`. Service owns request IDs and provenance.
   - Share extraction cleanup across providers; remove hard-coded Ollama metadata. Unsupported facts become unknown with visible diagnostics.
   - Retrieve existing indexed technical evidence from extracted or expert-corrected facts. Keep source IDs, pages, excerpts, review status, and version.
   - Assessment generates conditional candidates, trade-offs, relevant questions, and feasibility plan using retrieved evidence only. No evidence or missing critical facts yields explicit abstention, never baseline recommendations.
   - Require claim-linked evidence for consequential method, geometry, advantage, and limitation statements. Reject unknown citations and mismatched quotations; distinguish source claims, engineering hypotheses, and customer facts.
   - Validate units without conversions or invented limits. Flag contradictory constraints and unchecked compatibility.
   - Expose validation outcomes as `error`, `warning`, or `not_checked`. Citation membership and lexical screening never count as scientific validation.
   - Address recorded Phase 3 failures: unsupported fusion advantage, empty limitations, transferred application context, source optical paths presented as customer configuration, and invented acceptance thresholds. Unresolved claims require expert review or withholding.

4. **Update interfaces and user flow**
   - Preserve `assess_problem` and `assess_use_case` entry points; remove fallback parameter and legacy provider choices.
   - Extend extraction metadata for OpenAI; make report provider metadata accurate for successful and abstaining runs. Preserve existing customer/report fields where possible; document intentional legacy contract removal.
   - Replace offline UI instructions with natural-language enquiry guidance and active provider/model.
   - Preserve expert corrections; rerun retrieval and assessment without re-extracting corrected facts.
   - Show facts/quotes, gaps, candidate comparison, evidence details, validation diagnostics, and uncertainty. Preserve Markdown; add JSON download.
   - Clear stale reports/exports after input, correction, or provider changes and failed runs. Explain hosted data transmission before OpenAI submission.
   - Leave separate Document Q&A behavior intact.

### Implementation and verification

Deleted baseline module, parser/rule tests and exports. Added neutral guardrails;
Ollama/OpenAI now share extraction quote cleanup, domain output, source screening,
context handling and bounded structural repair. Added evidence claim links and
inspectable diagnostics; verbatim source-claim quotations and study-only geometry
are conservative lexical checks, not entailment validation. Customer quantity/unit
matching prevents numerical planning steps from borrowing unrelated source numbers.
Unsupported candidates are withheld. Qualitative LLM planning remains, with customer
criteria or TBD, independent references and held-out validation safeguards.

UI displays active provider/model, transmission disclosure, expert corrections,
Markdown/JSON downloads, source excerpts and validation diagnostics. Input/config
changes and failures invalidate reports; correction failures remain visible. The
separate Document Q&A retains behavior. Evaluator loads original D01–D08 enquiry
blocks (excludes worked answers), H01–H06, four original synthetic inputs, eight
synthetic paraphrases, five adversarial inputs and custom JSON. It uses assess_problem,
records per-case failures and continues; matching stale generated artifacts are cleared.
README, architecture, decisions, hackathon/demo docs, AGENTS, CI and settings updated.

Final software checks: `uv run pytest --basetemp=.cache/pytest-grounded-final`
**119 passed**; `uv run ruff check .` passed; `uv run ruff format --check .`
**46 files already formatted**. Streamlit smoke returned **HTTP 200/ok**. Tests require
writable TEMP/TMP; AppTest and live loopback require sandbox escalation on this host.
The initial sandbox temp/server errors were environmental; successful reruns above
are explicit. No approval review rejection occurred.

Local Ollama smoke on Gemma completed all stages and returned insufficient_information:
NIR candidates lacked linked claims and were removed. Raw JSON/Markdown saved under
ignored `.cache/grounded-live/`. Initial 2048 output-token smoke failed explicitly;
4096 completed. Default output budget raised to 4096 and timeout to 240 seconds;
context remains 16384. No paid calls or commits. OpenAI live smoke unavailable:
OPENAI_MODEL and OPENAI_API_KEY unset. Full local evaluation results recorded separately.

### Deviations and remaining risks

Source-claim bullets currently require verbatim quotations, a stricter lexical policy
than claim links alone. It may over-abstain; no semantic/scientific correctness is
claimed. Unit checks preserve text and screen quoted quantities; hardware compatibility
and scientific entailment remain not_checked. Expert D01/D02 and unfamiliar-enquiry
review remains required, including fusion benefit, cross-application transfer and
source optics/customer configuration issues. The complete scientific gate is not
accepted by this software entry. Historical reports and customer materials untouched.

Final review additionally removed unit-test dependency on ignored AFS materials,
using temporary DOCX/TXT fixtures, and bounded OpenAI extraction repair context.
These two regression tests bring final total to 119 passed.

## Completed — 2026-10-18 — Local evaluation execution

Exact original task text:

- Finish and review local evaluation records for synthetic, D01–D08, P01–P08,
  A01–A05 and H01–H06 using the identical production pipeline; preserve raw outputs.

Implementation: ran all 31 enquiries through unchanged local Ollama production pipeline,
preserved raw JSON/Markdown and summary under `.cache/grounded-evaluation-20261018/all/`.
Verification: command exit 0; 31 completed, 0 runtime failures. Results and limitations
recorded in docs/LLM_EVALUATION.md. No manual answer edits or paid requests.
Deviation: automated/software review only; expert scientific acceptance remains pending.

## Completed — 2026-10-18 — Unified hybrid assessment knowledge base

### Original task objective

Refactor the existing SpectroFiberAI repository to use **one unified knowledge base with hybrid retrieval (BM25 + semantic search)** for the main spectroscopy assessment.

Remove the separate Document Q&A functionality and introduce an **Update Knowledge Base** button in Streamlit.

### Implementation and verification

Reused ingestion, Ollama embeddings and SQLite in one assessment knowledge index.
Added incremental atomic updates, structure-aware chunks, targeted requirement queries,
RRF fusion, immutable assessment snapshots and revision-based report invalidation.
Removed standalone Q&A UI, generation adapter, models and prompt. Preserved provenance
and existing claim-level safety checks, including rejection of unsupported essential
rationales. Scientific acceptance gates in PLAN.md remain pending.

Verification: `uv run pytest` — 121 passed; `uv run ruff check .` — passed;
`uv run ruff format --check .` — passed (49 Python files). Streamlit AppTest
passed outside the Windows sandbox; temporary HTTP health returned 200/ok and stopped.
Real local Ollama smoke: 7 source documents, 197 chunks, 1,024-dimensional vectors;
repeat update generated zero new embeddings and reused 197. Real D01 completed through
hybrid retrieval, validation and exports with a provisional source-backed comparison.
This verifies software operation, not scientific suitability. Details and limitations:
[SEMANTIC_SEARCH.md](docs/SEMANTIC_SEARCH.md).

Deviation: retained existing AFS manifest ingestion alongside folder additions in the
same database; BM25 statistics are derived from committed chunks at retrieval time.
No new dependency, vector service or framework. Initial user edits remain uncommitted;
checks cover the combined working tree. Isolated staged tests: 110 passed and three
pre-existing HEAD failures, each independently confirmed against untouched HEAD.
No unrelated PLAN.md acceptance gate closed.

## Completed — 2026-10-18 — Assessment claim validation and report separation

### Original task objective

Refine grounded assessment validation so unsupported secondary claims do not erase supported exploratory candidates; improve conflict handling, study planning safeguards, and separate customer report content from technical diagnostics while retaining verified source provenance.

### Implementation and verification

Validated evidence claims individually, optionally searched the same verified knowledge index for exact source quotations, retained source/page/version provenance, and withheld candidates with detected direct cross-source contradictions. Tightened feasibility safeguards and ordered missing information by decision impact. Customer Markdown now omits internal source scope/review diagnostics; JSON and the diagnostics tab retain the audit trail. Added focused regression coverage and documented the approach and limits in `docs/ASSESSMENT_REVIEW.md`.

Verification: `uv run --offline ruff check .` — passed; `uv run --offline ruff format --check .` — passed (49 files); focused `pytest tests/test_claim_validation.py tests/test_assessment.py tests/test_guardrails.py` — 26 passed. Full suite and focused AppTest run stalled at `tests/test_app.py` under the Windows sandbox; AppTest requires execution outside the sandbox. `git diff --check` — passed. No expert scientific acceptance is claimed; original Phase 3 gate remains pending.

Deviation: accepted implementation refinement only; did not remove or rewrite the original Phase 3 task because the documented expert review gate remains open.

## Completed — 2026-10-18 — Optional local reranker

### Exact original task text

- Verify Python 3.14 CPU runtime and add an optional, cached local cross-encoder adapter with explicit model setup.
- Integrate opt-in reranking after fusion, preserve default retrieval, and expose snapshot-scoped report/UI diagnostics with visible RRF fallback.
- Add a reproducible retrieval benchmark with provisional fixtures, human-reviewed version-bound labels, and offline regression coverage.
- Document setup and scientific limits; run full acceptance checks and a separate real-model smoke; commit verified work.

### Implementation

Added the optional `rerank` extra and uv lock resolution, explicit model setup CLI,
lazy cached-only CPU assessment scoring, bounded query/passage token inputs and
validated scalar scores. Existing default retrieval remains unchanged; opt-in
retrieval reranks 30 fused distinct candidates before final byte/count selection.
Added nullable relevance scores, snapshot-scoped diagnostic histories, technical
Markdown/JSON output and configuration-driven Streamlit report invalidation.

The retrieval-only benchmark compares original retrieval, widened RRF and reranking
on one pinned index, records model/index identities and mode timings, and rejects
unreviewed or stale relevance labels. Six provisional questions cover the four
synthetic scenarios, limitations and irrelevant evidence; no relevance labels or
quality gains were invented. Setup, architectural decisions, limits and hackathon
strategy are documented. Source documents and index schema are unchanged.

### Verification

- `uv lock` and `uv sync --extra rag --extra llm --extra rerank` succeeded.
- `uv run pytest --basetemp=.cache/pytest-final` — 150 passed, including 10 AppTests.
- `uv run ruff check .` — passed.
- `uv run ruff format --check .` — passed, 52 files already formatted.
- `git diff --check` — passed.
- Python 3.14.4, sentence-transformers 5.7.0 and torch 2.14.1+cpu: explicit model
  download and fresh-process finite CPU inference succeeded. Resolved model
  revision: `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Fresh-process production retrieval with Hugging Face/Transformers offline flags
  returned 8 scored excerpts from 30 candidates and an applied diagnostic; a
  deliberately unavailable model returned 8 unscored excerpts with visible fallback.
- Cached-model BM25 and real Ollama hybrid corpus benchmark smokes completed.
  Hybrid run used index `123cdc83ecc9466b8e671afdb00ceda7` (197 chunks), six queries
  and one warm repetition each: means 1.90 s original, 1.73 s widened RRF, 2.50 s
  reranked; first model load plus scoring 7.54 s. Results remain ignored in
  `.cache/retrieval-benchmark-hybrid.json`; relevance metrics are null pending review.

### Deviations and remaining gates

Runtime dependency downloads and Streamlit tests required execution outside the
Windows sandbox; the sandbox blocked DNS, default cache/temp writes and asyncio's
local socket pair. Repo-local uv cache and test temp directories were used. These
were environment restrictions, not application regressions. No paid model calls
were made. Cold timing includes scoring rather than claiming isolated model-load
latency. Human relevance labels, default promotion, scientific acceptance and the
existing Phase 3–5 expert/demo gates remain pending; no implementation task is blocked.


## Completed — 2026-10-18 — Structure-aware milestone 1

### Original milestone text

### 1. Safe index evolution and vector reuse

- Add typed source blocks, sections and locations; introduce schema version 3 with validated document/section/block relationships.
- Continue reading version 2 indexes. Explicit knowledge-base updates create version 3 in staging, validate it, then atomically replace active database. Failed updates leave old index usable; pinned assessments retain their original snapshot.
- Separate extraction fingerprint from embedding cache key. Parser/version/settings changes trigger re-extraction, but vectors remain reusable when **exact embedding input and complete embedding configuration** match.
- Embedding identity covers provider, resolved model revision, preprocessing/options and vector dimension. Validate cached vectors before reuse.
- Keep existing text as embedding input initially. Store document titles and heading paths as metadata; defer changing indexed text to avoid combining ingestion and ranking changes.
- Preserve existing public IDs/contracts where unchanged; regenerate identities where passage content or provenance changes. Reject stale benchmark labels.

**Acceptance:** version 2 compatibility, atomic replacement/failure recovery, pinned snapshots, unchanged-input vector reuse across parser upgrades, and changed-input/configuration invalidation all pass. Demo runs against either supported index version.

### Implementation and verification

Sol reviewed typed v3 structure records, file/provenance anchors and atomic replacement. Schema 2 remains readable, including the existing real AFS index; snapshots remain pinned during migration. Embedding reuse uses exact input plus provider/endpoint/model/revision/preprocessing/truncation configuration and validated dimensions, independently of parser/chunking version.

Acceptance: 153 tests passed; `uv run ruff check .` passed; `uv run ruff format --check .` passed (53 files); Streamlit health smoke returned HTTP 200/ok. Full tests used an approved unsandboxed process after sandbox AppTest stalled.

Deviation: schema 2 has no complete embedding-configuration record, so its vectors are conservatively rebuilt on migration; subsequent v3 parser updates reuse identical inputs. Active demo index was not migrated automatically.


## Implemented — 2026-10-18 — Structure-aware milestone 2 (acceptance blocked)

### Original milestone text

### 2. PDF and Markdown ingestion

- Add locked Docling dependencies and explicit local-model setup. Verify Windows/Python 3.14 compatibility before switching PDF default.
- Configure CPU conversion, table extraction, local assets, OCR disabled and remote services disabled. Keep explicit `pypdf` backend available for compatibility and comparison; never silently switch backend after an error.
- Explicitly enable `HeadingHierarchyOptions(enabled=True)` and `generate_parsed_pages=True`; explicitly configure bookmark, numbering and style signals. Pin resolved settings in extraction fingerprint. Docling otherwise leaves PDF headings flat by default. [Official heading documentation](https://docling-project.github.io/docling/usage/heading_levels/)
- Treat inferred nesting as extraction output requiring validation. Check levels, source order, page bounds, parent relationships and numbering consistency; retain diagnostics and original detected levels.
- On invalid/ambiguous nesting, use flat page-local sections and suppress cross-section expansion. Do not invent corrected hierarchy. Visually compare extracted outlines with actual AFS pages before acceptance.
- Preserve PDF paragraphs, table headers, rows, captions and original block/page locations. Preserve Markdown heading hierarchy, paragraphs, lists, fenced blocks and tables.
- Keep 1,200-character target and existing overlap setting. Pack within structural boundaries; overlap applies only to split prose. Split large tables by rows, repeating source headers. Preserve and flag indivisible oversized rows.
- Keep TXT, DOCX and XLSX behavior unchanged during this milestone.

**Acceptance:** fixture coverage and real AFS inspection establish correct reading order, heading boundaries and table/value associations. Full suite and demo smoke pass before Docling becomes default.

### Implementation and verification

Sol reviewed structure records, source spans, PDF heading validation, local-only Docling setup, exact conversion status checking, and multi-page provenance handling. Markdown now preserves block structure. Docling 2.140.0 is locked and verified in isolated Windows CPython 3.14.4 with rag/rerank extras. Real S01/S04 production adapter runs completed; invalid heading nesting produced flat page-local sections. Ambiguous multi-page tables fail explicitly rather than inventing row-page mapping.

Verification: 171 tests passed; Ruff lint/format passed (59 files); Streamlit HTTP 200/ok. Normal tests mock Docling; separate real conversions used explicitly cached assets.

Not accepted: real S01 p.6 table mapping remains wrong after two focused configuration attempts. Keep pypdf default and original milestone in PLAN.md; no further parser tuning or scientific/default promotion. Parent/context and broader formats can proceed independently. See docs/STRUCTURE_CONTEXT_REVIEW.md.


## Completed — 2026-10-18 — Structure-aware milestone 3

### Original milestone text

### 3. Parent context, disabled by default

- Add `SPECTRO_RAG_PARENT_EXPANSION=false`. Disabled path retains existing selection and budgeting behavior.
- Attach context to its retrieved child using optional typed `context_blocks` containing text, block identity, exact locations and review metadata. Existing evidence fields remain valid; missing collections default to empty lists.
- Parents/context receive no independent retrieval score or evidence count. Never count child and parent as separate corroborating sources.
- With expansion enabled, reserve **30% of existing retrieval byte budget before child selection**. Rank/rerank children normally; bypass premature per-query byte truncation, then select children within remaining 70% and existing child-count limit.
- Allocate reserved capacity in child-rank order. Attach full section when it fits; otherwise construct contiguous source-block window around child, growing outward by block distance, preceding block first on ties.
- Preserve whole blocks and table header/row associations. Do not cross document or validated section boundaries. Deduplicate shared context blocks; retain each block's review status and precise locations.
- If no complete surrounding block fits, record omission. Reuse unused capacity for additional ranked children only after expansion allocation.
- Charge serialized context against retrieval and provider budgets. Final provider fitting drops complete optional context blocks before children; never silently slices source text.
- Verify context against pinned stored blocks before assessment. Permit quotations only from verified child/context text and bind quotations to exact contributing locations.
- Show attached context in diagnostics/UI and exports without presenting another independent source. Existing reports without context retain their current rendering.

**Acceptance:** reservation, windows, ordering, deduplication, provenance validation, count limits, byte limits and provider fitting pass. Expansion remains disabled after software acceptance.

### Implementation and verification

Sol reviewed disabled-by-default parent expansion, 30% reservation before child selection, whole sections/contiguous source-block windows, partial-anchor recovery, table-header associations, deduplication and serialized UTF-8 accounting. Context remains nested under a child with no independent evidence ID/score/corroboration count. Exact context block IDs and canonical source locations govern quotations. Whole optional contexts drop before children; claim-specific evidence additions cannot trim or invalidate earlier verified citations. Conditional UI/Markdown/JSON rendering preserves the no-context path.

Acceptance: 191 tests passed; `uv run ruff check .` passed; `uv run ruff format --check .` passed (61 files); Streamlit HTTP 200/ok. Full tests include AppTest and real synthetic Markdown ingestion through mocked assessment/export, plus tamper, budget and same-source limitation regressions. Real AFS BM25 comparison harness executes without generation or model requests.

Deviation: none to feature behavior. Real AFS usefulness/default promotion remains independent and pending in milestone 4; parent expansion remains false.


## Completed — 2026-10-18 — Structure-aware milestone 5

### Original milestone text

### 5. Broader format improvements

- Extend shared block packing to DOCX headings/paragraphs/tables, XLSX worksheets/rows/explicit headers, and TXT paragraphs with original line locations.
- Preserve cell positions, empty cells and customer units; do not infer absent headers or scientific meaning.
- Keep curated YAML technique profiles as structured records.
- Apply same provenance, vector-reuse, context-window and scientific-context tests.

**Acceptance:** format-specific regressions pass, existing examples/exports remain usable, and demo smoke succeeds.

### Implementation and verification

Sol reviewed shared DOCX/XLSX/TXT block packing: explicit heading/header metadata, paragraph/worksheet/row/line locations, empty cell positions, blank-row handling, side-by-side table handling, literal units and cached workbook values. Curated YAML remains unchanged. Regression coverage exercises format structure, canonical context and vector reuse across parser fingerprint changes.

Final compatibility audit also corrected a pre-existing evidence-ID weakness: normal updates reuse IDs only for matching text and citation provenance; changed parser passages receive deterministic content/location-bound IDs. Authorized tests now assert precise per-row/paragraph/table source locators.

Acceptance: 200 tests passed; Ruff lint passed; Ruff format check passed (63 files); Streamlit HTTP 200/ok. This includes the final synthetic PDF repeated-value/operating-condition provenance test.

Deviation: cached raw XLSX values are preserved; formulas are not evaluated and display number formats are not rendered. No inferred units or header meanings. Parser/default and real AFS usefulness gates remain pending independently.
