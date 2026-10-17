# Repository readiness audit

Audit date: 2026-10-16. Scope: source, models, knowledge, providers, RAG, prompts,
examples, tests, exports, dependency metadata, CI, safe environment template, and
Git ignore/configuration. Existing working changes were preserved. No private
customer documents, `.env`, credentials, Git remote URLs, or paid models were read
or called. The original attachment is not available in this checkout: sections A–I
below map the capabilities explicitly retained in PLAN.md, rather than claiming
line-by-line verification of an unavailable attachment.

## Implementation closure and final checks

The initial-state table below is retained as audit evidence. Intake now includes
compatible process/composition/detection/deployment/interference fields and field
provenance. `intake.correct_requirements` validates expert changes; Streamlit hides
pending exports. `prioritized_questions`, `_context`, `_configuration`, and `_experiment`
provide bounded consequential questions, conditional screening/configuration and
tailored studies with both outcome branches. Six separate fixtures and exports are
covered in `test_evaluation.py`; the full expert-edit demo is covered in `test_intake.py`.
Unknown-provider generation is rejected explicitly. No LLM extraction or automatic
retrieval-to-assessment integration was justified by this bounded demo.

Final commands on 2026-10-16 (README local UV_CACHE_DIR and TEMP/TMP):

- `.tools/Scripts/uv.exe sync --locked`: passed; `uv pip check`: compatible, 47 packages.
- `uv run pytest --basetemp=.cache/pytest-minimal`: 82 passed, 1 optional PDF skip.
- `uv sync --locked --extra rag --extra llm`: passed; `uv pip check`: compatible, 55 packages.
- `uv run --extra rag --extra llm pytest --basetemp=.cache/pytest-final2`: 83 passed
  after final objective-background/fluorescence-target regression fix.
- `uv run ruff check .`: passed; `uv run ruff format --check .`: 39 files formatted.
- `uv run python scripts/smoke_check.py`: HTTP 200/ok; verification server stopped.
- `uv run python scripts/evaluate.py`: six JSON/Markdown outputs in `.cache/evaluation`.

Default sandbox cache initially failed permission checks; sandbox AppTest later
stalled with local cache/temp. Completed checks ran outside sandbox after automatic
approval. Initial test formatting failure was corrected before final successful runs.
No new live model or paid API call was made; historical RAG rehearsal is separate.
An untouched pre-implementation whole-suite baseline was not captured: audit/source
work overlapped the first stalled run. The reproducible post-change results above
supersede that run without claiming a clean initial benchmark.

See EVALUATION_RESULTS.md for partial software rubric scores and scientific review
limits; DEMO_PLAN.md for presenter steps/recovery. Source verification does not
validate customer suitability. Human event rehearsal, expert review, proprietary
hardware/data and original attachment confirmation remain external follow-ups.

## Status and priority

States: **Complete**, **Partial**, **Missing**, **Not verified**, **Not needed for MVP**.
Complete describes the bounded software contract, never scientific validation.
Priorities: P0 essential demo, P1 credibility/usability, P2 optional, P3 future work.
The table records the starting audit; subsequent implementation closure belongs in
HISTORY.md and the evaluation/demo documents, with fresh check results below.

| Requirement | Current State | Evidence | Priority | Recommended Action |
| --- | --- | --- | --- | --- |
| A: intake schema, missing data, original text and units | Complete for bounded schema | `models.CustomerUseCase`, `DomainModel.normalize_unknown`; `test_models.py` checks nulls, units, zero, extra fields, JSON | P0 | Preserve contract and explicit unknowns. |
| A: process/composition, detection limit, deployment, provenance | Partial | Starting `CustomerUseCase` has installation constraints but lacks dedicated process/composition/detection/deployment/provenance fields | P0 | Add compatible optional fields and provenance; preserve customer text. |
| A: general prose understanding | Partial | `baseline.extract_use_case`, `PHRASES`, `_positive_text`; `test_plain_text_food_problem_extracts_only_literal_facts` | P0 | Demonstrate bounded supported prose plus labels/manual correction; do not promise unrestricted extraction. |
| B: targeted technical questions | Partial | `QUESTIONS`, `missing_fields`, critical-field abstention tests; starting implementation asks all 11 missing fields | P0 | Prioritize consequential gaps and relevant interference/access questions. |
| C: five technique profiles | Complete as starter data | `knowledge/techniques.yaml`, `test_starter_knowledge_covers_all_five_techniques`; all marked expert review required | P1 | Preserve provenance; verify passages before extending scientific rules. |
| C: contextual screening and changing selection | Partial | Starting `_rules` uses objective/target phrases; matrix/interference/installation do not affect selection | P0 | Small explicit justified rules; paired regression cases; distinguish target fluorescence from Raman background. |
| C: safe abstention | Complete for declared contract | `assess` critical fields and `no_rule_match`; `test_each_missing_critical_requirement_prevents_candidates`, `test_sparse_input_abstains`, `test_uv_vis_rule_and_explicit_no_match` | P0 | Preserve abstention; no forced positive technique. |
| D: probe/geometry and operating constraints | Partial | `TechniqueCandidate` supports optional configuration; starting `assess` leaves it unknown; YAML lists possibilities | P1 | Conditional configurations with contact, fiber, fouling and compatibility caveats; no guessed AFS limits. |
| E: experiment and acceptance criteria | Partial | Starting `_experiment` has blanks, reference, repeats, interference, held-out validation and unknown thresholds, but generic branches | P0 | Tailor objective, matrix controls, configuration, metrics and success/failure actions. |
| F: rationale, evidence, assumptions, uncertainty | Partial | `Evidence.require_source_provenance`, candidate rationale, global uncertainty; provenance is checked structurally, applicability unvalidated | P1 | Connect each candidate to customer facts and change conditions; keep heuristic/source distinction. |
| G: expert interaction and follow-up answers | Partial | Starting `app.py` allows raw text/reassessment and hides stale results; `test_blank_input_and_stale_report_behavior` | P0 | Structured validated corrections; recompute candidates/questions; stale JSON/export regression. |
| G: Markdown and JSON review | Complete for initial fields | `reporting.render_markdown`, JSON model dumps; `test_markdown_contains_full_report_and_safe_original_input`, AppTest download assertion | P0 | Keep new fields, evidence, and experiment branches consistent in all outputs. |
| H: assessment LLM adapter | Not needed for MVP | `providers.assess_problem` explicitly rejects OpenAI stub; `prompts/assessment.md` and `experiment.md` are reserved | P2 | Keep stub labeled; add only if a demonstrated extraction requirement cannot use offline/manual flow. |
| H: document Q&A and retrieval | Complete for bounded separate workflow | `rag.index_documents/search/ask`, `rag_models.DocumentAnswer`; `test_rag.py` checks reuse, rollback, metadata, embeddings, citation membership, abstention | P1 | Keep independent of technique knowledge and assessment; source passage review remains required. |
| H: provider boundary failures | Complete for mocked contracts after audit fix | `providers.generate_document_answer` explicit provider guard; tests for transport JSON/envelopes, timeout, missing model/credentials, invalid context, truncation, safe errors | P1 | Preserve fail-closed behavior and explicit provider selection; no paid CI calls. |
| H: prompt injection resistance and citation truth | Partial | `prompts/rag.md` treats question/excerpts as untrusted; JSON data framing; `rag.ask` checks retrieved IDs, not entailment | P1 | Inspect cited passages; do not promote output to validated knowledge. No prompt can guarantee factual support. |
| H: external data transfer | Complete disclosure, not policy approval | README and Document Q&A UI describe question/excerpt transfer; embeddings use configured Ollama endpoint | P1 | Confirm organizer/customer permission before real data. Endpoint may be remote if configured. |
| I: original synthetic examples | Complete | Four originals in `examples/scenarios.json`; `test_all_synthetic_scenarios_match_declared_rule_behavior` explicitly asserts four | P0 | Keep originals; six challenge cases belong in separate evaluation fixtures. |
| I: six challenge cases and scored evaluation | Missing at starting audit | Original examples lack all requested cases; `docs/EVALUATION.md` initially contains future rubric only | P0 | Add six outputs, bounded checks and reviewer notes; scientific plausibility needs an actual expert. |
| I: dedicated demo/handoff | Missing at starting audit | README contains basic walkthrough; no initial `docs/DEMO_PLAN.md` | P0 | Rehearse incomplete input → correction → changed candidates → export, with recovery. |
| Developer: Python/uv separation and lock | Complete configuration; environment verification below | `.python-version` 3.14; pyproject requires >=3.14,<3.15; lock requires ==3.14.*; `.tools` bootstrap vs `.venv` documented | P0 | Run locked sync and dependency check; do not equate metadata with successful install. |
| Developer: reproducible verification | Complete configuration | `.github/workflows/checks.yml` Windows Python 3.14, uv 0.13.1, locked sync, pytest, Ruff, startup smoke | P1 | Remote CI run remains unverified until pushed/run; optional RAG extras need separate local verification. |
| Developer: safe configuration and ignored artifacts | Complete for inspected patterns | `.env.example` placeholders only, not auto-loaded; `.gitignore` excludes `.env.*` except example, secrets, documents, cache/reports/tools/venv | P1 | Never commit secrets; ignore rules alone do not prove history contains none. No full secret-history audit performed. |
| Developer: Git platform settings | Complete observation | `git config --get core.autocrlf` and `core.ignorecase` both returned `true` | P2 | Preserve checkout settings; avoid gratuitous line-ending churn. |
| Scientific/industrial validation | Not verified | No expert labels, spectra, instruments, proprietary specifications or reference measurements supplied | P0 external dependency | Obtain expert review and representative samples; software tests cannot certify customer suitability. |
| OCR, hardware automation, training, vector service | Not needed for MVP | Text-based PDF/TXT/MD support; explicit scanned-PDF error, exact local cosine search, no hardware control | P3 | Expand only when real challenge data establishes need. |

## Provider and knowledge review

`KnowledgeSource.get_techniques` remains the assessment data boundary; YAML is
safe-loaded and validated for schema/unique IDs. RAG is a separate document store,
not a new automatically trusted technique source. Local similarity search is O(chunks
× dimensions), top four; scores are diagnostics, never confidence. Index updates
stage a new SQLite database and replace atomically; text and embeddings require the
same access care as source documents. A valid citation ID establishes location only.

Ollama uses a 120-second transport timeout and schema-shaped JSON request. API
generation uses 120 seconds, no retries, JSON-object mode and final Pydantic
validation. Context is conservatively bounded by UTF-8 bytes and output reserve;
this is not a tokenizer measurement. No automatic provider fallback occurs. Audit
found that directly constructing unknown `RagSettings.provider` bypassed environment
validation and selected the API branch. An explicit guard now rejects it before any
request. Mock regressions preserve this boundary.

Assessment prompts remain inactive. Customer/document text has no execution path;
untrusted text is still sent to the selected generation model and may influence its
answer. Citation membership and schema checks cannot prove semantic grounding or
prevent every prompt injection. Markdown escapes generated statements/excerpts.

## Official challenge checks

See [challenge source/date record](CHALLENGE.md). The official AFS article supports
only general vendor scope, not available equipment, exact operating limits or a
validated application recommendation. SchubKI homepage did not confirm the supplied
challenge title. Judging, data rights, hardware, network and expert review remain open.

## Reproducible verification — audit boundary

Commands ran from `C:\Projects\SpectroFiberAI`, with `UV_CACHE_DIR=.cache/uv` and
`TEMP`/`TMP=.cache/tmp`, using `.tools/Scripts/uv.exe` on Python 3.14.4.

| Command | New result on 2026-10-16 |
| --- | --- |
| `uv run pytest tests/test_rag.py tests/test_providers_and_reporting.py --basetemp=.cache/pytest-audit` | 33 passed, 1 skipped in 0.73 s; optional pypdf unavailable |
| `uv run ruff check src/spectro_ai/providers.py tests/test_rag.py` | Passed |
| `uv run ruff format --check src/spectro_ai/providers.py tests/test_rag.py` | Initially flagged appended test formatting; applied formatter, rerun passed (2 files) |
| Live API/paid generation | Not run; mocks only |
| New real Ollama/GPU rehearsal | Not run; prior README timing is historical, not this audit |
| Private/customer-document ingestion | Not run; synthetic fixtures only |

Full locked/minimal/optional installation, whole-suite tests, lint/format, health
smoke and synthetic input-to-export results are recorded during final acceptance.
Do not infer those results from this targeted run. Optional PDF tests skip without
`pypdf`; full optional coverage requires `uv sync --locked --extra rag --extra llm`.
