# Spectroscopy hackathon readiness — pending work

Updated: 2026-10-17. Seven earlier software readiness milestones and implementation
Phases 0–2 implemented and accepted;
exact original text, results and deviations are preserved in [HISTORY.md](HISTORY.md).
Current execution request ends after Phase 3; Phases 4–5 remain pending.
Audit: [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md).
Evaluation: [docs/EVALUATION_RESULTS.md](docs/EVALUATION_RESULTS.md).
Demo: [docs/DEMO_PLAN.md](docs/DEMO_PLAN.md).

# PLAN.md — Spectro-AI Hackathon Implementation Plan

> Execution plan for Codex. **Upgrade the existing repository rather than rewriting it.** First inspect the code and preserve useful work. Implement in small, testable increments. This is a plan, not evidence that any feature already exists.

## 0. Mission and non-negotiable constraints

Build a working, explainable **AI assistant for first-pass in-line spectroscopy solution design** for Art Fiber Systems (AFS). A customer describes an industrial measurement problem; an AFS expert receives a provisional, source-grounded assessment: stated facts, missing information, suitable spectroscopy approaches, high-level probe/measurement concepts, relevant spectral ranges, limitations, and a practical feasibility experiment.

The assistant is **decision support, not a validated measurement system**. It does not predict concentrations from spectra, promise equipment performance, or replace expert/sample validation.

### Challenge requirements to preserve

- Natural-language customer input, recommendations, clarification questions, source-backed explanations, a feasibility plan, and a demonstrable prototype.
- The **supplied AFS materials are the primary technical source**. Cite material IDs and exact page/slide references for consequential technical claims. If using external sources, comply with the brief's quoted-passage-and-link requirement.
- **No manually edited final evaluation answers or case-specific hard-coding for H01–H06.** The jury may submit a short unfamiliar enquiry live. Do not use scenario IDs, exact phrases, or worked-answer retrieval as decision logic.
- Unknown client facts remain unknown. Do not invent concentrations, limits, response times, wavelength coverage of actual components, detection limits, hardware compatibility, costs, or accuracy.
- A published application supports *testing a candidate*, not claiming the candidate is proven for this customer.
- `insufficient_information` or a conditional shortlist may be the correct outcome. If the request is beyond the supplied evidence, be explicit.
- Prefer a simple reproducible architecture. Do not introduce multiple agents, orchestration frameworks, service meshes, a separate API service, or a hosted vector database without measured need.

### Main evaluation criteria (from the challenge brief)

- Technical quality **30%**; use of AFS knowledge **20%**; feasibility-study plan **15%**; clarification/uncertainty **15%**; explainability **10%**; prototype/presentation **10%**.
- Optimize first for technical accuracy, grounded evidence, and useful experiments, not UI decoration.

### Source boundaries

Use supplied materials as follows:

| Source | Purpose |
| --- | --- |
| `S01_Multispectral_Laboratory.pdf` | Main technical source: method comparison p.6; method/probe coverage pp.9–12; use cases pp.13–18. |
| `S02_AFS_Process_Control_Slides.pdf` | Fiber transmission, probes and applications; many slides are image-heavy—check visually or use controlled transcription. |
| `S03_LED_Kidney_Study_2017.pdf`, `S04_Combi_MIR_Fluorescence_2021.pdf` | Research examples with limited demonstration scope, not universal performance evidence. |
| `S05_AFS_Flyer.pdf`, `S06_Laboratory_Link.txt` | Company/product and business context, not proof of analytical capability. |
| `AFS_method_comparison_table.xlsx` | Structured transcription of S01 Table 1; validate transcription against the PDF. |
| `AFS_Examples_and_Reference.docx` (D01–D08; worked D01/D02) | **Development/evaluation fixtures and guidance only**; never index worked answers in the technical-retrieval corpus. |
| `customer_scenarios_for_final_evaluation.txt` (H01–H06) | Final evaluation inputs; run through the same general pipeline and save untouched generated outputs. |
| `AI-based molecular media analysis by process-spectroscopy in-line.pdf` | Official challenge brief and evaluation rules. |

If any listed file is absent from the actual repo, report the gap; do **not** invent its content, citations, or expected output.

They are all located in the materials folder inside knowledge folder.

---

## 1. Proposed architecture

**Evidence-grounded LLM reasoning with deterministic guardrails**. Do not build a fixed `if sample == dairy: recommend NIR` decision tree.

```text
                         DOCUMENT PIPELINE (on changes)
AFS PDF / DOCX / TXT / XLSX -> extract by page/section -> source registry
                                       -> chunks with metadata -> search index
                                                       |
                                                       v
                        CUSTOMER ASSESSMENT PIPELINE
customer text -> LLM extraction -> Pydantic CustomerUseCase -> fact checks
                                             |
                                             v
                                 context-aware retrieval
                                             |
                                             v
                                retrieved AFS passages
                                             |
                                             v
                                 LLM technical assessment
                                             |
                                             v
                         Pydantic AssessmentReport + validators
                                             |
                              +--------------+--------------+
                              |              |              |
                          Streamlit        JSON/MD      evaluation CLI
```

### Responsibilities and separation of concerns

1. **LLM extractor:** Understand varied customer wording; extract only customer-stated facts, preferences, targets, and constraints. **No technique recommendation at this step.**
2. **Pydantic/normalizer:** Guarantee a consistent schema, explicit unknowns, units-as-given, and provenance to customer text. Detect invalid extraction.
3. **Retriever:** Find relevant passages from AFS technical documents. Retrieval is not scientific validation.
4. **LLM assessor:** Compare candidate methods and geometries using the case and retrieved evidence; produce missing questions, uncertainty, feasibility experiment, and evidence references.
5. **Deterministic checks:** Validate structure, citation IDs/pages, contradiction against stated facts, supported fields, and report consistency. Do not pretend automatic checks prove scientific accuracy.
6. **Human-review UI:** Present editable extracted facts, cited conclusions, assumptions, and unresolved uncertainties. Reassess after corrections.
7. **Offline fallback:** Remain usable without Ollama/network/keys; label its assessment as **limited** and never misrepresent it as equivalent to the evidence-grounded LLM mode.

Prefer **two main LLM calls** per assessment: extraction and technical assessment. Only add a separate reviewer after evaluation demonstrates its value.

---

## 2. First action for Codex: repository audit (no rewriting yet)

Read `AGENTS.md`, `pyproject.toml`, README, domain models, Streamlit entry point, prompts, technique YAML, RAG code, exporters, tests, and existing sample inputs. Run existing checks in the repo's documented way. Inspect availability of AFS materials.

Produce a short gap assessment before modifying code:

- Which current components already implement the target stages?
- Which type contracts or source formats can be reused?
- Is the existing RAG a research chatbot or a reusable retrieval service?
- Is the LLM adapter a stub or actually callable?
- Are extraction mistakes being distinguished from recommendation mistakes?
- Which tests depend on exact scenario wording or implementation details?
- What can be implemented safely in the hackathon time available?

**Do not delete working modules just to match the directory layout below.** Change APIs through adapters/migrations as needed, keep the offline baseline working, and preserve green tests.

**First deliverable:** brief `docs/gap-analysis.md` listing keep/change/add/defer, with links to current code paths and a minimal work sequence.

---

## 3. Target data contracts (P0)

Keep Pydantic as the internal language between stages. Extend existing models rather than duplicating overlapping types.

### `CustomerUseCase`

Represent:

- Raw customer text and stable case/request identifier (not an official scenario routing key).
- Target measurand(s) and purpose: concentration, trend, classification, moisture, etc.
- Sample/material and physical state; process description and measurement location.
- Mode: in situ/in-line/at-line/off-line **when stated**, otherwise unknown.
- Known process constraints: temperature, pressure, moisture, turbidity, particle effects, flow, contamination/fouling, contact/access/cleaning, response time.
- Independent reference measurement availability and details, if given.
- Customer preferences vs strict requirements; desired stage (exploratory, pilot, production).
- **Customer-fact provenance:** each extracted factual field or claim includes a short exact supporting quote/span from input where feasible.
- Unknown scalar fields = `None`; unknown collections = `[]`; preserve numerical values and customer units as text unless a transformation is explicitly documented.
- Ambiguity/contradiction markers. Keep **customer facts** distinct from model inference and domain knowledge.

### `RetrievedEvidence`

At minimum: `evidence_id`, `source_id`, `document_version`/hash, `page_or_section`, `excerpt`, `review_status`, `retrieval_score` (if available), and `source_scope`. Preserve the actual source page; do not invent page numbers from chunks.

### `AssessmentReport`

At minimum:

- `status`: `provisional` / `insufficient_information` / `outside_supported_scope`.
- Explicit customer facts, unknowns, important assumptions and uncertainty.
- Prioritized clarification questions with a **reason** (what decision the answer affects).
- Candidate techniques (optionally a combination), associated geometry/probe concept, limitations, relevant nominal spectral range **distinguished from actual hardware coverage**, and traceable `evidence_ids`.
- Candidate comparison, including why options might fail and why a combined method may or may not add value.
- First feasibility study: representative samples/process states, independent reference measurements, comparison procedure, appropriate validation on independent batches/runs, and customer-defined go/no-go criteria.
- Diagnostics/provenance: extraction provider, assessment provider, evidence set/version, validation warnings and report version.

Do not add arbitrary numeric confidence scores unless calibrated against evidence. Prefer explicit status/uncertainty reasons.

**Acceptance:** models serialize/deserialize consistently, unknowns survive JSON and Markdown export, and no presentation layer imports provider-specific model types.

---

## 4. Working LLM requirements extractor (P0)

Create/complete a provider-neutral interface:

```python
class UseCaseExtractor(Protocol):
    def extract(self, customer_text: str) -> CustomerUseCase: ...
```

Implement:

- A **real** LLM implementation using the currently chosen provider (Ollama is acceptable) and provider-supported structured JSON/schema output.
- Existing conservative offline implementation retained under the same interface.
- A focused extraction prompt stored under `prompts/` (or existing prompt location): only stated facts, no technical recommendations, no fabricated constraints, unknowns remain `None`, preserve units, exact input quotes for fact provenance.
- Pydantic validation, bounded repair/retry for malformed JSON, sensible timeouts and actionable error handling.
- Literal substring checks for claimed quotes; invalid/missing quote -> warning/review or re-extract, **not** uncritical trust. Semantic correctness is not proven by matching a quote.
- Visible `extraction_mode` and graceful fallback when the local model is unavailable. Never silently present fallback as LLM output.

**Acceptance tests:**

- Paraphrases such as "API", "active ingredient", and "potency" produce compatible target representations when justified.
- Distinguish explicit feed-frame location from an unstated location.
- An unspecified temperature/concentration/accuracy stays unknown.
- Malformed JSON, model timeout, and provider unavailability are handled clearly.
- A malicious customer instruction requesting fabrication is treated as customer text, not system authority.

---

## 5. Knowledge ingestion + retrieval (P0)

### 5.1 Source registry and provenance

Create/extend a registry of source IDs (`S01`–`S06`), source filenames, content hashes/versions, source type (reference, research paper, company context), review status, and allowed citation format. **Unreviewed does not mean verified.** Make source scope available to the assessor.

### 5.2 Ingest technical documents

- Parse PDF page-by-page, DOCX by headings/sections, TXT directly, and XLSX tables where useful. Use existing working dependencies; only add more when necessary and Python 3.14 compatible.
- Keep page/slide references after chunking. Store quoted/near-verbatim text excerpts separately from authored summaries.
- Detect poor/empty extraction (especially slide figures/table content). Mark those pages as needing manual review or a documented transcription; **do not hallucinate what an image-only slide says**.
- Chunk using page/section boundaries with modest overlap. Do not mix claims from different pages and attribute them to one page.
- Deduplicate by source content hash. Provide a `reindex` CLI command that can update changed documents without corrupting the registry.
- Separate the **scientific source corpus** from evaluation scenarios and worked answers. Exclude `AFS_Examples_and_Reference.docx` worked responses and H01–H06 from the technical retrieval index.
- Treat all retrieved text as **untrusted data**, never as instructions to the LLM or runtime.

### 5.3 Retrieval

Implement a small reusable interface such as:

```python
class EvidenceRetriever(Protocol):
    def search(self, use_case: CustomerUseCase, limit: int = 8) -> list[RetrievedEvidence]: ...
```

- Start with reproducible keyword/BM25 retrieval plus metadata filtering; semantic embeddings can be added if testing shows a gap.
- Build queries from material, measurands, process conditions, geometry constraints and synonyms. Avoid relying on industry label alone.
- Retrieve **competing/limiting evidence** too (not only passages praising one method).
- Log retrieved IDs, source references and retrieval mode in reports for debugging.
- Measure whether relevant evidence is retrieved on D01–D08; fix retrieval misses before adding complexity.
- Keep optional exploratory document Q&A separate from the decision pipeline.

**Acceptance:** a new document can be added, indexed, searched, and cited without writing scenario-specific application rules; unreviewed/research/demo scope remains visible.

---

## 6. Evidence-grounded technical assessor (P0)

Create/complete a provider-neutral interface:

```python
class TechnicalAssessor(Protocol):
    def assess(
        self,
        use_case: CustomerUseCase,
        evidence: list[RetrievedEvidence],
    ) -> AssessmentReport: ...
```

The LLM assessment prompt receives:

1. Validated `CustomerUseCase`, including unknowns and original customer statements.
2. The **retrieved excerpts** with immutable `evidence_ids`, source/page metadata, and review/scope context.
3. Structured `AssessmentReport` JSON schema.
4. Rules: compare candidate methods, give conditional recommendations, ask high-impact questions, distinguish nominal method ranges from selected instrument/fiber/probe coverage, distinguish Raman excitation (nm) from Raman shift (cm^-1), refuse performance guarantees, and propose a discriminating feasibility experiment.

Prefer a short technical comparison over long generic explanations. Do **not** require a fixed number of candidates. A combined technique is a hypothesis to compare against the best single method, never an automatic upgrade. A method can be considered provisionally even when implementation specs remain unknown, but must not be presented as a selected hardware product.

Use exact retrieved evidence IDs for report citations, not LLM-composed arbitrary bibliographic references. When evidence is insufficient, return a conditional shortlist with caveats or `insufficient_information`; do not force a definitive recommendation to fill a UI card.

**Acceptance tests:**

- D01 powder/API: reasonable NIR reflection / Raman comparison can be considered with proper caveats (not hardcoded answers).
- D02 opaque dairy: reflection evidence is weighed against turbidity/texture concerns.
- New/paraphrased queries use the same pipeline, with no scenario-ID routing.
- Unrelated evidence cannot be cited as supporting a claim just because a keyword matches.
- If documents contain no support, the model says so instead of inventing a source.

---

## 7. Deterministic guardrails and source checks (P0)

Implement validators **after** generation (and after extraction where applicable):

- Pydantic shape/type validation.
- Every referenced evidence ID belongs to the retrieved set; each source/page exists in registry.
- Reject or flag claims that turn unknown customer quantities into stated facts.
- Reject or flag unsupported exact accuracy, detection limit, response time, fiber length, prices, or component ratings.
- Enforce valid unit labeling, especially wavelength (nm / µm), wavenumber (cm^-1) and Raman shift vs excitation.
- Check the assessment includes meaningful risks, critical questions, and a first experiment with independent reference data whenever quantitative performance is to be evaluated.
- Warn when a measurement setup contradicts a **known and verified** process constraint. Do not infer unavailable component ratings.
- Indicate which support is an application demonstration vs an instrument specification vs an inference.
- Surface the limitations of checking: valid IDs and present text do **not** establish that a scientific interpretation is true. Allow human expert review.

Make validation outcomes inspectable (`error`, `warning`, `not_checked`). Repair only structural issues automatically with bounded retries. For questionable scientific claims, downshift certainty or request expert review instead of hiding the error.

No arbitrary numerical "confidence" meter.

---

## 8. Contextual questions + feasibility plan (P0)

These are parts of the `AssessmentReport` generated from the case and sources; they need not be separate LLM agents.

**Clarification questions:** Prioritize questions likely to change method choice, geometry, installation, or go/no-go evaluation. Each question includes why it matters. Avoid rote lists that repeat already-known customer facts.

**Experiment planner:** For each serious candidate, require:

1. Representative material/process states covering expected variation.
2. Independent, appropriately timed and identifiable reference assays or labels.
3. Measurements for each proposed method/geometry, with relevant physical variations.
4. Single-method baselines before data fusion where a combination is proposed.
5. Independent validation on held-out batches/runs/states; compare bias/error/trend tracking as appropriate.
6. Numerical acceptance criteria **to be provided by customer**, not invented by AI.

Adapt the plan to the problem type: concentration regression, classification, process trend or geometry comparison. Do not make every scenario produce the same experiment text.

---

## 9. UI, exports and expert correction loop (P1 after core works)

Keep Streamlit. There is no demonstrated need to migrate to FastAPI for this hackathon.

**Minimal demo flow:**

1. Customer enters an industrial enquiry or selects a development example (selection only populates the input text).
2. UI shows extracted facts with supporting quotations, unknowns and extraction mode.
3. Expert can edit/correct extracted facts and rerun retrieval/assessment.
4. UI displays decision status, 1–N method candidates, high-level probe/geometry, evidence citations, missing questions, risks and an experiment plan.
5. Evidence expander shows source ID, page, excerpt, scope, and review status (not merely a source filename).
6. Export validated `AssessmentReport` to reproducible JSON and readable Markdown.
7. Show warnings and provider failures honestly; no fabricated "validated" or "ready for production" badges.

Prioritize legibility and one compelling live demo, not heavy front-end customization. Keep all business logic outside Streamlit; `streamlit run ...` must just call the application services.

---

## 10. Evaluation and anti-overfitting (P0; continuous)

Make an evaluation runner that uses **the same assessment function as the UI**. Do not make different special-case pipelines for development, final evaluation, or live demo.

### Tests and cases

- All eight development enquiries D01–D08 (source supplied); use worked D01/D02 responses as **review references**, not training/retrieval documents or exact-output templates.
- At least one paraphrase of each development case.
- Incomplete/no-target cases requiring questions or abstention.
- Contradictory constraints, unfamiliar media, unsupported applications, document conflict, and prompt-injection text.
- New-document ingestion regression: introducing a relevant evidence source changes retrieval and may change a provisional conclusion appropriately, without changing scenario-specific Python code.
- H01–H06 through a clean, identical pipeline, saving generated JSON/Markdown outputs **without manual editing**. If files are not available, document that blocker; do not synthesize official H scenarios.

### Evaluation metrics (no fake scientific accuracy)

Track:

- Extraction fact precision/coverage and fabricated-fact count.
- Retrieval relevance, missing important source pages, and citation ID validity.
- Key claim-to-citation support audited manually on a test subset.
- Proportion of reports with meaningful uncertainty/questions and actionable experiments.
- Unjustified precise specifications/performance claims (**target zero**).
- Offline mode health, provider error handling and end-to-end run success.
- Human expert assessment for scientific defensibility. Do not label output as scientifically validated because unit tests pass.

Do not assert technique rankings are ground truth. Multiple conditional approaches may be defensible.

Provide CLI commands (adapt to current repo) equivalent to:

```bash
uv sync
uv run pytest
uv run python -m spectro_ai.eval --dataset development
uv run python -m spectro_ai.eval --dataset holdout --output evaluation/outputs/
uv run streamlit run <existing_app_entrypoint>
```

The `holdout` command must not change prompts/rules per case. Note actual CLI spelling in README; don't leave fictional commands documented as working.

---

## 11. Suggested code organization (adapt to current repo)

```text
src/spectro_ai/
  domain/             # Pydantic models and enums
  extraction/         # LLM and conservative offline extractors
  knowledge/          # source registry, document parsing, chunking, indexing, retrieval
  assessment/         # LLM assessor, constraints, validators
  reporting/          # JSON/Markdown serialization
  providers/          # Ollama / optional provider adapters
  prompts/            # versioned extraction/assessment prompts
  app.py              # Streamlit UI; thin presentation layer

data/                 # supplied source documents (where licensing permits)
knowledge/            # source manifests and curated technique vocabulary
evaluation/          # scenario fixtures, runner, reports (NOT RAG corpus)
tests/                # unit, integration, adversarial, UI smoke
docs/                 # architecture, gaps, setup, limitations, demo
```

Reuse existing folder names if the repo has a coherent layout. Keep dependencies lean; no new framework for functionality that is a couple of functions.

---

## 12. Implementation order and checkpoints

### Phase 3 — Grounded assessment (P0)

- Add structured LLM assessor, conditional method comparison, questions, experiment plan and evidence IDs.
- Keep deterministic checks and fallback behavior.

**Gate:** outputs on D01/D02 are defensible against worked examples without matching canned text; unfamiliar inputs can abstain.


#### Blocker — 2026-10-17 — live semantic gate not accepted

Implementation exists, but Phase 3 remains pending after two focused repair attempts.
Software checks pass (127 tests, Ruff lint/format, Streamlit HTTP 200/ok); these do
not establish scientific defensibility. Final D01 live output still presents an
unqualified fusion accuracy advantage, leaves candidate limitations empty, and
transfers FTIR/ATR extrusion evidence into a powder feed-frame candidate without
sufficient support. Earlier D02 output invented an acceptance threshold; generic
planning safeguards now remove invented criteria and prevent production next steps.
Final D02 also copies a source-typical 1–10 mm optical path into a proposed probe
configuration without clearly distinguishing it from a customer-selected setup.

Preserve the original task and gate above. Do not mark Phase 3 complete or proceed
to Phases 4–5 under this request. Further work needs a separately reviewed approach
to semantic reliability/expert assessment; additional prompt tuning stopped under
AGENTS.md failure handling after two unsuccessful repairs. Generated records remain
untouched in ignored `.cache/phase3-live`, `-v2`, and `-v3` directories. Baseline
remains default, credential-free, and independently tested. See HISTORY.md for
implementation and failed-gate details.

### Phase 4 — Validation + evaluation runner (P0)

- Add consistency/citation checks and transparent diagnostics.
- Add automated development, paraphrase, adversarial and held-out execution.
- Run through official final scenarios if present, without editing answers.

**Gate:** source/citation integrity tests and end-to-end checks pass; reports are reproducible and inspectable.

### Phase 5 — Presentation polish (P1)

- Editable extraction, evidence expanders, clean candidate comparison, exports.
- Write a 5-minute live demo script using the real pipeline, plus reproducible setup commands.
- Show one case change affecting the recommendation, and one appropriate insufficient-info result.

**Gate:** an unfamiliar customer text can be processed live with clearly indicated uncertainty; README setup works from a clean environment.

### Defer unless core gates pass

- Vector database server; extensive embedding/reranking experiments.
- Multi-agent frameworks or autonomous web research.
- FastAPI migration, authentication/accounts or persistent production infrastructure.
- Full spectral modeling / PLS calibration without measured spectral/reference datasets.
- Complex analytics dashboards or highly custom UI.

---

## 13. Definition of done

Implementation is ready for hackathon submission only when:

- [ ] Original customer enquiry -> extraction -> retrieval -> assessment -> validation -> UI/export works end-to-end.
- [ ] Pydantic schemas explicitly preserve unknowns and customer-fact provenance.
- [ ] A working LLM adapter is present **and** the conservative offline mode still runs without secrets/network.
- [ ] Important technical claims have traceable source IDs, pages and supporting excerpts; unsupported claims are flagged.
- [ ] Reports clearly state provisional status, scientific limitations, and missing information.
- [ ] First experiments contain independent references, comparisons, and customer-defined go/no-go criteria.
- [ ] D01–D08 and adversarial/paraphrase tests run; no scenario-ID or exact-phrase decision branches exist.
- [ ] H01–H06 are generated by the same pipeline and included unedited if supplied.
- [ ] New documentation can be reindexed without altering the assessor's case-specific code; source version/review status remains traceable.
- [ ] `uv` setup, tests, application command, and evaluation runner are documented and demonstrably work.
- [ ] README/architecture explain what is supported, what is provisional, and what still needs AFS/customer validation.

## 14. Codex execution discipline

1. Read this file and `AGENTS.md` first; honor existing repo conventions.
2. Inspect, propose minimal deltas, then implement **one phase at a time**.
3. After each phase: run targeted tests, fix regressions, and summarize changed files + remaining risks.
4. Prefer testable, typed, simple functions. Use interfaces only where they separate meaningful implementations (e.g., extractors/providers/retrievers).
5. Do not silently add credentials, telemetry, paid services, invented source documents, technical ratings, or hidden data dependencies.
6. Do not hard-code output recommendations based on D/H scenario IDs, exact phrases, or worked solution texts.
7. Preserve provenance and distinguish **customer facts**, **source claims**, **engineering hypotheses**, and **experimental validation needs**.
8. If files, specifications, or expert decisions are missing, document the blocker and choose a conservative fallback; do not fabricate missing inputs.
9. Stop adding features when the core pipeline and test/evaluation gates are sound. Spend remaining time on scientific correctness and the demo.

**Success is not an impressive-looking answer. Success is an assessment that an AFS expert can trace, challenge, correct, and use to plan the next feasibility experiment.**


## Constraints

Keep Python 3.14, uv, typed Pydantic domain models, Streamlit, credential-free offline
baseline, four originals, abstention and Markdown export. Unknown scalars remain None;
collections stay empty; customer quantities remain text. Distinguish verified general
principles, engineering hypotheses and assumptions. No invented hardware limits,
confidence or validation. No new framework, vector service, training, hardware control,
broad refactor or deployment stack. Larger retrieval integration remains P2 unless
reviewed evidence establishes need.

After accepting future work, append its exact original text, date, implementation,
verification and deviations to HISTORY.md before removing it from this plan.
