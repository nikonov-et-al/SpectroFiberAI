# Grounded LLM pipeline — pending acceptance

## Structure-aware ingestion and bounded scientific context

Implement five independently accepted milestones. Preserve existing retrieval APIs,
BM25/semantic/RRF behavior, reranker fallback, assessment errors/abstention, citation
guards and Markdown/JSON exports. Do not reinstate removed provider fallbacks.
Keep the demo functional after every milestone. Parent expansion defaults off.

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

### 4. Scientific-context regression and AFS usefulness gate

- Add offline end-to-end tests through ingestion → retrieval → prompt construction → mocked assessment → validation → exports.
- Cover table headers and values, customer/source units, device identity, operating conditions, calibration limits, nearby qualifications and contradictory/limiting passages.
- Include adversarial cases: value separated from unit; limitation immediately after claim; similar values on different pages; ambiguous headings; multi-page tables; tampered block references; repeated parent context falsely presented as corroboration.
- Assert exact document version and page/line/block provenance, no title-only quotation support, unchanged source-review status, and correct abstention/unsupported-claim behavior.
- Run expansion-off/on comparisons against pinned real AFS indexes. Review at least one table/specification question, one operating-condition question and one limitation question; record expected source passages before comparing outputs.
- Require useful surrounding context on reviewed cases, no observed citation/context regressions, and recorded latency/context-volume results. Retrieval metrics require human-reviewed, index-bound labels.
- Keep default disabled unless regression checks pass **and** real-document review demonstrates usefulness. Enabling default requires a separately recorded acceptance decision.

**Acceptance:** reproducible test results and reviewed AFS examples exist. Software success alone does not close scientific acceptance gates.

### 5. Broader format improvements

- Extend shared block packing to DOCX headings/paragraphs/tables, XLSX worksheets/rows/explicit headers, and TXT paragraphs with original line locations.
- Preserve cell positions, empty cells and customer units; do not infer absent headers or scientific meaning.
- Keep curated YAML technique profiles as structured records.
- Apply same provenance, vector-reuse, context-window and scientific-context tests.

**Acceptance:** format-specific regressions pass, existing examples/exports remain usable, and demo smoke succeeds.

Each milestone requires targeted tests, full pytest, Ruff lint/format, demo smoke,
Sol review and a Conventional Commit. Preserve unrelated work. Copy each accepted
milestone's exact original text into HISTORY.md before removing it here.


Updated: 2026-10-18. User-approved architecture replaces every earlier requirement
for an offline parser, baseline recommender, or automatic provider fallback.

Enquiry → Ollama/OpenAI extraction → allowlisted evidence retrieval → LLM assessment
→ deterministic validation → expert-reviewable report. Ollama defaults to a 16,384
context budget, 4,096 output tokens and 240 seconds per request. OpenAI is explicit;
no paid requests are authorized by merely having a key present.

## Accepted software and remaining gates

Offline parser/recommender/fallback removed. Both adapters, shared validation,
expert corrections, Markdown/JSON export and evaluation runner are implemented.
150 tests and Ruff lint/format passed after optional local reranking was added;
earlier local Streamlit HTTP 200/ok also passed. A local Ollama
smoke completed with explicit abstention after unsupported candidates were removed.
All 31 local evaluation inputs completed without runtime errors; see
[LLM_EVALUATION.md](docs/LLM_EVALUATION.md). This establishes software behavior, not
scientific acceptance. Strict quote guards can over-abstain; expert review remains open.

Optional English CPU reranking is implemented and remains disabled by default.
Explicit model setup, cached inference, visible RRF fallback and retrieval benchmark
runtime smokes passed. Human relevance labels and any default-promotion decision
remain pending; no retrieval quality improvement or expert acceptance is claimed.
Completed implementation tasks and exact original wording are in HISTORY.md.

- OpenAI live smoke remains blocked: OPENAI_MODEL and OPENAI_API_KEY unset.
  Obtain explicit authorization before any paid request.
- Expert review remains required for D01/D02 and unfamiliar enquiries. Check source
  application scope, fusion claims, geometry, customer-defined thresholds and study
  usefulness. Do not close Phase 3 based on automated checks or mere abstention.
- Human demo rehearsal remains pending. No source/hardware/customer performance
  claim becomes validated through this implementation.

## Original phase tasks and gates (still pending acceptance)

Original task wording and recorded 2026-10-17 blocker follow unchanged. Historical
references below to baseline/default/offline fallback are superseded by the explicit
2026-10-18 architecture above. Operational provider failures now produce errors;
scientific insufficiency produces neutral abstention. Do not reinstate baseline.

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

Completed implementation and verification belong in HISTORY.md. Keep these gates
pending until their own acceptance evidence exists.
