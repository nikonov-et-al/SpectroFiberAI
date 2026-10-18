# Hackathon plan

## Architecture update — 2026-10-18

User approved removal of offline parser, deterministic recommendations and all fallback.
Live model availability and expert scientific acceptance remain independent blockers;
Current update passes 119 tests, Ruff lint/format and local startup HTTP 200/ok.
All 31 local LLM evaluation inputs completed without runtime errors; results in
[LLM_EVALUATION.md](LLM_EVALUATION.md). Expert acceptance and OpenAI live smoke remain
open. Prior counts below are historical.

## Foundation implemented

Structure-aware ingestion is being delivered in independently verified milestones.
Schema 3 adds source block/section relationships while retaining schema 2 reading;
updates remain explicit and atomic. Markdown structure and opt-in local Docling
PDF conversion are implemented; pypdf stays default because real AFS table and
heading extraction failed the promotion gate after two focused configuration
attempts. Bounded parent context is implemented with 30% reserved retrieval capacity,
same-source block windows and exact context quotation provenance. Parent expansion defaults off until regression and real-source
review gates pass; this work does not establish scientific acceptance.
DOCX/XLSX/TXT also use source-block packing with explicit headers and original
paragraph/row/line positions. The offline scientific-context tests cover repeated
values under different operating conditions, exact citations, same-source
limitations, tampering and provider budgets. Real AFS usefulness remains unaccepted.
Final three-case pinned BM25 comparison stays within retrieval budgets but misses
the expected qualification passages; default provider fitting drops all attached
context. Keep expansion disabled. Detailed measurements and both promotion-gate
blockers are in [STRUCTURE_CONTEXT_REVIEW.md](STRUCTURE_CONTEXT_REVIEW.md).

- Python 3.14 with uv-managed locked dependencies.
- Typed requirements, candidates, evidence, feasibility plans, reports.
- LLM-only extraction and grounded assessment; deterministic missing-field checks and explicit abstention.
- Ollama and OpenAI schema-constrained extraction with quote checks; no offline fallback.
- Experimental Ollama/OpenAI assessment using the allowlisted AFS index. Reports bind citations to retrieved excerpts and preserve source/page/review metadata; unsupported and insufficient-information outcomes are labeled separately. Phase 3 live acceptance is pending: D01/D02 still show semantic transfer errors after two focused repairs. Ollama is default; no baseline path remains. This path is not scientifically validated for demo claims.
- Five preliminary technique profiles with provenance and expert-review flags.
- Streamlit demo, four synthetic scenarios, structured JSON, Markdown and JSON downloads.
- Unified SQLite knowledge base with persisted Ollama embeddings and BM25/RRF hybrid retrieval for assessment. Streamlit Update Knowledge Base performs incremental atomic updates; reports retain index revisions.
- Curated AFS source metadata and folder-based PDF/MD/TXT additions share chunks and citations. Standalone Document Q&A removed; expert acceptance remains open.
- Tests, Ruff, GitHub Actions checks.
- Structured expert answers/corrections, field provenance, consequential questions,
  contextual candidate conditions, conditional geometry, tailored study outcome branches.
- Six separate synthetic evaluations; see [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md).
  Ready-to-use workflow and recovery: [DEMO_PLAN.md](DEMO_PLAN.md).

The supplied challenge brief names ArtFiberSystems' **AI Assistant for In-line
Spectroscopy Solution Design** and lists the evaluation weights. AFS source files and
D01–D08/H01–H06 fixtures are present locally. Independent organizer confirmation,
actual hardware specifications, measured customer data, and achievable performance
remain unconfirmed. Source coverage and review status: [AFS_SOURCE_REVIEW.md](AFS_SOURCE_REVIEW.md)
and unresolved questions: [CHALLENGE.md](CHALLENGE.md).

Software checks: 127 tests passed, Ruff lint/format passed, and startup returned
HTTP 200/ok. Live D01/D02 Ollama assessment outputs are exploratory and failed Phase 3
semantic review after two focused repairs; see the limitations above. Automated
workflow rehearsal passes; human event rehearsal and scientific review remain pending.

## Open questions

Independent organizer confirmation, representative measured customer data,
instrument/hardware specifications, permitted external services, and live-demo
constraints. The supplied brief defines the rubric, and source files plus scenario
fixtures are present locally. See CHALLENGE.md. Development target: user-reported RTX
4070 with 12 GB VRAM; synthetic Ollama rehearsal passed; single-run measurements are
in HISTORY.md.

The first AFS evidence index run found 7 allowlisted files and 160 unique excerpts;
all eight saved D-case queries returned eight lexical hits and surfaced each case's
main application page in the top eight. This is retrieval coverage only. False
positives remain, D06 fiber-range guidance was not retrieved in the top eight, and no
retrieved excerpt is a scientific suitability judgment. Page coverage: [AFS source
review](AFS_SOURCE_REVIEW.md).

## Prioritized milestones

1. **Intake:** record outcome, criteria, assets, hardware, restrictions. Exit: concrete
   demo acceptance case.
2. **Expert alignment:** validate knowledge/rules; obtain representative cases,
   reference answers, and consequential failures.
3. **One workflow:** adapt extraction, rationale, and experiment to the challenge;
   demonstrate input through reviewable export.
4. **Evidence/reliability:** replace assumptions with official sources, add regressions,
   score reports using EVALUATION.md, rehearse failures.
5. **Rehearsal:** freeze proven path, fit allotted time, prepare local synthetic fallback.

Prioritize complete behavior over feature count. Resolve domain questions before
tuning AI against unverified labels.

## Optional stretch features

Expert validation of retrieved document content; instrument constraints; challenge
dataset exploration; German translation; hardware only if required/available.
Choose against the actual rubric.

## Demo checklist

- [ ] Confirm challenge and allowed assets with source/date.
- [ ] Run locked sync, pytest, Ruff lint/format on demo machine.
- [ ] Confirm local app/report generation without credentials.
- [ ] Rehearse food, polymer, fluorescence, and insufficient-information cases.
- [ ] Confirm structured output and Markdown match current input.
- [ ] Show provisional status, gaps, and evidence categories.
- [ ] Confirm configured LLM and evidence index after initial install.
- [ ] Rehearse customer question → gaps → candidates → testable experiment.
- [ ] Agree presenter, time limit, fallback, recovery steps.
- [ ] Have domain expert review scientific claims before pitching suitability.

## Risks

| Risk | Mitigation |
| --- | --- |
| Extraction misses language or semantic details | Show exact quotes, raw input, and gaps; review model extraction; show failures without fabricated fallback. |
| LLM output mistaken for scientific validation | Provisional status, evidence labels, abstention, expert review. |
| Sources do not describe actual instruments | Obtain AFS documents; validate compatibility separately. |
| No reference labels/samples | Keep thresholds unknown; prioritize reference acquisition. |
| No internet/API | Install local Ollama/models early; verify local generation. |
| Future SDK incompatible with Python 3.14 | Assess when known; record agreed runtime change. |
| Infrastructure consumes hackathon | Keep one process and local SQLite index; defer deployment/frameworks. |

## First three tasks when announced

Confirm acceptance criteria/assets; validate knowledge/rules with experts; implement
one challenge-specific workflow with regression coverage.

## Knowledge-base rehearsal

- [x] Install `rag` extra, local Ollama, and configured generation/embedding models.
- [ ] Index only authorized documents; inspect page/line provenance and search results.
- [x] Rehearse supported/unsupported questions with real Ollama; test unavailable models and rebuild offline.
- [ ] Confirm customer enquiry and evidence transfer is acceptable before choosing OpenAI assessment.
- [x] Measure GPU residency, post-run total VRAM, indexing time, and answer latency on the RTX 4070 (single synthetic run; no peak-memory claim).
- [ ] Review cited statements against source passages; similarity is not confidence.
- [ ] Preserve four original synthetic inputs through the LLM pipeline.

No challenge dataset, instrument specification, or scientific suitability has been
validated by adding retrieval. Scanned PDFs and complex tables need another workflow.
# Assessment refinement — 2026-10-18

## Optional reranker trial — 2026-10-18

Add an opt-in English CPU cross-encoder trial after fused retrieval. Default retrieval
remains unchanged. Enabled searches rank up to 30 distinct candidates before applying
the existing evidence count and context budget. Model failures retain RRF ordering
with visible technical diagnostics; embedding and assessment failures remain errors.
Ranking scores measure textual relevance, not scientific confidence or applicability.

Retrieval benchmarks compare original retrieval, widened RRF and reranking against
one pinned index. New benchmark questions are provisional and have no invented
relevance labels. Human review must bind labels to exact evidence/source versions
before precision, recall or MRR can support a quality claim. Real-model runtime
verification is separate from network-free software tests. No default promotion,
scientific acceptance, Phase 3 closure or new challenge facts follow from this work.

Runtime feasibility verified on this Windows host: Python 3.14.4,
sentence-transformers 5.7.0 and PyTorch 2.14.1+cpu. Explicit setup downloaded the
trial model; fresh-process CPU inference produced finite scores. Model revision:
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. This verifies execution only.

Six-query hybrid timing smoke against the existing 197-chunk index also completed:
one warm repetition per query, mean original retrieval 1.90 s, widened RRF 1.73 s,
reranked retrieval 2.50 s; first model-load-and-score 7.54 s. This small run is
runtime evidence, not a stable performance estimate. All relevance metrics remain
null pending human labels; no quality improvement is claimed. Generated output is
ignored at `.cache/retrieval-benchmark-hybrid.json`; reproduce with the README CLI.

Customer view separates exploratory technique assessment from production readiness;
technical diagnostics retain the validation audit trail. Claim-level validation and
targeted local retrieval preserve provenance while removing unsupported secondary
claims. Feasibility plans compare retained techniques with independent references,
representative variation, held-out evaluation and customer-agreed acceptance criteria.
These are software safeguards and study-design heuristics, not scientific acceptance.
Challenge facts, hardware capability and customer performance remain unconfirmed.
