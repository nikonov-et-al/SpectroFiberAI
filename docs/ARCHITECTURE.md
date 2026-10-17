# Architecture

## Boundaries

Streamlit owns presentation, example selection, and session results. It calls the same
assessment service as other callers. Baseline mode is fully offline. Ollama mode runs
structured local extraction, evidence retrieval, and a provisional grounded assessment.
Provider failures preserve a valid corrected case and visibly fall back to the
deterministic baseline; extraction failures alone use the offline literal extractor.

`models.py` defines CustomerUseCase, RetrievedEvidence, TechniqueCandidate,
FeasibilityPlan, AssessmentReport, and knowledge types. Scalars support unknowns; lists
default empty. Extra fields and invalid statuses are rejected. Quantities remain
supplied text; raw input, field quotes, extraction mode, and warnings are retained.

`baseline.py` recognizes labels and a small literal phrase vocabulary. Labels override
inference. Negated clauses are dropped; this is lexical matching, not general language
understanding. Objective, target, and matrix gate candidate generation. Rules inspect
extracted objective/property and contextual matrix, interference, sensitivity and
installation facts. All missing requirements remain visible; questions prioritize
critical gaps and relevant follow-ups. No rule match means abstention.

`intake.correct_requirements()` validates scalar expert edits against the domain model,
preserving raw input, untouched facts and customer units. Field provenance records
literal phrases, labeled source clauses, exact quotes, or expert edits. Collections
and extraction metadata are not exposed as UI text fields. Streamlit hides assessment
and exports while corrections are pending, then calls the same assessment service with
the corrected case without repeating extraction.

`KnowledgeSource.get_techniques()` returns validated profiles. YAML uses safe loading,
schema version 1, unique IDs, and provenance validation. Missing files, invalid YAML,
and bad schemas surface useful errors. Editable checkouts use root YAML; wheels
include a copy. Add/replace a source when authoritative documentation becomes available.

`reporting.py` renders deterministically without timestamps or network access. Raw
input is safely fenced; structured fields escape Markdown/HTML. Changed input hides
the stale result and export until reassessed.

## Scientific meaning

- **verified_source**: a real source was checked for the stated high-level principle.
  Title, URL, and check date are mandatory; customer suitability remains unverified.
- **engineering_heuristic**: provisional rule or screening note requiring expert review.
- **assumption**: an unconfirmed prerequisite, never a measured result.

Every profile requires expert review. Candidates are unranked and provisional. Numeric
confidence/performance, exact wavelength, and hardware probe specifications require
actual evidence. Geometry suggestions are conditional laboratory studies justified
by explicit state/access facts; they never assert AFS compatibility. Tailored plans
retain unknown thresholds/references, controls, held-out validation and outcome branches.

## Structured local extraction

`UseCaseExtractor` has an offline implementation and an Ollama adapter in
`providers.py`. Ollama receives a Pydantic JSON schema and the packaged extraction
prompt. Customer input is untrusted data. Every returned fact must carry a literal
input quote; absent or nonliteral quotes clear the field and add a warning. Numeric
values must remain exact substrings of their quote. This verifies quote membership,
not semantic correctness. Structural output gets at most one retry. Provider failure
uses conservative offline extraction with visible `extraction_mode` and warning.
`SPECTRO_AI_PROVIDER=baseline` remains default. Ollama assessment uses the separate
allowlisted AFS evidence index, a small typed assessor response, and candidate evidence
IDs. The service validates source/version/page membership and rejects nontechnical or
lexically unrelated citations. Lexical screening is only a necessary check, not
entailment or scientific validation. Report metadata records the method, report
version, evidence-set hash, review state, and warnings. The Ollama prompt and schema
cannot replace the service-owned customer facts or retrieved excerpts. `openai` remains
an explicit unimplemented assessment provider; optional OpenAI API mode belongs to
document Q&A. Deterministic engineering-planning safeguards replace invented acceptance
thresholds with customer-stated criteria (or TBD), add independent reference and
held-out validation steps, and limit next steps to expert feasibility review. Reports
disclose these process safeguards; they do not validate scientific suitability.

No backend service, model training, hardware driver,
deployment stack, or multi-agent framework. Baseline works locally without credentials
or networking after setup. Streamlit telemetry is disabled.

## Document RAG boundary

Document Q&A is independent of `AssessmentReport`, `assess_problem()`, YAML technique
profiles, and `KnowledgeSource`. Retrieved passages never automatically become
verified knowledge or applicability rules. Streamlit calls the same orchestration
as the CLI; ingestion, retrieval, and storage do not depend on Streamlit.

Local TXT/Markdown/PDF → deterministic chunks and locations → Ollama embeddings →
SQLite metadata/text/vectors. Query → local embedding → NumPy exact cosine search
(top four, stable ties) → supplied excerpts → selected generator → Pydantic answer.
Chunks target 1,200 characters with 200-character overlap; PDF pages stay separate.
SQLite index metadata binds schema, chunk settings, embedding identity/digest and
dimension. Generation model changes do not invalidate that index.

Index changes are atomic after extraction and embedding succeed. Unchanged chunks
reuse vectors; deleted sources disappear only from the index. Incompatible settings
require an explicit rebuild. Empty corpora can abstain; missing/corrupt indexes and
model failures are errors. No similarity threshold is presented as scientific truth.

Model adapters remain in `providers.py`, prompts ship as package resources. Ollama
uses `/api/embed` without truncation and schema-constrained generation. Embeddings
are unloaded before generation. External generation uses Chat Completions JSON mode;
both paths validate the same typed answer. Each answer statement cites retrieved
chunk IDs, which the application resolves to document paths and locations. Unknown
IDs and uncited statements fail validation. This checks references, not entailment.

The UI displays sources and external data handling, exports document answers
separately, and hides results after question/provider/index changes. Network errors
have finite timeouts and no fallback; credentials and raw provider errors stay out
of displayed/logged error messages. Default tests mock providers; real GPU/API
rehearsals are separate from offline regression checks.

Index writes stage a complete SQLite database before atomic replacement; explicit
rebuild can recover from a corrupt index. Generation bounds context conservatively
using UTF-8 bytes, dropping lowest-ranked whole excerpts until the input fits.
This may supply fewer than four passages and does not prove the model's actual
context capacity. A separate real Ollama rehearsal passed on the RTX 4070 with context 4096 and a synthetic TXT chunk; this does not validate larger-context workloads.

## Scientific source evidence

`evidence.py` provides a separate allowlisted source index for the assessment layer.
`knowledge/sources.yaml` records source IDs, file names, scope/type, review status,
citation format, and related-source metadata. Paths must resolve under the manifest's
materials root. The D worked-example document and H evaluation enquiries are excluded
by both their canonical names and fixture/evaluation scope tags. Sources not present in
the manifest are never indexed.

Reindexing uses pypdf for page-aware PDF text, line spans for TXT, section blocks for
DOCX, and worksheet row blocks for XLSX. Content hashes version each source; repeated
excerpt text is stored once with all source/page provenance retained. The index is
rebuilt in memory and atomically replaced, and absent source files are listed as gaps.
The retriever validates manifest/file versions and citation locations before local
BM25 search. `RetrievedEvidence` excerpts remain raw evidence; they do not populate
`TechniqueKnowledge` or prove scientific entailment. Short PDF extraction and all S02
slides carry manual-review flags. The S01 table workbook remains a separate,
page-6-linked transcription with its own hash and worksheet-row citations.

The Ollama assessor uses this index; baseline mode remains independent. Source coverage
can be checked mechanically; scientific defensibility and image-only support still
need an expert. Whole, lowest-ranked excerpts are removed when needed to fit the model
context; customer text and excerpts are never truncated. The Ollama assessment pipeline
is experimental and has not passed the Phase 3 live review. D01/D02 still overstated
source-level comparisons or transferred source details into customer-specific
configurations after two focused repairs. The offline baseline remains the default.
H01–H06 holdouts and Phases 4–5 have not run.
