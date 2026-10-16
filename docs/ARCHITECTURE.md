# Architecture

## Boundaries

Streamlit owns presentation, example selection, and session results. Provider selection
runs extraction/assessment for baseline or fails explicitly for unavailable providers.

`models.py` defines CustomerUseCase, TechniqueCandidate, FeasibilityPlan,
AssessmentReport, and evidence/knowledge types. Scalars support unknowns; lists
default empty. Extra fields and invalid statuses are rejected. Quantities remain
supplied text; raw input is retained.

`baseline.py` recognizes labels and a small literal phrase vocabulary. Labels override
inference. Negated clauses are dropped; this is lexical matching, not general language
understanding. Objective, target, and matrix gate candidate generation. Rules inspect
extracted objective/property. Every missing requirement generates a question. No rule
match means abstention.

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
confidence/performance, exact wavelength, geometry, and probe claims require actual
evidence before any future implementation. Knowledge geometries are possible options,
not selected customer configurations.

## Future LLM integration

Baseline is default. OpenAI model/key variables are checked for presence only; values
are not logged/returned. OpenAI selection always reports an unimplemented adapter.
Optional SDK and prompts prepare a seam, not working AI extraction. Future adapters
need typed validation, tests, explicit failure behavior, and agreed data handling.

No backend service, database, document ingestion, model training, hardware driver,
deployment stack, or multi-agent framework. Baseline works locally without credentials
or networking after setup. Streamlit telemetry is disabled.
