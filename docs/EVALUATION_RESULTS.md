# Synthetic workflow review — 2026-10-16

Reproduce: `uv run python scripts/evaluate.py`. Six full JSON and Markdown reports
are written to ignored `.cache/evaluation/`. `examples/evaluation.json` preserves
inputs, expected unknowns, status and method lists; four original examples stay intact.
`tests/test_evaluation.py` checks all six and export round trips.

This is software review, not spectroscopy expert review. Apply EVALUATION.md's 0–2
rubric only to observable structural behavior below. No combined score or confidence
is assigned. Plausibility, scientific evidence quality, and experiment appropriateness
require an actual domain reviewer and remain **Not reviewed** for every case.

| Case | Requirement completeness | Follow-up questions | Uncertainty handling | Review notes / next expert question |
| --- | --- | --- | --- | --- |
| Aqueous fermentation ethanol | 2 | 1 | 2 | Supplied ethanol/broth/state/mode preserved; Raman/MIR conditional. Can target response be separated from broth variability at the actual concentration? |
| Powder moisture | 2 | 1 | 2 | NIR hypothesis and diffuse reflectance study; presentation/calibration conditions explicit. Which batches and reference define moisture acceptance? |
| Reactor progress | 2 | 1 | 2 | Raman/MIR comparison; actual species and band changes unknown. Which reactant/product and independent conversion reference are appropriate? |
| Liquid contamination | 2 | 2 | 2 | Known 1 ppm request retained; target unknown → abstain. Name contaminant and matrix before technique selection. No 1 ppm capability claim. |
| Strong fluorescence background | 2 | 1 | 2 | Raman stays conditional; MIR comparison; no fluorescence-target candidate. Does background obscure usable Raman response on real samples? |
| Insufficient input | 2 | 2 | 2 | No candidates; three critical questions first. Establish decision, target and matrix. |

Question scores of 1 reflect useful deterministic prioritization without expert
validation of ordering. Completeness scores check supplied fixture facts and visible
unknowns, not complete real customer requirements. Uncertainty scores check explicit
abstention/conditional behavior, absent performance guesses and expert caveats.

Assumption transparency: **1** across cases; assumptions and heuristic categories
are visible, but an expert must identify omitted physical constraints. Practical
usefulness remains provisional: controls, references to agree, held-out validation,
metrics, agreed criteria and success/failure branches are inspectable; no study has
been conducted. Do not reinterpret structural scores as scientific suitability.

## Prohibited claims and review boundaries

No invented AFS model, exact wavelength, sensitivity, sample count, validated
performance, numerical confidence or selected reference method. Unknown numerical
targets stay unknown. Contamination has no forced positive match. Fluorescence as
interference differs from explicitly requested target emission. General citations
support principles only; retrieved Q&A never changes assessment rules automatically.

## Verification

Minimal locked install: 82 passed, 1 PDF test skipped because pypdf absent. Optional
`rag` + `llm` install: 83 passed, including PDF and mocked API paths. Both dependency
sets passed `uv pip check`. Ruff lint and format passed after a test-line formatting
fix. Startup health returned HTTP 200/ok and server stopped. Evaluation script emitted
all six reports. Full demo AppTest also covers structured corrections and stale export.

No new live model or paid API calls; historical RAG rehearsal remains in HISTORY.md.
Tests required execution outside sandbox after sandboxed AppTest stalled, even with
README local cache/temp settings. No code failure remained in the completed run.

## Readiness and next actions

Software supports a credential-free expert-assistance demo. Scientific and hardware
readiness remain unverified. Original audit attachment is unavailable; audit mapping
uses PLAN's retained capability list and records that limit.

1. Obtain organizer brief, judging/time constraints, authorized documents and equipment.
2. Have a spectroscopy expert review all six reports, source scope, rules and study plans.
3. Rehearse with presenter on event setup; acquire representative samples/reference
   values before claiming suitability or integration.

Architecture: retain Pydantic/domain separation, deterministic baseline and independent
Q&A. Add LLM extraction only after a demonstrated parsing need with validated outputs.
Add retrieval-to-assessment integration only after reviewed knowledge ingestion warrants
it; no new framework, hardware catalog, vector service or deployment stack is needed.
