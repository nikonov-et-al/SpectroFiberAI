# Assessment pipeline review — 2026-10-18

## Root causes established before editing

The prior evaluation record at
`.cache/grounded-evaluation-20261018/all/D01.json` contains relevant S01 evidence
from page 14 and page 15, including blend potency, feed-frame measurement, NIR,
Raman and their comparison. All three proposed techniques were removed with the
same diagnostic: `every advantage and limitation needs a linked evidence claim`.
This branch runs before the candidate relevance check. Retrieval was therefore
not the immediate cause of that D01 result. Five lower-ranked excerpts were
dropped for context, but the relevant application excerpts remained.

The old candidate loop required nonempty limitations and a claim mapping for every
advantage and limitation. It then required `claim == quote` and a literal quote
inside the cited excerpt. Any failure removed the entire candidate. Unsupported
geometry or probe configuration also removed the whole candidate. A generated
non-provisional status cleared all candidates, and removing even one candidate
replaced the retained candidates' study with a generic abstention plan.

The final evaluation record does not preserve the raw provider response. It proves
the failed branch and the method labels, but cannot establish which individual
bullet or missing limitation triggered it. Do not infer that missing customer
accuracy alone caused this run: the explicit entry gate checks only objective,
target and matrix. Optional requirements were nonetheless poorly prioritized and
the report had no explicit production-readiness distinction.

Retrieval previously ran only once. Export mixed source observations, internal
validation warnings, chunk identifiers, extraction logs and customer advice.

## Design and limits

Validate claims individually. A literal quote can replace a model paraphrase as an
attributed source observation; this does not certify the paraphrase or customer
suitability. Invalid citations and unsupported secondary claims must be removed.
Provisional candidates still require relevant technical support; no-evidence and
underspecified cases must abstain. Production readiness remains unestablished.

Local claim searches reuse the allowlisted index and its source/page/version
validation. Reindex after adding or changing a document. Retrieval relevance and
quote membership do not prove scientific entailment or resolve all contradictions.

Study comparisons are engineering planning guidance: representative variation,
independent references, batch/run held-out evaluation and customer-agreed criteria.
Combined methods are optional comparisons; no winner or performance is assumed.
Customer reports retain source references; the diagnostics view and JSON preserve
the technical audit trail.

Existing scientific acceptance gates in PLAN.md remain pending expert review.
This task does not restore the previously removed offline recommender; all
regressions remain credential-free and deterministic.

## D01 before and after

| Aspect | Prior saved evaluation | Fresh local Ollama run after claim refactor |
| --- | --- | --- |
| Status | Insufficient information | Provisional |
| Techniques | NIR, Raman and combined option all removed | NIR and Raman retained as exploratory candidates |
| Combination | No useful comparison retained | Optional combined-method comparison in study; no claimed fusion advantage |
| Requirements | Unknown accuracy/range and other gaps | Unknowns preserved; questions ordered by study impact |
| Study | Generic requirements-resolution plan | Same-sample comparison, representative batches/runs, independent reference, preprocessing-safe holdout and agreed criteria |
| Production | Not validated | Explicitly unestablished |

The fresh run used local `gemma4:12b-it-q4_K_M` through the actual extraction,
retrieval and assessment path. Raw responses and reports are stored under ignored
`.cache/d01-claim-review/`. These are audit artifacts, not source fixtures or
scientific validation. Its combined candidate contained a nonliteral quote
(`single-method` versus source `single -method`) and an anaphoric quote without
method names; the candidate was removed while the two independently supported
techniques and optional experiment comparison survived. No citation was invented
to rescue it.

Raw generation also predicted fusion superiority, supplied qualitative acceptance
criteria, and proposed production procurement. Deterministic final planning removed
those statements. Follow-up review corrected classification of a secondary source
observation and forced reference wording back to the customer fact. Captured-response
replay verifies those final presentation/validation corrections independently of
another model generation.

Remaining risks: lexical method/case matching and explicit-negation screening are
heuristics. They cannot establish scientific entailment, find every contradiction,
or guarantee retrieval of every relevant passage. A source can itself be wrong.
New instruments, matrices, sampling geometries and customer performance require
expert review and an actual study; no numerical specifications are inferred.

## Verification and changed files

Final verification on 2026-10-18:

- `uv run pytest --basetemp .cache/pytest-assessment-accepted`: **130 passed**.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed, 49 files already formatted.
- Fresh D01 local Ollama generation: provisional NIR and Raman; raw responses
  captured. Final captured-response replay passed after reference/classification
  corrections. Final customer report: `.cache/d01-claim-review/D01-final.md`.

Commands used `.tools/Scripts/uv.exe` with repository-local `UV_CACHE_DIR`.
Streamlit AppTest required execution outside the sandbox: Windows socket-pair
creation hung before application code inside the sandbox. Tests remain offline.

Implementation files: `src/spectro_ai/providers.py`, `guardrails.py`, `models.py`,
`evidence.py`, `reporting.py`, `prompts/assessment.md`, and `app.py`.
Regression files: `tests/test_claim_validation.py`, `test_assessment.py`,
`test_guardrails.py`, `test_evidence.py`, `test_providers_and_reporting.py`, and
`test_app.py`. Documentation: README, architecture, decision log, hackathon plan
and this review. Existing unrelated work was preserved; no commit was requested.
