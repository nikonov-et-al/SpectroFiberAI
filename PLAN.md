# Grounded LLM pipeline — pending acceptance

Updated: 2026-10-18. User-approved architecture replaces every earlier requirement
for an offline parser, baseline recommender, or automatic provider fallback.

Enquiry → Ollama/OpenAI extraction → allowlisted evidence retrieval → LLM assessment
→ deterministic validation → expert-reviewable report. Ollama defaults to a 16,384
context budget, 4,096 output tokens and 240 seconds per request. OpenAI is explicit;
no paid requests are authorized by merely having a key present.

## Accepted software and remaining gates

Offline parser/recommender/fallback removed. Both adapters, shared validation,
expert corrections, Markdown/JSON export and evaluation runner are implemented.
119 tests, Ruff lint/format, and local Streamlit HTTP 200/ok passed. A local Ollama
smoke completed with explicit abstention after unsupported candidates were removed.
All 31 local evaluation inputs completed without runtime errors; see
[LLM_EVALUATION.md](docs/LLM_EVALUATION.md). This establishes software behavior, not
scientific acceptance. Strict quote guards can over-abstain; expert review remains open.

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
