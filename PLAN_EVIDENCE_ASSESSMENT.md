# Step 1: Application-aware evidence assessment — revised

Approved for implementation 2026-10-18. This task-specific backlog leaves the
existing PLAN.md and its scientific acceptance gates intact. Completed milestone
text and verification move to HISTORY.md only after review.

## Goal and boundaries

Add one typed evidence-assessment pass between verified retrieval and synthesis.
Reuse models.py, providers.py, existing provider transports, provenance and report
surfaces. Add evidence_assessment.py for provider-independent validation and one
prompt. No retrieval rewrite, new dependency, cloud requirement, technique
mappings, ranking system or experiment-selector redesign. Step 2 owns selection.

Current pipeline: extraction/corrections → pinned BM25/semantic/RRF retrieval →
optional reranking/context → _verified_retrieval → fit_evidence → technical
synthesis → quote/lexical guards → feasibility safeguards → Markdown/JSON.
Relevant code is in models.py, evidence.py, providers.py and reporting.py; prompts
live in prompts/. _supports_candidate accepts method plus any target OR matrix
term; it cannot compare application conditions. Exact quotation membership does
not prove entailment. Claim repair can add evidence after synthesis, and existing
planning safeguards insert retained candidate methods into experiments.

Local extracted S01 pp.14–15 describes powder/feed-frame NIR/Raman; S02 p.14
describes API/hot-polymer ATR/FTIR and extrusion. This verifies wording, not source
truth, visual correctness or customer compatibility. Lexical overlap is a plausible
cause of the reported failure; its exact generation was not replayed.

## Contracts and policy

Keep authenticity, source entailment, applicability, source quality and customer
suitability separate. Models:

- EvidenceReference: evidence ID, exact quote, optional context-block ID; resolve
  provenance from verified evidence, never model-written metadata.
- EvidenceConditionComparison: dimension, nullable source value, customer-field
  references, match/different/unknown/conflicting, rationale and source references.
- EvidenceObservation: host ID, statement/references, named method/geometry,
  comparisons, supported/unsupported/uncertain entailment, applicability,
  limitations. Statement kinds: reported_result, proposed_capability,
  method_description, limitation. One source application scenario per observation;
  multiple observations may share excerpts. Cross-source reasoning is explicit.
- ApplicationSupportAssessment: host ID, named method, observation references,
  bounded conclusion, consideration basis and optional structured bridge.
- EvidenceAssessment: observations/supports, conflicts, per-input coverage,
  host-owned context availability, fingerprint and runtime diagnostics.

Applicability is direct_application, related_application, general_method or
insufficient_context. It is not ranking or customer performance validation.
Consideration basis:

- application_supported: supplied evidence explicitly supports relevant application
  consideration, including narrower objectives and explicit remaining unknowns;
  no universal requirement that every target/matrix/optional condition match.
- reasoned_exploratory: source-backed mechanism/capability premises plus customer
  facts, an explicit testable transfer hypothesis, known differences, missing
  conditions, failure mechanisms and next checks. Engineering reasoning is not
  a demonstrated source result. No exact application precedent is required.
- unsupported: no adequate application support or bounded bridge; diagnostics
  only, never retained recommendation. Keyword/industry overlap is insufficient.

Unresolved applicable contradictions cannot be hidden by an exploratory label.
Unknown scalars stay None, collections empty; quantities/units remain source text.
Unreviewed technical evidence may inform provisional assessments with visible
warnings. Review status never substitutes for scientific applicability or quality.
Existing business/pointer/transcription exclusions remain.

## Workflow and bypass prevention

1. Preserve essential-customer-fields gate and pinned index verification.
2. Fit complete evidence bundles including available context/locations/omissions;
   drop whole lower-ranked bundles instead of stripping qualifications.
3. One structured configured-provider call evaluates observations and application
   support. Initial malformed JSON has at most one structural retry.
4. Validate references, quoted values, customer anchors, coverage and categorical
   prerequisites. Host assigns IDs and binds exact customer/evidence/prompt input.
5. Synthesis receives assessed conclusions, quotes and metadata, not duplicate full
   raw excerpts; original evidence remains for checks/exports.
6. Require claim-use (source observation/application implication/exploratory
   hypothesis), observation IDs and support IDs where applicable. A valid ID does
   not grant permission for broader prose. Project final observations from verified
   quotes, implications from assessed bounded conclusions, hypotheses from bridges.
   Preserve limitations and labels; discard uncategorized generated assertions.
7. Every retained candidate references an assessed support record. Existing method
   checks are identity checks, not application authority. No lexical fallback.
8. Exclude exploratory candidates from automatic experiment-technique insertion.
   Exploratory-only reports retain candidates for expert screening but technique is
   None and plan gathers information. Existing neutral comparisons can use
   application-supported candidates without claiming priority. No new selector.

Context status is host-owned child-only/attached/unavailable, never a claim that
the entire source was examined. Expansion remains off by default. Missing context
cannot become a match. Self-contained explicit facts remain usable; missing
antecedents/headers/continuations/qualifications restrict dependent claims. Unknown
conditions must not be filled from titles or customer facts. Missing validation
context cannot establish demonstrated performance. Conditions outside supplied
context may be missed; disclose this limit.

Preserve max four claim-repair searches. Stage verified additions, assess augmented
evidence once without a retry, then revalidate all support/conflicts. Preserve
earlier bundles. Failed/oversized additions cannot repair claims or bypass judge.

Add optional fields to EvidenceClaim (observation_id, support_assessment_id,
claim_use), TechniqueCandidate (support reference/consideration basis) and
AssessmentReport (evidence_assessment). Extend technical assessor keyword contract
and add injected evidence callable. Old reports default assessment fields to None;
new report version grounded-{provider}-v2. Index v3/retrieval APIs unchanged;
no reindex. Customer corrections invalidate assessments.

Use existing configured context/output/timeout and temperature zero. Budget schema,
JSON, complete bundles, output/retry reserves. Byte checks are not token counts;
temperature zero is not semantic determinism. Record calls/latency/bytes/available
usage/omissions. If synthesis cannot fit all assessed qualifications, abstain rather
than silently drop them. No ensemble, agreement vote or persistent judge cache.

Tampered provenance fails; runtime/refusal/truncation errors remain visible provider
errors. Invalid references reject affected records with coverage diagnostics.
Scientific insufficiency abstains. Conflicts withhold affected support. Optional
repair failure preserves valid initial assessment with warning.

Deterministic checks verify membership, identity, linkage and category permissions;
they cannot guarantee semantic entailment, scenario binding, sound transfer,
complete qualifications or scientific truth. Model judgment stays visibly labeled.

## Pending milestones — exact approved text

| Milestone | Acceptance and Conventional Commit |
|---|---|
| 1. Contracts and permissions | Models, provenance resolution, context availability, support categories, bridge requirements and compatibility tests. `feat: define application evidence assessment contracts` |
| 2. Bounded evidence assessment | Prompt, provider adapter, complete-bundle fitting, coverage and failure diagnostics; mocked provider tests. `feat: assess retrieved evidence application context` |
| 3. Minimal integration guards | Observation-bound presentation, category-preserving candidates, conflict handling, claim repair and experiment bypass prevention. `feat: prevent unsupported evidence promotion` |
| 4. Semantic regressions and reporting | Generalization fixtures, exports, D01/S02 acceptance procedure and documentation. `test: cover evidence transfer and exploratory reasoning` |

Sol reviews; Luna implements bounded tasks. Run targeted and full pytest, Ruff lint
and Ruff format checks before acceptance. Commit accepted related work only.

## Tests and scientific acceptance

- D01/S01: close application relationship, source-described reflection geometry,
  unknown customer performance/configuration; no assumed fusion advantage.
- D01/S02 alone: authentic API/hot-polymer FTIR observation with explicit differences;
  no analyte-only prioritization/experiment selection. A genuine transfer bridge may
  permit exploration, never automatic priority/selection.
- Novel application: accept grounded mechanism/customer linkage/testable bridge;
  reject generic enthusiasm. Test IDs/method-label permutations, no preferred list.
- Same analyte/different matrix; same matrix/different analyte; incompatible stated
  conditions; general method premises; missing and contradictory context.
- Parent expansion off/unavailable, missing headers/qualifiers, later supplied
  context changing assessment; authentic quotes attached to broader claims;
  borrowed IDs, cross-scenario mixing, review warnings and units conflation.
- Multiple observations per excerpt, same-source conflicts, insufficient evidence,
  negative-only support, old-report compatibility, provider limits and repair bounds.
- Reuse FakeRetriever/FakeAssessor/context_index/repeated-PDF fixtures for offline
  ingestion → retrieval → assessment → synthesis → checks → exports integration.
- Preserve four synthetic examples, customer corrections, AppTest and Markdown/JSON.
  Mocked semantic outputs test enforcement, not actual model scientific reliability.

Run uv run pytest, uv run ruff check ., uv run ruff format --check . using local uv
when needed. Record pinned local Ollama smoke (calls, context volume, latency);
no paid requests without authorization. Update README, docs/DECISIONS.md and
docs/HACKATHON_PLAN.md while preserving unrelated work and existing blockers.

Done: inspectable assessments; justified novel exploration; unsupported promotion
and experiment bypass blocked; provenance/unknowns/review warnings preserved;
compatibility and checks pass. Scientific acceptance still requires human review of
D01/S02 and unfamiliar transfer cases. No model agreement or software test closes
existing Phase 3, retrieval or parent-context gates. Ranking/selection stay Step 2.
