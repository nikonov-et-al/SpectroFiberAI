# Application-aware evidence assessment

Step 1 adds a structured evidence pass, not a technique ranking system. Scientific
judgments remain model-assessed. Source authenticity, quotation membership, source
review, application relationship and customer suitability are separate questions.

Method and geometry labels need not occur verbatim in the supporting quotations.
Their semantic attribution remains model-assessed and requires expert review;
the host does not verify it through substring matching. Exact quotation membership,
source provenance, condition comparisons and internal support/observation method
identity checks remain enforced.

## Scientific review protocol

Before comparing outputs, pin the source files, index revision, excerpt/block IDs,
source review metadata, customer facts, prompt version and local model identity.
Record unavailable surrounding context. Never manufacture expected review labels
from the model's answer. Parent expansion remains off unless explicitly requested
for a comparison; it cannot recover context that is absent from the index.

Current local source identities (SHA256, inspected 2026-10-18):

- S01_Multispectral_Laboratory.pdf:
  `7026efe7266877b50238d9a9c91e93202242670ae9c776856efe93bb550a8935`
- S02_AFS_Process_Control_Slides.pdf:
  `f273a85e771c8907d6560a31ea6f7c9915d8ba8bae95cac06d751cc0ca843cee`

These hashes identify files, not scientific validity or visual extraction accuracy.
Use original pages when reviewing extracted context and table associations.

### Expected relationships to review

| Input evidence | Expected assessment boundary |
| --- | --- |
| S01 pp.14–15 | Powder blend/API feed-frame application is closely aligned with D01. NIR reflection and Raman are source descriptions. Customer performance, installed probe compatibility and fusion superiority are not established. Preserve page continuation explicitly. |
| S02 p.14 | API in hot polymer, diamond ATR/FTIR and extrusion differ from powder feed-frame monitoring. Authentic API wording alone cannot support application-specific prioritization or automatic experiment selection. |
| S02 plus genuine mechanism/transfer premises | A conditional, source-backed engineering hypothesis may justify exploration. Source application differences, failure mechanisms and necessary checks remain explicit. No automatic experiment priority follows. |
| Unfamiliar matrix/analyte without exact precedent | Accept exploration only with relevant source-backed premises, customer anchors, explicit inference, falsifiable hypothesis, gaps and checks. Reject generic enthusiasm and shared-keyword reasoning. |
| Missing or contradictory source context | Keep absent values unknown; withhold dependent conclusions or expose unresolved conflicts. An exact quote cannot override an unavailable qualification. |

Review each retained candidate's exact claim, not merely its method label. Try a
broader performance promise with the same valid IDs; it must not survive final
projection. Swap source IDs and use different methods/analytes/matrices to check
that application rules generalize without hardcoded D01 preferences.

Run parent-off/on and qualification-missing/restored variants using the same pinned
customer facts. Compare observations, consideration basis, limitations and export
warnings. Missing data must never be counted as evidence of a match. Pending source
review must remain visible without changing the applicability classification by
itself. Do not count attached blocks as independent corroboration.

## Runtime and failure interpretation

The configured provider performs the evidence pass; no cloud provider is required.
Requests reuse context/output/timeout settings. Complete bundles retain supplied
qualifications. Initial malformed output has one structural retry; optional staged
claim repair permits one augmented assessment. Timeouts, refusal and truncation are
errors, not scientific abstention or invitations to change providers.
Successful repair permits one re-synthesis against the augmented snapshot; no
recursive claim-repair loop follows. This preserves consistent IDs and conflicts.

Inspect request bytes, available token usage, dropped bundle IDs, calls and elapsed
time. Byte fitting is not tokenization, and temperature zero is not a correctness or
repeatability guarantee. Empty, invalid or oversized evidence cannot be silently
rescued by lexical matching.

## Acceptance status

Human scientific review is pending. Offline fixtures check enforcement of typed
relationships and guard behavior; they cannot prove real-model entailment or the
soundness of a transfer bridge. Local generation establishes runtime behavior only.
Existing Phase 3, retrieval-label, parser and parent-context usefulness gates remain
open. Candidate ranking and experiment-selection improvements belong to Step 2.

Implementation verification and local smoke results are recorded in HISTORY.md.

### Local development smoke observations — 2026-10-18

Pinned active index revision: `123cdc83ecc9466b8e671afdb00ceda7` (existing schema 2,
197 chunks). The index was read, not rebuilt or migrated. The S02 p.14 excerpt is
278 UTF-8 bytes and retains `needs_visual_review` status.

Initial default-budget request exceeded 4,096 output tokens and raised a provider
error. A compact-output revision completed in one request: 37.3 seconds, 9,842
serialized request/schema bytes, 966 reported prompt tokens and 1,583 output tokens.
Both observations were rejected: the model used context availability as a block ID
and customer values as field names. No support survived. Raw output also treated a
broad heading as direct application evidence while placing its hot-polymer context
in a separate observation. This is a semantic failure, not scientific acceptance;
format rejection happened to prevent promotion in that run.

The focused follow-up constrains transport anchor values and makes method descriptions
premises rather than application support. Source scenarios and nearby qualifications
must remain linked. Ignored runtime artifacts are under
`.cache/evidence-assessment-smoke/`; they are audit records, not reviewed labels.

Final v3 request completed in one call: 16.3 seconds, 11,348 request/schema bytes,
1,050 reported prompt tokens and 645 output tokens. Reference shape was valid.
The model still classified the broad API heading as direct application and omitted
the supplied polymer qualification. The generic method-description gate rejected
its application-support record. Captured-assessment replay with an adversarial
synthesis stub returned insufficient information, no candidates and no experiment
technique. This verifies software enforcement; it does **not** establish successful
scientific context interpretation. No additional live prompt tuning follows the two
focused repair attempts. The semantic gate remains blocked pending a separately
reviewed approach and human assessment. See `S02-final.json` and
`S02-final-raw-1.json` in the ignored audit directory.

Subsequent schema-only compaction shares the customer-field enum and removes
host-owned context fields from provider output. A read-only fitting check confirms
each real S01 p.15 chunk (1,048 and 1,168 bytes) now fits individually at the unchanged
16,384 context / 4,096 output defaults. This is budget feasibility, not a new model
run, full-page coverage guarantee or scientific acceptance.
