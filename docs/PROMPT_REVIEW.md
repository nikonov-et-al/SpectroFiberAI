# Prompt review — 2026-10-18

Reviewed all five prompts against the supplied critique, domain schemas, provider
request construction, evidence validation, and final candidate projection.
Changes clarify existing contracts; they do not establish scientific acceptance.

## Changes and judgments

| Prompt | Focused change |
| --- | --- |
| `extraction.md` | Preserve current, desired, conditional and exploratory meanings; require quotes to justify their destination field. Interest in a method does not mean installed equipment or a mandatory requirement. |
| `retrieval_queries.md` | Use distinct questions for capabilities and limiting evidence; allow meaning-preserving technical synonyms; avoid filling the budget with paraphrases. |
| `evidence_assessment.md` | Bound support across objective, matrix/state, geometry/location and relevant conditions. Require a grounded exploratory bridge for necessary transfer. Distinguish proposed capability from reported demonstration. Correct context-availability instructions to match observation and coverage schemas. |
| `assessment.md` | Preserve support scope, material differences, conflicts and unknowns. Keep unsupported options out of viable candidates; abstain when none remain. Separate source settings from conditional study proposals. |
| `experiment.md` | Retain future-only status. Add untrusted-input/JSON rules, objective/risk-dependent design, independent validation splits and explicit hypothesis-refutation checks. Candidate membership does not justify selection. |

The attached critique correctly identifies overgeneralization as a key risk, but
several suggestions already exist: complementary questions, limitation retrieval,
exact quotes, source-scenario separation, conflicts and exploratory bridges.
These received focused clarifications rather than replacement prompts.

The three-question limit is enforced by the schema, not just prompt wording.
Increasing it needs retrieval-budget evaluation; this review does not change it.
Likewise, adding rejected-option fields would change public contracts. Existing
support assessments already record rejection reasons; final provider validation
withholds unsupported candidates and records warnings.

Application support means evidence for considering a bounded application, not
proof of customer suitability. A source's proposed capability must remain labeled
as proposed. Requiring every consideration to have demonstrated results would
unnecessarily discard provisional technical evidence.

## Verification limits

Verification: 268 tests passed, including all 12 Streamlit AppTests. Ruff lint
passed and format check reported 75 files already formatted. Commands used local
uv with `run --offline --no-sync --python .venv/Scripts/python.exe` for `pytest`,
`ruff check .` and `ruff format --check .`, with repository-local uv/temp paths.
The full suite needed execution outside the sandbox: inside it, Python's local
socket-pair creation stalled before Streamlit application execution. The separate
sandbox run excluding AppTests passed all 256 remaining tests.

Existing offline tests exercise provider construction, schema/quote guards,
D01/S02 application mismatch, exploratory bridges, conflicts, insufficient context,
abstention and report exports. They use controlled model outputs: passing them
cannot show that an actual model follows the revised semantic instructions.

No new live model comparison or expert scientific review was run. The existing
Phase 3 and application-evidence scientific gates remain open. Deterministic
validation checks references and structural permissions; it cannot prove that all
necessary application dimensions were considered or that a transfer is sound.
Final candidates are projected from evidence assessments, so final-synthesis
wording alone cannot repair a scientifically weak evidence assessment.

Prompt additions consume request context. They may reduce retained evidence under
tight budgets; no retrieval coverage, latency or model-quality improvement is
claimed. The experiment prompt remains unused by the active baseline templates.
