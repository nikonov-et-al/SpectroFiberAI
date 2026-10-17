# Recommendation evaluation

Domain-expert review rubric, not validated scientific scoring or confidence estimates.
Score each dimension 0–2; retain reviewer notes and source provenance.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Scientific plausibility | Conflicts with principles | Plausible; constraints unexamined | Expert-reviewed fit to target, matrix, constraints |
| Requirement completeness | Critical facts omitted/invented | Critical facts captured; some details missed | Supplied facts preserved; missing requirements visible |
| Follow-up questions | Irrelevant/absent | Useful; important gaps missed | Targeted questions resolve consequential gaps |
| Assumption transparency | Assumptions presented as facts | Partly identified | All material assumptions explicit and traceable |
| Experiment appropriateness | Untestable; lacks reference | Testable outline; controls/criteria incomplete | Appropriate reference, controls, validation, metrics, criteria |
| Evidence quality | Fabricated/unsupported | Real general sources; applicability unverified | Authoritative relevant evidence with scope/provenance |
| Uncertainty handling | False certainty; no abstention | Incomplete caveats | Unknowns preserved; unsupported decisions deferred |

Total 0–14 supports comparison only. **Fabricated or unsupported scientific claims
require correction regardless of total.** A high score is not industrial validation.

## Procedure

1. Preserve input, expected facts, unknowns, and relevant source documents.
2. Generate report; have a spectroscopy expert score dimensions with notes.
3. Review gaps, incorrect candidate inclusion/exclusion, and experiment usefulness.
4. Add agreed regressions without treating synthetic expectations as scientific truth.

Six synthetic reviews, reproducible outputs, partial software scores and outstanding
expert questions are in [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md). Generate JSON
and Markdown with `uv run python scripts/evaluate.py`; fixtures are separate from the
four original examples. Do not assign scientific plausibility scores without an expert.

Current cases test software behavior. Future evaluation needs real challenge examples,
expert labels, representative edge cases, and failures. Compare providers separately.
