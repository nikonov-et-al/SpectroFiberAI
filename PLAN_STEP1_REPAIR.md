# Repair Step 1 budgeting and extraction

Software milestones 1–4 completed and reviewed on 2026-10-18. Exact original
milestone text, implementation, verification and deviations are archived in
[HISTORY.md](HISTORY.md). See [verification evidence](docs/STEP1_REPAIR_VERIFICATION.md).

No feasible implementation tasks remain under this repair plan.

## Remaining acceptance gates

- Scientific Step 1 acceptance stays open: the local D01 replay retained evidence
  and recorded an assessment, but citation/method guards withheld recommendations.
  Preserve the existing scientific blockers in PLAN.md; do not proceed to Step 2.
- Exact historical-run comparison is blocked by unavailable exported report/raw
  model response and historical runtime state. The matching repository D01 enquiry
  was replayed; budget flow and extraction were verified in separate local runs.
- Estimates remain approximate. Synthesis usage was not measured live because
  scientific support did not pass validation. Its overflow behavior is covered by
  offline tests; no new model/default promotion is implied.
