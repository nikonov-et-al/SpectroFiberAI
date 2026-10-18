# Step 1 reliability repair — verification

## Reproduction and budget policy

The previous evidence fitter added serialized UTF-8 bytes, reserved output tokens
and a fixed 256-unit margin, then compared the sum with a token context limit.
At default settings the empty-case prompt/schema/reserve already charged 15,128
units. An offline 1,200-byte excerpt without parent expansion charged 16,725 and
was discarded. Those byte-based charges did not demonstrate actual model context
exhaustion.

The repair estimates tokens separately for serialized messages and response schema
at one token per three bytes, rounded upward. The safety margin is the greater of
512 tokens and 15% of estimated input. A schema embedded in a prompt and also sent
as structured-output format is charged in both actual locations. Whole bundles are
removed from the lowest retrieval rank; remaining schema enums are rebuilt. No
qualification stripping, default context increase, tokenizer download, retrieval
rewrite or scientific guard relaxation is part of this repair.

Reports expose request estimates, output reserve, margins, fitting decisions,
evidence IDs and nullable provider usage. These are operational diagnostics, not
confidence or scientific acceptance. Actual provider counts are retained as
reported; estimates do not guarantee fit on other models or inputs.

## Local D01 replay — 2026-10-18

The matching D01 enquiry is available in
`knowledge/data/materials/AFS_Examples_and_Reference.docx`; the historical exported
report and raw model output supplied in the issue are unavailable. This is a
reproducible replay of the repository enquiry, not an exact recreation of the
historical model state or settings.

- Model: `gemma4:12b-it-q4_K_M`, context 16,384, output reserve 4,096.
- Pinned index revision: `123cdc83ecc9466b8e671afdb00ceda7`.
- Manifest: `0477ab5db9e086397152f8207be86d38347920adc3d61f3d55b7254dc1da6e1e`.
- Enquiry SHA-256: `8dcaaa9d6c96282edb3a3421ae2ec6fd5a3cf24fa415658ce3b2dd21a418902d`.
- Hybrid retrieval with multi-query explicitly enabled for this experiment; parent
  expansion disabled. Neither experiment's default was promoted.
- Initial dispatch exposed a query-planner import shadowing error after extraction;
  the import was corrected before the completed replay.
- Completed replay took 94.714 seconds, selected eight excerpts across five
  documents, admitted seven verified technical excerpts, and dropped none for
  request budget. S01 `0040` survived. S01 `0042` appeared among query candidates
  but was not in the final pooled selection; this was not a context-budget drop.

| Stage | Estimated input tokens | Provider-reported prompt tokens | Completion tokens |
| --- | ---: | ---: | ---: |
| Extraction | 5,258 | 2,164 | 872 |
| Retrieval query generation | 1,956 | 1,032 | 134 |
| Evidence assessment | 7,083 | 3,599 | 2,911 |

Query prompt usage above comes from the query result's own usage diagnostics. The
first integration version incorrectly let subsequent embedding requests overwrite
the request-budget entry with their token usage. The adapter now attributes usage
to chat requests only; a dedicated transport regression checks this boundary.

All measured chat prompt counts were below the initial estimates, so no estimator
coefficient increase is justified by this sample. Provider counts may reflect
backend caching/accounting; no downward calibration is made. Synthesis was not
reached, so this replay does not calibrate synthesis usage.

A structured evidence assessment was recorded. The final result remained
`evidence_abstention`, zero candidates: host validation rejected observations for
missing method support or unavailable source references, then rejected dependent
support records. This is a scientific/model-output acceptance limitation, separate
from evidence loss during request fitting. No scientific prompt or permission
guard was weakened to force candidates.

## Extraction acceptance

The first completed replay retained `Laboratory checks` and placed pharmaceutical
industry in unverified inferred context while keeping authoritative industry null.
It omitted deployment mode and put `first pilot` into desired stage. One focused
prompt repair explicitly separated those two destinations; no keyword recovery or
additional semantic retry was added.

An independent extraction replay on the same D01 input then completed in 20.135
seconds. Raw target fields and cleaned target fields agreed:

| Field | Value |
| --- | --- |
| Reference measurement | Laboratory checks |
| Deployment mode | first pilot |
| Desired stage | before we commit to production equipment |
| Authoritative industry | null |
| Unverified inferred industry | pharmaceutical |

The extraction estimate was 5,386 input tokens; Ollama reported 2,252 prompt and
891 completion tokens. No estimator coefficient increase was needed. An unrelated
`process` value was removed because its supporting quote was invalid/nonliteral;
that warning remained visible. This successful single replay establishes the
specific omission fix on D01, not extraction reliability on arbitrary enquiries.
The full scientific pipeline was not rerun after the focused extraction repair;
the evidence-flow and extraction observations above are separate verifications.

The host must preserve supported literal quotations, distinguish paraphrases from
literal wording, and never fill missing fields through an offline keyword parser.
Industry inference stays outside scientific customer facts and retrieval inputs.

## Final checks and remaining gates

Final verification used repository-local uv/cache/temp paths and Python 3.14.4:

- `uv run --offline pytest -q -o faulthandler_timeout=30`: **290 passed in 10.71s**,
  including Streamlit AppTest coverage and all four synthetic examples. A dedicated
  repository-local pytest base directory was supplied. The full run required
  approved execution outside the sandbox: Windows asyncio socketpair creation
  stalled inside the sandbox before AppTest application execution.
- `uv run --offline ruff check .`: **all checks passed**.
- `uv run --offline ruff format --check .`: **80 files already formatted**.
- `uv run --offline python scripts/smoke_check.py`: **HTTP 200/ok**; temporary
  Streamlit server stopped.
- `git diff --check`: **passed**.

Regressions include default-budget eight-bundle mock dispatch, intact attached
qualifications, Unicode and exact fit boundaries, shrinking schema enums, all-drop
reports, synthesis overflow, follow-up/structural retry budgets, embedding/chat
usage isolation, field provenance, inferred-industry exclusion, historical JSON
loading and the diagnostics download. The first full run found one new AppTest
assertion incorrectly reading file contents from a URL-only download proto; the
assertion was corrected and the final full suite passed.

Scientific usefulness, exact historical-run comparison, and Step 2 remain pending
regardless of software test results.
