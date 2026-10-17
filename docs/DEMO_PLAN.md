# Offline demo — 2026-10-16

All inputs are synthetic. Demonstrate expert assistance, not validated instrument
selection. Presenter and allotted time remain to be agreed with organizers.

## Preparation

Follow README install/cache settings. Run `uv run pytest`, Ruff checks,
`uv run python scripts/smoke_check.py`, and `uv run python scripts/evaluate.py`.
Launch `uv run streamlit run app.py --server.address=127.0.0.1
--browser.gatherUsageStats=false`. No key, model service, or network is needed for
assessment after dependencies are installed. Document Q&A is a separate optional demo.

## Presenter sequence

1. Enter `We need industrial monitoring.` and assess. Show abstention and three
   critical questions. No instrument recommendation is appropriate yet.
2. Expand **Expert review and follow-up answers**. Enter Objective `monitor ethanol`,
   Target `ethanol`, Matrix `aqueous fermentation broth`, State `liquid`, Deployment
   mode `in-line`. Apply corrections. Show two unranked conditional options: Raman
   and MIR/FTIR. Original text stays intact; edited facts say `expert edit`.
3. Read the water-interference condition for MIR. Set Interference to `strong
   fluorescence background; cells and bubbles` and Detection limit to `0.1 % v/v`
   (synthetic customer requirement, not achieved capability). Before applying, point
   out hidden stale export. Apply; Raman now explicitly requires background screening;
   sensitivity remains unknown. Fluorescence is not added as a target technique.
4. Compare mechanisms, assumptions and evidence. General source provenance does not
   validate this customer, AFS hardware, geometry, or performance. Configuration stays
   unknown until optical access/contact constraints justify a laboratory option.
5. Review tailored study: representative broth, blanks, references, interference,
   held-out batches, agreed thresholds, and success/failure next actions. No numerical
   acceptance criterion is invented. Reference method requires expert selection.
6. Download Markdown, compare with Structured report JSON. Edit raw input and show
   stale output disappears. Reassess to restart extraction.

Supported prose alternative: `We want to monitor ethanol in aqueous fermentation
broth.` Quantities and advanced requirements use labels or expert edits. The parser
supports bounded literal phrases; it is not unrestricted language understanding.

## Recovery and fallback

- Empty input: enter a synthetic example. Unsupported phrasing: use explicit labels.
- No rule match: show abstention and expert-led laboratory screening steps.
- Model/API failure: continue assessment; optional Q&A has explicit errors and no
  silent provider fallback.
- UI fails: run `uv run python scripts/evaluate.py`; show `.cache/evaluation` Markdown
  and JSON. Restart Streamlit; do not edit source during the presentation.
- Network unavailable: baseline remains local. Install dependencies before arrival.

## Rehearsal evidence and open work

AppTest covers incomplete input → structured answers → changed candidates → export,
pending-review hiding, raw-input stale hiding, and four original examples. Paired
domain tests cover background changes. Startup smoke covers HTTP health only.
These automate the workflow; human presenter rehearsal in the actual event setup
is still pending. Hardware, samples, expert approval, timing, and judging remain open.
