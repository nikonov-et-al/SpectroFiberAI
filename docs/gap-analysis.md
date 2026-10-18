> Historical analysis before 2026-10-18 architecture change. Offline parser/baseline proposals below are superseded; see README, ARCHITECTURE.md and PLAN.md for current behavior.

# Phase 0 — Repository gap analysis

## Keep

- [Pydantic domain models](../src/spectro_ai/models.py), deterministic offline [baseline](../src/spectro_ai/baseline.py), YAML knowledge boundary, expert corrections, Streamlit UI, and Markdown/JSON exports. Baseline makes no network calls.
- Four synthetic examples and abstention behavior. Preserve legacy report status values while adding any canonical statuses needed by later phases.
- Existing document Q&A [RAG](../src/spectro_ai/rag.py) as an exploratory feature; it is not the assessment retriever.

## Change

- Extend current flat contracts in `models.py`; do not duplicate domain types. Add customer-text quote provenance and assessment status/diagnostics compatibly.
- Replace the small literal phrase parser as the primary input path with structured extraction and a conservative offline fallback. Current extraction does not handle general D01 wording.
- Keep adapters in `providers.py`; OpenAI assessment currently raises an explicit unimplemented error, and Ollama currently serves document Q&A only.
- Keep source-backed claims, heuristics, assumptions, and customer facts visibly distinct. Current `techniques.yaml` uses general external references; it does not represent the supplied AFS corpus.

## Add

- Phase 1: structured extraction contract, callable Ollama assessment provider path, literal customer quote validation, and tests for unknowns, paraphrases, corrections, malformed output, and provider failures.
- Phase 2: source manifest and separate deterministic scientific retriever over page-aware AFS excerpts; exclude worked answers and holdout scenarios. Mark image-heavy or ambiguous pages for review.
- Phase 3: structured grounded assessor using retrieved evidence IDs, conditional candidate comparison, consequential questions, and an experiment plan; retain deterministic checks and offline fallback.

## Defer

- Embeddings/vector service, a new framework or API, extra reviewer agent, complex UI changes, and calibration/performance claims. Existing local RAG machinery is not evidence that a separate vector index is needed.
- Phase 4 holdout execution and evaluation runner; beyond this Phase 0 audit.

## Materials inventory

Canonical folder is `knowledge/data/materials/`. Every AFS file named in PLAN.md is present; there are **no missing AFS source filenames**:

- `S01_Multispectral_Laboratory.pdf`
- `S02_AFS_Process_Control_Slides.pdf`
- `S03_LED_Kidney_Study_2017.pdf`
- `S04_Combi_MIR_Fluorescence_2021.pdf`
- `S05_AFS_Flyer.pdf`
- `S06_Laboratory_Link.txt`
- `AFS_method_comparison_table.xlsx`
- `AFS_Examples_and_Reference.docx` (D01–D08 and worked D01/D02)
- `customer_scenarios_for_final_evaluation.txt` (H01–H06)

The same files also exist under `docs_schubKI/materials/`. The challenge brief is `docs_schubKI/AI-based molecular media analysis by process-spectroscopy in-line.pdf`. Prefer the canonical folder in code and docs. The method-comparison table should be checked against S01 p.6; flattened PDF table extraction needs review. S02 contains image-heavy slides and requires visual/manual review where used. Keep H scenarios and worked answers outside the technical retrieval corpus.

## Baseline snapshots (Python 3.14.4, offline)

| Input | Current result |
| --- | --- |
| Official D01 customer enquiry from `AFS_Examples_and_Reference.docx` | `insufficient_information`, no candidates. Literal parser recognizes only `physical_state=powder`; objective, target, matrix, industry, and feed-frame preference remain unrecognized. It asks the three generic objective/target/matrix questions. |
| `We need an industrial sensor.` | `insufficient_information`, no candidates; all 16 fields remain unknown and it asks the same three critical questions. |
| Existing synthetic food-moisture example | `provisional_candidates`, NIR candidate; objective/target/matrix and powder state recognized. Unknown accuracy and composition remain questions. |

The D01 shortfall is extraction coverage, not evidence-backed recommendation quality: current baseline cannot reach assessment for that wording. Current assessments use YAML rationale, are repeatable, and emit a deterministic Markdown report.

## Baseline checks

- `uv run pytest`: **83 passed** (3.12s; required local socket access for Streamlit AppTest).
- `uv run ruff check .`: **passed**.
- `uv run ruff format --check .`: **passed**, 39 files already formatted.

## Minimal sequence and gates

1. **Phase 1 — contracts/extraction:** extend current models compatibly; add actual structured Ollama extraction and conservative fallback. Gate: D01–D08 facts/unknowns stay grounded; correction service accepts scalar customer fields; provider errors are clear; baseline still passes offline.
2. **Phase 2 — scientific retrieval:** add manifest and page-aware AFS ingestion/retrieval, using stdlib/local dependencies where practical. Gate: development cases retrieve relevant AFS excerpts with stable source/page IDs, or report corpus gaps; keep evaluation answers out.
3. **Phase 3 — grounded assessment:** assess validated use case plus retrieved passages; emit conditional candidates, questions, risks, experiment and evidence IDs. Gate: D01/D02 can be reviewed against worked examples without answer templates; unrelated evidence cannot be cited; unfamiliar inputs can abstain; offline baseline remains usable.

No scientific recommendation or AFS hardware capability is established by this audit.
