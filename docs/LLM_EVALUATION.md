# LLM pipeline evaluation — 2026-10-18

Software execution record, **not scientific acceptance**. No expert scores, fact
precision estimates, or scientific confidence are inferred from report statuses.

## Run and checks

- Command: `uv run python -u scripts/evaluate.py --dataset all --output .cache/grounded-evaluation-20261018`.
- Provider: local Ollama, `gemma4:12b-it-q4_K_M`; context 16384, output budget 4096, timeout 240 seconds per request.
- Same production `assess_problem` pipeline for all 31 enquiries; no case-specific prompt/rule changes, manual answer edits, or hosted/paid requests.
- Original generated JSON/Markdown and `summary.json`: ignored `.cache/grounded-evaluation-20261018/all/`.
- 31 completed, 0 runtime failures. Abstention is a generated decision, not proof of correctness.
- Final software checks: 119 pytest tests passed, Ruff lint passed, Ruff format passed (46 files); local Streamlit health HTTP 200/ok.

## Generated status counts

| Dataset | Provisional | Insufficient information | Outside supported scope | Errors |
| --- | ---: | ---: | ---: | ---: |
| synthetic | 1 | 2 | 1 | 0 |
| development | 4 | 4 | 0 | 0 |
| paraphrase | 2 | 4 | 2 | 0 |
| adversarial | 0 | 4 | 1 | 0 |
| holdout | 2 | 4 | 0 | 0 |

## Case records

Method labels below are model-generated provisional study candidates, not expert-approved recommendations.

| Case | Dataset | Status | Retained method labels |
| --- | --- | --- | --- |
| food_moisture | synthetic | insufficient_information | None retained |
| polymer_identity | synthetic | outside_supported_scope | None retained |
| fluorescent_dye | synthetic | provisional | Fluorescence Spectroscopy |
| underspecified | synthetic | insufficient_information | None retained |
| D01 | development | insufficient_information | None retained |
| D02 | development | provisional | NIR Transmission; NIR Reflection |
| D03 | development | provisional | NIR Transmission; NIR Reflection |
| D04 | development | insufficient_information | None retained |
| D05 | development | provisional | NIR spectroscopy |
| D06 | development | insufficient_information | None retained |
| D07 | development | insufficient_information | None retained |
| D08 | development | provisional | Combined MIR ATR and Fluorescence |
| P01 | paraphrase | outside_supported_scope | None retained |
| P02 | paraphrase | provisional | NIR Reflection |
| P03 | paraphrase | provisional | NIR Transmission; NIR Reflection / Transflection |
| P04 | paraphrase | outside_supported_scope | None retained |
| P05 | paraphrase | insufficient_information | None retained |
| P06 | paraphrase | insufficient_information | None retained |
| P07 | paraphrase | insufficient_information | None retained |
| P08 | paraphrase | insufficient_information | None retained |
| A01 | adversarial | insufficient_information | None retained |
| A02 | adversarial | insufficient_information | None retained |
| A03 | adversarial | insufficient_information | None retained |
| A04 | adversarial | insufficient_information | None retained |
| A05 | adversarial | outside_supported_scope | None retained |
| H01 | holdout | insufficient_information | None retained |
| H02 | holdout | provisional | NIR Reflection |
| H03 | holdout | insufficient_information | None retained |
| H04 | holdout | insufficient_information | None retained |
| H05 | holdout | insufficient_information | None retained |
| H06 | holdout | provisional | NIR spectroscopy |

## Limitations and open acceptance

- Claim bullets must be exact source quotations; this conservative lexical requirement can over-abstain. Quote existence does not establish entailment or transferability.
- D01 abstained. D02 retained two conditional NIR geometries, with limitations quoted and acceptance criteria TBD; no fixed optical path was selected. This is not expert sign-off.
- Original development inputs produced four provisional reports; paraphrases produced two. Wording sensitivity requires expert review of extraction, retrieval and decisions. Paraphrases are synthetic review inputs, not official organizer cases.
- All five adversarial inputs abstained. This small set does not establish general prompt-injection resistance or scientific safety.
- Human review must verify independent reference assays, applicability of source applications, fusion claims, geometry and instrument compatibility. These remain unchecked scientific questions.
- OpenAI unit/SDK-boundary checks pass; live OpenAI run was unavailable because model and credentials are unset. No paid request was sent.
- Initial 2048-token Ollama smoke hit output limit; same enquiry completed with 4096. Defaults now use 4096 output tokens and 240 seconds. Larger inputs can still exceed budget or force whole evidence excerpts to be dropped.
