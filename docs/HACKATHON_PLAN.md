# Hackathon plan

## Foundation implemented

- Python 3.14 with uv-managed locked dependencies.
- Typed requirements, candidates, evidence, feasibility plans, reports.
- Literal extraction, missing-field questions, qualitative rules, explicit abstention.
- Five preliminary technique profiles with provenance and expert-review flags.
- Streamlit demo, four synthetic scenarios, structured JSON, Markdown download.
- Optional SDK and prompt/provider seams; OpenAI adapter unimplemented.
- Tests, Ruff, GitHub Actions checks.

## Unknowns

Problem, users, judging criteria, real data, references, hardware, AFS sources,
permitted services, networking, and demo constraints. See CHALLENGE.md.

## Prioritized milestones

1. **Intake:** record outcome, criteria, assets, hardware, restrictions. Exit: concrete
   demo acceptance case.
2. **Expert alignment:** validate knowledge/rules; obtain representative cases,
   reference answers, and consequential failures.
3. **One workflow:** adapt extraction, rationale, and experiment to the challenge;
   demonstrate input through reviewable export.
4. **Evidence/reliability:** replace assumptions with official sources, add regressions,
   score reports using EVALUATION.md, rehearse failures.
5. **Rehearsal:** freeze proven path, fit allotted time, prepare local synthetic fallback.

Prioritize complete behavior over feature count. Resolve domain questions before
tuning AI against unverified labels.

## Optional stretch features

Validated LLM extraction; authorized document retrieval; instrument constraints;
challenge dataset exploration; German translation; hardware only if required/available.
Choose against the actual rubric.

## Demo checklist

- [ ] Confirm challenge and allowed assets with source/date.
- [ ] Run locked sync, pytest, Ruff lint/format on demo machine.
- [ ] Confirm local app/report generation without credentials.
- [ ] Rehearse food, polymer, fluorescence, and insufficient-information cases.
- [ ] Confirm structured output and Markdown match current input.
- [ ] Show provisional status, gaps, and evidence categories.
- [ ] Confirm offline baseline after initial install.
- [ ] Rehearse customer question → gaps → candidates → testable experiment.
- [ ] Agree presenter, time limit, fallback, recovery steps.
- [ ] Have domain expert review scientific claims before pitching suitability.

## Risks

| Risk | Mitigation |
| --- | --- |
| Literal parser misses language/negation | Show raw input/gaps; use labels and reviewed regressions. |
| Rules mistaken for scientific validation | Provisional status, evidence labels, abstention, expert review. |
| Sources do not describe actual instruments | Obtain AFS documents; validate compatibility separately. |
| No reference labels/samples | Keep thresholds unknown; prioritize reference acquisition. |
| No internet/API | Install early; maintain offline baseline. |
| Future SDK incompatible with Python 3.14 | Assess when known; record agreed runtime change. |
| Infrastructure consumes hackathon | Keep one process; defer ingestion/deployment/frameworks. |

## First three tasks when announced

Confirm acceptance criteria/assets; validate knowledge/rules with experts; implement
one challenge-specific workflow with regression coverage.
