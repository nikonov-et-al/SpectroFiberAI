# Spectro-AI engineering guide

- Goal: a reliable SchubKI hackathon demo for spectroscopy use-case assessment.
  Challenge, data, hardware, and judging requirements remain unconfirmed.
- Python 3.14; manage dependencies with uv, pyproject.toml, and uv.lock. Keep
  `.tools` (uv bootstrap) separate from `.venv` (project). Never commit secrets.
- Keep typed Pydantic domain models and deterministic guardrails independent of
  Streamlit and provider code. Prefer small functions; no framework without need.
- Unknown scalar values are None; missing collections are empty lists. Preserve
  customer units as text. Never invent specifications, limits, citations, or confidence.
- Distinguish source-backed scientific statements, engineering heuristics, and
  assumptions. A verified source link does not validate a customer recommendation.
- Add methods in knowledge/techniques.yaml with provenance; add applicability rules
  only with explicit rationale and tests. New data sources implement KnowledgeSource.
- Keep model adapters in providers.py. Prompts live in prompts/; outputs must validate
  against the domain models. Assessment requires explicitly configured Ollama or OpenAI; tests must work without credentials/network.
- After changes run `uv run pytest`, `uv run ruff check .`, and
  `uv run ruff format --check .` (local uv may be under .tools/Scripts).
- Preserve four synthetic examples, insufficient-information behavior, Markdown
  export, and Streamlit AppTest coverage. Label new mock/placeholder capabilities.
- Update README and docs/DECISIONS.md when setup or architecture changes. Update
  docs/HACKATHON_PLAN.md when functionality or challenge facts change.

# AGENTS.md

Standing rules for AI coding agents in this repository. Keep milestones and detailed implementation plans in `PLAN.md` or task-specific documents, not here.

## 1. Scope and safety

- Work within the repository and inspect only what the task requires. Preserve its structure, unrelated changes, and existing project patterns.
- Change source, tests, configuration, and documentation as needed for the task. Avoid backup snapshots, generated files, dependency caches, and IDE metadata unless explicitly requested.
- Do not introduce frameworks, services, build systems, or broad refactors without a clear need.
- Preserve public contracts and persisted data compatibility unless the task explicitly requires a change. Escalate destructive actions, user-data changes, and migration decisions to the orchestrator; obtain user approval where required.
- Make reasonable, reversible assumptions for minor ambiguity and report them. Do not guess about irreversible or compatibility-sensitive changes.


## 2. Roles and delegation

all agents and subagetsn use caveman and ponytail skills

**Sol — orchestrator**
- Owns requirements, architecture, task sequencing,
  delegation, final review, acceptance, PLAN.md and HISTORY.md.
- Prefer delegating source-code implementation to Luna.
- Do not edit application source code directly during
  autonomous plan execution.
- Keep architectural, scientific validity, compatibility,
  migration, and data-safety decisions with Sol.
- Review worker results before accepting a task.
- Run integration checks when appropriate.

**Luna — implementation agent**
- Handles bounded features, bug fixes, tests, refactoring,
  documentation, and focused investigations.
- Implement both independent and sequential tasks.
- Inspect relevant code before editing.
- Preserve public contracts and existing behavior.
- Run relevant checks and report exact results.
- Never commit unless explicitly authorized.

**Delegation rules**
- Prefer Luna for implementation, including small tasks
  when operating in autonomous plan-execution mode.
- Use at most two concurrent Luna agents.
- Parallelize only independent tasks with non-overlapping
  file ownership.
- Sequential dependencies must be executed in order.
- No nested delegation.
- Each delegation must specify scope, allowed files,
  expected behavior, exclusions, and acceptance checks.
- Sol reviews changes before accepting them.
- Subagents inherit parent permissions; instructions
  are not a security boundary.

**Configuration**
- Set model versions and reasoning effort in
  .codex/config.toml and .codex/agents/*.toml.
- Keep runtime limits in Codex configuration.


## 3. Planning and delivery

- For substantial work, check `PLAN.md` when present and confirm the relevant step still matches the code. Choose the smallest useful step and define how to verify it. Skip formal plans for trivial changes.
- After accepting a planned step, add a `Completed` entry in `HISTORY.md` containing its **exact original text**, completion date, implementation summary, verification, and deviations. Only then remove that step from `PLAN.md`; keep `PLAN.md` for pending work.
- Keep changes small and focused. Follow existing layer boundaries and use project tooling or structured parsers when appropriate.
- Run relevant tests or checks, fix regressions caused by the change, and state precisely what passed, failed, or could not run. Never imply unrun checks passed.


### Autonomous PLAN.md execution

When explicitly asked to execute PLAN.md end-to-end:

1. Read the complete PLAN.md and relevant project context.
2. Determine dependencies and a safe execution order.
3. Work through all feasible pending tasks, not just one.
4. Select the smallest independently verifiable task.
5. Delegate implementation to Luna where available.
6. Validate the result against acceptance criteria.
7. Fix regressions before accepting the task.
8. Once accepted by Sol:
   - Append a Completed entry to HISTORY.md.
   - Preserve the exact original task text.
   - Record the date, implementation and verification.
   - Remove the completed task from PLAN.md.
9. Continue automatically with the next pending task.
10. Do not ask for approval between routine tasks.


- PLAN.md is the executable implementation backlog.
- docs/HACKATHON_PLAN.md tracks hackathon strategy,
  scope, milestones, and confirmed challenge facts.
- Keep both consistent without duplicating task history.


### Failure handling

- Investigate failed tests and attempt focused repairs.
- After two unsuccessful repair attempts, record the
  blocker and preserve the original pending task.
- Continue with other safe, independent tasks.
- Never mark unverified tasks as completed.
- Never fabricate scientific requirements, limits,
  measurements, citations, or customer specifications.
- Stop when a decision requires user approval,
  no safe executable tasks remain, or execution
  is interrupted by permissions or runtime limits.

### Verification

- Run targeted checks after implementation.
- Before accepting a milestone, run:
  - uv run pytest
  - uv run ruff check .
  - uv run ruff format --check .
- Report failures and unavailable checks accurately.

### Completion condition

Continue until PLAN.md has no feasible pending tasks,
or all remaining tasks have documented blockers.

Report completed work, blocked tasks, tests, and
remaining risks.


## 4. Commits

- Commit **only when explicitly requested**. Sol owns commits by default; stage only related files, and commit accepted, verified work without unrelated changes.
- Use Conventional Commits: `<type>: <summary>`. Types: `docs`, `chore`, `feat`, `fix`, `refactor`, `test`, `perf`, `build`.

## 5. Handoff

Keep reports concise:

- **Implementation:** summary, files changed, verification (commands and results), blockers, remaining risks.
- **Plan (when needed):** assumptions, numbered steps with checks, likely files, acceptance criteria, non-goals.
