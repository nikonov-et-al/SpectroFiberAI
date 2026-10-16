# Spectro-AI engineering guide

- Goal: a reliable SchubKI hackathon demo for spectroscopy use-case assessment.
  Challenge, data, hardware, and judging requirements remain unconfirmed.
- Python 3.14; manage dependencies with uv, pyproject.toml, and uv.lock. Keep
  `.tools` (uv bootstrap) separate from `.venv` (project). Never commit secrets.
- Keep typed Pydantic domain models and deterministic assessment independent of
  Streamlit and provider code. Prefer small functions; no framework without need.
- Unknown scalar values are None; missing collections are empty lists. Preserve
  customer units as text. Never invent specifications, limits, citations, or confidence.
- Distinguish source-backed scientific statements, engineering heuristics, and
  assumptions. A verified source link does not validate a customer recommendation.
- Add methods in knowledge/techniques.yaml with provenance; add applicability rules
  only with explicit rationale and tests. New data sources implement KnowledgeSource.
- Keep model adapters in providers.py. Prompts live in prompts/; outputs must validate
  against the domain models. Baseline must always work without credentials/network.
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

**Sol — orchestrator** owns task scope, architecture, planning, delegation, final diff review, acceptance, and plan/history updates. Handle tightly coupled or high-risk changes directly. Do not delegate final architectural, migration, data-safety, compatibility, or acceptance decisions.

**Luna — implementation agent** handles bounded fixes, isolated features, tests, mechanical refactors, documentation, and focused read-only investigations. Inspect relevant code first, stay within scope, preserve behavior unless instructed otherwise, run targeted checks, and report blockers or risks. Never commit unless explicitly authorized.

- Set model versions and reasoning effort in `.codex/config.toml` and `.codex/agents/*.toml`, not in this file.
- Use the project's Caveman and Ponytail skills when available: Sol uses `ultra` for both; Luna uses Caveman `wenyan-full` or `wenyan-ultra` and Ponytail `ultra`. Report unavailable skills; do not claim to have used them.
- Delegate only when independent work benefits from separate context or parallelism. Use at most two Luna agents concurrently, with no nested delegation or overlapping edits. Keep small or sequential tasks with Sol.
- Each delegation must state the objective, allowed and excluded files, expected behavior, and acceptance checks. Sol reviews the returned diff and verification before accepting it.
- Subagents inherit parent permissions; agent instructions are not a security boundary. Keep runtime limits in `.codex/config.toml` (including `agents.max_concurrent_threads_per_session` where supported).

## 3. Planning and delivery

- For substantial work, check `PLAN.md` when present and confirm the relevant step still matches the code. Choose the smallest useful step and define how to verify it. Skip formal plans for trivial changes.
- After accepting a planned step, add a `Completed` entry in `HISTORY.md` containing its **exact original text**, completion date, implementation summary, verification, and deviations. Only then remove that step from `PLAN.md`; keep `PLAN.md` for pending work.
- Keep changes small and focused. Follow existing layer boundaries and use project tooling or structured parsers when appropriate.
- Run relevant tests or checks, fix regressions caused by the change, and state precisely what passed, failed, or could not run. Never imply unrun checks passed.

## 4. Commits

- Commit **only when explicitly requested**. Sol owns commits by default; stage only related files, and commit accepted, verified work without unrelated changes.
- Use Conventional Commits: `<type>: <summary>`. Types: `docs`, `chore`, `feat`, `fix`, `refactor`, `test`, `perf`, `build`.

## 5. Handoff

Keep reports concise:

- **Implementation:** summary, files changed, verification (commands and results), blockers, remaining risks.
- **Plan (when needed):** assumptions, numbered steps with checks, likely files, acceptance criteria, non-goals.
