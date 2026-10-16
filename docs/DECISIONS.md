# Decision log

| Date | Decision | Reason / revisit trigger |
| --- | --- | --- |
| 2026-10-16 | MIT License | Allow free use, copying, modification, and distribution with attribution. |
| 2026-10-15 | Installed CPython 3.14 replaces requested 3.12 | Selected for the development environment; revisit if challenge SDK requires it. |
| 2026-10-15 | uv, pyproject, lockfile; separate tooling/app environments | Reproducibility without global changes. |
| 2026-10-15 | English throughout | Default language for initial demo; translation deferred. |
| 2026-10-15 | Streamlit in one process | Short path to a useful demo. |
| 2026-10-15 | Typed models and qualitative rules | Inspectable, reusable behavior while problem remains unknown. |
| 2026-10-15 | Require objective, target, matrix; otherwise abstain | Prevent unsupported instrument choices. |
| 2026-10-15 | Preserve quantities as text; no performance guesses | Units and acceptance requirements not established. |
| 2026-10-15 | YAML knowledge with provenance and source protocol | Easy expert edits; retrieval follows real documents. |
| 2026-10-15 | OpenAI explicitly unimplemented | Model/credentials cannot block baseline; no false feature claims. |
| 2026-10-15 | No Docker, vector store, agent framework, deployment | No confirmed need. |
| 2026-10-15 | Local binding and telemetry disabled | Predictable offline demo. |

Resolved versions live in uv.lock. Record architecture/setup changes here; update
related tests and documentation.
