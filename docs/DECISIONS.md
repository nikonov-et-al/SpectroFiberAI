# Decision log

| Date | Decision | Reason / revisit trigger |
| --- | --- | --- |
| 2026-10-18 | Evolve SQLite indexes atomically to schema 3 while retaining schema 2 readers; separate extraction and embedding identities | Source block/section relationships support grounded context. Parser changes require extraction, but identical embedding inputs under identical embedding configuration can reuse validated vectors. Existing snapshots and demo remain usable. |
| 2026-10-18 | Opt-in CPU cross-encoder reranking after fused retrieval; retain visible RRF fallback | English MiniLM trial changes textual relevance ordering only. Explicit model setup precedes cached-only assessment; optional dependencies do not affect default retrieval. Keep existing citation guards and expert acceptance gates; benchmark labels require human review. |
| 2026-10-18 | Unify assessment retrieval in one SQLite knowledge base with Ollama embeddings, BM25 and RRF; remove standalone Document Q&A | Reuse stored vectors and provenance, index only on explicit updates, preserve prior index on failure and bind assessments to immutable revisions. Exact cosine scan avoids another service; scientific acceptance remains independent. |
| 2026-10-18 | Validate source observations individually; separate provisional feasibility from production readiness and customer report from diagnostics | A missing secondary citation must not erase supported exploratory options. Literal source quotations retain provenance; lexical checks do not establish applicability. Candidate comparisons require independent references, held-out evaluation and customer-agreed criteria. |
| 2026-10-18 | Replace offline parser/recommender/fallback with sole evidence-grounded Ollama/OpenAI pipeline | User-approved architecture supersedes earlier baseline decisions. Keep deterministic guardrails and network-free tests; provider failures are errors, scientific gaps abstain. Expert scientific gate remains open. |
| 2026-10-17 | Keep grounded assessment as an opt-in Ollama adapter over the validated customer case and allowlisted evidence index; report abstention and fallback distinctly | Preserve offline baseline and customer corrections, while citations remain tied to retrieved source excerpts; lexical screening is only a necessary check, not semantic validation. |
| 2026-10-17 | Add separate allowlisted AFS evidence index with deterministic local BM25 | Preserve citations and review status without turning retrieved text into technique rules; revisit after expert review and measured retrieval gaps. |
| 2026-10-17 | Add schema-constrained local Ollama extraction with literal quote checks and visible offline fallback | Keep default credential-free mode; a quote check limits unsupported facts but does not establish semantic correctness. |
| 2026-10-16 | Validated structured expert corrections with field provenance | Preserve source text/units; recompute assessment and hide pending exports without adding a model dependency. |
| 2026-10-16 | Context changes screening conditions, not validated rankings | Matrix/interference/sensitivity and access require laboratory comparison; no guessed AFS limits. |
| 2026-10-16 | Separate six-case evaluation fixtures | Preserve original four-example contract; structural review cannot replace spectroscopy expert scores. |
| 2026-10-16 | Independent document Q&A; no assessment enrichment | Preserve deterministic assessment and scientific evidence boundaries. |
| 2026-10-16 | SQLite + NumPy exact search; optional pypdf | Small local corpus needs no vector service or orchestration framework. Revisit only after measured scale problems. |
| 2026-10-16 | Local Ollama embeddings; Ollama or explicit OpenAI-compatible generation | Generation provider can change without reindexing. API mode discloses question/excerpt transfer. |
| 2026-10-16 | Gemma 12B Q4, Qwen3 Embedding 0.6B, 4096 context | User's RTX 4070 12 GB target; synthetic rehearsal passed with context 4096; single-run residency/timings are recorded in history, without a throughput guarantee. |
| 2026-10-16 | Citation membership validation; no similarity confidence threshold | References do not prove correctness or scientific suitability. |
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
| 2026-10-15 | No Docker, vector store, agent framework, deployment | Initial baseline choice; local RAG index added 2026-10-16, other infrastructure remains deferred. |
| 2026-10-15 | Local binding and telemetry disabled | Predictable offline demo. |

Resolved versions live in uv.lock. Record architecture/setup changes here; update
related tests and documentation.
