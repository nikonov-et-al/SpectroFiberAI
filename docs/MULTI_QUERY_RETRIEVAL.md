# Experimental multi-query retrieval

Date: 2026-10-18. Engineering experiment; disabled by default. No scientific
applicability rule, hardware specification or confidence value is introduced.

## Design and compatibility

The production baseline already retrieves deterministic requirement groups. The
experiment generates complementary evidence questions rather than mere paraphrases.
A separate bounded request follows extraction/corrections and the existing essential
fact gate. Corrected validated facts take precedence over untrusted original text.
Generated plans never modify CustomerUseCase or enter assessment as source evidence.

One pinned index supplies all initial, claim-specific and citation-validation
operations. Each explicit question uses existing BM25/semantic/RRF search. Pooling
uses best reciprocal rank across question lists, bounded round-robin admission and
stable ties, with no extra score for repeated retrieval. Existing canonical source
representatives remain unchanged. Different passages from one document are not
independent corroboration. Optional cross-encoder scoring runs once on the merged
initial pool. Parent expansion reuses existing selection and context validation.

Single-query control is diagnostic only; compare primarily against current grouped
retrieval. Turning the feature off restores current behavior without migrations.
No vector database, framework, agent runtime or new dependency is required.

## Evaluation protocol

Use one explicitly prepared index snapshot. Bind labels to evidence ID, source ID,
source version and passage content hash; parser/chunk changes invalidate labels.
Review the union of candidates from all arms rather than only one system's hits.
Retain reviewer-confirmed empty relevance sets for unrelated questions. Unreviewed
quality metrics remain null, never inferred from source-review flags.

Benchmark cases may include `validated_requirements` copied literally from their
explicitly synthetic enquiry. This avoids an unreported extraction request and
lets the planner declare populated anchors. These fixture facts do not establish
extraction accuracy. Cases without essential facts preserve abstention behavior.
Human information-need labels belong in `information_needs` with version-bound
`relevant` records; mark `needs_reviewed` only after complete human review.

Compare grouped baseline, broad original-case query, complementary questions and
grouped questions under new pooling. Use identical final count/byte budgets,
candidate cap, embedding revision, reranker settings and parent-expansion setting.
Run parent expansion off/on separately. Equal context budgets do not mean equal
search compute; report query/embedding counts and candidate volume explicitly.
Use `paired_case_ids` and `paired_aggregate_metrics` for baseline/multi-query
comparisons; skipped insufficient cases cannot change the compared population.
Fallback timings include generation cost without modifying baseline timings.

Keep existing benchmark output fields compatible. Add Recall@8 because final
selection normally stops at eight; legacy Recall@10 is not silently renamed.
Measure Precision@5, Recall@8, MRR, candidate recall, labeled information-need
coverage, duplicate results and document concentration. Empty relevant sets use
false-positive retrieval/abstention checks rather than invented recall.

Use all four original synthetic examples, limitation/unrelated questions, new
explicitly synthetic feed-frame/reaction/contamination/product-quality examples,
and available AFS material. Review both promising approaches and limitations.
Preserve calibration needs, units, qualifications, measurement conditions, source
application scope and exact quotation locations. Similarity and quote membership
cannot validate customer applicability; retain abstention and expert review.

Measure cold and warm generation/search/rerank timing, actual provider token usage
when available, context bytes, process CPU time and peak process memory. Distinguish
retrieval-only saved-plan trials from live generation and complete assessment.
Generation timeout is 30 seconds by default; optional retrieval requests share a
30-second budget. CPU reranking is synchronous, not forcibly preempted.

## Acceptance and deferred work

Demo acceptance requires no observed safety/provenance regression, improved macro
Recall@8 and information-need coverage against grouped baseline, no Precision@5 or
MRR decline, reviewed useful promise/limitation evidence, and median added latency
at most 30 seconds. Record any deadline limitations. These are engineering gates,
not spectroscopy accuracy claims. Default remains off even after demo acceptance.

Human relevance labels and expert review remain required. Do not close existing
Phase 3–5 or parent-context usefulness gates through this experiment.

Defer recursive search, paraphrase ensembles, automatic method routing, new
applicability rules, learned fusion, fine-tuning, graph/vector services and hard
process-level cancellation. Extend only after measured need.

## Verification record

Software acceptance, 2026-10-18: full offline pytest passed 225 tests, including
Streamlit AppTests; Ruff lint and format checks passed (66 Python files). SQLite
integration exercises the optional path with parent expansion disabled/enabled,
exact source verification, untouched customer facts and both export views. A
temporary Streamlit server returned 200/ok and stopped. Windows sandbox loopback
restrictions required full tests and live runtime checks outside the sandbox.

Local Ollama (`gemma4:12b-it-q4_K_M`) generated three accepted complementary
evidence questions for the synthetic pharmaceutical example in 4.45 seconds.
Hybrid retrieval using `qwen3-embedding:0.6b` then took 2.60 seconds; one pinned
index (`123cdc83ecc9466b8e671afdb00ceda7`) supplied eight verified technical
excerpts, 7,604 UTF-8 excerpt bytes, across four source IDs. Parent expansion and
reranking were disabled in this smoke. Source count is not corroboration or a
quality metric. This did not execute a live technical assessment.

The ten-case BM25/no-reranker benchmark completed with six generated plans and
four insufficient-information skips. Generation took 2.84–4.05 seconds per
applicable case in this single run; provider input/output token counts were
recorded for all six. Whole benchmark-process peak working set was approximately
60 MB; this excludes the separate Ollama process and is not per-arm memory cost.
Warm runs reuse each plan and are explicitly retrieval-only. Ignored records:
`.cache/multi-query/benchmark-bm25.json` and `.cache/multi-query/smoke.json`.

Initial runtime checks exposed unconstrained anchor names and a first-use pinned
diagnostics bug; both were repaired and regression-covered. Prompt wording was
tightened to request corpus-answerable evidence questions rather than customer
follow-up answers. Deterministic checks still cannot establish semantic fidelity.

Human relevance labels remain absent: precision, recall, MRR and need coverage
are unavailable. M4, demo promotion and scientific acceptance remain pending;
default stays disabled. No quality improvement is claimed.
