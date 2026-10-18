# Architecture

Customer enquiry → structured LLM extraction → local AFS retrieval → structured LLM
technical assessment → deterministic checks → expert-reviewable report.

## Boundaries

Streamlit owns presentation and session state. `assess_problem` extracts requirements
and invokes `assess_use_case`; corrected requirements enter `assess_use_case` directly.
Both entry points use same provider policy, retrieval, validation, and report contracts.
`SPECTRO_AI_PROVIDER=ollama|openai` selects both generation stages; default Ollama.
There is no offline parser, rule-based recommender, or fallback provider.

Provider adapters remain in `providers.py`; SDK imports are lazy. Shared prompts live
in `prompts/` and ship in package data. Structural output repair is bounded to one
retry. Provider failures raise actionable errors and cannot become recommendations.
No evidence or missing critical facts produces neutral, inspectable abstention.

Pydantic models remain independent of provider SDKs and Streamlit. Unknown scalars
are `None`, collections empty, and customer quantities stay text. Service owns request
identity and provenance. Extracted facts need literal customer quotes; expert edits
have separate provenance. Quote membership does not prove correct interpretation.

## Evidence and validation

`evidence.py` ingests allowlisted AFS files into local BM25 index. Document parsers
remain necessary: PDF pages, text spans, DOCX sections and spreadsheet rows retain
locations, hashes, scope and review status. Evaluation answers never enter technical
corpus. Existing `KnowledgeSource` contract and curated YAML vocabulary remain.

Reports retain retrieved excerpts and citation IDs. Claim links must resolve to exact
retrieved quotations; source demonstrations remain limited to original application.
Engineering hypotheses and assumptions require explicit labeling. Deterministic
checks expose errors, warnings and unchecked claims. They cannot prove entailment,
hardware compatibility or customer performance. Source-typical dimensions and limits
are not customer-selected specifications. Customer acceptance thresholds must be
stated by customer; otherwise TBD. Plans require references and held-out validation.

## Presentation and compatibility

Expert corrections preserve original enquiry and unchanged provenance. Pending edits,
input changes, configuration changes and failures hide stale reports/exports. Markdown
escapes untrusted text; JSON exports domain model. OpenAI mode discloses transfer of
customer text/facts and evidence before submission. Separate Document Q&A retains its
own model and embedding configuration.

Legacy parser APIs, baseline provider and fallback parameter are intentionally removed.
Historical artifacts are preserved; no persisted user-data migration is performed.
Tests inject provider doubles and exercise same production orchestration without
credentials/network. Live provider smoke tests and expert scientific acceptance are
separate gates. Phase 3 scientific blocker remains open until human review.
