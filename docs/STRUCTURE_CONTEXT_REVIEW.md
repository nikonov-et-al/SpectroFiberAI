# Structure-aware context: AFS review protocol

Date: 2026-10-18. These expectations were recorded from original source text
before expansion-off/on comparisons. They are engineering review cases, not
human relevance labels or scientific acceptance of customer recommendations.

| Case | Question | Expected original source context |
| --- | --- | --- |
| Table/specification | What sample and geometry accompany the NIR transmission/transflection optical-path entry? | S01 p.6, Table 1: method, measurement technique, sample type and optical-path column must stay associated. Preserve original units as written. Do not transfer a source-typical range into a customer requirement. |
| Operating conditions | What makes NIR analysis possible and what analysis requirement accompanies its overlapping bands? | S01 p.10, section 5.2: sample scattering/optical geometry and overlapping broad bands requiring chemometrics belong to the same source discussion. |
| Limitations | Which sample constraints qualify the combined MIR/fluorescence probe's potential applications? | S04 p.4: fluorescent component plus suitability for ATR, liquid or soft solid sample, and tight contact with probe head. Retain the original biological-tissue study context; do not generalize measured performance. |

## Acceptance procedure

1. Bind each run to source hashes, index revision, parser identity and chunking
   settings. Keep source review flags unchanged.
2. Visually compare relevant original PDF pages against extracted tables,
   reading order, headings and inferred nesting.
3. Pin one index; run identical BM25 retrieval with expansion disabled/enabled.
   Record ranked child IDs, attached block IDs/locations, serialized context
   volume and warm latency. Repeat for semantic retrieval only if Ollama is
   explicitly available; no generation request is needed for this comparison.
4. Inspect whether attached context recovers the expected qualification without
   mixing documents, section boundaries, units or source locations. Check that
   attachment never counts as independent corroboration.
5. Record limitations and failed cases. Expert/human review remains open;
   automated or agent inspection cannot promote expansion by default.

## Results

Parent expansion remains disabled. Bounded comparisons remain pending.

### PDF parser gate — not accepted

Docling 2.140.0 installed successfully in isolated Windows CPython 3.14.4 with both
RAG and reranker extras. CPU conversion ran from explicitly prefetched local
assets with OCR and remote services disabled. The original S01 p.6/p.10 and S04
p.4 pages were rendered and visually inspected alongside JSON/Markdown exports.

S01 p.6 contains merged method cells and source line wraps such as `mu` and
`Sensitiv ity`; these must not be silently replaced with inferred units/labels.
More seriously, extraction assigns `diffuse reflection` to the UV/Visible cuvette
row where the original states `absorption`. Accurate/no-cell-matching and
fast/cell-matching were two focused repair attempts; neither resolves the error.
No parser-default promotion is accepted.

Full S01 outline inference assigns a table caption level 4, References level 6,
and inconsistent nesting around application cases. Disabling font-style or both
style signals does not establish correct structure. Validate inferred levels and
fall back to flat/page-local sections. S04 p.4's left-to-right column continuation
appears coherent in the inspected export; that observation does not validate all
pages or scientific recommendations.

Ignored reproducibility artifacts: `.cache/docling/review/` contains original-page
PNGs, baseline JSON/Markdown, full S01 outline, two table configuration attempts,
and heading-style comparisons. Active demo index remains unchanged.

Model assets: Heron `model.safetensors` SHA256
`00333A43451945AAF89DB8CA9C0A17E75D1537C17DB60FDB91AA95F4C7929E0C`;
TableFormer accurate SHA256
`2A7D6C924B3CD12FB99A09280CA9C33A89C5D60B93253617D2E088C1A40374D9`.
Actual application fingerprints must include configured artifacts and options.

### Final pinned BM25 comparison — usefulness not accepted

Run after final format/identity changes, with 7 real AFS source files and 196
deduplicated chunks. Active demo database was not replaced. Snapshot revision:
`9298e600933d4b35bdc1de85fde91245`; extraction fingerprint:
`e648d473f7b014de9ff2408c388b217aa315a193ea81659afc66f43fa3db7992`.
Source hashes and exact child/block identities are in the ignored JSON artifact
`.cache/structure-context/pypdf-review.json`.

Each mode used one pinned snapshot, BM25 without reranking, a warmup and three
timed repetitions. All modes returned 8 children. Times below are local warm
medians, not a service-level or scientific performance claim.

| Case | Off / on latency (ms) | Attached blocks | Serialized context bytes | Child + context bytes (on) |
| --- | --- | --- | --- | --- |
| Table/specification | 8.74 / 9.84 | 2 | 2,687 | 11,188 |
| Operating conditions | 8.61 / 10.63 | 2 | 3,460 | 11,913 |
| Limitations | 8.93 / 10.24 | 2 | 3,316 | 11,955 |

All totals stay within the 12,000-byte retrieval budget. Table search adds S01
p.6's caption block, but this does not repair or validate row/value relationships.
Operating-condition retrieval includes S01 p.10's first chunk, while the expected
overlapping-band/chemometrics qualification is in the following chunk and is not
recovered. Limitation retrieval reaches S04 p.4's final chunk after expansion
budget allocation, but not the expected fluorescent-component/ATR/tight-contact
qualification. Merely retrieving the expected page is not successful retrieval
of its relevant statement.

Canonical provider verification accepted all attached blocks with their source
locations and review metadata. It excluded the secondary-transcription child in
the table case (7 technical children retained there, 8 in the other cases). With
the existing default 16,384 context budget and 4,096 output reservation, complete
request fitting retained 3 children and **0 context blocks** in every case. This
is the required context-first dropping policy, not evidence of useful expansion.
Details are in ignored `.cache/structure-context/provider-fit-review.json`.

No generation request or semantic model request was made. Precision/recall remain
null because no human-reviewed relevance labels exist. The software regression
suite passes, but actual scientific-context usefulness and expert acceptance
remain open; parent expansion stays disabled. Improving query selection or
changing model/context budgets requires a separately reviewed task.
