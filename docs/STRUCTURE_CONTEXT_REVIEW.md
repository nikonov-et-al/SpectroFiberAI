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
