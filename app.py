"""Streamlit entry point; assessment stays in the Python package."""

import hashlib
import json
import os
from pathlib import Path

import streamlit as st

from spectro_ai import providers
from spectro_ai.intake import EDITABLE_FIELDS, correct_requirements
from spectro_ai.providers import assess_problem, assess_use_case
from spectro_ai.reporting import render_markdown
from spectro_ai.request_trace import RequestTrace, capture_requests

ROOT = Path(__file__).resolve().parent
EXAMPLES = json.loads((ROOT / "examples" / "scenarios.json").read_text(encoding="utf-8"))
EXAMPLE_TEXT = {example["title"]: example["problem"] for example in EXAMPLES}

st.set_page_config(page_title="Spectro-AI", page_icon="🔬", layout="wide")
st.title("Spectro-AI")
st.caption("SchubKI hackathon foundation · Evidence-grounded spectroscopy assessment")
st.warning(
    "LLM-generated assessment with deterministic checks. Scientific suitability requires "
    "expert review. All example cases are synthetic."
)


def _assessment_config() -> tuple[str, str, tuple[str, ...]]:
    provider = os.getenv("SPECTRO_AI_PROVIDER", "ollama").strip().lower()
    model = (
        os.getenv("OPENAI_MODEL", "")
        if provider == "openai"
        else os.getenv("OLLAMA_MODEL", "gemma4:12b-it-q4_K_M")
    )
    key_fingerprint = hashlib.sha256(os.getenv("OPENAI_API_KEY", "").encode()).hexdigest()
    signature = (
        provider,
        model,
        os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        os.getenv("OPENAI_BASE_URL", ""),
        key_fingerprint,
        os.getenv("SPECTRO_AI_CONTEXT", "16384"),
        os.getenv("SPECTRO_AI_TIMEOUT", "240"),
        os.getenv("SPECTRO_AI_OUTPUT_TOKENS", "4096"),
    )
    return provider, model or "model not configured", signature


def _clear_assessment(*, clear_review: bool = False) -> None:
    for key in (
        "report",
        "report_input",
        "report_signature",
        "correction_error",
        "request_trace",
        "trace_signature",
    ):
        st.session_state.pop(key, None)
    if clear_review:
        for field in EDITABLE_FIELDS:
            st.session_state.pop(f"review_{field}", None)


def input_changed() -> None:
    _clear_assessment(clear_review=True)


def select_example() -> None:
    st.session_state.problem = EXAMPLE_TEXT.get(st.session_state.example, "")
    _clear_assessment(clear_review=True)


def _render_request_trace(trace: RequestTrace | None) -> None:
    if trace is None or not trace.requests and not trace.queries:
        st.info("No LLM requests or retrieval queries have been captured yet.")
        return
    st.subheader("Retrieval queries")
    if not trace.queries:
        st.caption("No retrieval queries were recorded.")
    for index, item in enumerate(trace.queries, 1):
        st.markdown(f"**{index}. {item['kind']}**")
        st.code(item["query"], language="text")
    st.subheader("LLM requests")
    for index, item in enumerate(trace.requests, 1):
        title = (
            f"{index}. {item['stage']} · attempt {item['attempt']} · "
            f"{item['provider']}:{item['model']}"
        )
        with st.expander(title):
            request = item["request"]
            for message in request.get("messages", []):
                st.markdown(f"**{message.get('role', 'message')}**")
                st.code(str(message.get("content", "")), language="text")
            st.json(request)
            sources = []
            for message in request.get("messages", []):
                if message.get("role") != "user":
                    continue
                try:
                    payload = json.loads(message.get("content", ""))
                except TypeError, json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    sources.extend(
                        payload.get("retrieved_sources_untrusted_data", [])
                        or payload.get("evidence_bundles", [])
                        or payload.get("verified_source_metadata", [])
                    )
            st.markdown("**Sources present in this request**")
            if not sources:
                st.caption("No source evidence was attached to this request.")
            else:
                for source in sources:
                    locations = ", ".join(
                        loc.get("locator", "") for loc in source.get("source_locations", [])
                    )
                    location = source.get("page_or_section") or locations
                    st.markdown(
                        f"**{source.get('source_id', 'unknown source')}** · "
                        f"{source.get('evidence_id', '')} · "
                        f"{source.get('document_version', 'unknown version')} · "
                        f"{location}"
                    )
                    if source.get("excerpt"):
                        st.markdown(source["excerpt"])
                    for context in source.get("context_blocks", []):
                        context_locations = ", ".join(
                            item.get("locator", "") for item in context.get("source_locations", [])
                        )
                        st.caption(
                            "Attached context · same source · "
                            f"{context.get('block_id', 'context block')} · "
                            f"{context_locations}"
                        )
                        if context.get("text"):
                            st.markdown(context["text"])


provider, model, config_signature = _assessment_config()
st.caption(f"Assessment provider: {provider} · model: {model}")
if provider == "openai":
    st.warning(
        "OpenAI mode sends the customer enquiry and extracted requirements to the configured "
        "OpenAI endpoint. Retrieved source excerpts are also sent for technical assessment."
    )

knowledge_config_signature = (
    os.getenv("SPECTRO_RAG_INDEX", ".cache/rag/index.sqlite3"),
    os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
    os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    os.getenv("SPECTRO_RAG_EVIDENCE_COUNT", "8"),
    os.getenv("SPECTRO_RAG_CONTEXT", "4096"),
    os.getenv("SPECTRO_RAG_PARENT_EXPANSION", "false"),
    os.getenv("SPECTRO_RAG_RERANKER", "none"),
    os.getenv("SPECTRO_RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
    os.getenv("SPECTRO_RAG_RERANKER_REVISION", "main"),
    os.getenv("SPECTRO_RAG_RERANK_CANDIDATES", "30"),
    os.getenv("SPECTRO_RAG_MULTI_QUERY", "false"),
    os.getenv("SPECTRO_RAG_MULTI_QUERY_MAX_QUERIES", "4"),
    os.getenv("SPECTRO_RAG_MULTI_QUERY_TIMEOUT", "30"),
    os.getenv("SPECTRO_RAG_MULTI_QUERY_SEARCH_TIMEOUT", "30"),
)
multi_query_enabled = os.getenv("SPECTRO_RAG_MULTI_QUERY", "false").strip().casefold() in {
    "1",
    "true",
    "yes",
    "on",
}
st.caption(
    "Experimental multi-query retrieval: "
    + (
        "enabled · review diagnostics and benchmark before demo use"
        if multi_query_enabled
        else "disabled"
    )
)
try:
    retriever = providers.EvidenceRetriever()
    knowledge_status = retriever.knowledge_status()
except ValueError as exc:
    retriever = None
    knowledge_status = {"ready": False, "revision": None, "error": str(exc)}
st.subheader("Knowledge base")
if knowledge_status.get("ready"):
    st.caption(
        f"Ready · {knowledge_status.get('mode', 'hybrid').upper()} retrieval · "
        f"{knowledge_status.get('documents', 0)} documents · "
        f"{knowledge_status.get('chunks', 0)} chunks · "
        f"embeddings: {knowledge_status.get('embedding_model') or 'not configured'} · "
        f"last successful update: {knowledge_status.get('indexed_at') or 'unknown'}"
    )
else:
    st.info(
        knowledge_status.get("message")
        or "Knowledge index unavailable. Add supported files to documents/ and update it."
    )
if knowledge_status.get("error"):
    st.error(knowledge_status["error"])
if st.button("Update Knowledge Base", key="update_knowledge_base"):
    st.session_state.pop("knowledge_update_result", None)
    if retriever is None:
        st.error("Knowledge base settings are invalid. Correct them before updating.")
    else:
        try:
            with st.status("Updating knowledge base…", expanded=True) as update_status:
                result = retriever.update_knowledge_base()
                update_status.update(label="Knowledge base updated", state="complete")
            st.session_state.pop("report", None)
            st.session_state.pop("report_signature", None)
            st.session_state.pop("request_trace", None)
            st.session_state.pop("trace_signature", None)
            st.session_state.knowledge_update_result = result
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"Knowledge base update failed: {exc}")

update_result = st.session_state.get("knowledge_update_result")
if update_result:
    document_counts = update_result.get("documents", {})
    indexed_documents = sum(document_counts.get(key, 0) for key in ("new", "updated", "unchanged"))
    embedding_counts = update_result.get("embeddings", {})
    st.success(
        "Knowledge base updated successfully. "
        f"Indexed documents: {indexed_documents} · "
        f"New: {document_counts.get('new', 0)} · "
        f"Updated: {document_counts.get('updated', 0)} · "
        f"Removed: {document_counts.get('removed', 0)} · "
        f"Unchanged: {document_counts.get('unchanged', 0)} · "
        f"Indexed chunks: {update_result.get('chunks', 0)} · "
        f"New embeddings: {embedding_counts.get('new', 0)} · "
        f"Reused embeddings: {embedding_counts.get('reused', 0)}"
    )

st.selectbox(
    "Customer example",
    ["Custom input", *EXAMPLE_TEXT],
    key="example",
    on_change=select_example,
)
problem = st.text_area(
    "Customer problem",
    key="problem",
    height=220,
    placeholder="Describe what you want to measure and in which material.",
    on_change=input_changed,
)

_, _, current_config_signature = _assessment_config()
saved_signature = st.session_state.get("report_signature")
current_signature = (problem, current_config_signature, knowledge_config_signature)
current_revision = knowledge_status.get("revision")
saved_trace_signature = st.session_state.get("trace_signature")
if (saved_signature and saved_signature != current_signature) or (
    saved_trace_signature and saved_trace_signature[0] != current_signature
):
    _clear_assessment(clear_review=True)
    saved_trace_signature = None
    st.info("Input or provider settings changed. Assess again to refresh the report.")
if saved_trace_signature and saved_trace_signature[1] != current_revision:
    _clear_assessment(clear_review=True)
    saved_trace_signature = None
    st.info("Knowledge base changed. Assess again to refresh the report.")
saved_report = st.session_state.get("report")
if (saved_report is not None and saved_report.knowledge_index_revision) or (
    saved_trace_signature and saved_trace_signature[1]
):
    saved_revision = (
        saved_report.knowledge_index_revision
        if saved_report is not None and saved_report.knowledge_index_revision
        else saved_trace_signature[1]
    )
    if saved_revision and (
        not knowledge_status.get("ready")
        or not current_revision
        or saved_revision != current_revision
    ):
        _clear_assessment(clear_review=True)
        st.info("Knowledge base changed. Assess again to refresh the report.")

if st.button("Assess use case", type="primary"):
    _clear_assessment()
    trace = RequestTrace()
    if not problem.strip():
        st.error("Enter a customer problem or choose a synthetic example.")
    else:
        try:
            if retriever is None:
                raise ValueError("Knowledge base configuration is invalid.")
            with capture_requests(trace):
                report = assess_problem(problem, retriever=retriever)
            latest_revision = retriever.knowledge_status().get("revision")
            if (
                report.knowledge_index_revision
                and report.knowledge_index_revision != latest_revision
            ):
                raise ValueError("Knowledge base changed during assessment. Run assessment again.")
            st.session_state.report = report
            st.session_state.report_input = problem
            st.session_state.report_signature = current_signature
            st.session_state.request_trace = trace
            st.session_state.trace_signature = (current_signature, latest_revision)
        except ValueError as exc:
            st.session_state.request_trace = trace
            st.session_state.trace_signature = (current_signature, knowledge_status.get("revision"))
            for key in ("report", "report_input", "report_signature"):
                st.session_state.pop(key, None)
            st.error(str(exc))

report = st.session_state.get("report")
changed = False
if report is not None:
    with st.expander("Expert review and follow-up answers"):
        st.caption("Edit extracted facts or answer gaps. Blank values remain unknown.")
        edits = {}
        for field in EDITABLE_FIELDS:
            edits[field] = st.text_input(
                field.replace("_", " ").title(),
                value=getattr(report.extracted_requirements, field) or "",
                key=f"review_{field}",
            )
        pending = correct_requirements(report.extracted_requirements, edits)
        changed = pending != report.extracted_requirements
        if st.button("Apply expert corrections"):
            _clear_assessment()
            trace = RequestTrace()
            try:
                if retriever is None:
                    raise ValueError("Knowledge base configuration is invalid.")
                with capture_requests(trace):
                    report = assess_use_case(pending, retriever=retriever)
                latest_revision = retriever.knowledge_status().get("revision")
                if (
                    report.knowledge_index_revision
                    and report.knowledge_index_revision != latest_revision
                ):
                    raise ValueError(
                        "Knowledge base changed during assessment. Run assessment again."
                    )
                st.session_state.report = report
                st.session_state.report_input = problem
                st.session_state.report_signature = current_signature
                st.session_state.request_trace = trace
                st.session_state.trace_signature = (current_signature, latest_revision)
            except ValueError as exc:
                st.session_state.request_trace = trace
                st.session_state.trace_signature = (
                    current_signature,
                    knowledge_status.get("revision"),
                )
                st.session_state.correction_error = str(exc)
            st.rerun()
    if changed:
        st.info("Review changed. Apply expert corrections to refresh assessment and exports.")
        report = None

if st.session_state.get("correction_error"):
    st.error(st.session_state.correction_error)

if report is not None and st.session_state.get("report_input") == problem and not changed:
    markdown = render_markdown(report)
    structured = report.model_dump(mode="json")
    st.download_button(
        "Download assessment as Markdown",
        data=markdown,
        file_name="spectro-ai-assessment.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download assessment as JSON",
        data=json.dumps(structured, ensure_ascii=False, indent=2),
        file_name="spectro-ai-assessment.json",
        mime="application/json",
    )
    report_tab, diagnostics_tab, structured_tab, prompts_tab = st.tabs(
        ["Assessment", "Technical diagnostics", "Structured report", "Prompts and queries"]
    )
    with report_tab:
        st.markdown(markdown)
        for item in report.retrieved_evidence:
            if item.context_blocks:
                with st.expander(
                    f"Attached context · {item.source_id} {item.page_or_section or ''} · "
                    "same source"
                ):
                    st.caption(
                        f"Attached to {item.evidence_id}; context is not independent corroboration."
                    )
                    for block in item.context_blocks:
                        locations = ", ".join(
                            location.locator for location in block.source_locations
                        )
                        st.markdown(f"**{locations} · {block.review_status}**")
                        st.markdown(block.text)
    with diagnostics_tab:
        st.markdown(render_markdown(report, include_diagnostics=True))
        diagnostics = getattr(report, "validation_diagnostics", [])
        if diagnostics:
            st.subheader("Deterministic validation")
            for item in diagnostics:
                message = f"**{item.check}** · {item.message}"
                if item.evidence_ids:
                    message += " · Evidence: " + ", ".join(item.evidence_ids)
                if item.level == "error":
                    st.error(message)
                elif item.level == "warning":
                    st.warning(message)
                else:
                    st.info(message)
    with structured_tab:
        st.json(structured)
    with prompts_tab:
        _render_request_trace(st.session_state.get("request_trace"))
else:
    prompts_tab = st.tabs(["Prompts and queries"])[0]
    with prompts_tab:
        _render_request_trace(st.session_state.get("request_trace"))
