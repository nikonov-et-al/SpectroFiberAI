"""Streamlit entry point; assessment stays in the Python package."""

import hashlib
import json
import os
from pathlib import Path

import streamlit as st

from spectro_ai.intake import EDITABLE_FIELDS, correct_requirements
from spectro_ai.providers import RagSettings, assess_problem, assess_use_case
from spectro_ai.rag import ask, load_index, render_answer
from spectro_ai.reporting import render_markdown

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
    for key in ("report", "report_input", "report_signature", "correction_error"):
        st.session_state.pop(key, None)
    if clear_review:
        for field in EDITABLE_FIELDS:
            st.session_state.pop(f"review_{field}", None)


def input_changed() -> None:
    _clear_assessment(clear_review=True)


def select_example() -> None:
    st.session_state.problem = EXAMPLE_TEXT.get(st.session_state.example, "")
    _clear_assessment(clear_review=True)


provider, model, config_signature = _assessment_config()
st.caption(f"Assessment provider: {provider} · model: {model}")
if provider == "openai":
    st.warning(
        "OpenAI mode sends the customer enquiry and extracted requirements to the configured "
        "OpenAI endpoint. Retrieved source excerpts are also sent for technical assessment."
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
if saved_signature and saved_signature != (problem, current_config_signature):
    _clear_assessment(clear_review=True)
    st.info("Input or provider settings changed. Assess again to refresh the report.")

if st.button("Assess use case", type="primary"):
    _clear_assessment()
    if not problem.strip():
        st.error("Enter a customer problem or choose a synthetic example.")
    else:
        try:
            st.session_state.report = assess_problem(problem)
            st.session_state.report_input = problem
            st.session_state.report_signature = (problem, current_config_signature)
        except ValueError as exc:
            _clear_assessment()
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
            try:
                st.session_state.report = assess_use_case(pending)
                st.session_state.report_input = problem
                st.session_state.report_signature = (problem, current_config_signature)
            except ValueError as exc:
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
    report_tab, diagnostics_tab, structured_tab = st.tabs(
        ["Assessment", "Technical diagnostics", "Structured report"]
    )
    with report_tab:
        st.markdown(markdown)
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

st.divider()
st.header("Document Q&A")
st.caption("Local document retrieval; separate from spectroscopy assessment.")
try:
    rag_settings = RagSettings.from_env()
    provider = st.selectbox(
        "Document answer provider",
        ["ollama", "openai"],
        index=0 if rag_settings.provider == "ollama" else 1,
        key="rag_provider",
    )
    from dataclasses import replace

    rag_settings = replace(rag_settings, provider=provider)
    if provider == "openai":
        st.warning(
            "API mode sends your question and retrieved excerpts to the configured endpoint."
        )
    index_path = Path(rag_settings.index)
    revision = None
    try:
        metadata, _ = load_index(index_path)
        revision = metadata["revision"]
        st.caption(f"Index: {metadata['documents']} documents · {metadata['chunks']} chunks")
    except ValueError as exc:
        st.info(str(exc))
    question = st.text_input("Question about your documents", key="rag_question")
    signature = (
        question,
        rag_settings,
        revision,
        os.getenv("OPENAI_MODEL"),
        os.getenv("OPENAI_BASE_URL"),
    )
    if st.button("Ask documents", key="rag_ask"):
        st.session_state.pop("rag_result", None)
        try:
            st.session_state.rag_result = ask(question, index_path, rag_settings)
            st.session_state.rag_signature = signature
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get("rag_result")
    if result is not None and st.session_state.get("rag_signature") == signature:
        answer_markdown = render_answer(result)
        st.markdown(answer_markdown)
        st.download_button(
            "Download document answer as Markdown",
            answer_markdown,
            file_name="spectro-ai-document-answer.md",
            mime="text/markdown",
        )
except ValueError as exc:
    st.error(str(exc))
