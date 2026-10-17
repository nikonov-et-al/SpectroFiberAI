"""Streamlit entry point; assessment stays in the Python package."""

import json
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
st.caption("SchubKI hackathon foundation · Offline spectroscopy use-case screening")
st.warning(
    "Provisional rules and templates, not validated scientific analysis. "
    "All example cases are synthetic. Expert verification required."
)


def select_example() -> None:
    st.session_state.problem = EXAMPLE_TEXT.get(st.session_state.example, "")
    st.session_state.pop("report", None)
    st.session_state.pop("report_input", None)


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
)
with st.expander("How this baseline reads your input"):
    st.markdown(
        "Use one `Label: value` per line or separate labels with semicolons. Supported labels: "
        "Industry, Objective, Target, Matrix, Concentration range, Temperature, Pressure, "
        "State, Accuracy, Frequency, Constraints, Process, Composition, Detection limit, "
        "Deployment mode, Interference. Otherwise a small literal-phrase parser "
        "recognizes examples such as `measure moisture`, `food powder`, "
        "and `monitor ethanol` in `aqueous fermentation broth`. "
        "Unknown details stay unknown; negated clauses are excluded."
    )
if st.button("Assess use case", type="primary"):
    st.session_state.pop("report", None)
    if not problem.strip():
        st.error("Enter a customer problem or choose a synthetic example.")
    else:
        try:
            st.session_state.report = assess_problem(problem)
            st.session_state.report_input = problem
            st.session_state.pop("correction_error", None)
            for field in EDITABLE_FIELDS:
                st.session_state.pop(f"review_{field}", None)
        except ValueError as exc:
            st.error(str(exc))

report = st.session_state.get("report")
changed = False
if report is not None:
    if st.session_state.get("report_input") != problem:
        st.info("Input changed. Assess again to refresh the report.")
    else:
        with st.expander("Expert review and follow-up answers"):
            st.caption("Edit recognized facts or answer gaps. Blank values remain unknown.")
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
                st.session_state.pop("report", None)
                try:
                    st.session_state.report = assess_use_case(pending)
                    st.session_state.report_input = problem
                    st.session_state.pop("correction_error", None)
                except ValueError as exc:
                    st.session_state["correction_error"] = str(exc)
                st.rerun()
        if changed:
            st.info("Review changed. Apply expert corrections to refresh assessment and export.")
            report = None
if st.session_state.get("correction_error"):
    st.error(st.session_state["correction_error"])
if report is not None and st.session_state.get("report_input") == problem and not changed:
    markdown = render_markdown(report)
    st.download_button(
        "Download assessment as Markdown",
        data=markdown,
        file_name="spectro-ai-assessment.md",
        mime="text/markdown",
    )
    report_tab, structured_tab = st.tabs(["Assessment", "Structured report"])
    with report_tab:
        st.markdown(markdown)
    with structured_tab:
        st.json(report.model_dump(mode="json"))

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
    # Include effective configuration so settings changes cannot reveal stale results.
    import os

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
