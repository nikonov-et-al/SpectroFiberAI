"""Streamlit entry point; assessment stays in the Python package."""

import json
from pathlib import Path

import streamlit as st

from spectro_ai.providers import assess_problem
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
        "State, Accuracy, Frequency, Constraints. Otherwise a small literal-phrase parser "
        "recognizes examples such as `measure moisture` and `food powder`. "
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
        except ValueError as exc:
            st.error(str(exc))

report = st.session_state.get("report")
if report is not None:
    if st.session_state.get("report_input") != problem:
        st.info("Input changed. Assess again to refresh the report.")
    else:
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
