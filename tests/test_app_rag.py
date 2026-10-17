"""Document Q&A UI uses synthetic answers; no model/network required."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from spectro_ai import rag
from spectro_ai.rag_models import CitedStatement, DocumentAnswer, RagResult

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_document_answer_export_and_stale_state(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = {"revision": "first", "documents": 1, "chunks": 1}
    monkeypatch.setattr(rag, "load_index", lambda path: (metadata, []))
    monkeypatch.setattr(
        rag,
        "ask",
        lambda question, index, settings: RagResult(
            question=question,
            provider=settings.provider,
            answer=DocumentAnswer(
                status="answered",
                statements=[
                    CitedStatement(text="Synthetic document finding.", chunk_ids=["test-chunk"])
                ],
            ),
        ),
    )
    app = AppTest.from_file(str(APP)).run()
    app.text_input(key="rag_question").input("Synthetic question").run()
    app.button(key="rag_ask").click().run()
    assert not app.exception
    assert any("Synthetic document finding" in m.value for m in app.markdown)
    assert len(app.get("download_button")) == 1
    app.text_input(key="rag_question").input("Changed question").run()
    assert not app.get("download_button")
    app.button(key="rag_ask").click().run()
    metadata["revision"] = "second"
    app.run()
    assert not app.get("download_button")
    app.button(key="rag_ask").click().run()
    app.selectbox(key="rag_provider").select("openai").run()
    assert not app.get("download_button")
    assert any("sends your question" in w.value for w in app.warning)


def test_document_error_visible_without_assessment_change(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object) -> None:
        raise ValueError("Synthetic model unavailable")

    monkeypatch.setattr(rag, "ask", fail)
    app = AppTest.from_file(str(APP)).run()
    app.text_input(key="rag_question").input("Synthetic question").run()
    app.button(key="rag_ask").click().run()
    assert not app.exception
    assert any("Synthetic model unavailable" in e.value for e in app.error)
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[0].click().run()
    assert app.session_state["report"].candidate_techniques
