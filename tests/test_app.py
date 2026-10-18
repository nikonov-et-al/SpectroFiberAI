import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from spectro_ai import providers
from spectro_ai.models import (
    AssessmentReport,
    CustomerUseCase,
    EvidenceClaim,
    EvidenceContextBlock,
    FeasibilityPlan,
    RequestBudgetDiagnostic,
    RetrievedEvidence,
    SourceLocation,
    TechniqueCandidate,
    ValidationDiagnostic,
)
from spectro_ai.reporting import render_markdown
from spectro_ai.request_trace import record_query, record_request

APP = Path(__file__).resolve().parents[1] / "app.py"


class FakeKnowledgeRetriever:
    updates = 0
    revision = "revision-1"
    update_error = None

    def knowledge_status(self):
        return {
            "ready": True,
            "documents": 1,
            "chunks": 2,
            "embedding_model": "qwen3-embedding:0.6b",
            "indexed_at": "test time",
            "revision": type(self).revision,
        }

    def update_knowledge_base(self):
        type(self).updates += 1
        if type(self).update_error:
            raise ValueError(type(self).update_error)
        return {
            "documents": {"new": 1, "updated": 0, "removed": 0, "unchanged": 1},
            "chunks": 3,
            "embeddings": {"new": 1, "reused": 2},
        }


@pytest.fixture(autouse=True)
def knowledge_retriever(monkeypatch):
    FakeKnowledgeRetriever.updates = 0
    FakeKnowledgeRetriever.revision = "revision-1"
    FakeKnowledgeRetriever.update_error = None
    monkeypatch.setattr(providers, "EvidenceRetriever", FakeKnowledgeRetriever)


def _abstention(text: str) -> AssessmentReport:
    return AssessmentReport(
        extracted_requirements=CustomerUseCase(source_text=text),
        proposed_feasibility_experiment=FeasibilityPlan(),
        status="insufficient_information",
        assessment_method="evidence_abstention",
        assessment_provider=None,
    )


def test_four_synthetic_examples_use_injected_assessment(scenarios, monkeypatch) -> None:
    requests = []

    def assess(text: str, *, retriever=None) -> AssessmentReport:
        requests.append(text)
        return _abstention(text)

    monkeypatch.setattr(providers, "assess_problem", assess)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Prompts and queries"]
    assert app.title[0].value == "Spectro-AI"
    assert "Evidence-grounded" in app.caption[0].value
    for scenario in scenarios:
        app.selectbox[0].select(scenario["title"]).run()
        assert app.text_area[0].value == scenario["problem"]
        app.button[1].click().run()
        assert not app.exception
        assert [tab.label for tab in app.tabs] == [
            "Assessment",
            "Technical diagnostics",
            "Structured report",
            "Prompts and queries",
        ]
        assert app.json
        assert app.session_state["report"].candidate_techniques == []
        assert len(app.get("download_button")) == 3
    assert requests == [scenario["problem"] for scenario in scenarios]


def test_blank_input_and_stale_report_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "assess_problem", lambda text, **kwargs: _abstention(text))
    app = AppTest.from_file(str(APP)).run()
    app.button[1].click().run()
    assert app.error[0].value.startswith("Enter a customer problem")
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    assert app.json
    assert len(app.get("download_button")) == 3
    assert any(
        button.label == "Download technical diagnostics as Markdown"
        for button in app.get("download_button")
    )
    app.text_area[0].input("Changed problem.").run()
    assert not app.json
    assert not app.get("download_button")


def test_app_shows_attached_source_context_expander(monkeypatch: pytest.MonkeyPatch) -> None:
    def assess(text: str, *, retriever=None) -> AssessmentReport:
        report = _abstention(text)
        report.retrieved_evidence = [
            RetrievedEvidence(
                evidence_id="child-1",
                source_id="S01",
                document_version="version-1",
                page_or_section="p.2",
                excerpt="NIR table excerpt.",
                review_status="reviewed_for_phase2",
                source_scope="technical_reference; technical",
                context_blocks=[
                    EvidenceContextBlock(
                        block_id="block-1",
                        document_id="doc-1",
                        section_id="section-1",
                        source_id="S01",
                        filename="fixture.md",
                        document_version="version-1",
                        text="Adjacent operating condition: 25 °C.",
                        source_locations=[
                            SourceLocation(
                                source_id="S01",
                                filename="fixture.md",
                                document_version="version-1",
                                locator="L7-L7",
                                line_start=7,
                                line_end=7,
                            )
                        ],
                        review_status="reviewed_for_phase2",
                        source_scope="technical_reference; technical",
                    )
                ],
            )
        ]
        return report

    monkeypatch.setattr(providers, "assess_problem", assess)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    assert not app.exception
    assert any("Attached context" in expander.label for expander in app.expander)


def test_reranker_configuration_change_invalidates_cached_report(monkeypatch):
    monkeypatch.setenv("SPECTRO_RAG_RERANKER", "none")
    monkeypatch.setattr(providers, "assess_problem", lambda text, **kwargs: _abstention(text))
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    assert len(app.get("download_button")) == 3
    monkeypatch.setenv("SPECTRO_RAG_RERANKER", "cross_encoder")
    app.run()
    assert not app.get("download_button")
    assert any("settings changed" in item.value.lower() for item in app.info)


def test_multi_query_setting_invalidates_cached_report(monkeypatch):
    monkeypatch.setenv("SPECTRO_RAG_MULTI_QUERY", "false")
    monkeypatch.setattr(providers, "assess_problem", lambda text, **kwargs: _abstention(text))
    app = AppTest.from_file(str(APP)).run()
    assert any("Experimental multi-query retrieval: disabled" in item.value for item in app.caption)
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    assert len(app.get("download_button")) == 3
    monkeypatch.setenv("SPECTRO_RAG_MULTI_QUERY", "true")
    app.run()
    assert not app.get("download_button")
    assert any("settings changed" in item.value.lower() for item in app.info)
    assert any("Experimental multi-query retrieval: enabled" in item.value for item in app.caption)


def test_provider_error_clears_report_and_openai_warns_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "openai")
    calls = []

    def fail(text: str, *, retriever=None) -> None:
        calls.append(text)
        record_request(
            "openai",
            "test-model",
            {
                "messages": [{"role": "user", "content": text}],
                "response_format": {"type": "json_schema"},
            },
        )
        raise ValueError("Synthetic model unavailable")

    monkeypatch.setattr(providers, "assess_problem", fail)
    app = AppTest.from_file(str(APP)).run()
    assert any("sends the customer enquiry" in warning.value for warning in app.warning)
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    assert not app.exception
    assert "Synthetic model unavailable" in app.error[0].value
    assert [tab.label for tab in app.tabs] == ["Prompts and queries"]
    assert (
        app.session_state["request_trace"].requests[0]["request"]["messages"][0]["content"]
        == calls[0]
    )
    assert any(calls[0] in item.value for item in app.tabs[0].code)
    assert "report" not in app.session_state
    assert not app.get("download_button")
    assert calls == ["We measure moisture in flour."]


def test_prompts_tab_shows_request_queries_and_exact_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _abstention("We measure moisture in flour.")

    def assess(text: str, **kwargs) -> AssessmentReport:
        request = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "SYSTEM PROMPT"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "retrieved_sources_untrusted_data": [
                                {
                                    "source_id": "S01",
                                    "document_version": "v7",
                                    "page_or_section": "page 4",
                                    "source_locations": [{"locator": "p.4"}],
                                    "excerpt": "literal evidence excerpt",
                                    "context_blocks": [{"text": "attached context text"}],
                                }
                            ]
                        }
                    ),
                },
            ],
            "format": {"type": "object"},
        }
        record_request("ollama", "test-model", request)
        record_query("moisture spectroscopy", kind="retrieval")
        return report

    monkeypatch.setattr(providers, "assess_problem", assess)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()

    prompts_tab = app.tabs[3]
    assert any("S01" in item.value and "v7" in item.value for item in prompts_tab.markdown)
    assert any("literal evidence excerpt" in item.value for item in prompts_tab.markdown)
    assert any("attached context text" in item.value for item in prompts_tab.markdown)
    assert any("SYSTEM PROMPT" in item.value for item in prompts_tab.code)
    assert any("moisture spectroscopy" in item.value for item in prompts_tab.code)


def test_stale_trace_clears_when_input_or_settings_change(monkeypatch: pytest.MonkeyPatch) -> None:
    def assess(text: str, **kwargs) -> AssessmentReport:
        record_request("ollama", "test-model", {"messages": [{"role": "user", "content": text}]})
        return _abstention(text)

    monkeypatch.setattr(providers, "assess_problem", assess)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    assert app.session_state["request_trace"].requests
    app.text_area[0].input("Changed customer input.").run()
    assert "request_trace" not in app.session_state

    app.button[1].click().run()
    assert app.session_state["request_trace"].requests
    monkeypatch.setenv("OLLAMA_MODEL", "new-test-model")
    app.run()
    assert "request_trace" not in app.session_state


def test_provider_configuration_change_clears_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "assess_problem", lambda text, **kwargs: _abstention(text))
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    assert app.json
    monkeypatch.setenv("OLLAMA_MODEL", "changed-model")
    app.run()
    assert not app.json
    assert not app.get("download_button")
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "changed-openai-model")
    app.run()
    assert not app.json
    assert not app.get("download_button")
    assert any("Retrieved source excerpts" in warning.value for warning in app.warning)


def test_failed_expert_correction_preserves_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "assess_problem", lambda text, **kwargs: _abstention(text))

    def fail(use_case: CustomerUseCase, *, retriever=None) -> AssessmentReport:
        assert use_case.target_analyte_or_property == "moisture"
        raise ValueError("Synthetic correction failure")

    monkeypatch.setattr(providers, "assess_use_case", fail)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    app.text_input(key="review_target_analyte_or_property").input("moisture").run()
    app.button[2].click().run()
    assert not app.exception
    assert any("Synthetic correction failure" in error.value for error in app.error)
    assert not app.json
    assert not app.get("download_button")


def test_json_export_and_diagnostics_are_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _abstention("We measure moisture in flour.")
    report.request_budget_diagnostics = [
        RequestBudgetDiagnostic(
            stage="evidence assessment before fitting",
            context_limit=16384,
            output_reserve=4096,
            safety_margin=512,
            component_bytes={"messages": 1200},
            component_estimated_tokens={"messages": 400},
            fit_decision="fits",
        )
    ]
    report.candidate_techniques = [
        TechniqueCandidate(
            spectroscopy_method="NIR",
            evidence_claims=[
                EvidenceClaim(
                    claim="The source describes powder measurement.",
                    evidence_id="source-1-page-3",
                    quote="A literal source excerpt.",
                )
            ],
        )
    ]
    if hasattr(report, "validation_diagnostics"):
        report.validation_diagnostics = [
            ValidationDiagnostic(
                level="not_checked",
                check="scientific_suitability",
                message="Expert review required.",
            )
        ]
    monkeypatch.setattr(providers, "assess_problem", lambda text, **kwargs: report)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    assert not app.exception
    assert len(app.get("download_button")) == 3
    diagnostics_download = next(
        button
        for button in app.get("download_button")
        if button.label == "Download technical diagnostics as Markdown"
    )
    assert diagnostics_download.proto.url
    diagnostics_markdown = render_markdown(report, include_diagnostics=True)
    assert "utf8-bytes-div3-v1" in diagnostics_markdown
    assert "messages 1200 bytes / 400 estimated tokens" in diagnostics_markdown
    assert [tab.label for tab in app.tabs] == [
        "Assessment",
        "Technical diagnostics",
        "Structured report",
        "Prompts and queries",
    ]
    customer_text = "\n".join(item.value for item in app.tabs[0].markdown)
    assert "scientific_suitability" not in customer_text
    assert "source-1-page-3" not in customer_text
    assert any("A literal source excerpt." in item.value for item in app.markdown)
    structured = json.loads(app.json[0].value)
    assert structured["validation_diagnostics"][0]["check"] == "scientific_suitability"
    if hasattr(report, "validation_diagnostics"):
        assert any("scientific_suitability" in item.value for item in app.info)


def test_knowledge_update_is_explicit_and_document_qa_is_absent() -> None:
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any(button.label == "Update Knowledge Base" for button in app.button)
    assert "Document Q&A" not in [header.value for header in app.header]
    assert "Ask documents" not in [button.label for button in app.button]
    assert FakeKnowledgeRetriever.updates == 0
    app.button[0].click().run()
    assert not app.exception
    assert FakeKnowledgeRetriever.updates == 1
    assert any("New embeddings: 1" in item.value for item in app.success)


def test_knowledge_update_failure_and_no_automatic_repeat() -> None:
    FakeKnowledgeRetriever.update_error = "Synthetic indexing failure"
    app = AppTest.from_file(str(APP)).run()
    assert FakeKnowledgeRetriever.updates == 0
    assert not app.exception
    app.button[0].click().run()
    assert any("Synthetic indexing failure" in item.value for item in app.error)
    assert FakeKnowledgeRetriever.updates == 1
    app.run()
    assert FakeKnowledgeRetriever.updates == 1


def test_report_invalidated_when_index_revision_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _abstention("We measure moisture in flour.")
    report.knowledge_index_revision = "revision-1"

    def assess(text: str, **kwargs) -> AssessmentReport:
        record_request("ollama", "test-model", {"messages": [{"role": "user", "content": text}]})
        return report

    monkeypatch.setattr(providers, "assess_problem", assess)
    app = AppTest.from_file(str(APP)).run()
    app.text_area[0].input("We measure moisture in flour.").run()
    app.button[1].click().run()
    assert app.json
    FakeKnowledgeRetriever.revision = "revision-2"
    app.run()
    assert not app.json
    assert not app.get("download_button")
    assert "request_trace" not in app.session_state
    assert any("Knowledge base changed" in item.value for item in app.info)
