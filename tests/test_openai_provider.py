import json
import sys
from types import SimpleNamespace

import pytest

from spectro_ai.models import CustomerUseCase, RetrievedEvidence
from spectro_ai.providers import (
    ExtractionSettings,
    OllamaTechnicalAssessor,
    OpenAIUseCaseExtractor,
    ProviderError,
)


class FakeOpenAI:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[reply])


def install_client(monkeypatch, client):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_kwargs: client))


def choice(content, *, finish="stop", refusal=None):
    return SimpleNamespace(
        finish_reason=finish,
        message=SimpleNamespace(content=content, refusal=refusal),
    )


def test_openai_extraction_uses_strict_schema_and_shared_quote_validation(monkeypatch):
    payload = CustomerUseCase(
        measurement_objective="measure moisture",
        target_analyte_or_property="moisture",
        sample_matrix="flour",
        temperature="100 C",
        field_quotes={
            "measurement_objective": "measure moisture",
            "target_analyte_or_property": "moisture",
            "sample_matrix": "flour",
            "temperature": "100 C",
        },
    ).model_dump_json()
    client = FakeOpenAI([choice(payload)])
    install_client(monkeypatch, client)
    case = OpenAIUseCaseExtractor(ExtractionSettings(provider="openai", model="gpt-test")).extract(
        "We measure moisture in flour at 100 C."
    )
    assert case.temperature == "100 C"
    assert case.extraction_mode == "openai"
    schema = client.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False


def test_openai_extraction_repairs_malformed_json_once(monkeypatch):
    payload = CustomerUseCase(
        sample_matrix="flour", field_quotes={"sample_matrix": "flour"}
    ).model_dump_json()
    client = FakeOpenAI([choice("not-json"), choice(payload)])
    install_client(monkeypatch, client)
    case = OpenAIUseCaseExtractor(ExtractionSettings(provider="openai", model="gpt-test")).extract(
        "flour"
    )
    assert case.sample_matrix == "flour"
    assert len(client.calls) == 2


def test_openai_extraction_does_not_retry_past_context_budget(monkeypatch):
    client = FakeOpenAI([choice("x" * 20000), choice("{}")])
    install_client(monkeypatch, client)
    extractor = OpenAIUseCaseExtractor(
        ExtractionSettings(provider="openai", model="gpt-test", context=16384)
    )
    with pytest.raises(ProviderError, match="retry exceeds.*context budget"):
        extractor.extract("flour")
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "reply, message",
    [
        (choice("", refusal="policy"), "refused"),
        (choice("{}", finish="length"), "incomplete"),
        (RuntimeError("timeout"), "request failed"),
    ],
)
def test_openai_refusal_timeout_and_context_errors_are_visible(monkeypatch, reply, message):
    client = FakeOpenAI([reply])
    install_client(monkeypatch, client)
    extractor = OpenAIUseCaseExtractor(ExtractionSettings(provider="openai", model="gpt-test"))
    with pytest.raises(ProviderError, match=message):
        extractor.extract("flour")
    too_small = OpenAIUseCaseExtractor(
        ExtractionSettings(provider="openai", model="gpt-test", context=4096)
    )
    with pytest.raises(ProviderError, match="context budget"):
        too_small.extract("x" * 10000)


def test_openai_missing_optional_sdk_has_install_hint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ProviderError, match="uv sync --extra llm"):
        OpenAIUseCaseExtractor(ExtractionSettings(provider="openai", model="gpt-test")).extract(
            "flour"
        )


def test_openai_assessment_uses_shared_domain_output_and_metadata(monkeypatch):
    output = {
        "status": "insufficient_information",
        "suggested_technical_questions": [],
        "candidate_techniques": [],
        "recommendation_rationale": None,
        "proposed_feasibility_experiment": {},
        "assumptions_and_uncertainties": [],
    }
    client = FakeOpenAI([choice(json.dumps(output))])
    install_client(monkeypatch, client)
    use_case = CustomerUseCase(
        measurement_objective="measure protein",
        target_analyte_or_property="protein",
        sample_matrix="milk",
    )
    evidence = RetrievedEvidence(
        evidence_id="E1", source_id="S1", excerpt="protein milk", review_status="reviewed"
    )
    report = OllamaTechnicalAssessor(
        ExtractionSettings(provider="openai", model="gpt-test")
    ).assess(use_case, [evidence])
    assert report.assessment_provider == "openai"
    assert report.assessment_method == "openai_grounded"
    assert (
        "extracted_requirements"
        not in client.calls[0]["response_format"]["json_schema"]["schema"]["properties"]
    )
