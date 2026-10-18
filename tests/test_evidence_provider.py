import json

import pytest

from spectro_ai.models import (
    CustomerUseCase,
    EvidenceContextBlock,
    RetrievedEvidence,
    SourceLocation,
)
from spectro_ai.providers import ExtractionSettings, OllamaEvidenceAssessor, ProviderError
from spectro_ai.request_trace import RequestTrace, capture_requests


def _case() -> CustomerUseCase:
    return CustomerUseCase(
        measurement_objective="measure protein",
        target_analyte_or_property="protein",
        sample_matrix="milk powder",
        source_text="measure protein in milk powder",
    )


def _evidence() -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="source-1",
        source_id="S01",
        document_version="v1",
        page_or_section="p. 1",
        excerpt="NIR measured protein in milk powder.",
        review_status="unreviewed",
        source_scope="technical",
        context_blocks=[
            EvidenceContextBlock(
                block_id="block-2",
                filename="source.pdf",
                text="Measurements were exploratory; validation was not reported.",
                source_id="S01",
                document_version="v1",
                section_id="section-1",
                document_id="document-1",
                source_locations=[
                    SourceLocation(
                        source_id="S01",
                        filename="source.pdf",
                        document_version="v1",
                        locator="p. 1",
                        page_number=1,
                    )
                ],
                review_status="unreviewed",
            )
        ],
        context_omissions=["parent expansion disabled"],
    )


def _settings(**kwargs) -> ExtractionSettings:
    provider = kwargs.pop("provider", "ollama")
    return ExtractionSettings(
        provider=provider,
        model="test-model",
        context=kwargs.pop("context", 16384),
        output_tokens=kwargs.pop("output_tokens", 4096),
        **kwargs,
    )


def test_evidence_request_preserves_context_review_and_omissions() -> None:
    messages = OllamaEvidenceAssessor._messages(_case(), [_evidence()])
    request = json.loads(messages[1]["content"])
    bundle = request["evidence_bundles"][0]
    assert bundle["context_availability"] == "attached_context"
    assert bundle["context_blocks"][0]["text"].startswith("Measurements were exploratory")
    assert bundle["context_omissions"] == ["parent expansion disabled"]
    assert bundle["review_status"] == "unreviewed"
    assert "unknown" in messages[0]["content"].casefold()


def test_provider_schema_restricts_references_to_supplied_ids_and_fields() -> None:
    schema = OllamaEvidenceAssessor._schema([_evidence()])
    defs = schema["$defs"]
    ref = defs["EvidenceReference"]["properties"]
    assert ref["evidence_id"]["enum"] == ["source-1"]
    assert ref["context_block_id"]["anyOf"][0]["enum"] == ["block-2"]
    assert defs["EvidenceConditionComparison"]["properties"]["customer_fields"]["items"] == {
        "$ref": "#/$defs/CustomerFactField"
    }
    assert defs["CustomerFactField"]["enum"]
    assert "context_availability" not in defs["EvidenceObservation"]["properties"]
    assert "sample_matrix" in defs["CustomerFactField"]["enum"]


def test_fit_evidence_drops_whole_bundles_without_stripping_context() -> None:
    evidence = _evidence()
    assessor = OllamaEvidenceAssessor(_settings())
    retained, dropped = assessor.fit_evidence(_case(), [evidence])
    assert retained == [evidence]
    assert dropped == []


def test_fit_drops_low_ranked_full_bundle_atomically() -> None:
    first = _evidence()
    second = _evidence().model_copy(
        update={
            "evidence_id": "source-2",
            "source_id": "S02",
            "excerpt": "A different source excerpt.",
        }
    )
    probe = OllamaEvidenceAssessor(_settings())
    sizes = [
        len(
            json.dumps(
                probe._messages(_case(), items), ensure_ascii=False, separators=(",", ":")
            ).encode()
        )
        + len(json.dumps(probe._schema(items), separators=(",", ":")).encode())
        + probe.settings.output_tokens
        + 256
        for items in ([first], [first, second])
    ]
    assessor = OllamaEvidenceAssessor(_settings(context=max(sizes[0], (sizes[0] + sizes[1]) // 2)))
    retained, dropped = assessor.fit_evidence(_case(), [first, second])
    assert retained == [first]
    assert dropped == ["source-2"]
    assert retained[0].context_blocks == first.context_blocks


def test_request_snapshot_excludes_budget_dropped_bundle(monkeypatch) -> None:
    first = _evidence()
    second = _evidence().model_copy(
        update={"evidence_id": "source-2", "source_id": "S02", "excerpt": "large " * 100}
    )
    probe = OllamaEvidenceAssessor(_settings())
    sizes = [
        len(json.dumps(probe._messages(_case(), items), ensure_ascii=False).encode())
        + len(json.dumps(probe._schema(items)).encode())
        + probe.settings.output_tokens
        + 256
        for items in ([first], [first, second])
    ]
    assessor = OllamaEvidenceAssessor(_settings(context=max(sizes[0], (sizes[0] + sizes[1]) // 2)))
    retained, dropped = assessor.fit_evidence(_case(), [first, second])
    assert dropped == ["source-2"]

    monkeypatch.setattr(
        "spectro_ai.providers.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    trace = RequestTrace()
    with capture_requests(trace), pytest.raises(ProviderError, match="Ollama request failed"):
        assessor(_case(), retained)

    bundle_json = trace.requests[0]["request"]["messages"][1]["content"]
    assert json.loads(bundle_json)["evidence_bundles"] == [
        json.loads(assessor._messages(_case(), [first])[1]["content"])["evidence_bundles"][0]
    ]


def test_structural_retry_is_bounded_and_records_calls(monkeypatch) -> None:
    calls = []
    outputs = [
        "not json",
        '{"observations":[],"support_assessments":[],"conflicts":[],"coverage":[]}',
    ]

    def request(*args, **kwargs):
        calls.append(kwargs["payload"] if "payload" in kwargs else args[2])
        return {"done_reason": "stop", "message": {"content": outputs.pop(0)}}

    monkeypatch.setattr("spectro_ai.providers.ollama_request", request)
    result = OllamaEvidenceAssessor(_settings())(_case(), [_evidence()])
    assert len(calls) == 2
    assert result.diagnostics.call_count == 2
    assert result.coverage[0].evidence_id == "source-1"
    assert result.diagnostics.status == "partial"


def test_provider_failure_is_not_retried_as_structural_error(monkeypatch) -> None:
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ProviderError("model timed out")

    monkeypatch.setattr("spectro_ai.providers.ollama_request", fail)
    with pytest.raises(ProviderError, match="model timed out"):
        OllamaEvidenceAssessor(_settings())(_case(), [_evidence()])
    assert calls == 1


def test_invalid_output_retries_once_then_fails(monkeypatch) -> None:
    calls = 0

    def invalid(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"done_reason": "stop", "message": {"content": "not json"}}

    monkeypatch.setattr("spectro_ai.providers.ollama_request", invalid)
    with pytest.raises(ProviderError, match="invalid structured output"):
        OllamaEvidenceAssessor(_settings())(_case(), [_evidence()])
    assert calls == 2


def test_truncation_is_not_retried(monkeypatch) -> None:
    calls = 0

    def truncated(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"done_reason": "length", "message": {"content": "{}"}}

    monkeypatch.setattr("spectro_ai.providers.ollama_request", truncated)
    with pytest.raises(ProviderError, match="output-token limit"):
        OllamaEvidenceAssessor(_settings())(_case(), [_evidence()])
    assert calls == 1


def test_claim_repair_can_disable_second_structural_retry(monkeypatch) -> None:
    calls = 0

    def invalid(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"done_reason": "stop", "message": {"content": "not json"}}

    monkeypatch.setattr("spectro_ai.providers.ollama_request", invalid)
    with pytest.raises(ProviderError, match="invalid structured output"):
        OllamaEvidenceAssessor(_settings())(_case(), [_evidence()], allow_retry=False)
    assert calls == 1


def test_oversize_evidence_fails_before_provider_call(monkeypatch) -> None:
    calls = 0

    def should_not_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr("spectro_ai.providers.ollama_request", should_not_call)
    with pytest.raises(ProviderError, match="context budget"):
        OllamaEvidenceAssessor(_settings(context=4096))(_case(), [_evidence()])
    assert calls == 0


def test_openai_uses_same_schema_path_without_ollama(monkeypatch) -> None:
    calls = []

    def openai(settings, messages, schema, *, usage=None, temperature=None):
        calls.append((settings.provider, schema, temperature))
        usage.update(prompt_tokens=12, completion_tokens=3, call_count=1)
        return '{"observations":[],"support_assessments":[],"conflicts":[],"coverage":[]}'

    def ollama(*args, **kwargs):
        raise AssertionError("OpenAI path called Ollama")

    monkeypatch.setattr("spectro_ai.providers._openai_json", openai)
    monkeypatch.setattr("spectro_ai.providers.ollama_request", ollama)
    assessor = OllamaEvidenceAssessor(_settings(provider="openai"))
    result = assessor(_case(), [_evidence()])
    assert calls[0][0] == "openai"
    assert calls[0][2] == 0
    assert result.diagnostics.prompt_tokens == 12
    assert result.diagnostics.completion_tokens == 3
