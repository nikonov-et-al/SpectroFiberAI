import json
from math import ceil

import pytest

from spectro_ai.models import (
    CustomerUseCase,
    EvidenceAssessment,
    EvidenceContextBlock,
    RetrievedEvidence,
    SourceLocation,
)
from spectro_ai.providers import (
    ExtractionSettings,
    OllamaEvidenceAssessor,
    OllamaTechnicalAssessor,
    ProviderError,
    _budget_diagnostic,
    _estimate_tokens,
)
from spectro_ai.request_trace import (
    RequestTrace,
    capture_budget_diagnostics,
    capture_requests,
    request_stage,
)


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


def test_inferred_industry_is_excluded_from_assessment_customer_facts() -> None:
    case = _case().model_copy(
        update={
            "industry": None,
            "inferred_customer_context": [
                {
                    "field": "industry",
                    "value": "pharmaceutical",
                    "supporting_quote": "active pharmaceutical ingredient",
                    "rationale": "Unverified inference.",
                }
            ],
        }
    )
    evidence = _evidence()
    evidence_messages = OllamaEvidenceAssessor._messages(case, [evidence])
    technical_messages = OllamaTechnicalAssessor._messages(
        case,
        [evidence],
        evidence_assessment=EvidenceAssessment(input_fingerprint="f", assessor_version="v"),
    )
    for messages in (evidence_messages, technical_messages):
        payload = json.loads(messages[1]["content"])
        customer = payload.get("customer", payload.get("customer_case_untrusted_data", {}))
        facts = customer.get("validated_customer_facts", {})
        assert "industry" not in facts
        assert "inferred_customer_context" not in customer


def test_fit_evidence_drops_whole_bundles_without_stripping_context() -> None:
    evidence = _evidence()
    assessor = OllamaEvidenceAssessor(_settings())
    retained, dropped = assessor.fit_evidence(_case(), [evidence])
    assert retained == [evidence]
    assert dropped == []


def test_default_budget_keeps_one_1200_byte_excerpt_and_assesses_eight_bundles(
    monkeypatch,
) -> None:
    assessor = OllamaEvidenceAssessor(_settings())
    single = _evidence().model_copy(update={"excerpt": "x" * 1200, "context_blocks": []})
    retained, dropped = assessor.fit_evidence(_case(), [single])
    assert retained == [single]
    assert dropped == []
    bundles = [
        _evidence().model_copy(
            update={
                "evidence_id": f"source-{i}",
                "source_id": f"S{i:02}",
                "excerpt": f"Method condition finding {i}. " + "x" * 1120,
                "context_blocks": [
                    _evidence()
                    .context_blocks[0]
                    .model_copy(
                        update={
                            "block_id": f"qualification-{i}",
                            "text": "Exploratory result; validation not reported.",
                        }
                    )
                ]
                if i == 0
                else [],
            }
        )
        for i in range(8)
    ]
    retained, dropped = assessor.fit_evidence(_case(), bundles)
    assert retained == bundles
    assert dropped == []

    dispatched = []

    def request(*args, **kwargs):
        dispatched.append(kwargs["payload"] if "payload" in kwargs else args[2])
        return {
            "done_reason": "stop",
            "message": {
                "content": json.dumps(
                    {
                        "observations": [],
                        "support_assessments": [],
                        "conflicts": [],
                        "coverage": [],
                    }
                )
            },
        }

    monkeypatch.setattr("spectro_ai.providers.ollama_request", request)
    assessment = assessor(_case(), retained)
    assert len(dispatched) == 1
    sent_bundles = json.loads(dispatched[0]["messages"][1]["content"])["evidence_bundles"]
    assert [bundle["evidence_id"] for bundle in sent_bundles] == [
        item.evidence_id for item in bundles
    ]
    assert sent_bundles[0]["context_blocks"][0]["text"] == (
        "Exploratory result; validation not reported."
    )
    assert {item.evidence_id for item in assessment.coverage} == {
        item.evidence_id for item in bundles
    }


def test_estimator_uses_unicode_utf8_bytes_and_output_margin_boundary() -> None:
    settings = _settings(output_tokens=128)
    messages = [{"role": "user", "content": "é"}]
    schema = {"type": "string"}
    probe, _ = _budget_diagnostic("boundary", messages, schema, settings)
    assert probe["component_bytes"]["messages"] == len(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    evidence = _evidence()
    messages = OllamaEvidenceAssessor._messages(_case(), [evidence])
    schema = OllamaEvidenceAssessor._schema([evidence])
    sizes, fits = _budget_diagnostic("boundary", messages, schema, settings)
    needed = (
        sum(sizes["component_estimated_tokens"].values())
        + settings.output_tokens
        + sizes["safety_margin"]
    )
    assert fits
    accepted, dropped = OllamaEvidenceAssessor(
        _settings(context=needed, output_tokens=128)
    ).fit_evidence(_case(), [evidence])
    assert accepted == [evidence] and dropped == []
    accepted, dropped = OllamaEvidenceAssessor(
        _settings(context=needed - 1, output_tokens=128)
    ).fit_evidence(_case(), [evidence])
    assert accepted == [] and dropped == [evidence.evidence_id]


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
    sizes = []
    for items in ([first], [first, second]):
        messages = probe._messages(_case(), items)
        schema = probe._schema(items)
        input_tokens = _estimate_tokens(messages)[1] + _estimate_tokens(schema)[1]
        sizes.append(
            input_tokens + probe.settings.output_tokens + max(512, ceil(input_tokens * 0.15))
        )
    assessor = OllamaEvidenceAssessor(_settings(context=sum(sizes) // 2))
    with capture_budget_diagnostics() as diagnostics:
        retained, dropped = assessor.fit_evidence(_case(), [first, second])
    assert retained == [first]
    assert dropped == ["source-2"]
    assert retained[0].context_blocks == first.context_blocks
    before = next(item for item in diagnostics if item["stage"].endswith("before fitting"))
    after = next(item for item in reversed(diagnostics) if item["stage"].endswith("after fitting"))
    assert (
        before["component_bytes"]["response_schema"] > after["component_bytes"]["response_schema"]
    )


def test_request_snapshot_excludes_budget_dropped_bundle(monkeypatch) -> None:
    first = _evidence()
    second = _evidence().model_copy(
        update={"evidence_id": "source-2", "source_id": "S02", "excerpt": "large " * 100}
    )
    probe = OllamaEvidenceAssessor(_settings())
    sizes = []
    for items in ([first], [first, second]):
        messages = probe._messages(_case(), items)
        schema = probe._schema(items)
        input_tokens = _estimate_tokens(messages)[1] + _estimate_tokens(schema)[1]
        sizes.append(
            input_tokens + probe.settings.output_tokens + max(512, ceil(input_tokens * 0.15))
        )
    assessor = OllamaEvidenceAssessor(_settings(context=sum(sizes) // 2))
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
    assessor = OllamaEvidenceAssessor(_settings())
    with capture_budget_diagnostics() as diagnostics:
        with request_stage("evidence follow-up"):
            result = assessor(_case(), [_evidence()])
    assert len(calls) == 2
    assert result.diagnostics.call_count == 2
    assert result.coverage[0].evidence_id == "source-1"
    assert result.diagnostics.status == "partial"
    assert [item["stage"] for item in diagnostics] == [
        "evidence follow-up",
        "evidence follow-up",
    ]
    assert [item["attempt"] for item in diagnostics] == [1, 2]
    assert (
        diagnostics[1]["component_bytes"]["messages"]
        > diagnostics[0]["component_bytes"]["messages"]
    )


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
