from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError

import pytest

from spectro_ai.providers import ExtractionSettings, ollama_request
from spectro_ai.request_trace import (
    RequestTrace,
    capture_budget_diagnostics,
    capture_requests,
    record_budget_diagnostic,
    record_request,
    request_stage,
)


def test_request_snapshots_are_deep_copied_and_scoped():
    trace = RequestTrace()
    request = {"model": "local", "messages": [{"role": "user", "content": "case"}]}

    with capture_requests(trace):
        with request_stage("evidence assessment"):
            record_request("ollama", "local", request)
        request["messages"][0]["content"] = "mutated"
        record_request("ollama", "local", request)

    record_request("ollama", "local", request)

    assert [item["attempt"] for item in trace.requests] == [1, 1]
    assert [item["stage"] for item in trace.requests] == [
        "evidence assessment",
        "assessment",
    ]
    assert trace.requests[0]["request"]["messages"][0]["content"] == "case"
    assert len(trace.requests) == 2


def test_ollama_snapshot_matches_dispatched_payload(monkeypatch):
    import json
    from io import BytesIO

    from spectro_ai import providers

    payload = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "case"}],
        "format": {"type": "object"},
        "options": {"temperature": 0},
    }
    seen = {}

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        seen["payload"] = json.loads(request.data)
        seen["headers"] = dict(request.header_items())
        return Response(b'{"message":{"content":"{}"}}')

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    trace = RequestTrace()
    settings = ExtractionSettings(model="local-model")
    with capture_requests(trace), request_stage("extraction"):
        ollama_request(settings, "/api/chat", payload)

    assert seen["payload"] == payload
    assert "Authorization" not in seen["headers"]
    assert trace.requests[0]["request"] == payload
    assert trace.requests[0]["stage"] == "extraction"


def test_transport_failure_keeps_attempt_snapshot(monkeypatch):
    from spectro_ai import providers

    monkeypatch.setattr(
        providers, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline"))
    )
    trace = RequestTrace()
    payload = {"model": "local-model", "messages": [{"role": "user", "content": "case"}]}
    with capture_requests(trace), pytest.raises(ValueError, match="Ollama request failed"):
        ollama_request(ExtractionSettings(model="local-model"), "/api/chat", payload)

    assert len(trace.requests) == 1
    assert trace.requests[0]["request"] == payload


def test_context_traces_are_isolated_between_threads():
    def capture(content: str) -> RequestTrace:
        trace = RequestTrace()
        with capture_requests(trace):
            record_request("ollama", "local", {"messages": [content]})
        return trace

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(capture, ["first", "second"])

    assert first.requests[0]["request"]["messages"] == ["first"]
    assert second.requests[0]["request"]["messages"] == ["second"]


def test_embedding_usage_does_not_overwrite_chat_budget_usage(monkeypatch):
    from io import BytesIO

    from spectro_ai import providers

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/api/embed"):
            return Response(b'{"prompt_eval_count":999}')
        return Response(b'{"prompt_eval_count":42,"eval_count":7,"message":{"content":"ok"}}')

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    diagnostic = {
        "stage": "retrieval query generation",
        "provider_prompt_tokens": None,
        "provider_completion_tokens": None,
    }
    with capture_budget_diagnostics() as captured:
        record_budget_diagnostic(diagnostic)
        ollama_request(ExtractionSettings(), "/api/embed", {"model": "embed", "input": "q"})
        assert captured[0]["provider_prompt_tokens"] is None
        ollama_request(
            ExtractionSettings(),
            "/api/chat",
            {"model": "chat", "messages": [{"role": "user", "content": "q"}]},
        )
        assert captured[0]["provider_prompt_tokens"] == 42
        assert captured[0]["provider_completion_tokens"] == 7
