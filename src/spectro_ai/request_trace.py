"""Session-local snapshots of LLM requests and retrieval queries."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestTrace:
    requests: list[dict[str, Any]] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)


_ACTIVE: ContextVar[RequestTrace | None] = ContextVar("spectro_request_trace", default=None)
_STAGE: ContextVar[tuple[str, int]] = ContextVar("spectro_request_stage", default=("assessment", 1))


@contextmanager
def capture_requests(trace: RequestTrace):
    token = _ACTIVE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE.reset(token)


@contextmanager
def request_stage(stage: str, attempt: int = 1):
    token = _STAGE.set((stage, attempt))
    try:
        yield
    finally:
        _STAGE.reset(token)


def record_request(provider: str, model: str, request: dict[str, Any]) -> None:
    trace = _ACTIVE.get()
    if trace is None:
        return
    stage, attempt = _STAGE.get()
    attempt = max(
        attempt,
        1 + sum(item["stage"] == stage for item in trace.requests),
    )
    trace.requests.append(
        {
            "stage": stage,
            "attempt": attempt,
            "provider": provider,
            "model": model,
            "request": deepcopy(request),
        }
    )


def record_query(query: str, *, kind: str = "retrieval") -> None:
    trace = _ACTIVE.get()
    if trace is not None:
        trace.queries.append({"kind": kind, "query": query})
