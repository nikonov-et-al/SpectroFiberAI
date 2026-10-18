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
    budget_diagnostics: list[dict[str, Any]] = field(default_factory=list)


_ACTIVE: ContextVar[RequestTrace | None] = ContextVar("spectro_request_trace", default=None)
_STAGE: ContextVar[tuple[str, int]] = ContextVar("spectro_request_stage", default=("assessment", 1))
_BUDGETS: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "spectro_budget_diagnostics", default=None
)


@contextmanager
def capture_requests(trace: RequestTrace):
    token = _ACTIVE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE.reset(token)


@contextmanager
def capture_budget_diagnostics():
    token = _BUDGETS.set([])
    try:
        yield _BUDGETS.get()
    finally:
        _BUDGETS.reset(token)


@contextmanager
def request_stage(stage: str, attempt: int = 1):
    token = _STAGE.set((stage, attempt))
    try:
        yield
    finally:
        _STAGE.reset(token)


def current_request_stage() -> tuple[str, int]:
    return _STAGE.get()


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


def record_budget_diagnostic(diagnostic: dict[str, Any]) -> None:
    trace = _ACTIVE.get()
    if trace is not None:
        trace.budget_diagnostics.append(deepcopy(diagnostic))
    budgets = _BUDGETS.get()
    if budgets is not None:
        budgets.append(deepcopy(diagnostic))


def record_provider_usage(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    if prompt_tokens is None and completion_tokens is None:
        return
    trace = _ACTIVE.get()
    budgets = _BUDGETS.get()
    for items in (trace.budget_diagnostics if trace else None, budgets):
        if items:
            for key, value in (
                ("provider_prompt_tokens", prompt_tokens),
                ("provider_completion_tokens", completion_tokens),
            ):
                if value is not None:
                    items[-1][key] = value


def current_budget_diagnostics() -> list[dict[str, Any]]:
    budgets = _BUDGETS.get()
    if budgets is not None:
        return deepcopy(budgets)
    trace = _ACTIVE.get()
    return deepcopy(trace.budget_diagnostics) if trace is not None else []
