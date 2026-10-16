"""Explicit provider selection; the baseline never imports an LLM SDK."""

import os

from spectro_ai.baseline import assess, extract_use_case
from spectro_ai.models import AssessmentReport


class ProviderError(ValueError):
    """A provider is unknown, incomplete, or deliberately unavailable."""


def assess_problem(text: str) -> AssessmentReport:
    provider = os.getenv("SPECTRO_AI_PROVIDER", "baseline").strip().lower()
    if provider == "baseline":
        return assess(extract_use_case(text))
    if provider == "openai":
        configured = bool(os.getenv("OPENAI_MODEL")) and bool(os.getenv("OPENAI_API_KEY"))
        detail = "Configuration present." if configured else "OPENAI_MODEL/API key not configured."
        raise ProviderError(
            f"OpenAI adapter is not implemented. {detail} "
            "No request was sent. Set SPECTRO_AI_PROVIDER=baseline for the offline demo."
        )
    raise ProviderError(f"Unsupported provider {provider!r}. Use SPECTRO_AI_PROVIDER=baseline.")
