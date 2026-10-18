import pytest

from spectro_ai.guardrails import abstention_report
from spectro_ai.models import CustomerUseCase
from spectro_ai.providers import ProviderError, assess_problem
from spectro_ai.reporting import render_markdown


@pytest.mark.parametrize("provider", ["baseline", "other"])
def test_removed_provider_names_fail_with_migration_message(monkeypatch, provider):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", provider)
    with pytest.raises(ProviderError, match="baseline provider was removed|Unsupported provider"):
        assess_problem("an enquiry")


def test_openai_requires_explicit_model_and_credentials(monkeypatch):
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_MODEL and OPENAI_API_KEY"):
        assess_problem("an enquiry")


def test_markdown_renders_neutral_abstention_and_validation_diagnostics() -> None:
    source = "We measure moisture in flour.\nConstraints: avoid | <script> tags\n`````\n# input"
    report = abstention_report(
        CustomerUseCase(source_text=source),
        "No source evidence was retrieved; assessment abstained.",
    )
    markdown = render_markdown(report)
    assert source in markdown
    assert "## Extracted requirements" in markdown
    assert "## Missing information" in markdown
    assert "## Follow-up questions" in markdown
    assert "## Proposed feasibility experiment" in markdown
    assert "No source evidence was retrieved" in markdown
