import socket

import pytest

from spectro_ai.providers import ProviderError, assess_problem
from spectro_ai.reporting import render_markdown


def test_baseline_requires_no_credentials_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Baseline attempted network access")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    assert assess_problem("We measure moisture in flour.").candidate_techniques


@pytest.mark.parametrize("configured", [False, True])
def test_openai_is_explicitly_unimplemented_without_network_or_secret_leaks(
    monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "openai")
    if configured:
        monkeypatch.setenv("OPENAI_API_KEY", "secret-test-value")
        monkeypatch.setenv("OPENAI_MODEL", "placeholder-model")
    with pytest.raises(ProviderError, match="not implemented") as error:
        assess_problem("We measure moisture in flour.")
    assert "No request was sent" in str(error.value)
    assert "secret-test-value" not in str(error.value)


def test_unknown_provider_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "unrecognized")
    with pytest.raises(ProviderError, match="Unsupported provider"):
        assess_problem("anything")


def test_markdown_contains_full_report_and_safe_original_input() -> None:
    source = "We measure moisture in flour.\nConstraints: avoid | <script> tags\n`````\n# input"
    report = assess_problem(source)
    markdown = render_markdown(report)
    assert source in markdown
    assert "``````text" in markdown
    assert "## Extracted requirements" in markdown
    assert "## Missing information" in markdown
    assert "## Follow-up questions" in markdown
    assert "## Proposed feasibility experiment" in markdown
    assert "## Assumptions and uncertainties" in markdown
    assert "[verified_source]" in markdown
    assert "[engineering_heuristic]" in markdown
    assert "https://assets.thermofisher.com/" in markdown
    assert "avoid \\| &lt;script&gt; tags" in markdown
    assert markdown.endswith("\n")
