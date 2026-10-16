import json
from pathlib import Path

import pytest


@pytest.fixture
def scenarios() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "examples" / "scenarios.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def offline_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SPECTRO_AI_PROVIDER", "OPENAI_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
