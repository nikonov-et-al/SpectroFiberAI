"""Validated local knowledge with one small boundary for future data sources."""

from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import Field, ValidationError, model_validator

from spectro_ai.models import DomainModel, TechniqueKnowledge


class KnowledgeSource(Protocol):
    def get_techniques(self) -> list[TechniqueKnowledge]: ...


class KnowledgeDocument(DomainModel):
    schema_version: Literal[1]
    techniques: list[TechniqueKnowledge] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> KnowledgeDocument:
        ids = [technique.id for technique in self.techniques]
        if len(ids) != len(set(ids)):
            raise ValueError("Technique IDs must be unique")
        return self


def default_knowledge_path() -> Path:
    # Editable checkout uses the human-editable root file; wheels carry a copy.
    checkout = Path(__file__).resolve().parents[2] / "knowledge" / "techniques.yaml"
    return checkout if checkout.is_file() else Path(__file__).parent / "data" / "techniques.yaml"


class YamlKnowledgeSource:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else default_knowledge_path()

    def get_techniques(self) -> list[TechniqueKnowledge]:
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
            return KnowledgeDocument.model_validate(raw).techniques
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise ValueError(f"Cannot load spectroscopy knowledge from {self.path}: {exc}") from exc
