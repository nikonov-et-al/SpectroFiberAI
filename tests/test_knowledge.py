from pathlib import Path

import pytest
import yaml

from spectro_ai.baseline import assess
from spectro_ai.knowledge import KnowledgeSource, YamlKnowledgeSource, default_knowledge_path
from spectro_ai.models import CustomerUseCase


def test_starter_knowledge_covers_all_five_techniques() -> None:
    source: KnowledgeSource = YamlKnowledgeSource()
    techniques = source.get_techniques()
    assert {t.id for t in techniques} == {"raman", "nir", "mir", "uv_vis", "fluorescence"}
    for technique in techniques:
        assert technique.expert_review_required
        assert technique.possible_geometries
        assert any(e.source_url and e.checked_on for e in technique.supporting_evidence)


@pytest.mark.parametrize(
    "content",
    ["techniques: [", "{}", "schema_version: 2\ntechniques: []", "!!python/object:bad {}"],
)
def test_malformed_knowledge_gives_useful_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load spectroscopy knowledge"):
        YamlKnowledgeSource(path).get_techniques()


def test_missing_file_and_duplicate_ids_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot load"):
        YamlKnowledgeSource(tmp_path / "absent.yaml").get_techniques()
    raw = yaml.safe_load(default_knowledge_path().read_text(encoding="utf-8"))
    raw["techniques"].append(raw["techniques"][0])
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Technique IDs must be unique"):
        YamlKnowledgeSource(duplicate).get_techniques()


def test_custom_source_boundary_and_absent_rule_knowledge() -> None:
    class EmptySource:
        def get_techniques(self) -> list:
            return []

    case = CustomerUseCase(
        measurement_objective="measure moisture",
        target_analyte_or_property="moisture",
        sample_matrix="flour",
    )
    with pytest.raises(ValueError, match="rule-required technique: nir"):
        assess(case, EmptySource())
