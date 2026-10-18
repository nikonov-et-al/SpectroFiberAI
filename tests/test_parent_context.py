from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectro_ai.evidence import (
    EvidenceRetriever,
    canonical_context_block,
    evidence_context_block_bytes,
    evidence_context_blocks_bytes,
)
from spectro_ai.models import RetrievedEvidence, SourceBlock, SourceLocation, SourceSection
from spectro_ai.providers import RagSettings


def _fixture_index(
    texts: list[str],
    child_positions: list[int],
    *,
    hierarchy_valid: bool = True,
    table: bool = False,
) -> tuple[dict, list[tuple[float, RetrievedEvidence]]]:
    source_id, filename, version, document_id, section_id = "S01", "source.md", "v1", "DOC", "SEC"
    blocks = []
    locations = []
    block_ids = [f"B{number:02}" for number in range(len(texts))]
    for ordinal, text in enumerate(texts):
        location = SourceLocation(
            source_id=source_id,
            filename=filename,
            document_version=version,
            locator=f"lines {ordinal + 1}-{ordinal + 1}",
            line_start=ordinal + 1,
            line_end=ordinal + 1,
        )
        locations.append(location)
        is_caption = table and ordinal == 0
        is_header = table and ordinal == 1
        table_row = table and ordinal in {1, 2, 3}
        blocks.append(
            SourceBlock(
                block_id=block_ids[ordinal],
                document_id=document_id,
                section_id=section_id,
                source_id=source_id,
                filename=filename,
                document_version=version,
                ordinal=ordinal,
                text=text,
                location=location,
                locations=[location],
                block_type=(
                    "table_caption" if is_caption else ("table_row" if table_row else "paragraph")
                ),
                table_id="TABLE" if table and ordinal in {0, 1, 2, 3} else None,
                table_row_index=ordinal - 1 if table_row else None,
                is_table_header=is_header,
                table_header_block_ids=[block_ids[1]] if table and table_row else [],
                review_status="reviewed_source",
                source_scope="technical_reference; technical",
            ).model_dump(mode="json")
        )
    section = SourceSection(
        section_id=section_id,
        document_id=document_id,
        source_id=source_id,
        filename=filename,
        document_version=version,
        heading_path=["Method"],
        block_ids=block_ids,
        hierarchy_valid=hierarchy_valid,
    ).model_dump(mode="json")
    provenance = []
    chunks = []
    ranked = []
    for number, position in enumerate(child_positions, 1):
        evidence_id = f"E{number}"
        location = locations[position]
        provenance_item = {
            "evidence_id": evidence_id,
            "source_id": source_id,
            "filename": filename,
            "document_version": version,
            "page_or_section": location.locator,
            "review_status": "reviewed_source",
            "source_scope": "technical_reference; technical",
            "block_id": block_ids[position],
            "block_ids": [block_ids[position]],
            "section_id": section_id,
            "source_locations": [location.model_dump(mode="json")],
        }
        provenance.append(provenance_item)
        chunks.append(
            {
                "text": texts[position],
                "content_sha256": f"hash-{evidence_id}",
                "provenance": [provenance_item],
            }
        )
        ranked.append(
            (
                float(-number),
                RetrievedEvidence(
                    evidence_id=evidence_id,
                    source_id=source_id,
                    document_version=version,
                    page_or_section=location.locator,
                    excerpt=texts[position],
                    review_status="reviewed_source",
                    source_scope="technical_reference; technical",
                    source_locations=[location],
                ),
            )
        )
    index = {
        "schema_version": 3,
        "files": [
            {
                "source_id": source_id,
                "filename": filename,
                "document_version": version,
                "document_id": document_id,
                "source_type": "technical_reference",
                "scope": "technical",
                "review_status": "needs_source_review",
            }
        ],
        "sections": [section],
        "source_blocks": blocks,
        "chunks": chunks,
    }
    return index, ranked


def _retriever(
    tmp_path: Path, *, budget: int = 10000, enabled: bool | None = True
) -> EvidenceRetriever:
    return EvidenceRetriever(
        index_path=tmp_path / "unused.sqlite3",
        manifest_path=tmp_path / "unused.yaml",
        settings=RagSettings(),
        mode="bm25",
        context_budget=budget,
        parent_expansion=enabled,
    )


def test_enabled_retrieval_attaches_full_section_to_child_without_extra_evidence(tmp_path) -> None:
    index, ranked = _fixture_index(
        ["Background detail", "Claim and evidence", "Adjacent qualification", "Limitation"],
        [1],
    )
    retriever = _retriever(tmp_path)

    selected = retriever._select_with_parent_context(ranked, 1, index)

    assert len(selected) == 1
    assert selected[0].evidence_id == "E1"
    assert [block.block_id for block in selected[0].context_blocks] == ["B00", "B02", "B03"]
    assert selected[0].context_blocks[0].source_locations[0].line_start == 1
    assert selected[0].context_blocks[0].review_status == "reviewed_source"
    assert selected[0].context_blocks[0].source_scope == "technical_reference; technical"


def test_window_grows_preceding_first_and_respects_serialized_byte_budget(tmp_path) -> None:
    texts = ["prior block", "near prior", "retrieved child", "near after", "after block"]
    index, ranked = _fixture_index(texts, [2])
    canonical = canonical_context_block(index, "B01")
    one_block_cost = evidence_context_block_bytes(canonical)
    budget = one_block_cost * 10 // 3
    retriever = _retriever(tmp_path, budget=budget)

    selected = retriever._select_with_parent_context(ranked, 1, index)
    contexts = selected[0].context_blocks

    assert [block.block_id for block in contexts] == ["B01"]
    charged = len(selected[0].excerpt.encode("utf-8")) + evidence_context_blocks_bytes(contexts)
    assert charged <= budget


def test_window_context_remains_in_source_order(tmp_path) -> None:
    texts = ["far before", "near before", "child", "near after", "far after"]
    index, ranked = _fixture_index(texts, [2])
    one_block_cost = evidence_context_block_bytes(canonical_context_block(index, "B01"))
    selected = _retriever(tmp_path, budget=one_block_cost * 7)._select_with_parent_context(
        ranked, 1, index
    )

    context_ids = [block.block_id for block in selected[0].context_blocks]

    assert context_ids[:2] == ["B01", "B03"]
    assert context_ids == sorted(context_ids)


def test_partial_child_expands_to_complete_original_source_block(tmp_path) -> None:
    source_text = (
        "The sensor reports 4.2 μm under dry nitrogen. "
        "However, the stated range excludes humid samples."
    )
    index, ranked = _fixture_index([source_text], [0])
    excerpt = "The sensor reports 4.2 μm"
    index["chunks"][0]["text"] = excerpt
    provenance = index["chunks"][0]["provenance"][0]
    provenance["block_spans"] = [{"block_id": "B00", "start": 0, "end": len(excerpt)}]
    ranked[0] = (ranked[0][0], ranked[0][1].model_copy(update={"excerpt": excerpt}))
    retriever = _retriever(tmp_path, budget=4000)

    selected = retriever._select_with_parent_context(ranked, 1, index)

    assert selected[0].excerpt == excerpt
    assert len(selected[0].context_blocks) == 1
    assert selected[0].context_blocks[0].block_id == "B00"
    assert "excludes humid samples" in selected[0].context_blocks[0].text


def test_context_serialization_charge_includes_json_envelope_and_unicode(tmp_path) -> None:
    index, ranked = _fixture_index(["Background μm", "Claim μm", "Limit β"], [1])
    retriever = _retriever(tmp_path, budget=3000)
    selected = retriever._select_with_parent_context(ranked, 1, index)
    contexts = selected[0].context_blocks
    expected = len(
        json.dumps(
            {"context_blocks": [block.model_dump(mode="json") for block in contexts]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    assert evidence_context_blocks_bytes(contexts) == expected
    assert evidence_context_block_bytes(contexts[0]) == len(
        json.dumps(
            {
                "context_blocks": [contexts[0].model_dump(mode="json")],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert len(selected[0].excerpt.encode("utf-8")) + expected <= 3000


def test_no_context_fit_is_recorded_and_unused_capacity_can_add_children(tmp_path) -> None:
    index, ranked = _fixture_index(["a" * 60, "b" * 30, "surrounding qualification"], [0, 1])
    selected = _retriever(tmp_path, budget=100)._select_with_parent_context(ranked, 2, index)

    assert [item.evidence_id for item in selected] == ["E1", "E2"]
    assert selected[0].context_blocks == []
    assert selected[0].context_omissions
    assert selected[1].context_omissions
    assert sum(len(item.excerpt.encode("utf-8")) for item in selected) == 90


def test_shared_context_blocks_are_deduplicated_and_child_count_is_unchanged(tmp_path) -> None:
    index, ranked = _fixture_index(
        ["leading detail", "first child", "shared context", "second child", "final limit"],
        [1, 3],
    )
    selected = _retriever(tmp_path)._select_with_parent_context(ranked, 2, index)
    attached_ids = [block.block_id for item in selected for block in item.context_blocks]

    assert [item.evidence_id for item in selected] == ["E1", "E2"]
    assert len(attached_ids) == len(set(attached_ids))
    assert "B01" not in attached_ids and "B03" not in attached_ids


def test_invalid_heading_hierarchy_suppresses_expansion(tmp_path) -> None:
    index, ranked = _fixture_index(
        ["Before", "Retrieved child", "After"], [1], hierarchy_valid=False
    )

    selected = _retriever(tmp_path)._select_with_parent_context(ranked, 1, index)

    assert selected[0].context_blocks == []
    assert "hierarchy is invalid" in selected[0].context_omissions[0]


def test_table_context_rows_always_include_their_header(tmp_path) -> None:
    index, ranked = _fixture_index(
        ["caption", "Property | Value", "Sample | Oil", "Range | 400-700 nm"],
        [2],
        table=True,
    )
    selected = _retriever(tmp_path)._select_with_parent_context(ranked, 1, index)

    context_ids = {block.block_id for block in selected[0].context_blocks}
    if "B03" in context_ids:
        assert "B01" in context_ids


def test_canonical_context_block_rejects_dangling_and_tampered_ids() -> None:
    index, _ = _fixture_index(["One", "Two"], [0])

    assert canonical_context_block(index, "B01").text == "Two"
    with pytest.raises(ValueError, match="not present"):
        canonical_context_block(index, "unknown")
    index["sections"][0]["document_id"] = "other-document"
    with pytest.raises(ValueError, match="inconsistent"):
        canonical_context_block(index, "B01")


def test_default_disabled_and_candidate_mode_remain_child_only(tmp_path, monkeypatch) -> None:
    index, _ = _fixture_index(["Background", "Spectral child", "Qualification"], [1])
    monkeypatch.delenv("SPECTRO_RAG_PARENT_EXPANSION", raising=False)
    default = _retriever(tmp_path, enabled=None)
    enabled = _retriever(tmp_path, enabled=True)

    assert default.parent_expansion is False
    default_hits = default._search_many(["spectral"], 1, pinned_index=index)
    candidates = enabled._search_many(["spectral"], 1, pinned_index=index, candidate_mode=True)
    assert default_hits[0].context_blocks == []
    assert candidates[0].context_blocks == []


def test_parent_expansion_environment_must_be_a_boolean(monkeypatch) -> None:
    monkeypatch.setenv("SPECTRO_RAG_PARENT_EXPANSION", "sometimes")

    with pytest.raises(ValueError, match="must be a boolean"):
        _retriever(Path("."), enabled=None)
