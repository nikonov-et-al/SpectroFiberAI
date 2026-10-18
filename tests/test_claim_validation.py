import pytest
from test_assessment import FakeAssessor, FakeRetriever, assess_use_case, case, draft, evidence

from spectro_ai.models import EvidenceClaim


def test_unsupported_secondary_claim_does_not_remove_supported_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    generated = draft()
    generated.status = "insufficient_information"
    candidate = generated.candidate_techniques[0]
    candidate.advantages.append("Works with every installed probe.")
    candidate.limitations.append("Works with every installed probe.")
    candidate.evidence_claims.append(
        EvidenceClaim(
            claim="Works with every installed probe.",
            evidence_id="S01-p17",
            quote="This quote is not in the source.",
        )
    )
    fiber_quote = "Fiber-optic probes support remote measurements."
    candidate.evidence_claims.append(
        EvidenceClaim(
            claim=fiber_quote,
            evidence_id="S01-p17",
            quote=fiber_quote,
        )
    )
    source = FakeRetriever(
        evidence(
            excerpt=(
                "NIR diffuse reflection can measure protein in opaque milk samples. " + fiber_quote
            )
        )
    )
    report = assess_use_case(case(), retriever=source, assessor=FakeAssessor(generated))
    assert report.status == "provisional"
    assert len(report.candidate_techniques) == 1
    assert "Works with every installed probe." not in str(report.candidate_techniques[0])
    assert all(
        "Fiber-optic probes" not in item for item in report.candidate_techniques[0].advantages
    )


def test_valid_quote_replaces_model_paraphrase_and_invalid_ids_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    generated = draft(evidence_ids=["S01-p17", "not-real"])
    generated.candidate_techniques[0].evidence_claims[
        0
    ].claim = "NIR has proven accuracy for our milk line."
    report = assess_use_case(case(), retriever=FakeRetriever(), assessor=FakeAssessor(generated))
    candidate = report.candidate_techniques[0]
    assert report.status == "provisional"
    assert candidate.supporting_evidence_ids == ["S01-p17"]
    assert candidate.advantages == [
        "Source reports the cited measurement under the stated conditions."
    ]
    assert "proven accuracy" not in str(candidate)


def test_negative_only_source_does_not_promote_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    source = FakeRetriever(evidence(excerpt="NIR cannot measure protein in milk samples."))
    generated = draft()
    generated.candidate_techniques[0].evidence_claims[
        0
    ].quote = "NIR cannot measure protein in milk samples."
    report = assess_use_case(case(), retriever=source, assessor=FakeAssessor(generated))
    assert report.candidate_techniques == []
    assert report.status == "insufficient_information"


def test_candidate_plan_compares_methods_and_keeps_combination_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    generated = draft()
    other = generated.candidate_techniques[0].model_copy(
        update={"spectroscopy_method": "Raman"}, deep=True
    )
    other.evidence_claims[0].quote = "Raman measures protein in milk samples."
    generated.candidate_techniques.append(other)
    source = FakeRetriever(
        evidence(
            excerpt=(
                "NIR diffuse reflection can measure protein in opaque milk samples. "
                "Raman measures protein in milk samples."
            )
        )
    )
    report = assess_use_case(case(), retriever=source, assessor=FakeAssessor(generated))
    plan = report.proposed_feasibility_experiment
    assert report.status == "provisional"
    assert "NIR; Raman" == plan.proposed_measurement_technique
    assert any("each candidate" in step for step in plan.experimental_steps)
    assert any("held-out" in step and "before" in step for step in plan.experimental_steps)
    assert "not_established" == report.production_readiness


def test_claim_retrieval_requires_exact_quote_and_verified_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    first = evidence(excerpt="NIR notes for protein in milk samples.")
    second = evidence(
        excerpt="NIR diffuse reflection can measure protein in opaque milk samples."
    ).model_copy(
        update={
            "evidence_id": "S02-p4",
            "source_id": "S02",
            "document_version": "hash-v2",
            "page_or_section": "p.4",
        }
    )

    class ClaimRetriever(FakeRetriever):
        def search_claim(self, use_case, claim: str, limit: int = 4):
            return [second]

        def index_metadata(self):
            return {
                "manifest_version": "manifest-v1",
                "chunks": [
                    {
                        "text": item.excerpt,
                        "provenance": [
                            {
                                "evidence_id": item.evidence_id,
                                "source_id": item.source_id,
                                "document_version": item.document_version,
                                "page_or_section": item.page_or_section,
                                "review_status": item.review_status,
                                "source_scope": item.source_scope,
                            }
                        ],
                    }
                    for item in (first, second)
                ],
            }

    report = assess_use_case(
        case(), retriever=ClaimRetriever(first), assessor=FakeAssessor(draft())
    )
    assert report.status == "provisional"
    assert set(report.candidate_techniques[0].supporting_evidence_ids) == {
        "S01-p17",
        "S02-p4",
    }


def test_direct_cross_source_contradiction_withholds_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    positive = evidence()
    negative = evidence(excerpt="NIR cannot measure protein in milk samples.").model_copy(
        update={
            "evidence_id": "S02-p4",
            "source_id": "S02",
            "document_version": "hash-v2",
            "page_or_section": "p.4",
        }
    )

    class ConflictingRetriever(FakeRetriever):
        def search(self, use_case, limit: int = 8):
            return [positive, negative]

        def index_metadata(self):
            return {
                "manifest_version": "manifest-v1",
                "chunks": [
                    {
                        "text": item.excerpt,
                        "provenance": [
                            {
                                "evidence_id": item.evidence_id,
                                "source_id": item.source_id,
                                "document_version": item.document_version,
                                "page_or_section": item.page_or_section,
                                "review_status": item.review_status,
                                "source_scope": item.source_scope,
                            }
                        ],
                    }
                    for item in (positive, negative)
                ],
            }

    report = assess_use_case(
        case(), retriever=ConflictingRetriever(positive), assessor=FakeAssessor(draft())
    )
    assert report.candidate_techniques == []
    assert report.status == "insufficient_information"
    assert report.evidence_assessment is not None
    assert report.evidence_assessment.conflicts


def test_d01_s01_evidence_retains_nir_raman_and_optional_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRO_AI_PROVIDER", "ollama")
    method_quote = (
        "15 and widely implemented spectral techniques are Near -Infrared (NIR) "
        "and Raman spectroscopy."
    )
    combined_quote = (
        "The combination of these techniques is particularly effective because it captures "
        "complementary chemical information: NIR detects hydrogen bonds and asymmetric polar "
        "groups, while Raman provides information on the molecular backbone and symmetrical "
        "non-polar groups."
    )
    application_quote = (
        "As the powder blend passes the probe tip, the system captures spectra at high speeds "
        "without interrupting the production flow."
    )
    # Noncontiguous literal sentences from S01 p.15 retained as one compact fixture.
    source_text = f"{method_quote} {combined_quote} [...] {application_quote}"
    source = evidence(excerpt=source_text).model_copy(
        update={
            "evidence_id": "S01-S01_Multispectral_Laboratory-7026efe72668-0033",
            "source_id": "S01",
            "document_version": (
                "7026efe7266877b50238d9a9c91e93202242670ae9c776856efe93bb550a8935"
            ),
            "page_or_section": "p.15 (chunk 1/2)",
        }
    )

    class D01Retriever(FakeRetriever):
        def index_metadata(self):
            item = self.item
            return {
                "manifest_version": "manifest-v1",
                "chunks": [
                    {
                        "text": item.excerpt,
                        "provenance": [
                            {
                                "evidence_id": item.evidence_id,
                                "source_id": item.source_id,
                                "document_version": item.document_version,
                                "page_or_section": item.page_or_section,
                                "review_status": item.review_status,
                                "source_scope": item.source_scope,
                            }
                        ],
                    }
                ],
            }

    generated = draft(evidence_ids=[source.evidence_id])
    generated.status = "insufficient_information"
    first = generated.candidate_techniques[0]
    first.advantages = [method_quote]
    first.limitations = ["This unsupported secondary limitation is dropped."]
    first.evidence_claims = [
        EvidenceClaim(
            claim="NIR is a provisional candidate.",
            evidence_id=source.evidence_id,
            quote=method_quote,
        )
    ]
    for method, quote in (("Raman", method_quote), ("NIR + Raman", combined_quote)):
        candidate = first.model_copy(update={"spectroscopy_method": method}, deep=True)
        candidate.evidence_claims = [
            EvidenceClaim(
                claim=f"{method} is a provisional candidate.",
                evidence_id=source.evidence_id,
                quote=quote,
            )
        ]
        candidate.advantages = [quote]
        candidate.limitations = ["Unsupported secondary limitation."]
        generated.candidate_techniques.append(candidate)
    report = assess_use_case(
        case().model_copy(
            update={
                "source_text": None,
                "field_quotes": {},
                "measurement_objective": "Measure blend potency at the production feed frame.",
                "target_analyte_or_property": (
                    "Active Pharmaceutical Ingredient (API) concentration"
                ),
                "sample_matrix": "powder blend",
                "reference_measurement": "Laboratory checks",
            }
        ),
        retriever=D01Retriever(source),
        assessor=FakeAssessor(generated),
    )
    assert {item.spectroscopy_method for item in report.candidate_techniques} >= {
        "NIR",
        "Raman",
    }
    assert report.status == "provisional"
