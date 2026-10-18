from spectro_ai.evidence_assessment import evidence_fingerprint, validate_assessment
from spectro_ai.models import (
    ApplicationSupportAssessment,
    AssessmentReport,
    CustomerUseCase,
    EvidenceAssessment,
    EvidenceConditionComparison,
    EvidenceCoverage,
    EvidenceObservation,
    EvidenceReference,
    ExploratoryBridge,
    FeasibilityPlan,
    RetrievedEvidence,
)


def _source() -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="src-1",
        source_id="S1",
        document_version="v1",
        page_or_section="p. 1",
        excerpt="Raman measured protein concentration in aqueous solution.",
        review_status="needs_source_review",
    )


def _observation(key="model-observation", **changes) -> EvidenceObservation:
    data = {
        "observation_id": key,
        "scenario_id": "model-scenario",
        "source_statement": "Raman measured protein concentration in aqueous solution.",
        "references": [
            EvidenceReference(
                evidence_id="src-1",
                quote="Raman measured protein concentration in aqueous solution.",
            )
        ],
        "statement_kind": "reported_result",
        "method": "Raman",
        "condition_comparisons": [
            EvidenceConditionComparison(
                dimension="target_property",
                source_value="protein concentration",
                customer_fields=["measurement_objective"],
                comparison="match",
                rationale="Both concern protein concentration.",
                supporting_references=[
                    EvidenceReference(evidence_id="src-1", quote="protein concentration")
                ],
            )
        ],
        "entailment": "supported",
        "applicability": "related_application",
    }
    data.update(changes)
    return EvidenceObservation(**data)


def _assessment(**changes) -> EvidenceAssessment:
    data = {
        "observations": [_observation()],
        "support_assessments": [
            ApplicationSupportAssessment(
                support_assessment_id="model-support",
                method="Raman",
                basis="application_supported",
                observation_ids=["model-observation"],
                bounded_conclusion="Source reports Raman measurement in aqueous solution.",
            )
        ],
        "coverage": [
            EvidenceCoverage(
                evidence_id="src-1",
                disposition="assessed",
                context_availability="child_only",
            )
        ],
        "input_fingerprint": "model-supplied-fingerprint",
        "assessor_version": "model-supplied-version",
    }
    data.update(changes)
    return EvidenceAssessment(**data)


def test_validation_binds_ids_and_metadata_to_host_snapshot():
    source = _source()
    customer = CustomerUseCase(
        sample_matrix="powder", measurement_objective="protein concentration"
    )
    fingerprint = evidence_fingerprint(customer, [source])

    result = validate_assessment(
        _assessment(), customer, {source.evidence_id: source}, fingerprint, "ea-v1", "local-model"
    )

    observation_id = result.observations[0].observation_id
    support_id = result.support_assessments[0].support_assessment_id
    assert observation_id.startswith("observation-")
    assert support_id.startswith("support-")
    assert result.support_assessments[0].observation_ids == [observation_id]
    assert result.input_fingerprint == fingerprint
    assert result.assessor_version == "ea-v1"
    assert result.model_identity == "local-model"


def test_method_and_geometry_labels_can_be_semantic_and_missing_values_stay_unknown():
    source_text = "NIR diffuse reflection measured protein concentration in solution."
    source = _source().model_copy(update={"excerpt": source_text})
    observation = _observation(
        source_statement=source_text,
        references=[EvidenceReference(evidence_id="src-1", quote=source_text)],
        method="near infrared spectroscopy",
        geometry="diffuse reflectance",
    )
    support = ApplicationSupportAssessment(
        support_assessment_id="model-support",
        method="near infrared spectroscopy",
        basis="application_supported",
        observation_ids=["model-observation"],
        bounded_conclusion="Source reports a near infrared measurement.",
    )
    assessment = _assessment(observations=[observation], support_assessments=[support])
    customer = CustomerUseCase(
        measurement_objective="protein concentration", sample_matrix="powder"
    )

    valid = validate_assessment(
        assessment,
        customer,
        {source.evidence_id: source},
        "host-fingerprint",
        "ea-v1",
    )
    missing = validate_assessment(
        _assessment(
            observations=[observation.model_copy(update={"method": None, "geometry": None})],
            support_assessments=[],
        ),
        customer,
        {source.evidence_id: source},
        "host-fingerprint",
        "ea-v1",
    )
    fabricated_quote = validate_assessment(
        _assessment(
            observations=[
                observation.model_copy(
                    update={
                        "references": [EvidenceReference(evidence_id="src-1", quote="fiber probe")]
                    }
                )
            ],
            support_assessments=[],
        ),
        customer,
        {source.evidence_id: source},
        "host-fingerprint",
        "ea-v1",
    )

    assert valid.observations[0].method == "near infrared spectroscopy"
    assert valid.observations[0].geometry == "diffuse reflectance"
    assert len(valid.support_assessments) == 1
    assert missing.observations[0].geometry is None
    assert missing.observations[0].method is None
    assert fabricated_quote.observations == []
    assert any("source reference" in warning for warning in fabricated_quote.diagnostics.warnings)


def test_unknown_customer_condition_cannot_be_marked_match():
    source = _source()
    observation = _observation(
        condition_comparisons=[
            EvidenceConditionComparison(
                dimension="operating_conditions",
                source_value="protein concentration",
                customer_fields=["temperature"],
                comparison="match",
                rationale="Model claims match despite unknown customer temperature.",
                supporting_references=[
                    EvidenceReference(
                        evidence_id="src-1", quote="protein concentration in aqueous solution"
                    )
                ],
            )
        ]
    )
    result = validate_assessment(
        _assessment(observations=[observation], support_assessments=[]),
        CustomerUseCase(sample_matrix="powder", measurement_objective="protein concentration"),
        {source.evidence_id: source},
        "host-fingerprint",
        "ea-v1",
    )

    assert result.observations == []
    assert any("condition comparison" in warning for warning in result.diagnostics.warnings)


def test_unknown_comparison_enum_survives_scalar_unknown_normalization():
    comparison = EvidenceConditionComparison(
        dimension="matrix_state",
        source_value="unknown",
        customer_fields=["sample_matrix"],
        comparison="unknown",
        rationale="Source context does not identify matrix.",
    )

    assert comparison.comparison == "unknown"
    assert comparison.source_value is None


def test_authentic_quote_does_not_validate_unreferenced_or_broader_support():
    source = _source()
    observation = _observation(
        source_statement="Raman proves accurate powder measurement.",
        references=[
            EvidenceReference(
                evidence_id="src-1",
                quote="Raman measured protein concentration in aqueous solution.",
            )
        ],
        applicability="insufficient_context",
    )
    support = ApplicationSupportAssessment(
        support_assessment_id="model-support",
        method="Raman",
        basis="application_supported",
        observation_ids=["model-observation"],
        bounded_conclusion="Raman is accurate for powder measurement.",
    )
    result = validate_assessment(
        _assessment(observations=[observation], support_assessments=[support]),
        CustomerUseCase(sample_matrix="powder", measurement_objective="protein concentration"),
        {source.evidence_id: source},
        "host-fingerprint",
        "ea-v1",
    )

    assert result.observations
    assert result.support_assessments == []


def test_exploratory_bridge_requires_grounded_premise_customer_link_and_checks():
    source = _source()
    bridge = ExploratoryBridge(
        source_premises=[
            EvidenceReference(evidence_id="src-1", quote="Raman measured protein concentration")
        ],
        customer_fields=["measurement_objective", "sample_matrix"],
        hypothesis=(
            "Raman response may track protein concentration in powder if scattering is controlled."
        ),
        inference=(
            "A reported concentration response suggests testing whether analyte response "
            "persists in the target powder matrix."
        ),
        application_differences=["Source used aqueous solution; customer sample is powder."],
        unknown_gaps=["No source evidence establishes performance in this powder matrix."],
        possible_failure_mechanisms=["Powder scattering can change measured response."],
        checks=["Compare against a reference method across representative powder samples."],
    )
    support = ApplicationSupportAssessment(
        support_assessment_id="model-support",
        method="Raman",
        basis="reasoned_exploratory",
        observation_ids=["model-observation"],
        bounded_conclusion="Consider Raman only as a testable engineering hypothesis.",
        exploratory_bridge=bridge,
    )
    customer = CustomerUseCase(
        measurement_objective="Estimate protein concentration",
        sample_matrix="powder",
    )
    result = validate_assessment(
        _assessment(support_assessments=[support]),
        customer,
        {source.evidence_id: source},
        "host-fingerprint",
        "ea-v1",
    )

    assert len(result.support_assessments) == 1
    shallow = support.model_copy(update={"exploratory_bridge": None})
    rejected = validate_assessment(
        _assessment(support_assessments=[shallow]),
        customer,
        {source.evidence_id: source},
        "host-fingerprint",
        "ea-v1",
    )
    assert rejected.support_assessments == []


def test_conflicting_comparison_cannot_be_hidden_by_exploratory_bridge():
    source = _source()
    observation = _observation(
        condition_comparisons=[
            EvidenceConditionComparison(
                dimension="matrix_state",
                source_value="aqueous solution",
                customer_fields=["sample_matrix"],
                comparison="conflicting",
                rationale="The observation explicitly conflicts with the customer matrix.",
                supporting_references=[
                    EvidenceReference(evidence_id="src-1", quote="aqueous solution")
                ],
            )
        ]
    )
    bridge = ExploratoryBridge(
        source_premises=[EvidenceReference(evidence_id="src-1", quote="protein concentration")],
        customer_fields=["measurement_objective", "sample_matrix"],
        hypothesis="Test whether the Raman response transfers to powder.",
        inference="A solution result may motivate a controlled transfer experiment.",
        application_differences=["The customer sample is powder."],
        unknown_gaps=["No powder performance is reported."],
        possible_failure_mechanisms=["Scattering may alter response."],
        checks=["Compare against a reference method."],
    )
    support = ApplicationSupportAssessment(
        support_assessment_id="model-support",
        method="Raman",
        basis="reasoned_exploratory",
        observation_ids=["model-observation"],
        bounded_conclusion="Explore Raman as a conditional hypothesis.",
        exploratory_bridge=bridge,
    )
    result = validate_assessment(
        _assessment(observations=[observation], support_assessments=[support]),
        CustomerUseCase(sample_matrix="powder", measurement_objective="protein concentration"),
        {source.evidence_id: source},
        "host-fingerprint",
        "ea-v1",
    )

    assert result.support_assessments == []


def test_missing_context_is_coverage_gap_not_application_match():
    result = validate_assessment(
        _assessment(coverage=[]),
        CustomerUseCase(sample_matrix="powder", measurement_objective="protein concentration"),
        {"src-1": _source()},
        "host-fingerprint",
        "ea-v1",
    )

    assert result.coverage[0].disposition == "insufficient_context"
    assert result.coverage[0].context_availability == "child_only"
    assert result.observations[0].applicability == "related_application"


def test_legacy_report_defaults_new_assessment_to_none():
    report = AssessmentReport(
        extracted_requirements=CustomerUseCase(),
        proposed_feasibility_experiment=FeasibilityPlan(),
        status="insufficient_information",
    )
    assert report.evidence_assessment is None


def test_references_cannot_launder_across_source_scenarios():
    source_a = _source()
    source_b = source_a.model_copy(
        update={"evidence_id": "src-2", "source_id": "S2", "document_version": "v2"}
    )
    observation = _observation(
        references=[
            EvidenceReference(evidence_id="src-1", quote="protein concentration"),
            EvidenceReference(evidence_id="src-2", quote="protein concentration"),
        ]
    )

    result = validate_assessment(
        _assessment(observations=[observation], support_assessments=[]),
        CustomerUseCase(measurement_objective="protein concentration"),
        {"src-1": source_a, "src-2": source_b},
        "fingerprint",
        "ea-v1",
    )

    assert result.observations == []
    assert any("cross source scenarios" in warning for warning in result.diagnostics.warnings)


def test_context_quote_requires_named_attached_block():
    source = _source()
    observation = _observation(
        references=[
            EvidenceReference(
                evidence_id="src-1",
                quote="Raman measured protein concentration",
                context_block_id="missing-block",
            )
        ]
    )

    result = validate_assessment(
        _assessment(observations=[observation], support_assessments=[]),
        CustomerUseCase(measurement_objective="protein concentration"),
        {"src-1": source},
        "fingerprint",
        "ea-v1",
    )

    assert result.observations == []


def test_related_observation_with_matrix_difference_cannot_be_application_support():
    source = _source()
    observation = _observation(
        condition_comparisons=[
            EvidenceConditionComparison(
                dimension="target_property",
                source_value="protein concentration",
                customer_fields=["measurement_objective"],
                comparison="match",
                rationale="Target property matches.",
                supporting_references=[
                    EvidenceReference(evidence_id="src-1", quote="protein concentration")
                ],
            ),
            EvidenceConditionComparison(
                dimension="matrix_state",
                source_value="aqueous solution",
                customer_fields=["sample_matrix"],
                comparison="different",
                rationale="Source uses solution; customer has powder.",
                supporting_references=[
                    EvidenceReference(evidence_id="src-1", quote="aqueous solution")
                ],
            ),
        ]
    )
    support = _assessment().support_assessments
    result = validate_assessment(
        _assessment(observations=[observation], support_assessments=support),
        CustomerUseCase(measurement_objective="protein concentration", sample_matrix="powder"),
        {"src-1": source},
        "fingerprint",
        "ea-v1",
    )

    assert result.observations
    assert result.support_assessments == []


def test_general_method_and_limitation_cannot_support_application_or_bridge():
    source = _source()
    observation = _observation(
        statement_kind="limitation",
        applicability="general_method",
    )
    app_support = _assessment().support_assessments
    bridge_support = ApplicationSupportAssessment(
        support_assessment_id="bridge-support",
        method="Raman",
        basis="reasoned_exploratory",
        observation_ids=["model-observation"],
        bounded_conclusion="Explore Raman for customer powder samples.",
        exploratory_bridge=ExploratoryBridge(
            source_premises=[EvidenceReference(evidence_id="src-1", quote="protein concentration")],
            customer_fields=["measurement_objective", "sample_matrix"],
            hypothesis="The method may detect the target in powder.",
            inference="No supported capability follows from a source limitation.",
            possible_failure_mechanisms=["Scattering."],
            unknown_gaps=["Powder response is unknown."],
            checks=["Test against reference results."],
        ),
    )
    customer = CustomerUseCase(
        measurement_objective="protein concentration", sample_matrix="powder"
    )
    for support in [app_support[0], bridge_support]:
        result = validate_assessment(
            _assessment(observations=[observation], support_assessments=[support]),
            customer,
            {"src-1": source},
            "fingerprint",
            "ea-v1",
        )
        assert result.support_assessments == []


def test_general_method_cannot_be_application_precedent():
    source = _source()
    observation = _observation(applicability="general_method")
    result = validate_assessment(
        _assessment(observations=[observation]),
        CustomerUseCase(measurement_objective="protein concentration"),
        {"src-1": source},
        "fingerprint",
        "ea-v1",
    )

    assert result.observations
    assert result.support_assessments == []


def test_method_description_cannot_promote_application_even_when_labeled_direct():
    source = _source()
    observation = _observation(
        statement_kind="method_description",
        applicability="direct_application",
    )
    result = validate_assessment(
        _assessment(observations=[observation]),
        CustomerUseCase(measurement_objective="protein concentration"),
        {"src-1": source},
        "fingerprint",
        "ea-v1",
    )

    assert result.observations
    assert result.support_assessments == []


def test_supported_observation_cannot_be_borrowed_for_different_method():
    source = _source()
    support = _assessment().support_assessments[0].model_copy(update={"method": "FTIR"})
    result = validate_assessment(
        _assessment(support_assessments=[support]),
        CustomerUseCase(measurement_objective="protein concentration"),
        {"src-1": source},
        "fingerprint",
        "ea-v1",
    )

    assert result.observations
    assert result.support_assessments == []


def test_application_support_cannot_mix_another_methods_observation():
    source = _source().model_copy(
        update={
            "excerpt": (
                "Raman measured protein concentration in aqueous solution. "
                "FTIR is described as a general method."
            )
        }
    )
    general_method = _observation(
        key="model-ft-ir",
        source_statement="FTIR is described as a general method.",
        references=[
            EvidenceReference(evidence_id="src-1", quote="FTIR is described as a general method")
        ],
        method="FTIR",
        condition_comparisons=[],
        applicability="general_method",
    )
    support = ApplicationSupportAssessment(
        support_assessment_id="borrowed-support",
        method="FTIR",
        basis="application_supported",
        observation_ids=["model-observation", "model-ft-ir"],
        bounded_conclusion="FTIR has application support.",
    )
    result = validate_assessment(
        _assessment(observations=[_observation(), general_method], support_assessments=[support]),
        CustomerUseCase(measurement_objective="protein concentration"),
        {"src-1": source},
        "fingerprint",
        "ea-v1",
    )

    assert result.support_assessments == []


def test_known_operating_condition_difference_blocks_application_support():
    source = _source()
    observation = _observation(
        condition_comparisons=[
            EvidenceConditionComparison(
                dimension="target_property",
                source_value="protein concentration",
                customer_fields=["measurement_objective"],
                comparison="match",
                rationale="Target property matches.",
                supporting_references=[
                    EvidenceReference(evidence_id="src-1", quote="protein concentration")
                ],
            ),
            EvidenceConditionComparison(
                dimension="operating_conditions",
                source_value="aqueous solution",
                customer_fields=["sample_matrix"],
                comparison="different",
                rationale="Different physical conditions.",
                supporting_references=[
                    EvidenceReference(evidence_id="src-1", quote="aqueous solution")
                ],
            ),
        ]
    )
    result = validate_assessment(
        _assessment(observations=[observation]),
        CustomerUseCase(measurement_objective="protein concentration", sample_matrix="powder"),
        {"src-1": source},
        "fingerprint",
        "ea-v1",
    )

    assert result.support_assessments == []
