"""Small, deliberately conservative extraction and qualitative applicability rules."""

import re

from spectro_ai.knowledge import KnowledgeSource, YamlKnowledgeSource
from spectro_ai.models import (
    AssessmentReport,
    CustomerUseCase,
    Evidence,
    FeasibilityPlan,
    TechniqueCandidate,
)

QUESTIONS = {
    "measurement_objective": "What decision should the measurement support?",
    "target_analyte_or_property": "Which analyte or property must be measured?",
    "sample_matrix": "What material or mixture contains the target?",
    "industry": "Which industry and process are involved?",
    "concentration_range": "What target range and units must be covered?",
    "temperature": "What sample and operating temperatures apply?",
    "pressure": "What operating pressure and pressure constraints apply?",
    "physical_state": "Is the sample solid, liquid, gas, or multiphase?",
    "required_accuracy": "What error tolerance and reference method define acceptance?",
    "measurement_frequency": "How often must results be available?",
    "installation_constraints": "What access, contact, cleaning, and installation limits apply?",
}
CRITICAL_FIELDS = ("measurement_objective", "target_analyte_or_property", "sample_matrix")
LABELS = {
    "industry": "industry",
    "objective": "measurement_objective",
    "measurement objective": "measurement_objective",
    "target": "target_analyte_or_property",
    "target analyte or property": "target_analyte_or_property",
    "matrix": "sample_matrix",
    "sample matrix": "sample_matrix",
    "concentration range": "concentration_range",
    "temperature": "temperature",
    "pressure": "pressure",
    "state": "physical_state",
    "physical state": "physical_state",
    "accuracy": "required_accuracy",
    "required accuracy": "required_accuracy",
    "frequency": "measurement_frequency",
    "measurement frequency": "measurement_frequency",
    "constraints": "installation_constraints",
    "installation constraints": "installation_constraints",
}

# Each phrase is a literal fact or intention the user can actually state.
PHRASES = {
    "measurement_objective": (
        "measure moisture",
        "monitor moisture",
        "identify polymer",
        "identify polymers",
        "identify molecular composition",
        "measure fluorescence",
        "monitor fluorescence",
        "measure absorbance",
        "measure absorption",
        "measure color",
        "measure colour",
        "measure bulk composition",
        "measure protein",
        "measure fat",
        "quantify fluorescent dye",
    ),
    "target_analyte_or_property": (
        "moisture",
        "protein",
        "fat content",
        "bulk composition",
        "fluorescent dye",
        "polymer identity",
        "absorbance",
        "absorption",
        "color",
        "colour",
        "fluorescence",
    ),
    "sample_matrix": ("food powder", "flour", "polymer pellets", "water", "aqueous solution"),
    "physical_state": ("powder", "pellets", "liquid", "solid", "gas"),
}


def _positive_text(text: str) -> str:
    """Drop negated clauses rather than attempting general natural-language reasoning."""
    clauses = re.split(r"[;\n.!?]+|\bbut\b", text.casefold())
    negation = r"\b(?:no|not|never|without|cannot|can't|don't|doesn't|isn't|isnt)\b|\bnon[- ]"
    return "\n".join(clause for clause in clauses if not re.search(negation, clause))


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def extract_use_case(text: str) -> CustomerUseCase:
    """Read label:value clauses; otherwise use only the documented literal phrases."""
    values: dict[str, str | None] = {"source_text": text or None}
    for clause in re.split(r"[;\n]", text):
        label, separator, value = clause.partition(":")
        field = LABELS.get(label.strip().casefold())
        if separator and field:
            values[field] = value.strip()
    positive = _positive_text(text)
    for field, phrases in PHRASES.items():
        # Explicit labels, including explicitly unknown values, take precedence.
        if field not in values:
            match = next((phrase for phrase in phrases if _contains(positive, phrase)), None)
            if match:
                values[field] = match
    case = CustomerUseCase.model_validate(values)
    return case.model_copy(update={"unknown_or_missing_information": missing_fields(case)})


def missing_fields(use_case: CustomerUseCase) -> list[str]:
    return [field for field in QUESTIONS if getattr(use_case, field) is None]


def _rules(use_case: CustomerUseCase) -> dict[str, str]:
    # Match extracted objective/property, not arbitrary background in the raw problem.
    text = _positive_text(
        f"{use_case.measurement_objective or ''}; {use_case.target_analyte_or_property or ''}"
    )
    matches: dict[str, str] = {}
    if any(
        _contains(text, word) for word in ("moisture", "protein", "fat content", "bulk composition")
    ):
        matches["nir"] = (
            "Bulk-property/composition phrase matched; NIR calibration is a hypothesis."
        )
    if any(
        _contains(text, word)
        for word in ("identify", "identity", "identification", "functional group")
    ):
        matches["raman"] = "Molecular identification phrase matched; assess Raman fingerprints."
        matches["mir"] = "Molecular identification phrase matched; assess MIR fingerprints."
    if any(_contains(text, word) for word in ("absorption", "absorbance", "color", "colour")):
        matches["uv_vis"] = "Explicit absorption/color phrase matched; test optical response."
    if any(_contains(text, word) for word in ("fluorescence", "fluorescent")):
        matches["fluorescence"] = "Explicit fluorescence phrase matched; test target emission."
    return matches


def _experiment(use_case: CustomerUseCase, candidates: list[TechniqueCandidate]) -> FeasibilityPlan:
    if not candidates:
        return FeasibilityPlan(
            objective="Resolve requirements before selecting an instrument.",
            experimental_steps=[
                "Confirm objective, target, matrix, operating conditions, and acceptance criteria.",
                "Obtain representative samples and agree an independent reference measurement.",
                "Ask a domain expert to select techniques for an initial feasibility screen.",
            ],
            expected_limitations=["Technique and measurement performance are undetermined."],
            success_criteria=[
                "Critical requirements and an expert-reviewed screening plan agreed."
            ],
        )
    methods = ", ".join(candidate.spectroscopy_method for candidate in candidates)
    return FeasibilityPlan(
        objective=use_case.measurement_objective,
        hypothesis="A reproducible spectral response may support the stated target; test this.",
        proposed_measurement_technique=f"Compare provisional options: {methods}",
        sample_preparation="Agree handling with the expert; retain representative variability.",
        reference_measurement="Agree an independent reference method with an expert.",
        experimental_steps=[
            "Agree reference method, representative samples, blanks, and acceptance thresholds.",
            "Confirm sample safety, optical access, geometry, and instrument compatibility.",
            "Record blank/reference and repeated sample spectra under documented conditions.",
            "Vary relevant matrix and operating conditions; inspect interference and drift.",
            "For quantification, fit calibration on training samples; evaluate held-out batches.",
            "For identification, compare known materials and held-out unknown samples.",
            "Compare results with the reference; review failures before recommending deployment.",
        ],
        evaluation_metrics=[
            "Repeatability and drift under representative conditions",
            "Prediction error and bias versus reference, if quantifying",
            "Misidentification/rejection rate on held-out samples, if identifying",
            "Time to result and sensitivity to matrix/operating changes",
        ],
        expected_limitations=[
            "A small feasibility study does not establish industrial performance.",
            "Sample count, calibration coverage, and instrument settings remain to be agreed.",
        ],
        success_criteria=[
            f"Required accuracy: {use_case.required_accuracy or 'to be agreed'}.",
            f"Measurement frequency: {use_case.measurement_frequency or 'to be agreed'}.",
            "Agree reference, repeatability, and failure thresholds before testing.",
        ],
    )


def assess(
    use_case: CustomerUseCase, knowledge_source: KnowledgeSource | None = None
) -> AssessmentReport:
    missing = missing_fields(use_case)
    requirements = use_case.model_copy(update={"unknown_or_missing_information": missing})
    rules = {} if any(field in missing for field in CRITICAL_FIELDS) else _rules(requirements)
    candidates: list[TechniqueCandidate] = []
    if rules:
        source = knowledge_source if knowledge_source is not None else YamlKnowledgeSource()
        techniques = {technique.id: technique for technique in source.get_techniques()}
        for technique_id, rationale in rules.items():
            if technique_id not in techniques:
                raise ValueError(f"Knowledge is missing rule-required technique: {technique_id}")
            technique = techniques[technique_id]
            candidates.append(
                TechniqueCandidate(
                    spectroscopy_method=technique.method,
                    proposed_spectral_region=technique.spectral_region,
                    advantages=[technique.principle],
                    limitations=technique.typical_limitations + technique.sample_considerations,
                    required_assumptions=[
                        "Target response can be distinguished from matrix/interferences.",
                        "Compatible instrumentation and representative reference samples exist.",
                        "Geometry, probe, and performance require expert verification.",
                    ],
                    supporting_evidence=technique.supporting_evidence
                    + [Evidence(kind="engineering_heuristic", statement=rationale)],
                    preliminary_suitability_assessment="provisional_candidate",
                    rule_rationale=rationale,
                )
            )
    if any(field in missing for field in CRITICAL_FIELDS):
        status = "insufficient_information"
        rationale = "Insufficient information: objective, target, and matrix are required."
    elif candidates:
        status = "provisional_candidates"
        rationale = (
            "Unranked screening candidates from literal phrase rules; no validated recommendation. "
            "A source-backed principle does not establish suitability for this customer."
        )
    else:
        status = "no_rule_match"
        rationale = "No transparent baseline rule matched; expert assessment is required."
    return AssessmentReport(
        extracted_requirements=requirements,
        missing_information=missing,
        suggested_technical_questions=[QUESTIONS[field] for field in missing],
        candidate_techniques=candidates,
        recommendation_rationale=rationale,
        proposed_feasibility_experiment=_experiment(requirements, candidates),
        assumptions_and_uncertainties=[
            "Deterministic template baseline; no sophisticated scientific or LLM analysis.",
            "Extraction recognizes only explicit labels and a small set of literal phrases.",
            "Negated clauses are conservatively excluded from phrase matching.",
            "Starter knowledge and all applicability rules require domain-expert review.",
            "Detection limits, exact wavelengths, hardware compatibility, performance unverified.",
        ],
        status=status,
    )
