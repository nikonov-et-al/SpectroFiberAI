"""Small, deliberately conservative extraction and qualitative applicability rules."""

import re
from hashlib import sha256
from typing import Protocol

from spectro_ai.knowledge import KnowledgeSource, YamlKnowledgeSource
from spectro_ai.models import (
    UNKNOWN_VALUES,
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
    "process": "Which process stages and changes must be represented?",
    "composition": "Which matrix components and interferents may vary?",
    "detection_limit": "What lowest target level and units must be detected?",
    "deployment_mode": "Is monitoring in-line, at-line, or off-line?",
    "interference": "Is fluorescence, scattering, or other background observed?",
    "measurement_location": "Where can the measurement interface access the process?",
    "flow": "What flow conditions or changes apply at the measurement point?",
    "moisture": "How does sample moisture vary?",
    "turbidity": "How does turbidity or optical clarity vary?",
    "particle_effects": "How do particle size, density, or scattering vary?",
    "contamination_fouling": "What contamination or fouling is expected at the interface?",
    "access_cleaning": "What access, cleaning, and maintenance conditions apply?",
    "response_time": "What response time is required?",
    "reference_measurement": "Which independent reference measurements are available?",
    "desired_stage": "Is the intended stage exploratory, pilot, or production?",
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
    "process": "process",
    "composition": "composition",
    "detection limit": "detection_limit",
    "deployment mode": "deployment_mode",
    "interference": "interference",
    "measurement location": "measurement_location",
    "location": "measurement_location",
    "flow": "flow",
    "moisture": "moisture",
    "turbidity": "turbidity",
    "particle effects": "particle_effects",
    "fouling": "contamination_fouling",
    "contamination/fouling": "contamination_fouling",
    "access and cleaning": "access_cleaning",
    "response time": "response_time",
    "reference measurement": "reference_measurement",
    "references": "reference_measurement",
    "stage": "desired_stage",
    "desired stage": "desired_stage",
}

# Each phrase is a literal fact or intention the user can actually state.
PHRASES = {
    "measurement_objective": (
        "monitor ethanol",
        "monitor reaction progress",
        "detect contamination",
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
        "ethanol",
        "reaction progress",
        "contamination",
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
    "sample_matrix": (
        "aqueous fermentation broth",
        "reaction mixture",
        "food powder",
        "flour",
        "polymer pellets",
        "water",
        "aqueous solution",
    ),
    "physical_state": ("powder", "pellets", "liquid", "solid", "gas"),
}


def _positive_text(text: str) -> str:
    """Drop negated clauses rather than attempting general natural-language reasoning."""
    clauses = re.split(r"[;\n.!?]+|\bbut\b", text.casefold())
    negation = r"\b(?:no|not|never|without|cannot|can't|don't|doesn't|isn't|isnt)\b|\bnon[- ]"
    return "\n".join(clause for clause in clauses if not re.search(negation, clause))


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def _exact_quote(text: str, phrase: str) -> str | None:
    for clause in re.split(r"[;\n.!?]+|\bbut\b", text):
        if _positive_text(clause) and _contains(clause.casefold(), phrase.casefold()):
            match = re.search(re.escape(phrase), clause, re.IGNORECASE)
            if match:
                return match.group(0)
    return None


def extract_use_case(text: str) -> CustomerUseCase:
    """Read label:value clauses; otherwise use only the documented literal phrases."""
    values: dict[str, str | None] = {"source_text": text or None}
    provenance = {}
    quotes = {}
    for clause in re.split(r"[;\n]", text):
        label, separator, value = clause.partition(":")
        field = LABELS.get(label.strip().casefold())
        if separator and field:
            values[field] = value.strip()
            provenance[field] = f"customer label: {clause.strip()}"
            if value.strip().casefold() not in UNKNOWN_VALUES:
                quotes[field] = clause.strip()
    positive = _positive_text(text)
    for field, phrases in PHRASES.items():
        # Explicit labels, including explicitly unknown values, take precedence.
        if field not in values:
            match = next((phrase for phrase in phrases if _contains(positive, phrase)), None)
            if match:
                values[field] = match
                provenance[field] = f"customer literal phrase: {match}"
                quotes[field] = _exact_quote(text, match) or match
    request_id = "req_" + sha256((text or "").encode("utf-8")).hexdigest()[:16]
    case = CustomerUseCase.model_validate(
        {
            **values,
            "request_id": request_id,
            "field_provenance": provenance,
            "field_quotes": quotes,
        }
    )
    return case.model_copy(update={"unknown_or_missing_information": missing_fields(case)})


class UseCaseExtractor(Protocol):
    def extract(self, customer_text: str) -> CustomerUseCase: ...


class OfflineUseCaseExtractor:
    """Conservative, deterministic extractor used without a configured model."""

    def extract(self, customer_text: str) -> CustomerUseCase:
        return extract_use_case(customer_text)


def missing_fields(use_case: CustomerUseCase) -> list[str]:
    return [field for field in QUESTIONS if getattr(use_case, field) is None]


def prioritized_questions(use_case: CustomerUseCase) -> list[str]:
    """Ask consequential gaps first; retain all unknowns separately in the report."""
    critical = [field for field in CRITICAL_FIELDS if getattr(use_case, field) is None]
    if critical:
        return [QUESTIONS[field] for field in critical]
    fields = ["required_accuracy", "concentration_range", "physical_state", "composition"]
    target = (use_case.target_analyte_or_property or "").casefold()
    if any(word in target for word in ("contamin", "trace", "detect")):
        fields.insert(0, "detection_limit")
    if "raman" in _rules(use_case):
        fields.insert(0, "interference")
    fields += ["deployment_mode", "installation_constraints", "measurement_frequency"]
    if use_case.deployment_mode or use_case.installation_constraints:
        fields += ["temperature", "pressure", "process"]
    return [QUESTIONS[field] for field in fields if getattr(use_case, field) is None]


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
    target = _positive_text(use_case.target_analyte_or_property or "")
    objective = _positive_text(use_case.measurement_objective or "")
    emission_target = any(_contains(target, word) for word in ("fluorescence", "fluorescent"))
    background_target = any(word in target for word in ("background", "interfer"))
    emission_objective = any(
        _contains(objective, phrase)
        for phrase in ("measure fluorescence", "monitor fluorescence", "quantify fluorescent")
    )
    if (emission_target and not background_target) or emission_objective:
        matches["fluorescence"] = "Explicit fluorescence phrase matched; test target emission."
    matrix = (use_case.sample_matrix or "").casefold()
    if _contains(text, "ethanol") and any(
        word in matrix for word in ("water", "aqueous", "fermentation")
    ):
        matches["raman"] = (
            "Aqueous ethanol monitoring: test Raman target response against "
            "matrix variability; no sensitivity assumed."
        )
        matches["mir"] = (
            "Aqueous ethanol monitoring: compare MIR target absorption with "
            "water interference and suitable sampling."
        )
    if any(
        _contains(text, phrase)
        for phrase in ("reaction progress", "reactor progress", "conversion")
    ):
        matches["raman"] = (
            "Reaction monitoring: Raman is conditional on distinguishable "
            "reactant/product response and representative time samples."
        )
        matches["mir"] = (
            "Reaction monitoring: MIR is conditional on distinguishable "
            "changing absorption and compatible sampling."
        )
    return matches


def _configuration(use_case: CustomerUseCase, method: str) -> str | None:
    """Conditional study geometry; never an asserted hardware configuration."""
    constraints = (use_case.installation_constraints or "").casefold()
    state = (use_case.physical_state or "").casefold()
    if any(word in constraints for word in ("non-contact", "no contact", "without contact")):
        if method in ("Raman", "NIR"):
            return (
                "Non-contact optical study, conditional on suitable access and "
                "representative sampling."
            )
        return (
            "Contact forbidden: confirm a compatible optical window/path before selecting geometry."
        )
    if method == "NIR" and state in ("powder", "solid", "pellets"):
        return "Diffuse reflectance study; control presentation and sample heterogeneity."
    if method == "UV-Vis" and state == "liquid":
        return "Transmission study; establish blank, clarity, and a suitable optical path."
    if method == "MIR/FTIR" and "contact allowed" in constraints:
        return "ATR study conditional on suitable contact and chemical compatibility."
    if state == "liquid" and "immersion allowed" in constraints:
        return "Immersion study conditional on compatible probe materials and optical path."
    return None


def _context(use_case: CustomerUseCase, method: str) -> list[str]:
    """Engineering screening conditions; supporting principles live in technique YAML."""
    notes = []
    matrix = (use_case.sample_matrix or "").casefold()
    interference = _positive_text(
        f"{use_case.interference or ''}; {use_case.measurement_objective or ''}"
    )
    if method == "Raman" and any(word in interference for word in ("fluorescen", "background")):
        notes.append(
            "Reported background may obscure Raman: retain only conditionally "
            "until a blank/sample screen resolves it; compare MIR."
        )
    if method == "MIR/FTIR" and any(word in matrix for word in ("water", "aqueous")):
        notes.append(
            "Aqueous matrix: water absorption can interfere; establish usable "
            "target bands and sampling path before MIR selection."
        )
    if method == "NIR" and use_case.physical_state == "powder":
        notes.append(
            "Powder matrix: test presentation and particle variability with "
            "representative calibration batches."
        )
    if use_case.detection_limit:
        notes.append(
            f"Requested detection limit {use_case.detection_limit}: capability unknown; "
            "validate blanks and low-level samples before selection."
        )
    if use_case.interference:
        notes.append(
            f"Customer interference: {use_case.interference}; "
            "challenge target selectivity experimentally."
        )
    if use_case.installation_constraints or use_case.deployment_mode:
        notes.append(
            "Deployment conditional on optical access, contact, fiber "
            "transmission, material compatibility, fouling, cleaning, "
            "temperature and pressure ratings; AFS capability unconfirmed."
        )
    return notes


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
            success_criteria=["Expert confirms requirements and an appropriate screening study."],
            success_actions=["Review confirmed requirements and select a laboratory screen."],
            failure_actions=["Resolve missing facts; do not recommend deployment."],
        )
    target = use_case.target_analyte_or_property
    matrix = use_case.sample_matrix
    identification = any(
        word in (use_case.measurement_objective or "").casefold() for word in ("identif", "classif")
    )
    configurations = [c.measurement_geometry for c in candidates if c.measurement_geometry]
    return FeasibilityPlan(
        objective=(
            f"Test whether {use_case.measurement_objective} is feasible for {target} in {matrix}."
        ),
        hypothesis=(
            f"The response of {target} can be distinguished from {matrix} variability "
            "under representative conditions; unvalidated hypothesis."
        ),
        proposed_measurement_technique="Compare provisional options: "
        + ", ".join(c.spectroscopy_method for c in candidates),
        proposed_configuration="; ".join(configurations)
        or "Geometry and probe remain to be agreed after optical-access review.",
        sample_preparation=(
            f"Retain representative {matrix} variability; agree handling, blanks "
            "and independent reference samples before preparation."
        ),
        reference_measurement=(
            f"Agree an independent reference for {target} with an expert; "
            "no reference method has been selected."
        ),
        experimental_steps=[
            f"Document process {use_case.process or 'to be agreed'} and matrix composition "
            f"{use_case.composition or 'unknown'}; sample each relevant source of variation.",
            (
                f"Collect representative {matrix} samples spanning supplied target range "
                f"{use_case.concentration_range or 'to be agreed'}, "
                "matrix batches and process conditions."
            ),
            (
                "Include matrix blanks, known reference samples and repeats; agree "
                "sample count and acceptance thresholds before testing."
            ),
            (
                "Confirm safety, optical access, contact, path, fiber "
                "transmission, material compatibility, fouling, cleaning, "
                "temperature and pressure limits before installing a probe."
            ),
            (
                "Record blank/reference and repeated spectra with documented "
                "geometry and instrument settings."
            ),
            (
                "Challenge interference "
                f"({use_case.interference or 'unknown; screen matrix background'}); "
                "vary matrix composition and operating conditions separately from target."
            ),
            (
                "Compare known materials and held-out unknown samples; include "
                "out-of-library rejection."
            )
            if identification
            else (
                "Fit calibration using independent reference values; evaluate "
                "held-out batches separated from calibration."
            ),
            (
                "Test low-level samples and blanks against requested detection limit "
                f"{use_case.detection_limit or 'to be agreed'}; "
                "do not infer sensitivity from a technique name."
            ),
            (
                "Compare held-out results with independent reference; inspect "
                "failures, drift and time to result before deployment review."
            ),
        ],
        evaluation_metrics=[
            "Misidentification and rejection rates on held-out samples"
            if identification
            else "Held-out prediction error and bias versus independent reference",
            "Repeatability, drift and interference sensitivity",
            "Blank response, low-level discrimination and time to result",
        ],
        expected_limitations=[
            "A feasibility study does not establish industrial performance or AFS compatibility.",
            (
                "Sample counts, settings, reference method and operating limits "
                "require expert agreement."
            ),
        ],
        success_criteria=[
            f"Required accuracy: {use_case.required_accuracy or 'to be agreed'}.",
            f"Measurement frequency: {use_case.measurement_frequency or 'to be agreed'}.",
            f"Detection limit: {use_case.detection_limit or 'to be agreed'}.",
            (
                "Agree reference, repeatability, interference and failure "
                "thresholds before testing; require held-out evidence."
            ),
        ],
        success_actions=[
            (
                "Review held-out evidence with a domain expert; confirm hardware "
                "ratings and pilot under representative process conditions."
            )
        ],
        failure_actions=[
            (
                "Inspect interference, sampling and reference failures; revise "
                "geometry/calibration or compare alternatives, then repeat "
                "independent validation."
            )
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
                    limitations=technique.typical_limitations
                    + technique.sample_considerations
                    + _context(requirements, technique.method),
                    measurement_geometry=_configuration(requirements, technique.method),
                    potential_probe_configuration=(
                        (
                            "Provisional laboratory configuration; probe specifications and "
                            "AFS availability require confirmation."
                        )
                        if _configuration(requirements, technique.method)
                        else None
                    ),
                    required_assumptions=[
                        "Target response can be distinguished from matrix/interferences.",
                        "Compatible instrumentation and representative reference samples exist.",
                        "Geometry, probe, and performance require expert verification.",
                    ],
                    supporting_evidence=technique.supporting_evidence
                    + [Evidence(kind="engineering_heuristic", statement=rationale)],
                    preliminary_suitability_assessment="provisional_candidate",
                    rule_rationale=" ".join(
                        [
                            f"Customer target: {requirements.target_analyte_or_property}; "
                            f"matrix: {requirements.sample_matrix}.",
                            rationale,
                            *_context(requirements, technique.method),
                        ]
                    ),
                )
            )
    if any(field in missing for field in CRITICAL_FIELDS):
        status = "insufficient_information"
        rationale = "Insufficient information: objective, target, and matrix are required."
    elif candidates:
        status = "provisional_candidates"
        rationale = (
            "Unranked contextual screening candidates; no validated recommendation. "
            "A source-backed principle does not establish suitability for this customer."
        )
    else:
        status = "no_rule_match"
        rationale = "No transparent baseline rule matched; expert assessment is required."
    return AssessmentReport(
        extracted_requirements=requirements,
        missing_information=missing,
        suggested_technical_questions=prioritized_questions(requirements),
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
