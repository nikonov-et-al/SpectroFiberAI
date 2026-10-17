"""Domain types; no UI or model-provider dependencies."""

from datetime import date
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

UNKNOWN_VALUES = {
    "",
    "unknown",
    "not known",
    "not specified",
    "not yet determined",
    "not-yet-determined",
    "to be determined",
    "tbd",
    "n/a",
    "none",
    "null",
}


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def normalize_unknown(cls, value: object, info: ValidationInfo) -> object:
        if info.field_name == "source_text":
            return value
        if isinstance(value, str) and value.strip().casefold() in UNKNOWN_VALUES:
            return None
        return value


CUSTOMER_FACT_FIELDS = (
    "industry",
    "measurement_objective",
    "target_analyte_or_property",
    "sample_matrix",
    "concentration_range",
    "temperature",
    "pressure",
    "physical_state",
    "required_accuracy",
    "measurement_frequency",
    "installation_constraints",
    "process",
    "measurement_location",
    "composition",
    "detection_limit",
    "deployment_mode",
    "interference",
    "flow",
    "moisture",
    "turbidity",
    "particle_effects",
    "contamination_fouling",
    "access_cleaning",
    "response_time",
    "reference_measurement",
    "desired_stage",
)


class CustomerUseCase(DomainModel):
    FACT_FIELDS: ClassVar[tuple[str, ...]] = CUSTOMER_FACT_FIELDS
    source_text: Annotated[str, StringConstraints(strip_whitespace=False)] | None = None
    request_id: str | None = None
    industry: str | None = None
    measurement_objective: str | None = None
    target_analyte_or_property: str | None = None
    sample_matrix: str | None = None
    concentration_range: str | None = None
    temperature: str | None = None
    pressure: str | None = None
    physical_state: str | None = None
    required_accuracy: str | None = None
    measurement_frequency: str | None = None
    installation_constraints: str | None = None
    process: str | None = None
    measurement_location: str | None = None
    composition: str | None = None
    detection_limit: str | None = None
    deployment_mode: str | None = None
    interference: str | None = None
    flow: str | None = None
    moisture: str | None = None
    turbidity: str | None = None
    particle_effects: str | None = None
    contamination_fouling: str | None = None
    access_cleaning: str | None = None
    response_time: str | None = None
    reference_measurement: str | None = None
    desired_stage: str | None = None
    customer_preferences: list[str] = Field(default_factory=list)
    strict_requirements: list[str] = Field(default_factory=list)
    field_quotes: dict[str, str] = Field(default_factory=dict)
    field_provenance: dict[str, str] = Field(default_factory=dict)
    unknown_or_missing_information: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    extraction_mode: Literal["offline", "ollama"] = "offline"
    extraction_warnings: list[str] = Field(default_factory=list)


class Evidence(DomainModel):
    kind: Literal["verified_source", "engineering_heuristic", "assumption"]
    statement: str
    source_title: str | None = None
    source_url: HttpUrl | None = None
    checked_on: date | None = None

    @model_validator(mode="after")
    def require_source_provenance(self) -> Evidence:
        if self.kind == "verified_source" and not (
            self.source_title and self.source_url and self.checked_on
        ):
            raise ValueError("Verified sources require a title, URL, and checked_on date")
        return self


class TechniqueKnowledge(DomainModel):
    id: str
    method: str
    spectral_region: str | None = None
    principle: str
    common_applications: list[str] = Field(min_length=1)
    typical_limitations: list[str] = Field(min_length=1)
    sample_considerations: list[str] = Field(min_length=1)
    possible_geometries: list[str] = Field(min_length=1)
    supporting_evidence: list[Evidence] = Field(min_length=1)
    expert_review_required: Literal[True] = True


class TechniqueCandidate(DomainModel):
    spectroscopy_method: str
    proposed_spectral_region: str | None = None
    measurement_geometry: str | None = None
    potential_probe_configuration: str | None = None
    advantages: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    required_assumptions: list[str] = Field(default_factory=list)
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    preliminary_suitability_assessment: Literal["provisional_candidate", "undetermined"] = (
        "undetermined"
    )
    rule_rationale: str | None = None


class FeasibilityPlan(DomainModel):
    objective: str | None = None
    hypothesis: str | None = None
    proposed_measurement_technique: str | None = None
    sample_preparation: str | None = None
    reference_measurement: str | None = None
    experimental_steps: list[str] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)
    expected_limitations: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    proposed_configuration: str | None = None
    success_actions: list[str] = Field(default_factory=list)
    failure_actions: list[str] = Field(default_factory=list)


class RetrievedEvidence(DomainModel):
    evidence_id: str
    source_id: str
    document_version: str | None = None
    page_or_section: str | None = None
    excerpt: str
    review_status: str
    retrieval_score: float | None = None
    source_scope: str | None = None


class AssessmentReport(DomainModel):
    extracted_requirements: CustomerUseCase
    missing_information: list[str] = Field(default_factory=list)
    suggested_technical_questions: list[str] = Field(default_factory=list)
    candidate_techniques: list[TechniqueCandidate] = Field(default_factory=list)
    recommendation_rationale: str | None = None
    proposed_feasibility_experiment: FeasibilityPlan
    assumptions_and_uncertainties: list[str] = Field(default_factory=list)
    status: Literal[
        "insufficient_information",
        "provisional_candidates",
        "no_rule_match",
        "provisional",
        "outside_supported_scope",
    ]
    assessment_method: Literal[
        "deterministic_baseline", "ollama_grounded", "offline_fallback", "evidence_abstention"
    ] = "deterministic_baseline"
    assessment_provider: Literal["baseline", "ollama"] | None = "baseline"
    evidence_set_version: str | None = None
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    report_version: str = "baseline-v1"
    assessment_warnings: list[str] = Field(default_factory=list)
