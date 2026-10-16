"""Domain types; no UI or model-provider dependencies."""

from datetime import date
from typing import Annotated, Literal

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


class CustomerUseCase(DomainModel):
    source_text: Annotated[str, StringConstraints(strip_whitespace=False)] | None = None
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
    unknown_or_missing_information: list[str] = Field(default_factory=list)


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


class AssessmentReport(DomainModel):
    extracted_requirements: CustomerUseCase
    missing_information: list[str] = Field(default_factory=list)
    suggested_technical_questions: list[str] = Field(default_factory=list)
    candidate_techniques: list[TechniqueCandidate] = Field(default_factory=list)
    recommendation_rationale: str | None = None
    proposed_feasibility_experiment: FeasibilityPlan
    assumptions_and_uncertainties: list[str] = Field(default_factory=list)
    status: Literal["insufficient_information", "provisional_candidates", "no_rule_match"]
    assessment_method: Literal["deterministic_baseline"] = "deterministic_baseline"
