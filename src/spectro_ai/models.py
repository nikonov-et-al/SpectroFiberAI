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


class SourceLocation(DomainModel):
    source_id: str
    filename: str
    document_version: str
    locator: str
    page_number: int | None = Field(default=None, ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_line_range(self) -> SourceLocation:
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("Source line locations require both start and end")
        if self.line_start is not None and self.line_end < self.line_start:
            raise ValueError("Source line end must not precede its start")
        return self


class SourceSection(DomainModel):
    section_id: str
    document_id: str
    source_id: str
    filename: str
    document_version: str
    heading_path: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    parent_section_id: str | None = None
    detected_heading_level: int | None = None
    hierarchy_valid: bool = True
    diagnostics: list[str] = Field(default_factory=list)


SourceBlockType = Literal[
    "paragraph",
    "heading",
    "list",
    "code",
    "table_caption",
    "table_row",
    "caption",
]


class SourceBlock(DomainModel):
    block_id: str
    document_id: str
    section_id: str
    source_id: str
    filename: str
    document_version: str
    ordinal: int = Field(ge=0)
    text: str
    location: SourceLocation
    locations: list[SourceLocation] = Field(default_factory=list)
    block_type: SourceBlockType = "paragraph"
    detected_heading_level: int | None = None
    table_id: str | None = None
    table_row_index: int | None = Field(default=None, ge=0)
    is_table_header: bool = False
    table_header_block_ids: list[str] = Field(default_factory=list)
    oversized: bool = False

    @model_validator(mode="after")
    def valid_table_metadata(self) -> SourceBlock:
        if self.table_row_index is not None and self.table_id is None:
            raise ValueError("Table row metadata requires a table identity")
        if self.is_table_header and self.block_type != "table_row":
            raise ValueError("Only table rows can be marked as table headers")
        if self.table_header_block_ids and self.table_id is None:
            raise ValueError("Table header references require a table identity")
        return self


class SourceBlockSpan(DomainModel):
    block_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_span(self) -> SourceBlockSpan:
        if self.end <= self.start:
            raise ValueError("Source block span must be non-empty")
        return self


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
    extraction_mode: Literal["ollama", "openai"] = "ollama"
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


class EvidenceClaim(DomainModel):
    claim: str
    evidence_id: str
    quote: str


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
    evidence_claims: list[EvidenceClaim] = Field(default_factory=list)
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
    rerank_score: float | None = None
    source_scope: str | None = None


class RetrievalDiagnostic(DomainModel):
    status: Literal["disabled", "applied", "fallback", "skipped"] = "disabled"
    model_revision: str | None = None
    candidate_count: int = 0
    elapsed_ms: float = 0
    failure_reason: str | None = None


class ValidationDiagnostic(DomainModel):
    level: Literal["error", "warning", "not_checked"]
    check: str
    message: str
    evidence_ids: list[str] = Field(default_factory=list)


class AssessmentReport(DomainModel):
    extracted_requirements: CustomerUseCase
    missing_information: list[str] = Field(default_factory=list)
    suggested_technical_questions: list[str] = Field(default_factory=list)
    candidate_techniques: list[TechniqueCandidate] = Field(default_factory=list)
    recommendation_rationale: str | None = None
    proposed_feasibility_experiment: FeasibilityPlan
    assumptions_and_uncertainties: list[str] = Field(default_factory=list)
    status: Literal["insufficient_information", "provisional", "outside_supported_scope"]
    production_readiness: Literal["not_assessed", "not_established"] = "not_established"
    assessment_method: Literal["ollama_grounded", "openai_grounded", "evidence_abstention"] = (
        "evidence_abstention"
    )
    assessment_provider: Literal["ollama", "openai"] | None = None
    evidence_set_version: str | None = None
    knowledge_index_revision: str | None = None
    retrieved_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    report_version: str = "grounded-v1"
    assessment_warnings: list[str] = Field(default_factory=list)
    validation_diagnostics: list[ValidationDiagnostic] = Field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
