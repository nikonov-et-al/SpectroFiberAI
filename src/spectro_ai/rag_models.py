"""Document retrieval types, separate from scientific assessment evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RagModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DocumentChunk(RagModel):
    chunk_id: str
    document: str
    file_hash: str
    text: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)


class RetrievalHit(RagModel):
    chunk: DocumentChunk
    score: float


class CitedStatement(RagModel):
    text: str = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1)


class DocumentAnswer(RagModel):
    status: Literal["answered", "insufficient_evidence"]
    statements: list[CitedStatement] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_status(self) -> DocumentAnswer:
        if (self.status == "answered") != bool(self.statements):
            raise ValueError("Answered requires cited statements; abstention requires none")
        return self


class RagResult(RagModel):
    question: str
    provider: str
    answer: DocumentAnswer
    sources: list[RetrievalHit] = Field(default_factory=list)
