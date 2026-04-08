from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


AnswerKind = Literal["number", "percent", "unknown"]
SourceType = Literal["local_pdf"]
OperandKind = Literal["number", "percent"]
CompletionStatus = Literal["complete", "incomplete"]


def _default_request_id() -> str:
    return uuid4().hex[:12]


class DiligenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    company: str | None = None
    fiscal_year: int | None = None
    request_id: str = Field(default_factory=_default_request_id)
    qid: int | None = None

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def _normalize_fiscal_year(cls, value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return value
        try:
            return int(str(value).strip())
        except ValueError:
            return None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    source_type: SourceType
    source_label: str
    source_path: str
    page_number: int
    source_url: str | None = None


class EvidenceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    source_type: SourceType
    source_label: str
    source_path: str
    page_number: int
    snippet: str
    score: float = 0.0
    match_reasons: list[str] = Field(default_factory=list)
    source_url: str | None = None

    def to_citation(self) -> Citation:
        return Citation(
            citation_id=self.citation_id,
            source_type=self.source_type,
            source_label=self.source_label,
            source_path=self.source_path,
            page_number=self.page_number,
            source_url=self.source_url,
        )


class DiligenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer: str | None = None
    answer_kind: AnswerKind = "unknown"
    explanation: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_human_review: bool = False
    errors: list[str] = Field(default_factory=list)


class IntakeAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str | None = None
    fiscal_year: int | None = None
    task_type: str = "unknown"
    required_metrics: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_dir: str | None = None
    fiscal_year_dir: str | None = None
    candidate_files: list[str] = Field(default_factory=list)
    primary_candidate_file: str | None = None
    secondary_candidate_files: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    active_local_queries: list[str] = Field(default_factory=list)


class CoverageAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_formula: str = ""
    required_operands: list[str] = Field(default_factory=list)
    found_operands: list[str] = Field(default_factory=list)
    missing_operands: list[str] = Field(default_factory=list)
    follow_up_local_queries: list[str] = Field(default_factory=list)
    enough_evidence_to_answer: bool = False
    reasoning_notes: list[str] = Field(default_factory=list)


class ReasonedOperand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: OperandKind = "number"
    value: float
    citation_id: str


class ReasonedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_kind: AnswerKind = "unknown"
    proposed_answer: str = ""
    formula: str = ""
    operands: list[ReasonedOperand] = Field(default_factory=list)
    explanation: str
    citation_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    completion_status: CompletionStatus = "complete"
    missing_operands: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recomputed_answer: str | None = None
    matches_proposed_answer: bool = False
    format_ok: bool = False
    citation_ids_valid: bool = False
    coverage_complete: bool = True
    issues: list[str] = Field(default_factory=list)
    relative_percent_error: float | None = None
