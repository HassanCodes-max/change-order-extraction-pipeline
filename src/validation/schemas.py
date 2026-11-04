"""Pydantic schemas: the single source of truth for the extraction data model.

Three layers of models live here:

* ``FIELD_SPECS`` — a registry describing every field (type, label, whether it
  is required). Prompts, JSON schemas, validators and the UI are all driven
  from this registry so the field list is defined exactly once.
* ``RawField`` / ``RawExtraction`` — the *untrusted* output of an extractor
  (LLM or rules): raw strings plus evidence, before normalization.
* ``ExtractedField`` / ``ExtractionResult`` — the *validated* final output with
  typed values, confidence scores, evidence and validation issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Field registry
# ---------------------------------------------------------------------------


class FieldKind(str, Enum):
    """Data type of a field, which selects its normalizer."""

    TEXT = "text"
    LONG_TEXT = "long_text"
    DATE = "date"
    MONEY = "money"
    DAYS = "days"
    STATUS = "status"


@dataclass(frozen=True)
class FieldSpec:
    """Static description of one extractable field."""

    name: str
    label: str
    kind: FieldKind
    required: bool
    description: str


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("project_name", "Project Name", FieldKind.TEXT, True,
              "Name of the construction project."),
    FieldSpec("project_number", "Project Number", FieldKind.TEXT, False,
              "Project / job number or code assigned to the project."),
    FieldSpec("change_order_number", "Change Order Number", FieldKind.TEXT, True,
              "Identifier of this change order (e.g. 'CO-014', '14')."),
    FieldSpec("original_contract_number", "Original Contract Number", FieldKind.TEXT, False,
              "Number of the original contract / agreement this change order modifies."),
    FieldSpec("contractor_name", "Contractor Name", FieldKind.TEXT, False,
              "Company performing the work (general contractor)."),
    FieldSpec("client_name", "Client / Owner Name", FieldKind.TEXT, False,
              "Owner or client who contracted the work."),
    FieldSpec("change_order_date", "Change Order Date", FieldKind.DATE, True,
              "Date the change order was issued or dated."),
    FieldSpec("description_of_changes", "Description of Changes", FieldKind.LONG_TEXT, True,
              "What work is being added, removed or modified."),
    FieldSpec("reason_for_change", "Reason for Change", FieldKind.LONG_TEXT, False,
              "Why the change is needed (owner request, unforeseen condition, design error...)."),
    FieldSpec("original_contract_amount", "Original Contract Amount", FieldKind.MONEY, False,
              "Contract sum before any change orders (or before this change order, as labeled)."),
    FieldSpec("additional_cost", "Additional Cost", FieldKind.MONEY, False,
              "Amount this change order ADDS to the contract."),
    FieldSpec("deducted_amount", "Deducted Amount", FieldKind.MONEY, False,
              "Amount this change order DEDUCTS / credits from the contract."),
    FieldSpec("revised_contract_amount", "Revised Contract Amount", FieldKind.MONEY, False,
              "New contract sum including this change order."),
    FieldSpec("schedule_impact_days", "Schedule Impact (days)", FieldKind.DAYS, False,
              "Change to contract time in days (positive = extension, negative = reduction, 0 = no change)."),
    FieldSpec("requested_completion_date", "Requested Completion Date", FieldKind.DATE, False,
              "Requested, revised or new completion date."),
    FieldSpec("approval_status", "Approval Status", FieldKind.STATUS, False,
              "Approval state: approved, approved_with_conditions, pending, rejected or void."),
    FieldSpec("prepared_by", "Prepared By", FieldKind.TEXT, False,
              "Person or company who prepared / submitted the change order."),
    FieldSpec("approved_by", "Approved By", FieldKind.TEXT, False,
              "Person who approved / signed off the change order."),
)

FIELD_NAMES: tuple[str, ...] = tuple(spec.name for spec in FIELD_SPECS)
SPEC_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPECS}
REQUIRED_FIELDS: frozenset[str] = frozenset(s.name for s in FIELD_SPECS if s.required)
MONEY_FIELDS: tuple[str, ...] = tuple(s.name for s in FIELD_SPECS if s.kind is FieldKind.MONEY)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FieldStatus(str, Enum):
    """How a value relates to the source document."""

    STATED = "stated"            # explicitly written in the document
    INFERRED = "inferred"        # derived by reasoning; not written verbatim
    CONFLICTING = "conflicting"  # the document gives two or more different values
    INVALID = "invalid"          # text was found but cannot be normalized (e.g. Feb 30)
    MISSING = "missing"          # not present in the document


class ApprovalStatus(str, Enum):
    """Normalized approval states."""

    APPROVED = "approved"
    APPROVED_WITH_CONDITIONS = "approved_with_conditions"
    PENDING = "pending"
    REJECTED = "rejected"
    VOID = "void"


ExtractionSource = Literal["llm", "rules", "llm+rules", "none"]
RawStatus = Literal["stated", "inferred", "missing", "conflicting"]
Severity = Literal["info", "warning", "error"]


# ---------------------------------------------------------------------------
# Raw (untrusted) extractor output
# ---------------------------------------------------------------------------


class RawField(BaseModel):
    """One field as reported by an extractor, before normalization.

    Values are kept as strings exactly as the extractor reported them; typed
    normalization happens later in :mod:`src.validation.validators`.
    """

    model_config = ConfigDict(extra="ignore")

    value: str | None = None
    evidence: str | None = None
    status: RawStatus = "missing"
    alternatives: list[str] = Field(default_factory=list)
    model_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str | None = None

    @field_validator("value", "evidence", "note", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = str(v)
        if isinstance(v, str):
            v = v.strip()
            # "None" is deliberately NOT treated as missing: "Deduct: None" or
            # "Schedule impact: None" carry meaning (zero) and are handled by
            # the field normalizers.
            if not v or v.lower() in {"null", "n/a", "not stated", "not provided", "unknown"}:
                return None
        return v

    @field_validator("alternatives", mode="before")
    @classmethod
    def _clean_alternatives(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(a).strip() for a in v if a is not None and str(a).strip()]

    @field_validator("model_confidence", mode="before")
    @classmethod
    def _percent_to_unit(cls, v: Any) -> Any:
        # Some models answer 95 instead of 0.95; accept and rescale.
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 1.0 < v <= 100.0:
            return v / 100.0
        return v

    @model_validator(mode="after")
    def _status_consistency(self) -> "RawField":
        if self.value is None:
            if self.status == "conflicting" and self.alternatives:
                # Promote the first alternative so the conflict stays visible.
                self.value = self.alternatives[0]
                self.alternatives = self.alternatives[1:]
            else:
                self.status = "missing"
                self.evidence = None
        elif self.status == "missing":
            # A value was supplied while claiming "missing": be conservative.
            self.status = "inferred"
        # Never list the chosen value as its own alternative.
        self.alternatives = [a for a in self.alternatives if a != self.value]
        return self


class RawExtraction(BaseModel):
    """Complete output of one extractor run."""

    fields: dict[str, RawField] = Field(default_factory=dict)
    #: Supporting values that are not output fields but help validation,
    #: e.g. ``prior_change_orders_net`` (previously approved change orders).
    auxiliary: dict[str, RawField] = Field(default_factory=dict)
    engine: str
    model: str | None = None
    attempts: int = 1
    is_change_order: bool | None = None
    document_notes: str | None = None

    def get(self, name: str) -> RawField:
        """Return the raw field ``name`` or an empty (missing) field."""
        return self.fields.get(name) or RawField()


# ---------------------------------------------------------------------------
# Validated output
# ---------------------------------------------------------------------------

T = TypeVar("T")


class ExtractedField(BaseModel, Generic[T]):
    """A single validated field with value, confidence and provenance."""

    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str | None = None
    status: FieldStatus = FieldStatus.MISSING
    raw_value: str | None = None
    source: ExtractionSource = "none"
    alternatives: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    confidence_factors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "ExtractedField[T]":
        if self.value is None:
            if self.status in (FieldStatus.STATED, FieldStatus.INFERRED):
                raise ValueError(f"status '{self.status.value}' requires a non-null value")
            if self.confidence != 0.0:
                raise ValueError("a null value must have confidence 0.0")
        if self.status is FieldStatus.MISSING and self.value is not None:
            raise ValueError("status 'missing' requires a null value")
        return self


class ChangeOrderFields(BaseModel):
    """All extracted change-order fields. Every key is always present."""

    model_config = ConfigDict(extra="forbid")

    project_name: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    project_number: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    change_order_number: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    original_contract_number: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    contractor_name: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    client_name: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    change_order_date: ExtractedField[date] = Field(default_factory=ExtractedField[date])
    description_of_changes: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    reason_for_change: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    original_contract_amount: ExtractedField[float] = Field(default_factory=ExtractedField[float])
    additional_cost: ExtractedField[float] = Field(default_factory=ExtractedField[float])
    deducted_amount: ExtractedField[float] = Field(default_factory=ExtractedField[float])
    revised_contract_amount: ExtractedField[float] = Field(default_factory=ExtractedField[float])
    schedule_impact_days: ExtractedField[int] = Field(default_factory=ExtractedField[int])
    requested_completion_date: ExtractedField[date] = Field(default_factory=ExtractedField[date])
    approval_status: ExtractedField[ApprovalStatus] = Field(
        default_factory=ExtractedField[ApprovalStatus]
    )
    prepared_by: ExtractedField[str] = Field(default_factory=ExtractedField[str])
    approved_by: ExtractedField[str] = Field(default_factory=ExtractedField[str])


class ValidationIssue(BaseModel):
    """A problem or observation found during validation."""

    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Severity
    message: str
    fields: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Metadata about the processed input document."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    input_type: Literal["pdf", "text", "image"]
    page_count: int = Field(ge=0)
    text_layer_pages: list[int] = Field(default_factory=list)
    ocr_pages: list[int] = Field(default_factory=list)
    ocr_mean_confidence: float | None = Field(default=None, ge=0.0, le=100.0)
    character_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class ExtractionInfo(BaseModel):
    """Metadata about how extraction was performed."""

    model_config = ConfigDict(extra="forbid")

    engine: Literal["anthropic", "openai", "rules"]
    model: str | None = None
    llm_attempts: int = Field(default=0, ge=0)
    fallback_used: bool = False
    fallback_reason: str | None = None
    cross_checked_with_rules: bool = False
    processing_seconds: float = Field(default=0.0, ge=0.0)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResultSummary(BaseModel):
    """Aggregate view over all fields, for dashboards and triage."""

    model_config = ConfigDict(extra="forbid")

    total_fields: int
    extracted: int
    stated: int
    inferred: int
    conflicting: int
    invalid: int
    missing: int
    missing_required: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    review_reasons: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Top-level JSON document produced by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    document: DocumentInfo
    extraction: ExtractionInfo
    fields: ChangeOrderFields
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    summary: ResultSummary

    def field_items(self) -> list[tuple[FieldSpec, ExtractedField[Any]]]:
        """Return ``(spec, field)`` pairs in canonical order."""
        return [(spec, getattr(self.fields, spec.name)) for spec in FIELD_SPECS]
