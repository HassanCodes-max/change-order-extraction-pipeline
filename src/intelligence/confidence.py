"""Evidence-based, explainable field-level confidence scoring.

Confidence answers: *how likely is this value correct, given what we could
verify?* It is computed deterministically from observable signals rather than
taken from the model's self-assessment, which is only one dampening input.

    score = base(status)
            x evidence grounding      (is the quote really in the document?)
            x value support           (does the quote contain the value?)
            x normalization quality   (ambiguous date, unit conversion...)
            x source text quality     (OCR confidence of the page)
            x extractor agreement     (LLM vs. independent rule-based pass)
            x cross-field validation  (arithmetic, date order, status/approver)
            x model self-confidence   (dampening only: 0.75 + 0.25 * p)

Each applied factor is recorded in ``confidence_factors`` so a reviewer can
see exactly why a score is what it is. A null value always scores 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.extraction.text_processor import EvidenceMatch
from src.validation.schemas import FieldStatus

CrossCheck = Literal["agree", "disagree", "recovered", "unconfirmed", "single_source"]

#: Starting point by status, before any verification.
BASE_BY_STATUS: dict[FieldStatus, float] = {
    FieldStatus.STATED: 0.90,
    FieldStatus.INFERRED: 0.65,
    FieldStatus.CONFLICTING: 0.45,
}
EVIDENCE_FACTORS: dict[EvidenceMatch, float] = {
    EvidenceMatch.EXACT: 1.00,
    EvidenceMatch.FUZZY: 0.92,
    EvidenceMatch.WEAK: 0.75,
    EvidenceMatch.NOT_FOUND: 0.45,
    EvidenceMatch.NONE: 0.60,
}
CROSS_CHECK_FACTORS: dict[CrossCheck, float] = {
    "agree": 1.0,         # handled as a bonus below
    "disagree": 0.80,
    "recovered": 0.85,    # LLM missed it; rules found it
    "unconfirmed": 1.0,   # rules found nothing (common for prose); neutral
    "single_source": 1.0,
}
AGREEMENT_BONUS = 0.4     # closes 40% of the remaining gap to 1.0
MAX_CONFIDENCE = 0.99


@dataclass
class ConfidenceInputs:
    """Signals gathered for one field."""

    status: FieldStatus
    has_value: bool
    evidence_match: EvidenceMatch = EvidenceMatch.NONE
    value_supported: bool | None = None
    ambiguous: bool = False
    converted: bool = False
    ocr_confidence: float | None = None   # 0-100 OCR confidence of the value's words, if OCR'd
    cross_check: CrossCheck = "single_source"
    model_confidence: float | None = None
    rules_only_long_text: bool = False
    adjustments: list[tuple[float, str]] = field(default_factory=list)


def score_field(inp: ConfidenceInputs) -> tuple[float, list[str]]:
    """Return ``(confidence, factor_explanations)`` for one field."""
    if not inp.has_value or inp.status in (FieldStatus.MISSING, FieldStatus.INVALID):
        return 0.0, [f"no usable value (status: {inp.status.value}) -> 0.00"]

    factors: list[str] = []
    score = BASE_BY_STATUS.get(inp.status, 0.5)
    factors.append(f"base for '{inp.status.value}': {score:.2f}")

    def apply(mult: float, reason: str, always: bool = False) -> None:
        nonlocal score
        if mult != 1.0 or always:
            score *= mult
            factors.append(f"x{mult:.2f} {reason}")

    # Grounding is always listed so a passing check is visible, not just failures.
    apply(EVIDENCE_FACTORS[inp.evidence_match], f"evidence match in source: {inp.evidence_match.value}",
          always=True)
    if inp.value_supported is False and inp.status is FieldStatus.STATED:
        apply(0.75, "value not found inside its evidence quote")
    if inp.ambiguous:
        apply(0.80, "ambiguous normalization (e.g. month/day order)")
    if inp.converted:
        apply(0.95, "value interpreted/converted (e.g. weeks->days, 'None'->0)")
    if inp.ocr_confidence is not None:
        ocr_factor = round(max(0.5, min(0.99, 0.5 + 0.5 * inp.ocr_confidence / 100.0)), 2)
        apply(ocr_factor, f"value read by OCR (word confidence {inp.ocr_confidence:.0f}/100)")
    if inp.rules_only_long_text:
        apply(0.85, "free-text boundaries determined by rules only")
    apply(CROSS_CHECK_FACTORS[inp.cross_check], f"extractor cross-check: {inp.cross_check}")
    for mult, reason in inp.adjustments:
        apply(mult, reason)
    if inp.model_confidence is not None:
        apply(round(0.75 + 0.25 * inp.model_confidence, 3),
              f"model self-reported confidence {inp.model_confidence:.2f}")
    if inp.cross_check == "agree":
        before = score
        score = score + (1.0 - score) * AGREEMENT_BONUS
        factors.append(f"+{score - before:.2f} independent rule-based extraction agrees")

    score = max(0.0, min(MAX_CONFIDENCE, score))
    factors.append(f"= {score:.2f}")
    return round(score, 2), factors
