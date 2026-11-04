"""Merge LLM output with the independent rule-based pass, field by field.

The LLM is the primary extractor when available. The rule-based pass is used to:

* confirm values (agreement raises confidence),
* expose disagreements (kept as alternatives, lowering confidence),
* recover fields the LLM missed (flagged for verification),
* surface conflicts in the document the LLM did not report,
* replace LLM values whose evidence cannot be found in the document when
  the rules found a grounded value instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.extraction.text_processor import EvidenceMatch, locate_evidence, text_similarity, tokenize
from src.intelligence.confidence import CrossCheck
from src.validation.schemas import (
    FIELD_NAMES,
    SPEC_BY_NAME,
    ExtractionSource,
    FieldKind,
    RawExtraction,
    RawField,
)
from src.validation.validators import comparison_key

_GROUNDED = (EvidenceMatch.EXACT, EvidenceMatch.FUZZY)


@dataclass
class Reconciled:
    """The chosen raw value for one field plus how it was decided."""

    raw: RawField
    source: ExtractionSource
    cross_check: CrossCheck
    notes: list[str] = field(default_factory=list)


def _contained(a: str, b: str) -> bool:
    """True if most tokens of the shorter text appear in the longer one."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return False
    short, long_ = (ta, set(tb)) if len(ta) <= len(tb) else (tb, set(ta))
    return sum(t in long_ for t in short) / len(short) >= 0.8


def values_agree(name: str, a: str, b: str, date_order: str = "MDY") -> bool:
    """Do two raw values for ``name`` denote the same thing?"""
    kind = SPEC_BY_NAME[name].kind
    if kind is FieldKind.LONG_TEXT:
        return text_similarity(a, b) >= 0.6 or _contained(a, b)
    if comparison_key(name, a, date_order) == comparison_key(name, b, date_order):
        return True
    if kind is FieldKind.TEXT:
        # Names/identifiers: tolerate punctuation and suffix differences
        # ("ABC Holdings, Inc." vs "ABC Holdings Inc"), but not different numbers.
        ta, tb = tokenize(a), tokenize(b)
        same_digits = [t for t in ta if t.isdigit()] == [t for t in tb if t.isdigit()]
        return same_digits and (text_similarity(a, b) >= 0.75 or _contained(a, b))
    return False


def _union(values: list[str], exclude: str | None) -> list[str]:
    out: list[str] = []
    for v in values:
        if v and v != exclude and v not in out:
            out.append(v)
    return out


def _reconcile_one(name: str, llm: RawField, rules: RawField, pages: list[str], date_order: str) -> Reconciled:
    if llm.value is not None and rules.value is not None:
        if values_agree(name, llm.value, rules.value, date_order):
            chosen = llm.model_copy(deep=True)
            notes: list[str] = []
            if rules.status == "conflicting" and llm.status != "conflicting":
                chosen.status = "conflicting"
                chosen.alternatives = _union(chosen.alternatives + [rules.value] + rules.alternatives, chosen.value)
                notes.append("rule-based pass found several different values for this field in the document")
            return Reconciled(chosen, "llm+rules", "agree", notes)

        llm_grounded = locate_evidence(llm.evidence, pages).match in _GROUNDED
        rules_grounded = locate_evidence(rules.evidence, pages).match in _GROUNDED
        if not llm_grounded and rules_grounded:
            chosen = rules.model_copy(deep=True)
            chosen.alternatives = _union(chosen.alternatives + [llm.value], chosen.value)
            return Reconciled(chosen, "rules", "disagree", [
                f"LLM value '{llm.value}' is not supported by any text in the document; "
                "the grounded rule-based value was used instead"
            ])
        chosen = llm.model_copy(deep=True)
        chosen.alternatives = _union(chosen.alternatives + [rules.value] + rules.alternatives, chosen.value)
        if rules.status == "conflicting":
            chosen.status = "conflicting"
        return Reconciled(chosen, "llm", "disagree", [
            f"rule-based pass read '{rules.value}' for this field; verify which is correct"
        ])

    if llm.value is not None:
        return Reconciled(llm.model_copy(deep=True), "llm", "unconfirmed")

    if rules.value is not None:
        return Reconciled(rules.model_copy(deep=True), "rules", "recovered", [
            "LLM reported this field as missing; value recovered by the rule-based pass, please verify"
        ])

    return Reconciled(RawField(), "none", "single_source")


def reconcile(
    llm: RawExtraction | None,
    rules: RawExtraction,
    pages: list[str],
    date_order: str = "MDY",
) -> tuple[dict[str, Reconciled], RawField]:
    """Return per-field decisions and the chosen ``prior_change_orders_net`` value."""
    rules_aux = rules.auxiliary.get("prior_change_orders_net") or RawField()
    if llm is None:
        decided = {
            name: Reconciled(
                rules.get(name).model_copy(deep=True),
                "rules" if rules.get(name).value is not None else "none",
                "single_source",
            )
            for name in FIELD_NAMES
        }
        return decided, rules_aux

    decided = {name: _reconcile_one(name, llm.get(name), rules.get(name), pages, date_order)
               for name in FIELD_NAMES}
    llm_aux = llm.auxiliary.get("prior_change_orders_net") or RawField()
    return decided, (llm_aux if llm_aux.value is not None else rules_aux)
