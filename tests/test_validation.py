"""Tests for normalizers, cross-field checks, schemas, confidence scoring and JSON output."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from src.extraction.text_processor import EvidenceMatch
from src.intelligence.confidence import ConfidenceInputs, score_field
from src.validation.schemas import ApprovalStatus, ExtractedField, FieldStatus, RawField
from src.validation.validators import (
    normalize_date,
    normalize_days,
    normalize_money,
    normalize_status,
    normalize_value,
    validate_cross_fields,
    value_supported_by_evidence,
)

# --------------------------------------------------------------------------- dates


@pytest.mark.parametrize("raw, expected", [
    ("March 3, 2025", date(2025, 3, 3)),
    ("Mar. 3 2025", date(2025, 3, 3)),
    ("3rd of March 2025", date(2025, 3, 3)),
    ("14 day of March, 2025", date(2025, 3, 14)),
    ("2025-06-10", date(2025, 6, 10)),
    ("08/15/2025", date(2025, 8, 15)),
    ("15.08.25", date(2025, 8, 15)),       # day > 12 -> unambiguous D/M/Y
    ("Dated: September 30, 2025.", date(2025, 9, 30)),
])
def test_normalize_date_formats(raw, expected):
    result = normalize_date(raw)
    assert result.ok and result.value == expected and not result.ambiguous


def test_ambiguous_numeric_date_is_flagged_and_respects_date_order():
    mdy = normalize_date("03/04/2025", "MDY")
    dmy = normalize_date("03/04/2025", "DMY")
    assert mdy.value == date(2025, 3, 4) and mdy.ambiguous
    assert dmy.value == date(2025, 4, 3) and dmy.ambiguous


@pytest.mark.parametrize("raw", ["02/30/2026", "February 30, 2025", "2025-13-01"])
def test_invalid_calendar_dates_are_rejected(raw):
    result = normalize_date(raw)
    assert not result.ok and result.value is None


def test_partial_and_garbage_dates_are_rejected():
    assert "no day" in normalize_date("March 2025").notes[0]
    assert not normalize_date("next Tuesday").ok


# --------------------------------------------------------------------------- money


@pytest.mark.parametrize("raw, expected", [
    ("$12,500.00", 12500.0),
    ("USD 3,480,000", 3480000.0),
    ("$12500", 12500.0),
    ("12,500", 12500.0),
    ("12.500,00", 12500.0),
    ("$12.5k", 12500.0),
    ("Additional cost - $12,500", 12500.0),   # spaced dash is a separator, not a minus
    ("$0.00", 0.0),
])
def test_normalize_money(raw, expected):
    result = normalize_money(raw)
    assert result.ok and result.value == expected


def test_negative_and_credit_amounts():
    assert normalize_money("($3,200.00)").value == -3200.0
    assert normalize_money("-$1,000").value == -1000.0
    deduct = normalize_value("deducted_amount", "(3,200.00)")
    assert deduct.value == 3200.0 and any("positive magnitude" in n for n in deduct.notes)


def test_money_words_and_failures():
    none = normalize_money("None")
    assert none.ok and none.value == 0.0 and none.converted
    assert not normalize_money("twelve thousand dollars").ok
    multi = normalize_money("$5,000 (was $4,000)")
    assert multi.value == 5000.0 and "multiple amounts" in multi.notes[0]
    assert "EUR" in normalize_money("EUR 1,000.00").notes[0]


# --------------------------------------------------------------------------- days


@pytest.mark.parametrize("raw, expected", [
    ("five (5) calendar days", 5),
    ("Project completion extended by five (5) calendar days.", 5),
    ("10 days", 10),
    ("fourteen days", 14),
    ("twenty-one days", 21),
    ("two weeks", 14),
    ("No change to contract time", 0),
    ("None", 0),
    ("reduced by 3 days", -3),
    ("7", 7),
])
def test_normalize_days(raw, expected):
    result = normalize_days(raw)
    assert result.ok and result.value == expected


def test_days_edge_cases():
    assert not normalize_days("2 months").ok
    assert any("working" in n for n in normalize_days("10 working days").notes)
    assert normalize_days("two weeks").converted
    assert not normalize_days("to be determined").ok


# --------------------------------------------------------------------------- status


@pytest.mark.parametrize("raw, expected", [
    ("Approved", ApprovalStatus.APPROVED),
    ("APPROVED AS NOTED", ApprovalStatus.APPROVED_WITH_CONDITIONS),
    ("Submitted for owner review", ApprovalStatus.PENDING),
    ("Not yet approved", ApprovalStatus.PENDING),
    ("Rejected - resubmit", ApprovalStatus.REJECTED),
    ("Void", ApprovalStatus.VOID),
    ("[ ] Approved   [x] Rejected", ApprovalStatus.REJECTED),
])
def test_normalize_status(raw, expected):
    assert normalize_status(raw).value == expected


def test_unknown_status_is_not_guessed():
    assert not normalize_status("see attached").ok


def test_value_supported_by_evidence():
    assert value_supported_by_evidence("additional_cost", 12500.0, "Additional Cost: $12,500.00")
    assert not value_supported_by_evidence("additional_cost", 15000.0, "Additional Cost: $12,500.00")
    assert value_supported_by_evidence("change_order_date", date(2025, 3, 3), "Date: March 3, 2025")
    assert value_supported_by_evidence("project_name", "ABC Building", "Project: ABC Building")
    assert value_supported_by_evidence("project_name", "ABC Building", None) is None


# --------------------------------------------------------------------------- cross-field


def _base_values(**overrides):
    values = {"project_name": "P", "change_order_number": "1", "change_order_date": date(2025, 3, 1),
              "description_of_changes": "work"}
    values.update(overrides)
    return values


def test_contract_arithmetic_consistent():
    report = validate_cross_fields(_base_values(original_contract_amount=1000.0, additional_cost=200.0,
                                                deducted_amount=50.0, revised_contract_amount=1150.0))
    assert [i.code for i in report.issues] == ["CONTRACT_ARITHMETIC_OK"]
    assert all(f > 1 for f, _ in report.adjustments["revised_contract_amount"])


def test_contract_arithmetic_mismatch_is_reported_not_corrected():
    values = _base_values(original_contract_amount=1_000_000.0, additional_cost=45_000.0,
                          revised_contract_amount=1_054_000.0)
    report = validate_cross_fields(values)
    issue = next(i for i in report.issues if i.code == "CONTRACT_ARITHMETIC_MISMATCH")
    assert issue.severity == "error" and "$1,045,000.00" in issue.message and "$9,000.00" in issue.message
    assert values["revised_contract_amount"] == 1_054_000.0  # untouched
    assert report.adjustments["revised_contract_amount"][0][0] < 1


def test_prior_change_orders_explain_difference():
    values = _base_values(original_contract_amount=100_000.0, additional_cost=2_500.0,
                          revised_contract_amount=107_500.0)
    assert any(i.code == "CONTRACT_ARITHMETIC_MISMATCH" for i in validate_cross_fields(values).issues)
    report = validate_cross_fields(values, prior_change_orders_net=5_000.0)
    assert any(i.code == "CONTRACT_ARITHMETIC_OK" and "previously approved" in i.message for i in report.issues)


def test_missing_revised_amount_is_not_invented():
    report = validate_cross_fields(_base_values(original_contract_amount=1000.0, additional_cost=100.0))
    issue = next(i for i in report.issues if i.code == "REVISED_AMOUNT_NOT_STATED")
    assert issue.severity == "info" and "not inserted" in issue.message


def test_required_fields_and_other_checks():
    report = validate_cross_fields({
        "change_order_date": date(2025, 5, 1), "requested_completion_date": date(2025, 4, 1),
        "approval_status": ApprovalStatus.APPROVED, "original_contract_amount": -5.0,
    })
    codes = {i.code for i in report.issues}
    assert {"REQUIRED_FIELD_MISSING", "COMPLETION_BEFORE_CO_DATE", "APPROVED_WITHOUT_APPROVER",
            "NON_POSITIVE_CONTRACT_AMOUNT", "NO_COST_CHANGE_FOUND"} <= codes
    missing = {f for i in report.issues if i.code == "REQUIRED_FIELD_MISSING" for f in i.fields}
    assert missing == {"project_name", "change_order_number", "description_of_changes"}


# --------------------------------------------------------------------------- schemas


def test_extracted_field_enforces_bounds_and_consistency():
    with pytest.raises(ValidationError):
        ExtractedField[float](value=1.0, confidence=1.5, status=FieldStatus.STATED)
    with pytest.raises(ValidationError):
        ExtractedField[float](value=None, confidence=0.0, status=FieldStatus.STATED)
    with pytest.raises(ValidationError):
        ExtractedField[float](value=None, confidence=0.4, status=FieldStatus.INVALID)
    with pytest.raises(ValidationError):
        ExtractedField[str](value="x", confidence=0.5, status=FieldStatus.MISSING)
    with pytest.raises(ValidationError):
        ExtractedField[float](value="not a number", confidence=0.5, status=FieldStatus.STATED)
    ok = ExtractedField[date](value="2025-03-03", confidence=0.9, status=FieldStatus.STATED)
    assert ok.value == date(2025, 3, 3)


def test_raw_field_coercions():
    assert RawField(value="  ", status="stated").status == "missing"
    assert RawField(value="N/A", status="stated").value is None
    assert RawField(value="x", status="missing").status == "inferred"   # conservative
    assert RawField(value="x", model_confidence=95).model_confidence == 0.95
    conflict = RawField(value=None, status="conflicting", alternatives=["A", "B"])
    assert conflict.value == "A" and conflict.alternatives == ["B"]


# --------------------------------------------------------------------------- confidence


def _score(**kw):
    defaults = dict(status=FieldStatus.STATED, has_value=True, evidence_match=EvidenceMatch.EXACT,
                    value_supported=True)
    defaults.update(kw)
    return score_field(ConfidenceInputs(**defaults))[0]


def test_confidence_is_zero_without_value():
    assert _score(has_value=False, status=FieldStatus.MISSING) == 0.0
    assert _score(status=FieldStatus.INVALID) == 0.0


def test_confidence_orders_by_evidence_quality():
    exact, fuzzy, weak, absent = (_score(evidence_match=m) for m in (
        EvidenceMatch.EXACT, EvidenceMatch.FUZZY, EvidenceMatch.WEAK, EvidenceMatch.NOT_FOUND))
    assert exact > fuzzy > weak > absent
    assert absent < 0.5  # ungrounded evidence is never trusted


def test_confidence_reflects_status_and_agreement():
    assert _score() > _score(status=FieldStatus.INFERRED) > _score(status=FieldStatus.CONFLICTING)
    assert _score(cross_check="agree") > _score() > _score(cross_check="disagree")


def test_confidence_penalizes_ocr_ambiguity_and_validation():
    assert _score(ocr_confidence=55) < _score(ocr_confidence=95) < _score()
    assert _score(ambiguous=True) < _score()
    assert _score(adjustments=[(0.7, "mismatch")]) < _score()
    assert _score(model_confidence=0.3) < _score(model_confidence=0.95)


def test_confidence_bounds_and_explanations():
    score, factors = score_field(ConfidenceInputs(
        status=FieldStatus.STATED, has_value=True, evidence_match=EvidenceMatch.EXACT,
        cross_check="agree", adjustments=[(1.05, "ok")], model_confidence=1.0))
    assert 0.0 <= score <= 0.99
    assert factors[0].startswith("base") and factors[-1].startswith("=")


# --------------------------------------------------------------------------- JSON


def test_json_output_round_trip(settings):
    from src.pipeline import ChangeOrderPipeline
    from src.serialization import from_json, save_result, to_compact_json, to_json

    result = ChangeOrderPipeline(settings).process_text(
        "Change Order No: CO-9\nProject: Elm Plaza\nDate: 2025-02-01\nDescription: Add bollards.\n"
        "Additional Cost: $2,000.00")
    payload = to_json(result)
    data = json.loads(payload)
    assert set(data) == {"schema_version", "document", "extraction", "fields", "validation_issues", "summary"}
    assert data["fields"]["change_order_date"]["value"] == "2025-02-01"
    assert from_json(payload) == result

    compact = json.loads(to_compact_json(result))
    assert compact["additional_cost"] == {"value": 2000.0, "confidence": result.fields.additional_cost.confidence,
                                          "evidence": "Additional Cost: $2,000.00", "status": "stated"}
    path = save_result(result, settings.output_dir)
    assert from_json(path.read_text(encoding="utf-8")) == result
