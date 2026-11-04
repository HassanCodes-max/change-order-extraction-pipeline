"""Tests for document ingestion (PDF, OCR, text) and the rule-based extractor."""

from __future__ import annotations

import pymupdf
import pytest

from scripts.generate_samples import (
    CO_007_SCANNED,
    CO_014_STANDARD,
    CO_031_CONFLICTING,
    CO_045_MESSY,
    make_scanned_pdf,
    make_text_pdf,
)
from src.extraction.document import DocumentProcessingError
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_processor import (
    EvidenceMatch,
    WordBox,
    assemble_lines,
    clean_text,
    locate_evidence,
    looks_like_garbage,
)
from src.intelligence.rule_extractor import RuleBasedExtractor
from tests.conftest import requires_ocr


class _NoOCR:
    """Stand-in for an OCR engine that is not installed."""

    available = False
    unavailable_reason = "Tesseract not installed (test stub)"


# --------------------------------------------------------------------------- text processing


def test_clean_text_normalizes_unicode_and_hyphenation():
    raw = "“Owner” request – exten-\nsion of\ttime\r\n\n\n\nNext"
    assert clean_text(raw) == '"Owner" request - extension of    time\n\nNext'


def test_clean_text_keeps_identifier_hyphen_breaks():
    assert clean_text("CO-\n014") == "CO-\n014"


def test_locate_evidence_exact_fuzzy_and_missing():
    pages = ["Change Order No:   CO-014\nAdditional Cost:   $12,500.00"]
    assert locate_evidence("change order no: CO-014", pages).match is EvidenceMatch.EXACT
    # OCR-style corruption of one token ("0rder") still grounds as fuzzy.
    fuzzy = locate_evidence("Change 0rder No: CO-014 Additional Cost: $12,500.00", pages)
    assert fuzzy.match is EvidenceMatch.FUZZY and fuzzy.score >= 0.85
    # Only a fragment overlapping: weak.
    assert locate_evidence("Change Order No CO-099 for the new canopy work", pages).match in (
        EvidenceMatch.WEAK, EvidenceMatch.NOT_FOUND)
    assert locate_evidence("Deducted Amount: $4,000", pages).match is EvidenceMatch.NOT_FOUND
    assert locate_evidence(None, pages).match is EvidenceMatch.NONE


def test_assemble_lines_keeps_columns_on_one_row_despite_skew():
    words = [WordBox(10, 100, 60, 110, "Owner:"), WordBox(200, 103, 260, 113, "ABC"),
             WordBox(265, 103, 300, 113, "Corp"), WordBox(10, 120, 60, 130, "Date:")]
    assert assemble_lines(words) == "Owner:   ABC Corp\nDate:"


def test_garbage_detection():
    assert looks_like_garbage("~~ ;;| ,.,. #@! ^^ %% &* () {} []")
    assert not looks_like_garbage("Change Order No CO-014 dated March 3, 2025 for $12,500.00")


# --------------------------------------------------------------------------- PDF extraction


def test_text_pdf_extraction_uses_text_layer(settings, tmp_path):
    pdf = make_text_pdf(CO_014_STANDARD)
    doc = PDFExtractor(settings, ocr=_NoOCR()).extract(pdf, "co.pdf")
    assert doc.ocr_pages == [] and doc.text_layer_pages == [1]
    # Label and value columns are reconstructed onto one line.
    assert "Change Order No.:   CO-014" in doc.text
    assert "Revised Contract Amount:   $1,262,500.00" in doc.text


def test_scanned_pdf_without_ocr_gives_actionable_error(settings):
    pdf = make_scanned_pdf(CO_007_SCANNED)
    with pytest.raises(DocumentProcessingError, match="scanned/image-only.*Tesseract"):
        PDFExtractor(settings, ocr=_NoOCR()).extract(pdf, "scan.pdf")


@requires_ocr
def test_scanned_pdf_ocr(settings):
    pdf = make_scanned_pdf(CO_007_SCANNED)
    doc = PDFExtractor(settings).extract(pdf, "scan.pdf")
    assert doc.ocr_pages == [1]
    assert doc.mean_ocr_confidence is not None and doc.mean_ocr_confidence > 70
    assert "CO-007" in doc.text
    assert "$858,950.00" in doc.text
    page = doc.pages[0]
    assert page.word_confidences and page.ocr_confidence_for("CO-007") is not None


@requires_ocr
def test_poor_quality_scan_reports_lower_confidence(settings):
    clean = PDFExtractor(settings).extract(make_scanned_pdf(CO_007_SCANNED), "a.pdf")
    noisy = PDFExtractor(settings).extract(
        make_scanned_pdf(CO_007_SCANNED, dpi=150, noise=0.05, blur=0.9, rotation=1.2), "b.pdf")
    assert noisy.mean_ocr_confidence < clean.mean_ocr_confidence
    assert "CO-007" in noisy.text  # recovered by the stronger-denoise retry


@requires_ocr
def test_unreadable_scan_is_never_silent(settings):
    pdf = make_scanned_pdf(CO_007_SCANNED, dpi=60, noise=0.35, blur=2.5, rotation=3)
    try:
        doc = PDFExtractor(settings).extract(pdf, "c.pdf")
    except DocumentProcessingError as exc:
        assert "No text could be extracted" in str(exc)
    else:
        assert doc.warnings, "fragments from an unreadable scan must carry quality warnings"


@requires_ocr
def test_force_ocr_on_text_pdf(settings):
    doc = PDFExtractor(settings).extract(make_text_pdf(CO_014_STANDARD), "co.pdf", force_ocr=True)
    assert doc.ocr_pages == [1]
    assert "CO-014" in doc.text


def test_corrupt_pdf_raises(settings):
    with pytest.raises(DocumentProcessingError, match="Could not open"):
        PDFExtractor(settings, ocr=_NoOCR()).extract(b"%PDF-1.7 this is not really a pdf", "bad.pdf")


def test_encrypted_pdf_raises(settings, tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Change Order No: CO-1")
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
    doc.close()
    with pytest.raises(DocumentProcessingError, match="password-protected"):
        PDFExtractor(settings, ocr=_NoOCR()).extract(path)


# --------------------------------------------------------------------------- rule-based extractor


def _rules(text: str):
    return RuleBasedExtractor().extract(clean_text(text.replace("\t", "    ")))


def test_rules_extract_standard_form():
    r = _rules(CO_014_STANDARD)
    expected = {
        "project_name": "ABC Commercial Building",
        "project_number": "2024-118",
        "change_order_number": "CO-014",
        "original_contract_number": "GC-2024-0457",
        "contractor_name": "Summit Ridge Builders, LLC",
        "client_name": "ABC Holdings Inc.",
        "change_order_date": "March 3, 2025",
        "additional_cost": "$12,500.00",
        "revised_contract_amount": "$1,262,500.00",
        "approval_status": "Approved",
        "approved_by": "Robert Smith, Owner's Representative",
    }
    for name, value in expected.items():
        assert r.get(name).value == value, name
        assert r.get(name).status == "stated"
    assert r.get("description_of_changes").value.startswith("Furnish and install twelve (12)")
    assert "LP-2" in r.get("description_of_changes").value  # multi-line block captured
    assert r.is_change_order is True


def test_rules_handle_messy_formatting():
    r = _rules(CO_045_MESSY)
    assert r.get("project_name").value == "harbor view condos - phase 2"
    assert r.get("project_number").value == "HV-PH2-17"
    assert r.get("change_order_date").value == "3rd of March 2025"
    assert r.get("contractor_name").value == "Atlas Building Group"
    assert r.get("deducted_amount").value == "(3,200.00)"
    assert r.get("schedule_impact_days").value == "two weeks"
    assert "trench drain" in r.get("description_of_changes").value


def test_rules_detect_conflicting_values():
    r = _rules(CO_031_CONFLICTING)
    co = r.get("change_order_number")
    assert co.status == "conflicting"
    assert {co.value, *co.alternatives} == {"CO-031", "CO-013"}
    days = r.get("schedule_impact_days")
    assert days.status == "conflicting"
    # Blank signature line is not treated as an approver.
    assert r.get("approved_by").value is None


def test_rules_skip_signature_dates_and_read_checkboxes():
    text = ("Change Order No: 7\nDate: 01/10/2025\nApproved by: J. Ruiz     Date: 01/20/2025\n"
            "[ ] Rejected   [x] Approved as noted")
    r = _rules(text)
    date_field = r.get("change_order_date")
    assert date_field.value == "01/10/2025" and date_field.status == "stated"
    assert r.get("approval_status").value.lower() == "approved as noted"


def test_rules_report_missing_rather_than_guess():
    r = _rules("Change Order No: 3\nProject: Pine St Garage\nAdditional cost: $1,200")
    for name in ("client_name", "revised_contract_amount", "approved_by", "change_order_date"):
        assert r.get(name).value is None and r.get(name).status == "missing"


def test_rules_aia_prose():
    text = ("The original Contract Sum was $100,000.00. Net change by previously authorized Change "
            "Orders: $5,000.00. The Contract Sum will be increased by this Change Order in the amount of "
            "$2,500.00. The new Contract Sum including this Change Order will be $107,500.00. "
            "The Contract Time will be increased by seven (7) days.")
    r = _rules(text)
    assert r.get("original_contract_amount").value == "$100,000.00"
    assert r.get("additional_cost").value == "$2,500.00"
    assert r.get("revised_contract_amount").value == "$107,500.00"
    assert "7" in r.get("schedule_impact_days").value
    assert r.auxiliary["prior_change_orders_net"].value == "$5,000.00"
