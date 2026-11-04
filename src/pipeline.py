"""End-to-end orchestration: document -> text -> extraction -> validation -> result.

Stages are deliberately separated:

1. **Ingestion** (:mod:`src.extraction`) — PDF text layer / OCR / raw text.
2. **Extraction** (:mod:`src.intelligence`) — LLM (if configured) plus the
   deterministic rule-based pass, reconciled field by field.
3. **Validation** (:mod:`src.validation`) — normalization, grounding,
   cross-field checks, confidence scoring, Pydantic schema validation.
4. **Serialization** (:mod:`src.serialization`) — JSON output.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.config import PROVIDER_RULES, Settings
from src.extraction.document import DocumentProcessingError, DocumentText, PageText
from src.extraction.ocr_extractor import OCRExtractor
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_processor import EvidenceMatch, clean_text, locate_evidence
from src.intelligence.confidence import ConfidenceInputs, score_field
from src.intelligence.llm_extractor import LLMError, LLMExtractor, build_llm_extractor
from src.intelligence.reconcile import Reconciled, reconcile
from src.intelligence.rule_extractor import RuleBasedExtractor
from src.validation.schemas import (
    FIELD_SPECS,
    REQUIRED_FIELDS,
    ChangeOrderFields,
    DocumentInfo,
    ExtractionInfo,
    ExtractionResult,
    FieldKind,
    FieldStatus,
    RawExtraction,
    ResultSummary,
    ValidationIssue,
)
from src.validation.validators import (
    normalize_money,
    normalize_value,
    validate_cross_fields,
    value_supported_by_evidence,
)

logger = logging.getLogger(__name__)

REVIEW_CONFIDENCE_THRESHOLD = 0.6
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_TEXT_SUFFIXES = {".txt", ".text", ".md", ".eml"}


class ChangeOrderPipeline:
    """Processes change-order documents into validated :class:`ExtractionResult` objects."""

    def __init__(
        self,
        settings: Settings | None = None,
        provider: str | None = None,
        llm_extractor: LLMExtractor | None = None,
        ocr: OCRExtractor | None = None,
    ) -> None:
        """
        Args:
            settings: runtime settings (defaults to environment).
            provider: override the provider (``anthropic``, ``openai``, ``rules``).
            llm_extractor: inject a pre-built extractor (used by tests).
            ocr: inject an OCR extractor.
        """
        self.settings = settings or Settings.from_env()
        self.ocr = ocr or OCRExtractor(self.settings)
        self.pdf_extractor = PDFExtractor(self.settings, self.ocr)
        self.rules = RuleBasedExtractor(self.settings.date_order)
        self.provider = llm_extractor.engine if llm_extractor else (provider or self.settings.resolve_provider())
        self._llm = llm_extractor
        self._llm_init_error: str | None = None

    # ------------------------------------------------------------------ inputs

    def load_bytes(self, data: bytes, source_name: str, force_ocr: bool = False) -> DocumentText:
        """Load an uploaded file's bytes, dispatching on the file extension."""
        suffix = Path(source_name).suffix.lower()
        if suffix == ".pdf":
            return self.load_pdf(data, source_name, force_ocr=force_ocr)
        if suffix in _IMAGE_SUFFIXES:
            return self.load_image(data, source_name)
        if suffix in _TEXT_SUFFIXES:
            return self.load_text(data.decode("utf-8", errors="replace"), source_name)
        raise DocumentProcessingError(f"Unsupported file type '{suffix}'. Use PDF, an image, or a .txt file.")

    def load_file(self, path: str | Path, force_ocr: bool = False) -> DocumentText:
        """Load a PDF, image or text file from disk."""
        path = Path(path)
        if not path.is_file():
            raise DocumentProcessingError(f"File not found: {path}")
        return self.load_bytes(path.read_bytes(), path.name, force_ocr=force_ocr)

    def load_pdf(self, data: bytes, source_name: str = "document.pdf", force_ocr: bool = False) -> DocumentText:
        """Extract text from PDF bytes (text layer, with per-page OCR fallback)."""
        self._check_size(len(data), source_name)
        return self.pdf_extractor.extract(data, source_name=source_name, force_ocr=force_ocr)

    def load_image(self, data: bytes, source_name: str = "image") -> DocumentText:
        """OCR a scanned page image (PNG/JPG/TIFF)."""
        self._check_size(len(data), source_name)
        if not self.ocr.available:
            raise DocumentProcessingError(f"Images require OCR, which is unavailable: {self.ocr.unavailable_reason}")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise DocumentProcessingError(f"'{source_name}' is not a readable image: {exc}") from exc
        result = self.ocr.ocr_image(image)
        text = clean_text(result.text)
        if not text:
            raise DocumentProcessingError(f"OCR found no readable text in '{source_name}'.")
        warnings = []
        if result.mean_confidence is not None and result.mean_confidence < 70:
            warnings.append(f"Low OCR confidence ({result.mean_confidence:.0f}/100); values may contain errors.")
        page = PageText(1, text, "ocr", result.mean_confidence, list(result.words))
        return DocumentText(source_name, "image", [page], warnings)

    def load_text(self, text: str, source_name: str = "pasted_text") -> DocumentText:
        """Wrap raw text (pasted or from a .txt file) as a one-page document."""
        self._check_size(len(text.encode("utf-8", errors="ignore")), source_name)
        cleaned = clean_text(text)
        if not cleaned:
            raise DocumentProcessingError("The text input is empty.")
        return DocumentText(source_name, "text", [PageText(1, cleaned, "text")])

    # Convenience wrappers: load + process in one call.

    def process_file(self, path: str | Path, force_ocr: bool = False) -> ExtractionResult:
        """Process a PDF, image or text file from disk."""
        return self.process_document(self.load_file(path, force_ocr=force_ocr))

    def process_pdf(self, data: bytes, source_name: str = "document.pdf", force_ocr: bool = False) -> ExtractionResult:
        """Process PDF bytes."""
        return self.process_document(self.load_pdf(data, source_name, force_ocr=force_ocr))

    def process_image(self, data: bytes, source_name: str = "image") -> ExtractionResult:
        """Process a scanned page image."""
        return self.process_document(self.load_image(data, source_name))

    def process_text(self, text: str, source_name: str = "pasted_text") -> ExtractionResult:
        """Process raw text."""
        return self.process_document(self.load_text(text, source_name))

    def _check_size(self, n_bytes: int, name: str) -> None:
        limit = self.settings.max_upload_mb * 1024 * 1024
        if n_bytes > limit:
            raise DocumentProcessingError(
                f"'{name}' is {n_bytes / 1_048_576:.1f} MB, above the {self.settings.max_upload_mb} MB limit."
            )

    # --------------------------------------------------------------- core

    def _get_llm(self) -> LLMExtractor | None:
        if self.provider == PROVIDER_RULES:
            return None
        if self._llm is None and self._llm_init_error is None:
            try:
                self._llm = build_llm_extractor(self.settings, self.provider)
            except LLMError as exc:
                self._llm_init_error = str(exc)
        return self._llm

    def process_document(self, document: DocumentText) -> ExtractionResult:
        """Run extraction and validation on already-extracted text."""
        started = time.perf_counter()
        text = document.text
        rules_raw = self.rules.extract(text)

        llm_raw: RawExtraction | None = None
        fallback_reason: str | None = None
        attempts = 0
        llm = self._get_llm()
        if self.provider != PROVIDER_RULES:
            if llm is None:
                fallback_reason = f"{self.provider} provider could not be initialised: {self._llm_init_error}"
            else:
                try:
                    llm_raw = llm.extract(text)
                    attempts = llm_raw.attempts
                except LLMError as exc:
                    attempts = llm.last_attempts
                    fallback_reason = f"{llm.engine} ({llm.model}) extraction failed: {exc}"
                    logger.warning("LLM extraction failed; using rule-based fallback: %s", exc)

        decided, prior_raw = reconcile(llm_raw, rules_raw, document.page_texts, self.settings.date_order)
        fields, issues = self._build_fields(document, decided, prior_raw, rules_only=llm_raw is None)

        issues = self._document_issues(document, llm_raw, rules_raw, fallback_reason) + issues
        validated_fields = ChangeOrderFields.model_validate(fields)

        extraction = ExtractionInfo(
            engine=llm_raw.engine if llm_raw else "rules",  # type: ignore[arg-type]
            model=llm_raw.model if llm_raw else None,
            llm_attempts=attempts,
            fallback_used=fallback_reason is not None,
            fallback_reason=fallback_reason,
            cross_checked_with_rules=llm_raw is not None,
            processing_seconds=round(time.perf_counter() - started, 3),
        )
        doc_info = DocumentInfo(
            source_name=document.source_name,
            input_type=document.input_type,
            page_count=len(document.pages),
            text_layer_pages=document.text_layer_pages,
            ocr_pages=document.ocr_pages,
            ocr_mean_confidence=document.mean_ocr_confidence,
            character_count=len(text),
            warnings=list(document.warnings),
        )
        result = ExtractionResult(
            document=doc_info,
            extraction=extraction,
            fields=validated_fields,
            validation_issues=issues,
            summary=_summarize(validated_fields, issues, extraction, doc_info),
        )
        # Round-trip through JSON to guarantee the output is serializable and schema-valid.
        return ExtractionResult.model_validate_json(result.model_dump_json())

    # ------------------------------------------------------------ field build

    def _build_fields(
        self,
        document: DocumentText,
        decided: dict[str, Reconciled],
        prior_raw: Any,
        rules_only: bool,
    ) -> tuple[dict[str, dict[str, Any]], list[ValidationIssue]]:
        order = self.settings.date_order
        pages = document.page_texts
        staged: dict[str, dict[str, Any]] = {}
        inputs: dict[str, ConfidenceInputs] = {}
        values: dict[str, Any] = {}
        issues: list[ValidationIssue] = []

        for spec in FIELD_SPECS:
            rec = decided[spec.name]
            raw = rec.raw
            notes = list(rec.notes) + ([raw.note] if raw.note else [])
            entry: dict[str, Any] = {
                "value": None, "confidence": 0.0, "evidence": raw.evidence, "status": FieldStatus.MISSING,
                "raw_value": raw.value, "source": rec.source, "alternatives": list(raw.alternatives),
                "notes": notes, "confidence_factors": [],
            }
            if raw.value is None:
                entry.update(evidence=None, source="none")
                entry["notes"].append("not found in the document")
                staged[spec.name] = entry
                continue

            norm = normalize_value(spec.name, raw.value, order)
            entry["notes"].extend(norm.notes)
            status = FieldStatus(raw.status)
            if not norm.ok:
                entry["status"] = FieldStatus.INVALID
                staged[spec.name] = entry
                issues.append(ValidationIssue(
                    code="INVALID_VALUE", severity="warning", fields=[spec.name],
                    message=f"{spec.label}: '{raw.value}' could not be normalized ({'; '.join(norm.notes)}). "
                            "Value set to null; the raw text is kept in raw_value.",
                ))
                continue

            values[spec.name] = norm.value
            entry.update(value=norm.value, status=status)
            location = locate_evidence(raw.evidence, pages)
            ocr_conf = None
            if location.page_index is not None:
                ocr_conf = document.pages[location.page_index].ocr_confidence_for(raw.value)
            elif document.ocr_pages:
                ocr_conf = document.mean_ocr_confidence
            inputs[spec.name] = ConfidenceInputs(
                status=status,
                has_value=True,
                evidence_match=location.match,
                value_supported=value_supported_by_evidence(spec.name, norm.value, raw.evidence, order),
                ambiguous=norm.ambiguous,
                converted=norm.converted,
                ocr_confidence=ocr_conf,
                cross_check=rec.cross_check,
                model_confidence=raw.model_confidence,
                rules_only_long_text=rules_only and spec.kind is FieldKind.LONG_TEXT,
            )
            staged[spec.name] = entry
            issues.extend(_field_issues(spec.name, spec.label, status, location.match, rec, norm.ambiguous))

        prior_value = None
        if prior_raw is not None and getattr(prior_raw, "value", None):
            prior_norm = normalize_money(prior_raw.value)
            prior_value = prior_norm.value if prior_norm.ok else None
        cross = validate_cross_fields(values, self.settings.amount_tolerance, prior_value)
        issues.extend(cross.issues)

        for name, inp in inputs.items():
            inp.adjustments = cross.adjustments.get(name, [])
            score, factors = score_field(inp)
            staged[name]["confidence"] = score
            staged[name]["confidence_factors"] = factors
        for entry in staged.values():
            if not entry["confidence_factors"]:
                status = FieldStatus(entry["status"]).value
                entry["confidence_factors"] = [f"no usable value (status: {status}) -> 0.00"]
        return staged, issues

    def _document_issues(
        self,
        document: DocumentText,
        llm_raw: RawExtraction | None,
        rules_raw: RawExtraction,
        fallback_reason: str | None,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if fallback_reason:
            issues.append(ValidationIssue(
                code="LLM_FALLBACK", severity="warning",
                message=f"LLM extraction unavailable; deterministic rule-based extraction was used. {fallback_reason}",
            ))
        elif self.provider == PROVIDER_RULES:
            issues.append(ValidationIssue(
                code="DEMO_MODE", severity="info",
                message="Deterministic rule-based mode (no LLM configured). Layout-dependent: well-labeled "
                        "documents extract reliably; free-form prose may be missed.",
            ))
        is_co = rules_raw.is_change_order
        if llm_raw is not None and llm_raw.is_change_order is not None:
            is_co = llm_raw.is_change_order
        if is_co is False:
            issues.append(ValidationIssue(
                code="NOT_A_CHANGE_ORDER", severity="warning",
                message="This document does not appear to be a construction change order; results may be meaningless.",
            ))
        if llm_raw and llm_raw.document_notes:
            issues.append(ValidationIssue(code="MODEL_NOTE", severity="info",
                                          message=f"Model note: {llm_raw.document_notes}"))
        if document.ocr_pages:
            conf = document.mean_ocr_confidence
            issues.append(ValidationIssue(
                code="OCR_USED", severity="warning" if conf is not None and conf < 70 else "info",
                message=f"OCR was used on page(s) {', '.join(map(str, document.ocr_pages))}"
                        + (f" (mean word confidence {conf:.0f}/100)." if conf is not None else "."),
            ))
        for warning in document.warnings:
            issues.append(ValidationIssue(code="DOCUMENT_WARNING", severity="warning", message=warning))
        return issues


def _field_issues(
    name: str, label: str, status: FieldStatus, match: EvidenceMatch, rec: Reconciled, ambiguous: bool
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if status is FieldStatus.CONFLICTING:
        alts = ", ".join(f"'{a}'" for a in rec.raw.alternatives) or "other values"
        issues.append(ValidationIssue(
            code="CONFLICTING_VALUES", severity="warning", fields=[name],
            message=f"{label}: the document gives conflicting values ('{rec.raw.value}' vs {alts}).",
        ))
    if match is EvidenceMatch.NOT_FOUND:
        issues.append(ValidationIssue(
            code="UNGROUNDED_EVIDENCE", severity="warning", fields=[name],
            message=f"{label}: the supporting evidence could not be found in the document text "
                    "(possible hallucination or OCR mismatch).",
        ))
    elif match is EvidenceMatch.NONE:
        issues.append(ValidationIssue(code="NO_EVIDENCE", severity="warning", fields=[name],
                                      message=f"{label}: no supporting evidence was provided."))
    if rec.cross_check == "disagree":
        issues.append(ValidationIssue(code="EXTRACTOR_DISAGREEMENT", severity="info", fields=[name],
                                      message=f"{label}: LLM and rule-based extraction disagree; "
                                              f"alternatives: {', '.join(rec.raw.alternatives)}."))
    if rec.cross_check == "recovered":
        issues.append(ValidationIssue(code="RECOVERED_BY_RULES", severity="info", fields=[name],
                                      message=f"{label}: missed by the LLM, recovered by rule-based extraction."))
    if status is FieldStatus.INFERRED:
        issues.append(ValidationIssue(code="INFERRED_VALUE", severity="info", fields=[name],
                                      message=f"{label}: value is inferred, not explicitly stated."))
    if ambiguous:
        issues.append(ValidationIssue(code="AMBIGUOUS_DATE", severity="info", fields=[name],
                                      message=f"{label}: numeric date is ambiguous (month/day order)."))
    return issues


def _summarize(
    fields: ChangeOrderFields,
    issues: list[ValidationIssue],
    extraction: ExtractionInfo,
    document: DocumentInfo,
) -> ResultSummary:
    items = [(spec, getattr(fields, spec.name)) for spec in FIELD_SPECS]
    count = lambda status: sum(1 for _, f in items if f.status is status)  # noqa: E731
    extracted = [f for _, f in items if f.value is not None]
    missing_required = [spec.name for spec, f in items if spec.name in REQUIRED_FIELDS and f.value is None]
    low = [spec.label for spec, f in items if f.value is not None and f.confidence < REVIEW_CONFIDENCE_THRESHOLD]

    reasons: list[str] = []
    if missing_required:
        reasons.append(f"missing required field(s): {', '.join(missing_required)}")
    if any(i.code == "CONTRACT_ARITHMETIC_MISMATCH" for i in issues):
        reasons.append("contract amounts are arithmetically inconsistent")
    if count(FieldStatus.CONFLICTING):
        reasons.append(f"{count(FieldStatus.CONFLICTING)} field(s) have conflicting values")
    if count(FieldStatus.INVALID):
        reasons.append(f"{count(FieldStatus.INVALID)} field(s) could not be normalized")
    if low:
        reasons.append(f"low confidence (<{REVIEW_CONFIDENCE_THRESHOLD}): {', '.join(low)}")
    if any(i.code == "UNGROUNDED_EVIDENCE" for i in issues):
        reasons.append("some evidence could not be located in the source text")
    if extraction.fallback_used:
        reasons.append("LLM unavailable; rule-based fallback used")
    if document.ocr_mean_confidence is not None and document.ocr_mean_confidence < 70:
        reasons.append("low OCR quality")
    if any(i.code == "NOT_A_CHANGE_ORDER" for i in issues):
        reasons.append("document may not be a change order")

    return ResultSummary(
        total_fields=len(items),
        extracted=len(extracted),
        stated=count(FieldStatus.STATED),
        inferred=count(FieldStatus.INFERRED),
        conflicting=count(FieldStatus.CONFLICTING),
        invalid=count(FieldStatus.INVALID),
        missing=count(FieldStatus.MISSING),
        missing_required=missing_required,
        overall_confidence=round(mean(f.confidence for f in extracted), 2) if extracted else 0.0,
        needs_review=bool(reasons),
        review_reasons=reasons,
    )
