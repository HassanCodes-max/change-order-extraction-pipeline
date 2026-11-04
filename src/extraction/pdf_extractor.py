"""PDF text extraction with PyMuPDF, falling back to OCR per page.

Each page is handled independently: pages with a usable text layer are read
directly; pages that are (nearly) empty — typical of scans — are rasterized
and OCR'd. Mixed documents (typed pages + scanned signature page) therefore
work naturally.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf  # PyMuPDF

from src.config import Settings
from src.extraction.document import DocumentProcessingError, DocumentText, PageText
from src.extraction.ocr_extractor import OCRExtractor, OCRUnavailableError
from src.extraction.text_processor import WordBox, assemble_lines, clean_text, looks_like_garbage

logger = logging.getLogger(__name__)


def reconstruct_lines(page: pymupdf.Page) -> str:
    """Rebuild visual lines from the page's word boxes.

    PyMuPDF's default block ordering separates label/value columns of a
    form into different blocks; grouping words by position keeps
    ``Label: value`` rows together (see :func:`assemble_lines`).
    """
    words = page.get_text("words")
    return assemble_lines([WordBox(w[0], w[1], w[2], w[3], w[4]) for w in words])


class PDFExtractor:
    """Extracts text from PDFs, using OCR for pages without a text layer."""

    def __init__(self, settings: Settings, ocr: OCRExtractor | None = None) -> None:
        self.settings = settings
        self.ocr = ocr or OCRExtractor(settings)

    def extract(
        self,
        source: bytes | str | Path,
        source_name: str | None = None,
        force_ocr: bool = False,
    ) -> DocumentText:
        """Extract per-page text from a PDF given as bytes or a path.

        Args:
            source: PDF bytes or a filesystem path.
            source_name: display name used in the output (defaults to file name).
            force_ocr: OCR every page even if it has a text layer.

        Raises:
            DocumentProcessingError: the PDF is unreadable, encrypted, or
                yields no text at all.
        """
        name = source_name or (Path(source).name if isinstance(source, (str, Path)) else "document.pdf")
        try:
            if isinstance(source, (bytes, bytearray)):
                doc = pymupdf.open(stream=bytes(source), filetype="pdf")
            else:
                doc = pymupdf.open(str(source))
        except Exception as exc:  # PyMuPDF raises FileDataError/RuntimeError/etc.
            raise DocumentProcessingError(
                f"Could not open '{name}' as a PDF ({exc}). The file may be corrupted or not a PDF."
            ) from exc

        with doc:
            if doc.needs_pass:
                raise DocumentProcessingError(
                    f"'{name}' is password-protected. Remove the password and upload it again."
                )
            if doc.page_count == 0:
                raise DocumentProcessingError(f"'{name}' contains no pages.")

            warnings: list[str] = []
            page_limit = min(doc.page_count, self.settings.max_pdf_pages)
            if doc.page_count > page_limit:
                warnings.append(
                    f"Document has {doc.page_count} pages; only the first {page_limit} were processed "
                    "(MAX_PDF_PAGES)."
                )

            pages = [self._extract_page(doc[i], i + 1, force_ocr, warnings) for i in range(page_limit)]

        result = DocumentText(source_name=name, input_type="pdf", pages=pages, warnings=warnings)
        if not result.text.strip():
            reason = self.ocr.unavailable_reason
            hint = f" OCR is unavailable: {reason}" if reason else " OCR found no readable text."
            raise DocumentProcessingError(
                f"No text could be extracted from '{name}'. It appears to be scanned/image-only.{hint}"
            )
        return result

    def _extract_page(
        self, page: pymupdf.Page, number: int, force_ocr: bool, warnings: list[str]
    ) -> PageText:
        text = clean_text(reconstruct_lines(page))
        has_text_layer = len(text.replace(" ", "").replace("\n", "")) >= self.settings.ocr_min_chars_per_page

        if has_text_layer and not force_ocr:
            if looks_like_garbage(text):
                warnings.append(
                    f"Page {number}: the embedded text layer looks garbled (possible bad font encoding "
                    "or a low-quality OCR layer). Consider re-running with 'Force OCR'."
                )
            return PageText(number=number, text=text, method="text")

        if not self.ocr.available:
            warnings.append(
                f"Page {number} has no usable text layer and OCR is unavailable: {self.ocr.unavailable_reason}"
            )
            return PageText(number=number, text=text, method="text" if text else "empty")

        try:
            result = self.ocr.ocr_pdf_page(page)
        except (OCRUnavailableError, RuntimeError, OSError) as exc:
            logger.exception("OCR failed on page %s", number)
            warnings.append(f"Page {number}: OCR failed ({exc}).")
            return PageText(number=number, text=text, method="text" if text else "empty")

        ocr_text = clean_text(result.text)
        if not ocr_text:
            warnings.append(f"Page {number}: OCR produced no text (blank page or unreadable scan).")
            return PageText(number=number, text=text, method="text" if text else "empty")

        conf = result.mean_confidence
        if conf is not None and conf < 70:
            warnings.append(
                f"Page {number}: low OCR confidence ({conf:.0f}/100). Values from this page may contain "
                "recognition errors."
            )
        if looks_like_garbage(ocr_text):
            warnings.append(f"Page {number}: OCR output looks mostly like noise; scan quality may be too poor.")
        return PageText(number=number, text=ocr_text, method="ocr", ocr_confidence=conf,
                        word_confidences=list(result.words))
