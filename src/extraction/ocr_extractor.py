"""Tesseract OCR fallback for scanned pages and images.

The Tesseract binary is optional: if it is not installed, :class:`OCRExtractor`
reports itself unavailable and the pipeline surfaces a clear warning instead
of failing silently.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from src.config import Settings
from src.extraction.text_processor import WordBox, assemble_lines

logger = logging.getLogger(__name__)

_WINDOWS_CANDIDATES = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
)


#: Tokens below this confidence with no letters/digits are treated as scanner speckle.
SPECKLE_CONFIDENCE = 60.0
#: A first OCR pass below this mean confidence triggers a stronger-denoise retry.
RETRY_BELOW_CONFIDENCE = 75.0


class OCRUnavailableError(RuntimeError):
    """Raised when OCR is requested but Tesseract is not usable."""


@dataclass(frozen=True)
class OCRResult:
    """Text recognized from one image."""

    text: str
    mean_confidence: float | None  # 0-100, over recognized words
    word_count: int
    words: tuple[tuple[str, float], ...] = ()  # (word, confidence)


def _ocr_quality(result: "OCRResult") -> float:
    """Amount of confidently recognized text: sum of word confidences."""
    return sum(conf for _, conf in result.words) / 100.0


def find_tesseract(configured: str | None) -> str | None:
    """Locate the Tesseract executable: explicit setting, PATH, then common Windows paths."""
    if configured:
        return configured if Path(configured).exists() or shutil.which(configured) else None
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _WINDOWS_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


class OCRExtractor:
    """Wraps pytesseract with preprocessing and confidence reporting."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._unavailable_reason: str | None = None

    @cached_property
    def available(self) -> bool:
        """True if pytesseract is importable and the Tesseract binary runs."""
        try:
            import pytesseract
        except ImportError:
            self._unavailable_reason = "pytesseract is not installed (pip install pytesseract)."
            return False
        cmd = find_tesseract(self.settings.tesseract_cmd)
        if cmd is None:
            self._unavailable_reason = (
                "Tesseract OCR binary not found. Install it (Windows: "
                "`winget install UB-Mannheim.TesseractOCR`) or set TESSERACT_CMD."
            )
            return False
        pytesseract.pytesseract.tesseract_cmd = cmd
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # pytesseract raises several unrelated types
            self._unavailable_reason = f"Tesseract at {cmd!r} could not be executed: {exc}"
            return False
        return True

    @property
    def unavailable_reason(self) -> str | None:
        """Human-readable reason OCR is unavailable (None if available)."""
        return None if self.available else self._unavailable_reason

    @staticmethod
    def preprocess(image: Image.Image, denoise: int = 3) -> Image.Image:
        """Grayscale, auto-contrast and median denoise (kernel ``denoise``)."""
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray, cutoff=1)
        return gray.filter(ImageFilter.MedianFilter(size=denoise)) if denoise > 1 else gray

    def _run(self, image: Image.Image, denoise: int) -> OCRResult:
        import pytesseract

        data = pytesseract.image_to_data(
            self.preprocess(image, denoise),
            lang=self.settings.ocr_language,
            config=f"--psm {self.settings.ocr_psm}",
            output_type=pytesseract.Output.DICT,
        )
        return self._assemble(data)

    def ocr_image(self, image: Image.Image) -> OCRResult:
        """OCR a PIL image, retrying with stronger denoising if the first pass is poor.

        Light denoising preserves thin glyphs on clean scans; heavy speckle
        noise needs a larger median kernel. The second pass only runs when
        the first one is empty or below ``RETRY_BELOW_CONFIDENCE``, and the
        result with more confidently-read words wins.
        """
        if not self.available:
            raise OCRUnavailableError(self.unavailable_reason or "OCR unavailable")
        first = self._run(image, denoise=3)
        if first.text.strip() and (first.mean_confidence or 0) >= RETRY_BELOW_CONFIDENCE:
            return first
        second = self._run(image, denoise=5)
        logger.info("OCR retry with stronger denoise: conf %s -> %s", first.mean_confidence, second.mean_confidence)
        return max((first, second), key=_ocr_quality)

    @staticmethod
    def _assemble(data: dict[str, list]) -> OCRResult:
        """Rebuild lines geometrically and compute mean word confidence.

        Tesseract's own block/line grouping splits two-column forms (labels
        and values) into separate blocks, especially on skewed scans, so
        words are regrouped by position with :func:`assemble_lines`.
        Low-confidence punctuation-only tokens (scanner speckle) are dropped.
        """
        boxes: list[WordBox] = []
        confidences: list[float] = []
        words: list[tuple[str, float]] = []
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            if not word:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < SPECKLE_CONFIDENCE and not any(ch.isalnum() for ch in word):
                continue
            if conf >= 0:
                confidences.append(conf)
                words.append((word, conf))
            left, top = int(data["left"][i]), int(data["top"][i])
            boxes.append(WordBox(left, top, left + int(data["width"][i]), top + int(data["height"][i]), word))

        mean_conf = round(sum(confidences) / len(confidences), 1) if confidences else None
        return OCRResult(text=assemble_lines(boxes), mean_confidence=mean_conf,
                         word_count=len(confidences), words=tuple(words))

    def ocr_pdf_page(self, page: "pymupdf.Page") -> OCRResult:  # noqa: F821
        """Rasterize a PyMuPDF page at the configured DPI and OCR it."""
        import pymupdf

        pix = page.get_pixmap(dpi=self.settings.ocr_dpi, colorspace=pymupdf.csGRAY, alpha=False)
        image = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        return self.ocr_image(image)
