"""In-memory representation of extracted document text, shared by all loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.extraction.text_processor import tokenize

PageMethod = Literal["text", "ocr", "empty"]
InputType = Literal["pdf", "text", "image"]


class DocumentProcessingError(Exception):
    """Raised when a document cannot be read or yields no usable text.

    The message is user-facing and should say what to do next.
    """


@dataclass
class PageText:
    """Text of one page and how it was obtained."""

    number: int  # 1-based
    text: str
    method: PageMethod
    ocr_confidence: float | None = None  # mean Tesseract word confidence, 0-100
    #: (word, confidence 0-100) for OCR pages; lets scoring use the confidence
    #: of the specific words a value came from rather than the page average.
    word_confidences: list[tuple[str, float]] = field(default_factory=list)

    def ocr_confidence_for(self, text: str | None) -> float | None:
        """Mean OCR confidence of the words in ``text`` (page mean as fallback)."""
        if self.method != "ocr":
            return None
        by_token: dict[str, float] = {}
        for word, conf in self.word_confidences:
            for token in tokenize(word):
                by_token[token] = min(conf, by_token.get(token, 100.0))
        matched = [by_token[t] for t in tokenize(text or "") if t in by_token]
        return round(sum(matched) / len(matched), 1) if matched else self.ocr_confidence


@dataclass
class DocumentText:
    """All text extracted from one input, page by page."""

    source_name: str
    input_type: InputType
    pages: list[PageText]
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Full document text, pages separated by a blank line."""
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def page_texts(self) -> list[str]:
        return [p.text for p in self.pages]

    @property
    def ocr_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.method == "ocr"]

    @property
    def text_layer_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.method == "text"]

    @property
    def mean_ocr_confidence(self) -> float | None:
        confs = [p.ocr_confidence for p in self.pages if p.method == "ocr" and p.ocr_confidence is not None]
        return round(sum(confs) / len(confs), 1) if confs else None
