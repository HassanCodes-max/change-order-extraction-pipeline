"""Text cleaning and evidence-grounding utilities.

Two responsibilities:

1. :func:`clean_text` normalizes messy extracted/OCR'd text (unicode quirks,
   hyphenation, whitespace) while preserving line structure and wide gaps,
   which the rule-based extractor uses as column separators.
2. :func:`locate_evidence` checks whether an evidence quote returned by an
   extractor actually occurs in the source document. This is the primary
   defence against LLM hallucination: ungrounded evidence lowers confidence
   and raises a validation warning.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Sequence

_CHAR_REPLACEMENTS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "•": "-", "●": "-", "▪": "-",
    "☐": "[ ]", "☑": "[x]", "☒": "[x]",
    "­": "",  # soft hyphen
}
_ZERO_WIDTH_RE = re.compile(r"[​-‍﻿]")
_HYPHEN_BREAK_RE = re.compile(r"([a-z])-\n([a-z])")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def clean_text(text: str) -> str:
    """Normalize text while keeping line breaks and wide column gaps.

    * NFKC unicode normalization; smart quotes/dashes/bullets to ASCII.
    * Joins words hyphenated across a line break (``exten-\\nsion``).
    * Tabs become 4 spaces (column gaps are preserved, not collapsed).
    * Trailing whitespace stripped; runs of blank lines collapsed to one.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _CHAR_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    # Only join hyphenation when the continuation is lowercase (avoids
    # merging "CO-\n014" style identifiers split across lines).
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class WordBox:
    """A positioned word from a PDF text layer or OCR."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def assemble_lines(words: Sequence[WordBox]) -> str:
    """Rebuild visual lines from positioned words.

    Words are clustered into rows by vertical centre (tolerant to slight scan
    skew), ordered left-to-right, and joined with a single space — or three
    spaces where the horizontal gap indicates a new column. A vertical gap
    larger than one line height becomes a blank line (paragraph break).

    Both the PDF and OCR paths use this, so ``Label    Value`` form rows stay
    together regardless of how the source engine grouped its blocks.
    """
    words = [w for w in words if w.text.strip()]
    if not words:
        return ""
    heights = sorted(w.y1 - w.y0 for w in words if w.y1 > w.y0)
    med_h = heights[len(heights) // 2] if heights else 10.0
    tolerance = 0.5 * med_h

    rows: list[list[WordBox]] = []
    centers: list[float] = []
    for w in sorted(words, key=lambda w: ((w.y0 + w.y1) / 2, w.x0)):
        c = (w.y0 + w.y1) / 2
        if rows and abs(c - centers[-1]) <= tolerance:
            rows[-1].append(w)
            centers[-1] = sum((x.y0 + x.y1) / 2 for x in rows[-1]) / len(rows[-1])
        else:
            rows.append([w])
            centers.append(c)

    lines: list[str] = []
    prev_bottom: float | None = None
    for row in rows:
        row.sort(key=lambda w: w.x0)
        top = min(w.y0 for w in row)
        if prev_bottom is not None and top - prev_bottom > med_h:
            lines.append("")
        parts = [row[0].text]
        for prev, cur in zip(row, row[1:], strict=False):
            parts.append("   " if cur.x0 - prev.x1 > 1.2 * med_h else " ")
            parts.append(cur.text)
        lines.append("".join(parts))
        prev_bottom = max(w.y1 for w in row)
    return "\n".join(lines)


def normalize_for_match(text: str) -> str:
    """Aggressive normalization used only for comparing strings."""
    text = clean_text(text).lower()
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class EvidenceMatch(str, Enum):
    """How well an evidence quote matches the source text."""

    EXACT = "exact"          # verbatim (after whitespace/quote normalization)
    FUZZY = "fuzzy"          # most tokens appear contiguously (OCR noise, reflow)
    WEAK = "weak"            # partial overlap only
    NOT_FOUND = "not_found"  # evidence does not occur in the document
    NONE = "none"            # no evidence supplied


@dataclass(frozen=True)
class EvidenceLocation:
    """Result of grounding an evidence quote in the source document."""

    match: EvidenceMatch
    score: float              # 0..1 fraction of evidence tokens matched contiguously
    page_index: int | None    # 0-based page where it was best matched


FUZZY_THRESHOLD = 0.85
WEAK_THRESHOLD = 0.5


def _token_coverage(evidence_tokens: list[str], page_tokens: list[str]) -> float:
    """Fraction of evidence tokens covered by matching blocks, in order."""
    if not evidence_tokens:
        return 0.0
    matcher = SequenceMatcher(None, page_tokens, evidence_tokens, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(evidence_tokens)


def locate_evidence(evidence: str | None, pages: Sequence[str]) -> EvidenceLocation:
    """Ground ``evidence`` against the document ``pages``.

    Returns the best match across pages. Matching is whitespace/case/quote
    insensitive; a fuzzy token-level fallback tolerates OCR errors and
    line reflow.
    """
    if not evidence or not evidence.strip():
        return EvidenceLocation(EvidenceMatch.NONE, 0.0, None)

    needle = normalize_for_match(evidence)
    # Evidence quotes sometimes contain an ellipsis joining two fragments.
    fragments = [f.strip() for f in re.split(r"\s*(?:\.\.\.|…)\s*", needle) if f.strip()]

    ev_tokens = tokenize(evidence)
    best_score, best_page = 0.0, None
    for idx, page in enumerate(pages):
        haystack = normalize_for_match(page)
        if fragments and all(f in haystack for f in fragments):
            return EvidenceLocation(EvidenceMatch.EXACT, 1.0, idx)
        if not ev_tokens:
            continue
        score = _token_coverage(ev_tokens, tokenize(page))
        if score > best_score:
            best_score, best_page = score, idx

    if best_score >= FUZZY_THRESHOLD:
        return EvidenceLocation(EvidenceMatch.FUZZY, best_score, best_page)
    if best_score >= WEAK_THRESHOLD:
        return EvidenceLocation(EvidenceMatch.WEAK, best_score, best_page)
    return EvidenceLocation(EvidenceMatch.NOT_FOUND, best_score, best_page)


def text_similarity(a: str, b: str) -> float:
    """Token-level similarity in [0, 1] (Dice coefficient over token sets)."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


def looks_like_garbage(text: str) -> bool:
    """Heuristic: is this text mostly OCR noise rather than language?

    Flags text where under 50% of tokens look like real words or numbers.
    """
    tokens = re.findall(r"\S+", text)
    if len(tokens) < 5:
        return False
    plausible = sum(
        1 for t in tokens
        if re.fullmatch(r"[A-Za-z]{2,}[.,:;)]?|[\$\(]?[\d,.\-/]+[.,:;)%]?|[A-Za-z]+-\d+|&|-", t)
    )
    return plausible / len(tokens) < 0.5
