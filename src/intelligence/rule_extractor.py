"""Deterministic, offline field extractor based on labels and phrase patterns.

Roles in the system:

* **Demo mode** — the complete pipeline runs without any API key.
* **Fallback** — used automatically if the LLM call fails.
* **Cross-check** — its independent results raise or lower LLM confidence and
  reveal conflicts the LLM may have glossed over.

It understands three layouts:

1. ``Label: value`` lines (and ``Label   value`` columns separated by wide gaps);
2. labels whose value sits on the following line(s) (form fields, headings);
3. common contract prose, e.g. AIA-style "The Contract Sum will be increased
   by this Change Order in the amount of $12,500.00".

It is intentionally conservative: it only reports text it can quote verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.extraction.text_processor import text_similarity
from src.validation.schemas import SPEC_BY_NAME, FieldKind, RawExtraction, RawField
from src.validation.validators import comparison_key

ENGINE_NAME = "rules"
MAX_EVIDENCE_CHARS = 400
MAX_BLOCK_LINES = 12

# ---------------------------------------------------------------------------
# Label vocabulary. Order inside a list does not matter: the longest matching
# label wins, so "Project Number" beats "Project".
# ---------------------------------------------------------------------------

_LABELS: dict[str, list[str]] = {
    "project_name": [r"proj(?:ect)?\.?(?:\s+name|\s+title)?", r"job\s+name", r"project\s*/\s*job"],
    "project_number": [r"project\s*(?:no\.?|number|num\.?|#|id|code)", r"job\s*(?:no\.?|number|#)",
                       r"proj\.?\s*(?:no\.?|#)"],
    "change_order_number": [r"change\s+order\s*(?:no\.?|number|num\.?|#|id)", r"change\s+order(?=\s*:)",
                            r"c\.?\s?o\.?\s*(?:no\.?|number|#)", r"co\s+#"],
    "original_contract_number": [r"(?:original\s+)?contract\s*(?:no\.?|number|#|id|ref(?:erence)?)",
                                 r"agreement\s*(?:no\.?|number|#)", r"subcontract\s*(?:no\.?|number|#)"],
    "contractor_name": [r"(?:general\s+)?contractor(?:\s+name)?", r"from\s*\(contractor\)", r"builder"],
    "client_name": [r"owner(?:\s+name)?", r"client(?:\s+name)?", r"customer(?:\s+name)?",
                    r"to\s*\(owner\)", r"employer"],
    "change_order_date": [r"(?:change\s+order|c\.?o\.?)\s+date", r"date(?:\s+(?:issued|of\s+issue|submitted|prepared))?",
                          r"issue\s+date", r"dated"],
    "description_of_changes": [r"description(?:\s+of\s+(?:the\s+)?(?:proposed\s+)?(?:changes?|work))?",
                               r"scope\s+of\s+(?:the\s+)?(?:change|work)", r"proposed\s+change",
                               r"change\s+description", r"work\s+description"],
    "reason_for_change": [r"reason(?:\s+for\s+(?:the\s+)?change)?", r"justification", r"cause\s+of\s+change",
                          r"basis\s+of\s+change"],
    "original_contract_amount": [r"orig(?:inal|\.)?\s+contract\s+(?:amount|sum|value|price|total)",
                                 r"original\s+contract"],
    "additional_cost": [r"additional\s+(?:cost|amount|work|charges?)", r"cost\s+of\s+(?:this\s+)?change",
                        r"(?:this\s+)?change\s+order\s+(?:amount|total|value|cost)",
                        r"(?:net\s+)?(?:contract\s+)?increase", r"add(?:ition)?\s+amount"],
    "deducted_amount": [r"deduct(?:ed|ion|ions)?(?:\s+amount)?", r"credit(?:\s+amount)?",
                        r"(?:net\s+)?(?:contract\s+)?decrease", r"less\s+credit"],
    "revised_contract_amount": [r"(?:revised|new|adjusted|updated)\s+contract\s+(?:amount|sum|value|price|total)",
                                r"contract\s+(?:sum|amount|value)\s+(?:including|incl\.?|after)\s+(?:this\s+)?change(?:\s+order)?",
                                r"new\s+contract\s+total"],
    "schedule_impact_days": [r"schedule\s+(?:impact|change|extension|adjustment)",
                             r"(?:contract\s+)?time\s+(?:impact|change|adjustment|extension)",
                             r"(?:additional|extension\s+of)\s+(?:calendar\s+|working\s+)?days", r"delay"],
    "requested_completion_date": [r"(?:requested|revised|new|adjusted|substantial|required)\s+completion(?:\s+date)?",
                                  r"completion\s+date"],
    "approval_status": [r"(?:approval\s+)?status", r"approval"],
    "prepared_by": [r"prepared\s+by", r"submitted\s+by", r"originated\s+by", r"requested\s+by"],
    "approved_by": [r"approved\s+by", r"authori[sz]ed\s+by", r"accepted\s+by"],
}
# Auxiliary (not output) values used by validation.
_AUX_LABELS: dict[str, list[str]] = {
    "prior_change_orders_net": [r"(?:net\s+change\s+by\s+)?previous(?:ly)?\s+(?:authorized\s+|approved\s+|executed\s+)?change\s+orders?(?:\s+total)?",
                                r"prior\s+change\s+orders?(?:\s+total)?"],
}

# Optional bullet/numbering, plus stray OCR punctuation before a label.
_PREFIX = r"^\s*['\"~_|`,.]*\s*(?:[-*]\s+|\(?\d{1,2}[.)]\s+|[a-z][.)]\s+)?"
_SEP = r"(?P<sep>\s*[:=]\s*|\s+-{1,2}\s+|\s*\.{3,}\s*|(?<=[#.])\s*|(?<=\bdated)\s+|\s*$)"


def _compile(labels: dict[str, list[str]]) -> dict[str, re.Pattern[str]]:
    return {
        name: re.compile(_PREFIX + r"(?P<label>" + "|".join(f"(?:{p})" for p in pats) + r")" + _SEP + r"(?P<value>.*)$",
                         re.I)
        for name, pats in labels.items()
    }


_FIELD_RES = _compile(_LABELS)
_AUX_RES = _compile(_AUX_LABELS)
_ALL_RES = {**_FIELD_RES, **_AUX_RES}

# ---------------------------------------------------------------------------
# Prose patterns
# ---------------------------------------------------------------------------

_MONEY_TXT = r"\(?-?(?:US\$|USD|\$|€|£)\s?\d[\d,]*(?:\.\d{1,2})?\)?|\(?\d{1,3}(?:,\d{3})+(?:\.\d{2})?\)?"
_DATE_TXT = (r"(?:[A-Z][a-z]{2,8}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
             r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?[A-Z][a-z]{2,8},?\s+\d{4}"
             r"|\d{4}-\d{2}-\d{2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})")
_DAYS_TXT = (r"(?:\w+(?:-\w+)?\s+)?\(?\d*\)?\s*(?:calendar\s+|working\s+|business\s+)?(?:days?|weeks?)")

_PROSE: list[tuple[str, re.Pattern[str]]] = [
    ("change_order_number", re.compile(
        r"\bchange\s+order\s+(?:no\.?\s*|number\s*|#\s*)?(?P<v>(?:[A-Z]{1,4}-?)?\d{1,5}[A-Z]?)\b(?!\s*[/.]\d)", re.I)),
    ("original_contract_amount", re.compile(
        rf"original\s+contract\s+(?:sum|amount|value|price)\s+(?:was|is|of|:)?\s*(?P<v>{_MONEY_TXT})", re.I)),
    ("prior_change_orders_net", re.compile(
        rf"(?:net\s+change\s+by\s+)?previously\s+(?:authorized|approved)\s+change\s+orders?\s*(?:was|is|:|of|totals?)?\s*(?P<v>{_MONEY_TXT})", re.I)),
    ("additional_cost", re.compile(
        rf"contract\s+(?:sum|amount|price)\s+will\s+be\s+increased\s+by\s+(?:this\s+change\s+order\s+)?"
        rf"(?:in\s+the\s+amount\s+of\s+)?(?P<v>{_MONEY_TXT})", re.I)),
    ("additional_cost", re.compile(
        rf"(?:additional\s+(?:cost|charge|amount)\s+of|an?\s+(?:add|increase)\s+of|cost\s+(?:of|for)\s+this\s+(?:change|work)\s+(?:is|will\s+be))\s+(?P<v>{_MONEY_TXT})", re.I)),
    ("deducted_amount", re.compile(
        rf"contract\s+(?:sum|amount|price)\s+will\s+be\s+decreased\s+by\s+(?:this\s+change\s+order\s+)?"
        rf"(?:in\s+the\s+amount\s+of\s+)?(?P<v>{_MONEY_TXT})", re.I)),
    ("deducted_amount", re.compile(rf"(?:a\s+)?credit\s+of\s+(?P<v>{_MONEY_TXT})", re.I)),
    ("revised_contract_amount", re.compile(
        rf"new\s+contract\s+(?:sum|amount|price)\s+(?:including\s+this\s+change\s+order\s+)?(?:will\s+be|is|of)\s+(?P<v>{_MONEY_TXT})", re.I)),
    ("schedule_impact_days", re.compile(
        rf"contract\s+time\s+will\s+be\s+(?:increased|extended|decreased|reduced)\s+by\s+(?P<v>{_DAYS_TXT})", re.I)),
    ("schedule_impact_days", re.compile(
        rf"(?:extend(?:s|ed|ing)?|extension\s+of|delay(?:s|ed)?|add(?:s|ed|ing)?)\s+(?:the\s+)?"
        rf"(?:contract\s+time|schedule|project\s+schedule|completion(?:\s+date)?|project)?\s*(?:by\s+)?(?P<v>{_DAYS_TXT})",
        re.I)),
    ("schedule_impact_days", re.compile(
        r"(?P<v>no\s+(?:change|impact|extension|adjustment)\s+(?:to|on|of|in)\s+(?:the\s+)?(?:contract\s+time|schedule|completion\s+date))",
        re.I)),
    ("schedule_impact_days", re.compile(r"contract\s+time\s+will\s+(?P<v>(?:be\s+)?unchanged)", re.I)),
    ("requested_completion_date", re.compile(
        rf"(?:substantial\s+)?completion(?:\s+date)?\s+(?:will\s+be|is|of|revised\s+to|extended\s+to|moved\s+to|shall\s+be)\s+"
        rf"(?:on\s+or\s+before\s+)?(?P<v>{_DATE_TXT})", re.I)),
]

# A schedule phrase must carry a quantity (digit, number word) or a "no change" marker.
_HAS_QUANTITY = re.compile(
    r"\d|\b(no|unchanged|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|twenty|thirty|forty|sixty|ninety)\b",
    re.I,
)
_CHECKBOX_RE = re.compile(
    r"\[\s*[xX✓✔]\s*\]\s*(?P<v>approved(?:\s+as\s+noted)?|rejected|pending|void|revise\s+and\s+resubmit)", re.I
)
_STAMP_RE = re.compile(r"^\W*(?P<v>APPROVED|REJECTED|VOID|PENDING(?:\s+APPROVAL)?)\W*$")
_CHANGE_ORDER_HINT = re.compile(r"change\s+order|\bc\.?o\.?\s*(?:#|no)|change\s+directive|variation\s+order", re.I)
_SIGNATURE_HINT = re.compile(r"\b(signature|signed|sign|approved\s+by|prepared\s+by|submitted\s+by|accepted\s+by|by:)\b", re.I)
_PLACEHOLDER_RE = re.compile(r"^[\s_.\-/]*$")


@dataclass
class _Candidate:
    value: str
    evidence: str
    line_no: int


def _split_segments(line: str) -> list[str]:
    """Split a line into column segments on runs of 3+ spaces."""
    return [s for s in re.split(r"\s{3,}", line.strip()) if s.strip()]


def _match_label(segment: str, patterns: dict[str, re.Pattern[str]]) -> tuple[str, re.Match[str]] | None:
    """Return the field whose label matches ``segment`` (longest label wins)."""
    best: tuple[str, re.Match[str]] | None = None
    for name, pattern in patterns.items():
        m = pattern.match(segment)
        if m and (best is None or len(m["label"]) > len(best[1]["label"])):
            best = (name, m)
    return best


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _match_label(s, _ALL_RES) is not None:
        return True
    short = len(s) <= 45
    return short and (s.endswith(":") or (s.isupper() and len(s.split()) <= 6))


def _trim(evidence: str) -> str:
    evidence = evidence.strip()
    return evidence if len(evidence) <= MAX_EVIDENCE_CHARS else evidence[: MAX_EVIDENCE_CHARS - 3].rstrip() + "..."


class RuleBasedExtractor:
    """Extracts all fields with regular expressions over the cleaned text."""

    def __init__(self, date_order: str = "MDY") -> None:
        self.date_order = date_order

    def extract(self, text: str) -> RawExtraction:
        lines = text.split("\n")
        candidates: dict[str, list[_Candidate]] = {}

        def add(name: str, value: str, evidence: str, line_no: int) -> None:
            value = value.strip().strip(";,").strip()
            if not value or _PLACEHOLDER_RE.match(value):
                return
            candidates.setdefault(name, []).append(_Candidate(value, _trim(evidence), line_no))

        self._extract_labeled(lines, add)
        self._extract_prose(lines, add)
        self._extract_status_marks(lines, add)

        fields = {name: self._resolve(name, candidates.get(name, [])) for name in SPEC_BY_NAME}
        aux = {name: self._resolve(name, candidates.get(name, [])) for name in _AUX_LABELS}
        return RawExtraction(
            fields=fields,
            auxiliary=aux,
            engine=ENGINE_NAME,
            model=None,
            attempts=0,
            is_change_order=bool(_CHANGE_ORDER_HINT.search(text)),
            document_notes=None,
        )

    # -- strategies ---------------------------------------------------------

    def _extract_labeled(self, lines: list[str], add) -> None:
        for i, line in enumerate(lines):
            segments = _split_segments(line)
            for j, seg in enumerate(segments):
                hit = _match_label(seg, _ALL_RES)
                if hit is None:
                    continue
                name, m = hit
                kind = SPEC_BY_NAME[name].kind if name in SPEC_BY_NAME else FieldKind.MONEY
                value = m["value"].strip()
                evidence_lines = [line.strip()]

                # Generic "Date:" on a signature row is a signature date, not the CO date.
                if name == "change_order_date" and m["label"].lower().startswith("date") and (
                    any(_SIGNATURE_HINT.search(s) for k, s in enumerate(segments) if k != j)
                    or (i > 0 and _SIGNATURE_HINT.search(lines[i - 1]) and not value)
                ):
                    continue

                if not value and j + 1 < len(segments) and _match_label(segments[j + 1], _ALL_RES) is None:
                    value = segments[j + 1]
                    # The neighbouring segment is consumed as this label's value.
                    segments[j + 1] = ""
                elif not value and (":" in (m["sep"] or "") or kind is FieldKind.LONG_TEXT):
                    value, used = self._following_value(lines, i, kind)
                    evidence_lines += used

                if kind is FieldKind.LONG_TEXT and value:
                    # Wrapped paragraphs continue on following non-heading lines.
                    continuation, used = self._continuation(lines, i + len(evidence_lines))
                    if continuation:
                        value = f"{value} {continuation}"
                        evidence_lines += used
                add(name, value, "\n".join(evidence_lines), i)

    def _following_value(self, lines: list[str], i: int, kind: FieldKind) -> tuple[str, list[str]]:
        """Value printed below its label (form layout)."""
        for k in range(i + 1, min(i + 3, len(lines))):
            nxt = lines[k].strip()
            if not nxt:
                continue
            if _is_heading(nxt):
                return "", []
            first = _split_segments(nxt)[0]
            return (nxt if kind is FieldKind.LONG_TEXT else first), lines[i + 1:k + 1]
        return "", []

    @staticmethod
    def _continuation(lines: list[str], start: int) -> tuple[str, list[str]]:
        parts: list[str] = []
        for k in range(start, min(start + MAX_BLOCK_LINES, len(lines))):
            nxt = lines[k].strip()
            if not nxt or _is_heading(nxt):
                break
            parts.append(nxt)
        return " ".join(parts), [lines[k] for k in range(start, start + len(parts))]

    def _extract_prose(self, lines: list[str], add) -> None:
        # Search paragraph-joined text so sentences wrapped across lines still match.
        for i, line in enumerate(lines):
            window = " ".join(part.strip() for part in lines[i:i + 2])
            for name, pattern in _PROSE:
                for m in pattern.finditer(window):
                    # Only report matches that start on this line (avoid duplicates).
                    if m.start() >= len(line.strip()) + 1:
                        continue
                    value = m["v"]
                    if name == "schedule_impact_days" and not _HAS_QUANTITY.search(value):
                        continue
                    add(name, value, m[0], i)

    def _extract_status_marks(self, lines: list[str], add) -> None:
        for i, line in enumerate(lines):
            for m in _CHECKBOX_RE.finditer(line):
                add("approval_status", m["v"], line.strip(), i)
            if (m := _STAMP_RE.match(line.strip())) is not None:
                add("approval_status", m["v"], line.strip(), i)

    # -- resolution -----------------------------------------------------------

    def _resolve(self, name: str, cands: list[_Candidate]) -> RawField:
        """Collapse candidates into one RawField, flagging true conflicts."""
        if not cands:
            return RawField()
        # Auxiliary values are amounts; normalize them like any money field.
        norm_name = name if name in SPEC_BY_NAME else "original_contract_amount"
        kind = SPEC_BY_NAME[norm_name].kind
        distinct: list[_Candidate] = []
        for cand in cands:
            if kind is FieldKind.LONG_TEXT:
                same = any(text_similarity(cand.value, d.value) >= 0.6 or cand.value in d.value
                           for d in distinct)
            else:
                key = comparison_key(norm_name, cand.value, self.date_order)
                same = any(comparison_key(norm_name, d.value, self.date_order) == key for d in distinct)
            if not same:
                distinct.append(cand)

        first = distinct[0]
        if len(distinct) == 1 or kind is FieldKind.LONG_TEXT:
            note = None
            if len(distinct) > 1:
                note = "several passages match this field; the first labeled passage was used"
            return RawField(value=first.value, evidence=first.evidence, status="stated", note=note)
        others = [d.value for d in distinct[1:]]
        return RawField(
            value=first.value,
            evidence=first.evidence,
            status="conflicting",
            alternatives=others,
            note=(f"document states different values for this field: '{first.value}' (line {first.line_no + 1})"
                  + "".join(f", '{d.value}' (line {d.line_no + 1})" for d in distinct[1:])),
        )
