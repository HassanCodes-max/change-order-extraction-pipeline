"""Deterministic normalization and cross-field validation.

Normalizers turn raw extractor strings into typed values and *report* what
they did (notes) instead of silently rewriting data. Cross-field checks never
modify values; they emit :class:`ValidationIssue` objects and per-field
confidence penalties.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from src.extraction.text_processor import normalize_for_match, tokenize
from src.validation.schemas import (
    REQUIRED_FIELDS,
    SPEC_BY_NAME,
    ApprovalStatus,
    FieldKind,
    ValidationIssue,
)


@dataclass
class Normalized:
    """Outcome of normalizing one raw value."""

    value: Any = None
    ok: bool = False
    notes: list[str] = field(default_factory=list)
    ambiguous: bool = False   # parsed, but another reading was possible
    converted: bool = False   # unit conversion or interpretation applied (weeks->days, "None"->0)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


_EDGE_JUNK_RE = re.compile(r"^[\s'\"~_|`*,;:.-]+(?=\w)|(?<=[\w.)])[\s'\"~_|`*,;:-]+$")


def normalize_text(raw: str) -> Normalized:
    """Collapse whitespace; strip label punctuation, stray OCR marks and placeholders."""
    value = re.sub(r"\s+", " ", raw).strip()
    if not value or re.fullmatch(r"[_\-.\s]+", value):
        return Normalized(ok=False, notes=["value is only a blank placeholder (e.g. '______')"])
    cleaned = _EDGE_JUNK_RE.sub("", value)
    cleaned = re.sub(r"\.{2,}$", ".", cleaned).strip()
    notes = []
    if cleaned != value.strip(":;, "):
        notes.append(f"stray leading/trailing characters removed from '{value}'")
    if not cleaned:
        return Normalized(ok=False, notes=[f"'{raw}' contains no usable text"])
    return Normalized(value=cleaned, ok=True, notes=notes)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_RE = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_NUMERIC_RE = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})\b")
_MDY_TEXT_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DMY_TEXT_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+of\s+|of\s+)?({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.I
)
_DATE_PATTERNS = (_ISO_RE, _MDY_TEXT_RE, _DMY_TEXT_RE, _NUMERIC_RE)


def _expand_year(y: int) -> int:
    return y + (2000 if y < 70 else 1900) if y < 100 else y


def _build_date(y: int, m: int, d: int, raw: str) -> Normalized:
    try:
        return Normalized(value=date(y, m, d), ok=True)
    except ValueError:
        return Normalized(ok=False, notes=[f"'{raw}' is not a valid calendar date"])


def normalize_date(raw: str, date_order: str = "MDY") -> Normalized:
    """Parse common date formats into a :class:`datetime.date`.

    Supports ISO (2025-03-14), numeric (03/14/2025, 14.03.25), and written
    forms (March 14, 2025 / 14th March 2025 / 14 day of March, 2025).
    Numeric dates where both parts are <= 12 are read using ``date_order``
    and flagged as ambiguous.
    """
    text = raw.strip()
    if m := _ISO_RE.search(text):
        return _build_date(int(m[1]), int(m[2]), int(m[3]), m[0])
    if m := _MDY_TEXT_RE.search(text):
        return _build_date(int(m[3]), _MONTHS[m[1].lower()], int(m[2]), m[0])
    if m := _DMY_TEXT_RE.search(text):
        return _build_date(int(m[3]), _MONTHS[m[2].lower()], int(m[1]), m[0])
    if m := _NUMERIC_RE.search(text):
        a, b, y = int(m[1]), int(m[2]), _expand_year(int(m[3]))
        if a > 12 and b <= 12:
            return _build_date(y, b, a, m[0])  # unambiguously D/M/Y
        if b > 12 and a <= 12:
            return _build_date(y, a, b, m[0])  # unambiguously M/D/Y
        month, day = (a, b) if date_order == "MDY" else (b, a)
        result = _build_date(y, month, day, m[0])
        if result.ok and a != b:
            result.ambiguous = True
            result.notes.append(
                f"'{m[0]}' is ambiguous (month/day order); interpreted as {date_order} per DATE_ORDER"
            )
        return result
    if re.search(rf"\b({_MONTH_ALT})\b", text, re.I) and re.search(r"\b\d{4}\b", text):
        return Normalized(ok=False, notes=[f"'{raw}' has a month and year but no day"])
    return Normalized(ok=False, notes=[f"could not parse '{raw}' as a date"])


def find_dates(text: str, date_order: str = "MDY") -> list[date]:
    """All parseable dates mentioned in ``text``."""
    found: list[date] = []
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            n = normalize_date(m[0], date_order)
            if n.ok:
                found.append(n.value)
    return found


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

# A minus sign only counts when it touches the amount ("-$1,200", "$-1,200");
# a spaced dash is usually a separator ("Additional cost - $12,500").
_MONEY_RE = re.compile(
    r"(?:(?P<paren>\()\s*|(?P<minus>-))?(?P<cur>US\$|USD|CAD|C\$|EUR|GBP|\$|€|£)?\s*(?P<neg2>-)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d{1,3}(?:\.\d{3})+,\d{1,2}|\d+(?:[.,]\d{1,2})?)"
    r"(?P<mult>\s*(?:k|K|thousand|million|M)\b)?(?P<close>\s*\))?"
)
_CURRENCY_CODES = {"US$": "USD", "USD": "USD", "$": "USD", "CAD": "CAD", "C$": "CAD",
                   "EUR": "EUR", "€": "EUR", "GBP": "GBP", "£": "GBP"}
_ZERO_WORDS = re.compile(r"^\s*(none|nil|zero|no\s+(?:cost|charge|change|deduction|credit)|-+)\s*\.?$", re.I)


@dataclass(frozen=True)
class MoneyMatch:
    amount: Decimal
    negative: bool
    currency: str | None
    text: str


def _parse_number(num: str) -> Decimal:
    if "," in num and "." in num:
        # Whichever separator comes last is the decimal separator.
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")
        else:
            num = num.replace(",", "")
    elif "," in num:
        num = num.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", num) else num.replace(",", ".")
    return Decimal(num)


def find_money(text: str) -> list[MoneyMatch]:
    """All monetary amounts in ``text``.

    Only numbers with a currency marker, thousands separators or cents are
    treated as money, so day counts and item numbers are not mistaken for amounts.
    """
    out: list[MoneyMatch] = []
    for m in _MONEY_RE.finditer(text):
        num = m["num"]
        has_marker = bool(m["cur"]) or "," in num or "." in num or bool(m["mult"])
        if not has_marker:
            continue
        try:
            amount = _parse_number(num)
        except InvalidOperation:
            continue
        mult = (m["mult"] or "").strip().lower()
        if mult in {"k", "thousand"}:
            amount *= 1000
        elif mult in {"m", "million"}:
            amount *= 1_000_000
        negative = bool(m["neg2"]) or bool(m["minus"]) or (bool(m["paren"]) and bool(m["close"]))
        out.append(MoneyMatch(amount, negative, _CURRENCY_CODES.get(m["cur"] or ""), m[0].strip()))
    return out


def normalize_money(raw: str) -> Normalized:
    """Parse a currency amount into a float rounded to cents.

    Negative or parenthesized amounts are returned negative; callers decide
    what a negative means for their field. Words like "None" map to 0.
    """
    if _ZERO_WORDS.match(raw):
        return Normalized(value=0.0, ok=True, converted=True, notes=[f"'{raw.strip()}' interpreted as 0.00"])
    matches = find_money(raw)
    if not matches:
        # Fall back to a bare integer such as "12500".
        bare = re.fullmatch(r"\s*\$?\s*(\d+)\s*", raw)
        if bare:
            matches = [MoneyMatch(Decimal(bare[1]), False, "USD" if "$" in raw else None, raw.strip())]
        else:
            return Normalized(ok=False, notes=[f"no numeric amount found in '{raw}'"])
    first = matches[0]
    notes: list[str] = []
    if len({m.amount for m in matches}) > 1:
        notes.append(f"multiple amounts in value ({', '.join(m.text for m in matches)}); first was used")
    if first.currency and first.currency != "USD":
        notes.append(f"currency is {first.currency} (not USD)")
    if re.search(r"\d\s*(?:k|K|thousand|million|M)\b", first.text):
        notes.append(f"shorthand '{first.text}' expanded")
    value = float((-first.amount if first.negative else first.amount).quantize(Decimal("0.01")))
    return Normalized(value=value, ok=True, notes=notes)


# ---------------------------------------------------------------------------
# Schedule impact (days)
# ---------------------------------------------------------------------------

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_NUMBER_WORD_RE = re.compile(
    r"\b(?:(?P<tens>" + "|".join(_TENS) + r")(?:[\s-](?P<unit1>" + "|".join(_UNITS) + r"))?"
    r"|(?P<unit>" + "|".join(_UNITS) + r"))\b",
    re.I,
)
_NO_IMPACT_RE = re.compile(
    r"\b(no\s+(?:change|impact|schedule\s+impact|time\s+extension|extension|additional\s+(?:time|days)|"
    r"delay)|none|zero\s+days|not\s+affected|unchanged)\b",
    re.I,
)
_REDUCTION_RE = re.compile(r"\b(reduc|decreas|shorten|accelerat|earlier|less|minus)", re.I)


def _word_number(text: str) -> int | None:
    m = _NUMBER_WORD_RE.search(text)
    if not m:
        return None
    if m["tens"]:
        return _TENS[m["tens"].lower()] + (_UNITS[m["unit1"].lower()] if m["unit1"] else 0)
    return _UNITS[m["unit"].lower()]


def normalize_days(raw: str) -> Normalized:
    """Convert a schedule-impact phrase into a signed number of days.

    Handles digits, number words ("five (5) calendar days"), weeks (x7),
    explicit "no change" (0) and reductions (negative). Months are rejected
    because they cannot be converted to days without calendar context.
    """
    text = raw.strip()
    notes: list[str] = []
    converted = False
    sign = -1 if (_REDUCTION_RE.search(text) or re.match(r"^\s*-\s*\d", text)) else 1

    m = re.search(r"(\d+)\s*(?:\)\s*)?(calendar|working|business|work)?\s*(day|week|month)s?\b", text, re.I)
    if m:
        number, qualifier, unit = int(m[1]), (m[2] or "").lower(), m[3].lower()
    else:
        number = _word_number(text)
        um = re.search(r"\b(calendar|working|business|work)?\s*(day|week|month)s?\b", text, re.I)
        qualifier = (um[1] or "").lower() if um else ""
        unit = um[2].lower() if um else None
        if number is None:
            bare = re.fullmatch(r"\s*[+-]?\s*(\d+)\s*", text)
            if bare:
                number, unit = int(bare[1]), "day"
                notes.append("bare number assumed to be days")
            elif _NO_IMPACT_RE.search(text):
                return Normalized(value=0, ok=True, converted=True, notes=[f"'{text}' interpreted as 0 days"])
            else:
                return Normalized(ok=False, notes=[f"no day count found in '{text}'"])
        if unit is None:
            unit = "day"
            notes.append("no unit given; assumed days")

    if unit == "month":
        return Normalized(ok=False, notes=[f"'{text}' is in months; cannot convert to days without a calendar"])
    if unit == "week":
        number *= 7
        converted = True
        notes.append("converted weeks to days (x7)")
    if qualifier in {"working", "business", "work"}:
        notes.append(f"stated in {qualifier} days; not converted to calendar days")
    if sign < 0:
        notes.append("interpreted as a schedule reduction (negative)")
    return Normalized(value=sign * number, ok=True, notes=notes, converted=converted)


# ---------------------------------------------------------------------------
# Approval status
# ---------------------------------------------------------------------------

_STATUS_RULES: tuple[tuple[re.Pattern[str], ApprovalStatus], ...] = (
    (re.compile(r"\b(void|voided|withdrawn|cancel+ed|rescinded)\b", re.I), ApprovalStatus.VOID),
    (re.compile(r"\b(reject\w*|denied|declined|disapproved|not\s+approved)\b", re.I), ApprovalStatus.REJECTED),
    (re.compile(r"\b(pending|awaiting|submitted|under\s+review|for\s+(?:approval|review)|"
                r"not\s+yet\s+approved|draft|proposed|in\s+review|open)\b", re.I), ApprovalStatus.PENDING),
    (re.compile(r"\b(approved\s+(?:as\s+noted|with\s+(?:conditions|comments|changes))|conditional\w*)\b",
                re.I), ApprovalStatus.APPROVED_WITH_CONDITIONS),
    (re.compile(r"\b(approved|accepted|executed|authori[sz]ed|signed)\b", re.I), ApprovalStatus.APPROVED),
)


def normalize_status(raw: str) -> Normalized:
    """Map free-text approval wording onto :class:`ApprovalStatus`."""
    # Checkbox style: "[x] Approved  [ ] Rejected" -> only the ticked option counts.
    ticked = re.findall(r"\[\s*[xX✓✔]\s*\]\s*([A-Za-z][A-Za-z ]{2,30})", raw)
    candidates = ticked or [raw]
    for text in candidates:
        for pattern, status in _STATUS_RULES:
            if pattern.search(text):
                return Normalized(value=status, ok=True)
    return Normalized(ok=False, notes=[f"approval wording '{raw}' does not map to a known status"])


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def normalize_value(field_name: str, raw: str, date_order: str = "MDY") -> Normalized:
    """Normalize ``raw`` according to the kind of ``field_name``."""
    kind = SPEC_BY_NAME[field_name].kind
    if kind in (FieldKind.TEXT, FieldKind.LONG_TEXT):
        return normalize_text(raw)
    if kind is FieldKind.DATE:
        return normalize_date(raw, date_order)
    if kind is FieldKind.MONEY:
        result = normalize_money(raw)
        if result.ok and result.value < 0 and field_name == "deducted_amount":
            result.value = abs(result.value)
            result.notes.append("deduction written as a negative/credit; stored as a positive magnitude")
        return result
    if kind is FieldKind.DAYS:
        return normalize_days(raw)
    if kind is FieldKind.STATUS:
        return normalize_status(raw)
    raise ValueError(f"unknown field kind for {field_name}")  # pragma: no cover


def comparison_key(field_name: str, raw: str, date_order: str = "MDY") -> Any:
    """A normalized key used to decide whether two raw values agree."""
    n = normalize_value(field_name, raw, date_order)
    if n.ok:
        return n.value
    return normalize_for_match(raw)


def value_supported_by_evidence(
    field_name: str, value: Any, evidence: str | None, date_order: str = "MDY"
) -> bool | None:
    """Does the evidence quote actually contain the normalized value?

    Returns None when there is no evidence to check.
    """
    if not evidence or value is None:
        return None
    kind = SPEC_BY_NAME[field_name].kind
    if kind in (FieldKind.TEXT, FieldKind.LONG_TEXT):
        v_tokens = tokenize(str(value))
        e_tokens = set(tokenize(evidence))
        if not v_tokens:
            return None
        return sum(t in e_tokens for t in v_tokens) / len(v_tokens) >= 0.8
    if kind is FieldKind.DATE:
        return value in find_dates(evidence, date_order)
    if kind is FieldKind.MONEY:
        amounts = {float(m.amount.quantize(Decimal("0.01"))) for m in find_money(evidence)}
        if _ZERO_WORDS.search(evidence) or re.search(r"\bnone\b", evidence, re.I):
            amounts.add(0.0)
        return abs(value) in amounts or value in amounts
    if kind is FieldKind.DAYS:
        n = normalize_days(evidence)
        return n.ok and abs(n.value) == abs(value)
    if kind is FieldKind.STATUS:
        n = normalize_status(evidence)
        return n.ok and n.value == value
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Cross-field validation
# ---------------------------------------------------------------------------


@dataclass
class CrossFieldReport:
    """Issues plus per-field multiplicative confidence adjustments."""

    issues: list[ValidationIssue] = field(default_factory=list)
    adjustments: dict[str, list[tuple[float, str]]] = field(default_factory=dict)

    def adjust(self, fields: list[str], factor: float, reason: str) -> None:
        for name in fields:
            self.adjustments.setdefault(name, []).append((factor, reason))

    def issue(self, code: str, severity: str, message: str, fields: list[str]) -> None:
        self.issues.append(ValidationIssue(code=code, severity=severity, message=message, fields=fields))


def _fmt(amount: Decimal) -> str:
    return f"${amount:,.2f}"


def validate_cross_fields(
    values: dict[str, Any],
    tolerance: float = 0.01,
    prior_change_orders_net: float | None = None,
) -> CrossFieldReport:
    """Run consistency checks across fields.

    ``values`` maps field name -> normalized value (None if absent). Values
    are never modified here; disagreements are reported and penalized.
    ``prior_change_orders_net`` (net of previously approved change orders, if
    the document lists it) lets the arithmetic check handle AIA-style
    documents where revised = original + prior changes + this change.
    """
    report = CrossFieldReport()
    tol = Decimal(str(tolerance))
    prior = Decimal(str(prior_change_orders_net)) if prior_change_orders_net is not None else None

    # 1. Required fields.
    for name in sorted(REQUIRED_FIELDS):
        if values.get(name) is None:
            report.issue("REQUIRED_FIELD_MISSING", "error",
                         f"Required field '{SPEC_BY_NAME[name].label}' was not found or could not be parsed.",
                         [name])

    # 2. Contract arithmetic: original + additional - deducted == revised.
    def dec(name: str) -> Decimal | None:
        v = values.get(name)
        return None if v is None else Decimal(str(v))

    original, added, deducted, revised = (dec(n) for n in
                                          ("original_contract_amount", "additional_cost",
                                           "deducted_amount", "revised_contract_amount"))
    money_fields = ["original_contract_amount", "additional_cost", "deducted_amount", "revised_contract_amount"]
    if original is not None and revised is not None and (added is not None or deducted is not None):
        expected = original + (added or Decimal(0)) - (deducted or Decimal(0))
        assumed = [SPEC_BY_NAME[n].label for n, v in (("additional_cost", added), ("deducted_amount", deducted))
                   if v is None]
        assumption = f" (assuming {', '.join(assumed)} = 0)" if assumed else ""
        present_money = [n for n in money_fields if values.get(n) is not None]
        if abs(expected - revised) <= tol:
            report.issue("CONTRACT_ARITHMETIC_OK", "info",
                         f"Revised contract amount {_fmt(revised)} equals original {_fmt(original)} "
                         f"+ additions - deductions{assumption}.", money_fields)
            report.adjust(present_money, 1.05, "cross-field: contract arithmetic consistent")
        elif prior is not None and abs(expected + prior - revised) <= tol:
            report.issue("CONTRACT_ARITHMETIC_OK", "info",
                         f"Revised contract amount {_fmt(revised)} equals original {_fmt(original)} "
                         f"+ previously approved change orders {_fmt(prior)} + additions - deductions"
                         f"{assumption}.", money_fields)
            report.adjust(present_money, 1.05, "cross-field: contract arithmetic consistent (incl. prior COs)")
        else:
            report.issue("CONTRACT_ARITHMETIC_MISMATCH", "error",
                         f"Revised contract amount {_fmt(revised)} does not equal original {_fmt(original)} "
                         f"+ additions - deductions = {_fmt(expected)}{assumption}; difference "
                         f"{_fmt(revised - expected)}"
                         + (f" (also inconsistent when including prior change orders {_fmt(prior)})"
                            if prior is not None else "")
                         + ". Values are reported as written; verify the source.",
                         money_fields)
            report.adjust(present_money, 0.7, "cross-field: contract arithmetic mismatch")
    elif original is not None and revised is None and (added is not None or deducted is not None):
        expected = original + (added or Decimal(0)) - (deducted or Decimal(0))
        report.issue("REVISED_AMOUNT_NOT_STATED", "info",
                     f"Revised contract amount is not stated. For reference only, original + additions - "
                     f"deductions = {_fmt(expected)} (not inserted into the output).",
                     ["revised_contract_amount"])

    # 3. Sign / magnitude sanity.
    for name in ("original_contract_amount", "revised_contract_amount"):
        if (v := dec(name)) is not None and v <= 0:
            report.issue("NON_POSITIVE_CONTRACT_AMOUNT", "error",
                         f"{SPEC_BY_NAME[name].label} is {_fmt(v)}; a contract sum must be positive.", [name])
            report.adjust([name], 0.6, "cross-field: non-positive contract amount")
    if added is not None and added < 0:
        report.issue("NEGATIVE_ADDITIONAL_COST", "warning",
                     f"Additional cost is negative ({_fmt(added)}); this may actually be a deduction.",
                     ["additional_cost", "deducted_amount"])
        report.adjust(["additional_cost"], 0.8, "cross-field: negative additional cost")
    if original is not None and original > 0:
        change = (added or Decimal(0)) + (deducted or Decimal(0))
        if change > original * Decimal("0.5"):
            report.issue("LARGE_CHANGE_RATIO", "warning",
                         f"This change order moves the contract by {change / original:.0%} of the original "
                         "amount; unusually large changes deserve review.",
                         ["additional_cost", "deducted_amount", "original_contract_amount"])

    # 4. Dates.
    co_date, completion = values.get("change_order_date"), values.get("requested_completion_date")
    if isinstance(co_date, date) and isinstance(completion, date) and completion < co_date:
        report.issue("COMPLETION_BEFORE_CO_DATE", "warning",
                     f"Requested completion date {completion.isoformat()} is before the change order date "
                     f"{co_date.isoformat()}.", ["change_order_date", "requested_completion_date"])
        report.adjust(["change_order_date", "requested_completion_date"], 0.8,
                      "cross-field: completion date precedes change order date")
    for name in ("change_order_date", "requested_completion_date"):
        d = values.get(name)
        if isinstance(d, date) and not (1980 <= d.year <= date.today().year + 15):
            report.issue("IMPLAUSIBLE_DATE", "warning",
                         f"{SPEC_BY_NAME[name].label} {d.isoformat()} is outside the plausible range.", [name])
            report.adjust([name], 0.6, "cross-field: implausible year")

    # 5. Approval status vs. signatures.
    status, approver = values.get("approval_status"), values.get("approved_by")
    if status in (ApprovalStatus.APPROVED, ApprovalStatus.APPROVED_WITH_CONDITIONS) and approver is None:
        report.issue("APPROVED_WITHOUT_APPROVER", "warning",
                     "Status is approved but no approver name was found.", ["approval_status", "approved_by"])
        report.adjust(["approval_status"], 0.9, "cross-field: approved but no approver found")
    if status in (ApprovalStatus.PENDING, ApprovalStatus.REJECTED) and approver is not None:
        report.issue("APPROVER_WITH_UNAPPROVED_STATUS", "warning",
                     f"An approver ('{approver}') is named but the status is '{status.value}'. The approver "
                     "field may be a placeholder or the status may be outdated.",
                     ["approval_status", "approved_by"])
        report.adjust(["approval_status", "approved_by"], 0.85, "cross-field: approver vs. status inconsistency")

    # 6. Financial content present at all?
    if added is None and deducted is None:
        report.issue("NO_COST_CHANGE_FOUND", "warning",
                     "Neither an additional cost nor a deducted amount was found.",
                     ["additional_cost", "deducted_amount"])

    return report
