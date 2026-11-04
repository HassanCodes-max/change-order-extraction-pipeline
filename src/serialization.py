"""JSON serialization of extraction results.

Kept separate from extraction and validation so output formats can evolve
independently. All writers produce strict JSON (no NaN/Infinity) and every
full report is re-validated against the Pydantic schema before it is written.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.validation.schemas import ExtractionResult


def to_json(result: ExtractionResult, indent: int = 2) -> str:
    """Full report as JSON (document info, fields, issues, summary)."""
    return result.model_dump_json(indent=indent)


def to_compact_dict(result: ExtractionResult) -> dict[str, Any]:
    """Fields only, as ``{field: {value, confidence, evidence, status}}``."""
    data = result.model_dump(mode="json")["fields"]
    return {
        name: {k: f[k] for k in ("value", "confidence", "evidence", "status")}
        for name, f in data.items()
    }


def to_compact_json(result: ExtractionResult, indent: int = 2) -> str:
    """Compact field map as strict JSON."""
    return json.dumps(to_compact_dict(result), indent=indent, ensure_ascii=False, allow_nan=False)


def from_json(text: str) -> ExtractionResult:
    """Parse and validate a full JSON report."""
    return ExtractionResult.model_validate_json(text)


def _safe_stem(name: str) -> str:
    stem = Path(name).stem or "document"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:80]


def save_result(result: ExtractionResult, output_dir: Path, stem: str | None = None) -> Path:
    """Write the full report to ``output_dir`` and return its path."""
    payload = to_json(result)
    from_json(payload)  # guarantee what we write is schema-valid
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{_safe_stem(stem or result.document.source_name)}_{timestamp}.json"
    path.write_text(payload, encoding="utf-8")
    return path
