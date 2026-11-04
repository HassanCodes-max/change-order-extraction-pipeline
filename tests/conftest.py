"""Shared fixtures: settings, sample documents, and fake LLM providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.generate_samples import generate
from src.config import Settings
from src.extraction.ocr_extractor import OCRExtractor
from src.validation.schemas import FIELD_NAMES


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Deterministic settings independent of the developer's environment."""
    return Settings(llm_provider="rules", output_dir=tmp_path / "outputs", llm_max_attempts=3)


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Freshly generated synthetic samples (same generator as sample_documents/)."""
    out = tmp_path_factory.mktemp("samples")
    generate(out)
    return out


def _ocr_available() -> bool:
    return OCRExtractor(Settings()).available


requires_ocr = pytest.mark.skipif(not _ocr_available(), reason="Tesseract OCR is not installed")


def field_payload(values: dict[str, tuple[str | None, str | None, str, float]],
                  is_change_order: bool = True, notes: str | None = None) -> str:
    """Build an LLM JSON response. ``values``: name -> (value, evidence, status, confidence)."""
    fields: dict[str, Any] = {}
    for name in FIELD_NAMES:
        value, evidence, status, conf = values.get(name, (None, None, "missing", 0.0))
        fields[name] = {"value": value, "evidence": evidence, "status": status,
                        "alternatives": [], "model_confidence": conf, "note": None}
    return json.dumps({
        "fields": fields,
        "prior_change_orders_net": {"value": None, "evidence": None, "status": "missing",
                                    "alternatives": [], "model_confidence": 0.0, "note": None},
        "is_change_order": is_change_order,
        "document_notes": notes,
    })


class FakeProvider:
    """LLM provider returning scripted responses (or raising scripted errors)."""

    def __init__(self, responses: list[Any], name: str = "anthropic", model: str = "fake-model") -> None:
        self.responses = list(responses)
        self.name = name
        self.model = model
        self.calls = 0

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, str]:
        self.calls += 1
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item, self.model
