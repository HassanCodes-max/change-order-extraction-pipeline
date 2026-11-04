"""Application configuration loaded from environment variables.

All secrets (API keys) come from the environment or a local ``.env`` file and
are never hardcoded. Every setting has a safe default so the application runs
out of the box in deterministic (demo) mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Provider identifiers accepted by ``LLM_PROVIDER``.
PROVIDER_AUTO = "auto"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDER_RULES = "rules"  # deterministic, offline "demo" mode

_PROVIDER_ALIASES = {
    "": PROVIDER_AUTO,
    "auto": PROVIDER_AUTO,
    "anthropic": PROVIDER_ANTHROPIC,
    "claude": PROVIDER_ANTHROPIC,
    "openai": PROVIDER_OPENAI,
    "rules": PROVIDER_RULES,
    "demo": PROVIDER_RULES,
    "local": PROVIDER_RULES,
    "none": PROVIDER_RULES,
}


class ConfigurationError(ValueError):
    """Raised when environment configuration is invalid."""


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_str(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean (true/false), got {value!r}")


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    value = _env_str(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    value = _env_str(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings. Build with :meth:`Settings.from_env`."""

    # --- LLM provider -----------------------------------------------------
    llm_provider: str = PROVIDER_AUTO
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str | None = "medium"
    anthropic_refusal_fallback: bool = True
    openai_api_key: str | None = None
    openai_model: str | None = None
    openai_base_url: str | None = None
    llm_timeout_seconds: float = 120.0
    llm_max_attempts: int = 3
    llm_sdk_max_retries: int = 2
    llm_max_tokens: int = 16000
    llm_max_input_chars: int = 200_000

    # --- Document processing ---------------------------------------------
    tesseract_cmd: str | None = None
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    ocr_psm: int = 3
    ocr_min_chars_per_page: int = 40
    max_pdf_pages: int = 50
    max_upload_mb: int = 25

    # --- Validation ------------------------------------------------------
    date_order: str = "MDY"  # how to read ambiguous numeric dates like 03/04/2025
    amount_tolerance: float = 0.01

    # --- Output ----------------------------------------------------------
    output_dir: Path = PROJECT_ROOT / "outputs"

    @classmethod
    def from_env(cls, load_env_file: bool = True) -> "Settings":
        """Create settings from environment variables (and ``.env`` if present)."""
        if load_env_file:
            load_dotenv(PROJECT_ROOT / ".env", override=False)

        raw_provider = (_env_str("LLM_PROVIDER", PROVIDER_AUTO) or "").lower()
        if raw_provider not in _PROVIDER_ALIASES:
            raise ConfigurationError(
                f"LLM_PROVIDER must be one of auto, anthropic, openai, rules; got {raw_provider!r}"
            )

        date_order = (_env_str("DATE_ORDER", "MDY") or "MDY").upper()
        if date_order not in {"MDY", "DMY"}:
            raise ConfigurationError(f"DATE_ORDER must be MDY or DMY, got {date_order!r}")

        effort = _env_str("ANTHROPIC_EFFORT", "medium")
        if effort is not None and effort.lower() in {"none", "default", "off"}:
            effort = None

        output_dir = Path(_env_str("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")) or "outputs")
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

        return cls(
            llm_provider=_PROVIDER_ALIASES[raw_provider],
            anthropic_api_key=_env_str("ANTHROPIC_API_KEY"),
            anthropic_model=_env_str("ANTHROPIC_MODEL", "claude-opus-5-5") or "claude-opus-5-5",
            anthropic_effort=effort,
            anthropic_refusal_fallback=_env_bool("ANTHROPIC_REFUSAL_FALLBACK", True),
            openai_api_key=_env_str("OPENAI_API_KEY"),
            openai_model=_env_str("OPENAI_MODEL"),
            openai_base_url=_env_str("OPENAI_BASE_URL"),
            llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 120.0, minimum=1.0),
            llm_max_attempts=_env_int("LLM_MAX_ATTEMPTS", 3, minimum=1),
            llm_sdk_max_retries=_env_int("LLM_SDK_MAX_RETRIES", 2, minimum=0),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 16000, minimum=1024),
            llm_max_input_chars=_env_int("LLM_MAX_INPUT_CHARS", 200_000, minimum=1000),
            tesseract_cmd=_env_str("TESSERACT_CMD"),
            ocr_language=_env_str("OCR_LANGUAGE", "eng") or "eng",
            ocr_dpi=_env_int("OCR_DPI", 300, minimum=72),
            ocr_psm=_env_int("OCR_PSM", 3, minimum=0),
            ocr_min_chars_per_page=_env_int("OCR_MIN_CHARS_PER_PAGE", 40, minimum=0),
            max_pdf_pages=_env_int("MAX_PDF_PAGES", 50, minimum=1),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 25, minimum=1),
            date_order=date_order,
            amount_tolerance=_env_float("AMOUNT_TOLERANCE", 0.01, minimum=0.0),
            output_dir=output_dir,
        )

    def resolve_provider(self) -> str:
        """Return the concrete provider to use, resolving ``auto``.

        ``auto`` picks Anthropic if ``ANTHROPIC_API_KEY`` is set, otherwise OpenAI
        if both ``OPENAI_API_KEY`` and ``OPENAI_MODEL`` are set, otherwise the
        deterministic rule-based engine.
        """
        if self.llm_provider != PROVIDER_AUTO:
            return self.llm_provider
        if self.anthropic_api_key:
            return PROVIDER_ANTHROPIC
        if self.openai_api_key and self.openai_model:
            return PROVIDER_OPENAI
        return PROVIDER_RULES

    def model_for(self, provider: str) -> str | None:
        """Return the configured model id for ``provider``."""
        if provider == PROVIDER_ANTHROPIC:
            return self.anthropic_model
        if provider == PROVIDER_OPENAI:
            return self.openai_model
        return None


def get_settings() -> Settings:
    """Convenience accessor used by the app and CLI."""
    return Settings.from_env()
