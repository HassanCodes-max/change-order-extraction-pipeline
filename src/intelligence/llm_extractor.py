"""LLM-backed semantic extraction with structured output, validation and retries.

Providers are pluggable via ``LLM_PROVIDER``:

* ``anthropic`` (default when ``ANTHROPIC_API_KEY`` is set) — Claude via the
  official Anthropic SDK, using ``output_config.format`` JSON-schema
  structured output and the server-side refusal fallback.
* ``openai`` — any OpenAI-API model via the official OpenAI SDK
  (``OPENAI_BASE_URL`` allows compatible endpoints), using strict
  ``json_schema`` response format.

Error policy:

* transient failures (rate limits, 5xx, network) and malformed output are
  retried with exponential backoff (the SDKs also retry HTTP errors
  internally);
* configuration errors (bad key, unknown model, bad request) and refusals are
  not retried — the pipeline falls back to the rule-based extractor and says so.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from pydantic import ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import PROVIDER_ANTHROPIC, PROVIDER_OPENAI, Settings
from src.intelligence.prompt_templates import SYSTEM_PROMPT, build_user_prompt, extraction_json_schema
from src.validation.schemas import FIELD_NAMES, RawExtraction, RawField

logger = logging.getLogger(__name__)

ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(Exception):
    """Base class for LLM extraction failures."""


class LLMConfigurationError(LLMError):
    """Misconfiguration (missing/invalid key, unknown model, rejected request). Not retried."""


class LLMRefusalError(LLMError):
    """The model declined the request. Not retried."""


class LLMInputTooLargeError(LLMError):
    """Document exceeds ``LLM_MAX_INPUT_CHARS``; we refuse to silently truncate it."""


class LLMTransientError(LLMError):
    """Temporary failure (rate limit, server error, network). Retried."""


class LLMOutputError(LLMError):
    """The model returned unusable output (invalid JSON, schema mismatch, truncation). Retried."""


class LLMProvider(Protocol):
    """Minimal provider interface: return a JSON string conforming to ``schema``."""

    name: str
    model: str

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, str]:
        """Return ``(json_text, model_actually_used)``."""
        ...


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Claude via the Anthropic Messages API with JSON-schema structured output."""

    name = PROVIDER_ANTHROPIC

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.model = settings.anthropic_model
        if client is not None:
            self.client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is in requirements
            raise LLMConfigurationError("The 'anthropic' package is not installed.") from exc
        try:
            self.client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key,  # None -> SDK resolves env / `ant auth` profile
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_sdk_max_retries,
            )
        except Exception as exc:
            raise LLMConfigurationError(f"Could not create Anthropic client: {exc}") from exc

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, str]:
        import anthropic

        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.settings.anthropic_effort:
            output_config["effort"] = self.settings.anthropic_effort
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.llm_max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        try:
            if self.settings.anthropic_refusal_fallback:
                # On a policy decline the API re-runs the request on a suitable
                # fallback model within the same call.
                response = self.client.beta.messages.create(
                    **request, betas=[ANTHROPIC_FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = self.client.messages.create(**request)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMConfigurationError(f"Anthropic rejected the credentials: {exc}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMConfigurationError(f"Anthropic model '{self.model}' not found: {exc}") from exc
        except anthropic.BadRequestError as exc:
            raise LLMConfigurationError(
                f"Anthropic rejected the request (check ANTHROPIC_MODEL / ANTHROPIC_EFFORT / "
                f"ANTHROPIC_REFUSAL_FALLBACK): {exc}"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMTransientError(f"Anthropic rate limit: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMTransientError(f"Anthropic server error {exc.status_code}: {exc}") from exc
            raise LLMConfigurationError(f"Anthropic API error {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise LLMTransientError(f"Could not reach Anthropic API: {exc}") from exc

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusalError(f"Model declined the request (category: {category or 'unspecified'}).")
        if stop_reason == "max_tokens":
            raise LLMOutputError("Response was truncated at max_tokens; increase LLM_MAX_TOKENS.")

        # Keep only text produced after the last fallback switch, if any.
        texts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "fallback":
                texts = []
            elif block_type == "text":
                texts.append(block.text)
        if not texts:
            raise LLMOutputError("Response contained no text block.")
        return "".join(texts), getattr(response, "model", None) or self.model


class OpenAIProvider:
    """OpenAI (or OpenAI-API-compatible endpoint) with strict JSON-schema output."""

    name = PROVIDER_OPENAI

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        if not settings.openai_model:
            raise LLMConfigurationError("OPENAI_MODEL must be set when LLM_PROVIDER=openai.")
        self.model = settings.openai_model
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError("The 'openai' package is not installed (pip install openai).") from exc
        try:
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_sdk_max_retries,
            )
        except Exception as exc:
            raise LLMConfigurationError(f"Could not create OpenAI client: {exc}") from exc

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, str]:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "change_order_extraction", "schema": schema, "strict": True},
                },
                max_completion_tokens=self.settings.llm_max_tokens,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise LLMConfigurationError(f"OpenAI rejected the credentials: {exc}") from exc
        except openai.NotFoundError as exc:
            raise LLMConfigurationError(f"OpenAI model '{self.model}' not found: {exc}") from exc
        except openai.BadRequestError as exc:
            raise LLMConfigurationError(f"OpenAI rejected the request: {exc}") from exc
        except openai.RateLimitError as exc:
            raise LLMTransientError(f"OpenAI rate limit: {exc}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMTransientError(f"OpenAI server error {exc.status_code}: {exc}") from exc
            raise LLMConfigurationError(f"OpenAI API error {exc.status_code}: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LLMTransientError(f"Could not reach OpenAI API: {exc}") from exc

        choice = response.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise LLMRefusalError(f"Model declined the request: {refusal}")
        if choice.finish_reason == "length":
            raise LLMOutputError("Response was truncated; increase LLM_MAX_TOKENS.")
        if not choice.message.content:
            raise LLMOutputError("Response had no content.")
        return choice.message.content, getattr(response, "model", None) or self.model


# ---------------------------------------------------------------------------
# Output parsing / validation
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def _coerce_confidence(item: dict[str, Any]) -> dict[str, Any]:
    conf = item.get("model_confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        item["model_confidence"] = None
    elif conf > 1.0:
        item["model_confidence"] = min(conf / 100.0, 1.0) if conf <= 100 else 1.0
    elif conf < 0:
        item["model_confidence"] = 0.0
    return item


def parse_llm_payload(text: str, engine: str, model: str | None, attempts: int) -> RawExtraction:
    """Validate the model's JSON against the expected structure.

    Raises:
        LLMOutputError: invalid JSON, wrong shape, or too many fields absent.
    """
    if m := _FENCE_RE.match(text):  # tolerate fenced JSON from non-strict endpoints
        text = m.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"Model output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("fields"), dict):
        raise LLMOutputError("Model output lacks a 'fields' object.")

    raw_fields = data["fields"]
    absent = [n for n in FIELD_NAMES if n not in raw_fields]
    if len(absent) > len(FIELD_NAMES) // 2:
        raise LLMOutputError(f"Model output is missing most fields: {absent}")

    try:
        fields = {
            name: RawField.model_validate(_coerce_confidence(dict(raw_fields[name])))
            if isinstance(raw_fields.get(name), dict) else RawField()
            for name in FIELD_NAMES
        }
        aux_raw = data.get("prior_change_orders_net")
        auxiliary = {"prior_change_orders_net": RawField.model_validate(_coerce_confidence(dict(aux_raw)))
                     if isinstance(aux_raw, dict) else RawField()}
    except ValidationError as exc:
        raise LLMOutputError(f"Model output failed schema validation: {exc}") from exc

    is_co = data.get("is_change_order")
    notes = data.get("document_notes")
    return RawExtraction(
        fields=fields,
        auxiliary=auxiliary,
        engine=engine,
        model=model,
        attempts=attempts,
        is_change_order=is_co if isinstance(is_co, bool) else None,
        document_notes=notes if isinstance(notes, str) and notes.strip() else None,
    )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class LLMExtractor:
    """Runs a provider with retries and returns a validated :class:`RawExtraction`."""

    def __init__(self, provider: LLMProvider, settings: Settings, wait_max_seconds: float = 20.0) -> None:
        self.provider = provider
        self.settings = settings
        self.wait_max_seconds = wait_max_seconds
        self.last_attempts = 0

    @property
    def engine(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    def extract(self, document_text: str) -> RawExtraction:
        """Extract fields from ``document_text``.

        Raises:
            LLMError subclasses after retries are exhausted or on non-retryable errors.
        """
        if len(document_text) > self.settings.llm_max_input_chars:
            raise LLMInputTooLargeError(
                f"Document has {len(document_text):,} characters, above LLM_MAX_INPUT_CHARS="
                f"{self.settings.llm_max_input_chars:,}. It was not truncated; split the document or raise the limit."
            )
        schema = extraction_json_schema()
        user_prompt = build_user_prompt(document_text)
        self.last_attempts = 0

        retrying = Retrying(
            stop=stop_after_attempt(self.settings.llm_max_attempts),
            wait=wait_exponential(multiplier=1, min=0 if self.wait_max_seconds == 0 else 1,
                                  max=self.wait_max_seconds),
            retry=retry_if_exception_type((LLMTransientError, LLMOutputError)),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                self.last_attempts = attempt.retry_state.attempt_number
                logger.info("LLM extraction attempt %d via %s/%s", self.last_attempts,
                            self.provider.name, self.provider.model)
                text, model_used = self.provider.complete_json(SYSTEM_PROMPT, user_prompt, schema)
                return parse_llm_payload(text, self.provider.name, model_used, self.last_attempts)
        raise LLMError("unreachable")  # pragma: no cover


def build_llm_extractor(settings: Settings, provider_name: str) -> LLMExtractor:
    """Create an extractor for ``provider_name`` (``anthropic`` or ``openai``)."""
    if provider_name == PROVIDER_ANTHROPIC:
        return LLMExtractor(AnthropicProvider(settings), settings)
    if provider_name == PROVIDER_OPENAI:
        return LLMExtractor(OpenAIProvider(settings), settings)
    raise LLMConfigurationError(f"Unknown LLM provider: {provider_name!r}")
