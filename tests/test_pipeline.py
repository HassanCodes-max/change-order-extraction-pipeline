"""End-to-end pipeline tests, including the LLM path with fake providers/clients."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from scripts.generate_samples import CO_014_STANDARD
from src.cli import main as cli_main
from src.config import ConfigurationError, Settings
from src.extraction.document import DocumentProcessingError
from src.intelligence.llm_extractor import (
    AnthropicProvider,
    LLMConfigurationError,
    LLMExtractor,
    LLMOutputError,
    LLMRefusalError,
    LLMTransientError,
    OpenAIProvider,
    parse_llm_payload,
)
from src.intelligence.prompt_templates import extraction_json_schema
from src.pipeline import ChangeOrderPipeline
from src.validation.schemas import FIELD_NAMES, ApprovalStatus, FieldStatus
from tests.conftest import FakeProvider, field_payload, requires_ocr

CO_014_TEXT = CO_014_STANDARD.replace("\t", "   ")

GOOD_LLM_VALUES = {
    "project_name": ("ABC Commercial Building", "Project:   ABC Commercial Building", "stated", 0.97),
    "project_number": ("2024-118", "Project No.:   2024-118", "stated", 0.95),
    "change_order_number": ("CO-014", "Change Order No.:   CO-014", "stated", 0.98),
    "original_contract_number": ("GC-2024-0457", "Original Contract No.: GC-2024-0457", "stated", 0.95),
    "contractor_name": ("Summit Ridge Builders, LLC", "Contractor:   Summit Ridge Builders, LLC", "stated", 0.95),
    "client_name": ("ABC Holdings Inc.", "Owner:   ABC Holdings Inc.", "stated", 0.95),
    "change_order_date": ("March 3, 2025", "Date:   March 3, 2025", "stated", 0.96),
    "description_of_changes": (
        "Furnish and install twelve (12) additional LED high-bay fixtures in the warehouse area, including "
        "conduit, wiring and circuiting to panel LP-2.",
        "Furnish and install twelve (12) additional LED high-bay fixtures in the\nwarehouse area",
        "stated", 0.93),
    "reason_for_change": ("Owner request to increase lighting levels in the warehouse.",
                          "Reason for Change: Owner request to increase lighting levels", "stated", 0.92),
    "original_contract_amount": ("$1,250,000.00", "Original Contract Amount:   $1,250,000.00", "stated", 0.97),
    "additional_cost": ("$12,500.00", "Additional Cost:   $12,500.00", "stated", 0.97),
    "deducted_amount": ("$0.00", "Deducted Amount:   $0.00", "stated", 0.95),
    "revised_contract_amount": ("$1,262,500.00", "Revised Contract Amount:   $1,262,500.00", "stated", 0.97),
    "schedule_impact_days": ("five (5) calendar days", "extended by five (5) calendar days", "stated", 0.92),
    "requested_completion_date": ("08/15/2025", "Requested Completion Date:   08/15/2025", "stated", 0.93),
    "approval_status": ("Approved", "Status:   Approved", "stated", 0.95),
    "prepared_by": ("Jane Doe, Project Manager", "Prepared by: Jane Doe, Project Manager", "stated", 0.95),
    "approved_by": ("Robert Smith, Owner's Representative",
                    "Approved by: Robert Smith, Owner's Representative", "stated", 0.95),
}


def _llm_pipeline(settings, responses, **kw) -> tuple[ChangeOrderPipeline, FakeProvider]:
    provider = FakeProvider(responses, **kw)
    extractor = LLMExtractor(provider, settings, wait_max_seconds=0)
    return ChangeOrderPipeline(settings, llm_extractor=extractor), provider


# --------------------------------------------------------------------------- rules mode on samples


def test_standard_pdf_end_to_end(settings, samples_dir):
    result = ChangeOrderPipeline(settings).process_file(samples_dir / "co_014_standard.pdf")
    f = result.fields
    assert result.extraction.engine == "rules"
    assert f.change_order_number.value == "CO-014"
    assert f.change_order_date.value == date(2025, 3, 3)
    assert f.additional_cost.value == 12500.0
    assert f.revised_contract_amount.value == 1262500.0
    assert f.schedule_impact_days.value == 5
    assert f.approval_status.value is ApprovalStatus.APPROVED
    assert result.summary.extracted == 18 and not result.summary.needs_review
    assert any(i.code == "CONTRACT_ARITHMETIC_OK" for i in result.validation_issues)
    for _, field in result.field_items():
        assert field.evidence and 0.5 < field.confidence <= 0.99
        assert field.confidence_factors[-1].startswith("=")


@requires_ocr
def test_scanned_pdf_end_to_end(settings, samples_dir):
    result = ChangeOrderPipeline(settings).process_file(samples_dir / "co_007_scanned.pdf")
    f = result.fields
    assert result.document.ocr_pages == [1]
    assert f.change_order_number.value == "CO-007"
    assert f.revised_contract_amount.value == 858950.0
    assert f.deducted_amount.value == 2100.0
    assert f.approval_status.value is ApprovalStatus.PENDING
    assert any(i.code == "OCR_USED" for i in result.validation_issues)
    # OCR'd values never reach the confidence of a clean text layer value.
    text_result = ChangeOrderPipeline(settings).process_file(samples_dir / "co_014_standard.pdf")
    assert f.change_order_number.confidence < text_result.fields.change_order_number.confidence
    assert any("OCR" in factor for factor in f.change_order_number.confidence_factors)


def test_missing_fields_are_null_with_zero_confidence(settings, samples_dir):
    result = ChangeOrderPipeline(settings).process_file(samples_dir / "co_022_missing_fields.txt")
    f = result.fields
    for name in ("client_name", "original_contract_amount", "revised_contract_amount",
                 "schedule_impact_days", "approved_by", "prepared_by"):
        field = getattr(f, name)
        assert field.value is None and field.confidence == 0.0 and field.status is FieldStatus.MISSING
        assert field.evidence is None
    assert f.additional_cost.value == 9850.0
    assert f.approval_status.value is ApprovalStatus.PENDING
    assert result.summary.missing == 11


def test_conflicting_document(settings, samples_dir):
    result = ChangeOrderPipeline(settings).process_file(samples_dir / "co_031_conflicting.pdf")
    f = result.fields
    codes = [i.code for i in result.validation_issues]
    assert f.change_order_number.status is FieldStatus.CONFLICTING
    assert f.change_order_number.alternatives == ["CO-013"]
    assert f.schedule_impact_days.status is FieldStatus.CONFLICTING
    assert "CONTRACT_ARITHMETIC_MISMATCH" in codes and "APPROVED_WITHOUT_APPROVER" in codes
    # Contradictory source values are reported as written, never "fixed".
    assert f.revised_contract_amount.value == 1054000.0
    assert f.change_order_number.confidence < 0.5
    assert f.revised_contract_amount.confidence < f.project_name.confidence
    assert result.summary.needs_review and result.summary.conflicting == 2


def test_messy_document_with_invalid_date(settings, samples_dir):
    result = ChangeOrderPipeline(settings).process_file(samples_dir / "co_045_messy_format.txt")
    f = result.fields
    assert f.change_order_date.value == date(2025, 3, 3)
    assert f.original_contract_amount.value == 3480000.0
    assert f.deducted_amount.value == 3200.0
    assert f.schedule_impact_days.value == 14
    assert f.approval_status.value is ApprovalStatus.APPROVED_WITH_CONDITIONS
    completion = f.requested_completion_date
    assert completion.status is FieldStatus.INVALID and completion.value is None
    assert completion.raw_value == "02/30/2026" and completion.confidence == 0.0
    assert any(i.code == "INVALID_VALUE" for i in result.validation_issues)


def test_narrative_document_rules_mode_is_honest(settings, samples_dir):
    result = ChangeOrderPipeline(settings).process_file(samples_dir / "co_052_narrative_email.txt")
    f = result.fields
    assert f.additional_cost.value == 6420.0 and f.revised_contract_amount.value == 518420.0
    assert f.schedule_impact_days.value == 3
    # The walkthrough date is not the change-order date; rules must not grab it.
    assert f.change_order_date.value is None
    assert "change_order_date" in result.summary.missing_required


# --------------------------------------------------------------------------- input handling


def test_text_input_errors(settings):
    pipeline = ChangeOrderPipeline(settings)
    with pytest.raises(DocumentProcessingError, match="empty"):
        pipeline.process_text("   \n  ")
    with pytest.raises(DocumentProcessingError, match="Unsupported"):
        pipeline.process_file(__file__)
    with pytest.raises(DocumentProcessingError, match="not a readable image|OCR"):
        pipeline.process_image(b"definitely not an image", "x.png")


def test_non_change_order_document_is_flagged(settings):
    result = ChangeOrderPipeline(settings).process_text("INVOICE #4411\nAmount due: $300.00\nThank you!")
    assert any(i.code == "NOT_A_CHANGE_ORDER" for i in result.validation_issues)
    assert result.summary.needs_review


# --------------------------------------------------------------------------- LLM path (fake provider)


def test_llm_extraction_agrees_with_rules(settings):
    pipeline, provider = _llm_pipeline(settings, [field_payload(GOOD_LLM_VALUES)])
    result = pipeline.process_text(CO_014_TEXT, "co14.txt")
    rules_result = ChangeOrderPipeline(settings).process_text(CO_014_TEXT, "co14.txt")
    assert result.extraction.engine == "anthropic" and result.extraction.cross_checked_with_rules
    assert result.extraction.llm_attempts == 1 and provider.calls == 1
    f = result.fields
    assert f.change_order_number.source == "llm+rules"
    assert any("agrees" in x for x in f.change_order_number.confidence_factors)
    assert f.change_order_number.confidence > rules_result.fields.change_order_number.confidence
    assert f.schedule_impact_days.value == 5
    assert not result.summary.needs_review


def test_llm_hallucination_is_caught(settings):
    values = dict(GOOD_LLM_VALUES)
    # Invented value with invented evidence, and an invented field the document lacks.
    values["additional_cost"] = ("$15,000.00", "Additional work: $15,000", "stated", 0.99)
    values["approved_by"] = ("Maria Lopez", "Approved by: Maria Lopez", "stated", 0.9)
    text = CO_014_TEXT.replace("Approved by: Robert Smith, Owner's Representative\n", "")
    pipeline, _ = _llm_pipeline(settings, [field_payload(values)])
    result = pipeline.process_text(text)
    f = result.fields
    # Rules found the grounded value, so it replaces the ungrounded LLM value.
    assert f.additional_cost.value == 12500.0 and f.additional_cost.source == "rules"
    assert "$15,000.00" in f.additional_cost.alternatives
    # No grounded alternative: keep it, but flag and score it low.
    assert f.approved_by.value == "Maria Lopez" and f.approved_by.confidence < 0.5
    codes = {(i.code, tuple(i.fields)) for i in result.validation_issues}
    assert ("UNGROUNDED_EVIDENCE", ("approved_by",)) in codes
    assert result.summary.needs_review


def test_llm_missed_field_is_recovered_and_flagged(settings):
    values = dict(GOOD_LLM_VALUES)
    values["project_number"] = (None, None, "missing", 0.0)
    pipeline, _ = _llm_pipeline(settings, [field_payload(values)])
    f = pipeline.process_text(CO_014_TEXT).fields
    assert f.project_number.value == "2024-118" and f.project_number.source == "rules"
    assert any("recovered" in n for n in f.project_number.notes)


def test_llm_inferred_value_scores_below_stated(settings):
    values = dict(GOOD_LLM_VALUES)
    values["client_name"] = ("ABC Holdings Inc.", "Owner:   ABC Holdings Inc.", "inferred", 0.8)
    pipeline, _ = _llm_pipeline(settings, [field_payload(values)])
    result = pipeline.process_text(CO_014_TEXT)
    assert result.fields.client_name.status is FieldStatus.INFERRED
    assert result.fields.client_name.confidence < result.fields.contractor_name.confidence
    assert result.summary.inferred == 1


def test_llm_retries_invalid_output_then_succeeds(settings):
    pipeline, provider = _llm_pipeline(settings, ["not json at all", field_payload(GOOD_LLM_VALUES)])
    result = pipeline.process_text(CO_014_TEXT)
    assert provider.calls == 2 and result.extraction.llm_attempts == 2
    assert not result.extraction.fallback_used


def test_llm_persistent_failure_falls_back_to_rules(settings):
    pipeline, provider = _llm_pipeline(settings, [LLMTransientError("503 overloaded")])
    result = pipeline.process_text(CO_014_TEXT)
    assert provider.calls == settings.llm_max_attempts == 3
    assert result.extraction.engine == "rules" and result.extraction.fallback_used
    assert "503 overloaded" in result.extraction.fallback_reason
    assert any(i.code == "LLM_FALLBACK" for i in result.validation_issues)
    assert result.fields.change_order_number.value == "CO-014"  # still useful output


def test_llm_configuration_error_is_not_retried(settings):
    pipeline, provider = _llm_pipeline(settings, [LLMConfigurationError("invalid x-api-key")])
    result = pipeline.process_text(CO_014_TEXT)
    assert provider.calls == 1 and result.extraction.fallback_used


def test_oversized_input_is_not_truncated(settings):
    small = Settings(llm_provider="rules", llm_max_input_chars=1000, output_dir=settings.output_dir)
    provider = FakeProvider([field_payload(GOOD_LLM_VALUES)])
    pipeline = ChangeOrderPipeline(small, llm_extractor=LLMExtractor(provider, small, wait_max_seconds=0))
    result = pipeline.process_text(CO_014_TEXT + "\nNotes: " + "x " * 1000)
    assert provider.calls == 0 and "not truncated" in result.extraction.fallback_reason


def test_parse_llm_payload_validation():
    with pytest.raises(LLMOutputError):
        parse_llm_payload('{"fields": {}}', "anthropic", "m", 1)
    with pytest.raises(LLMOutputError):
        parse_llm_payload("[1, 2]", "anthropic", "m", 1)
    fenced = "```json\n" + field_payload(GOOD_LLM_VALUES) + "\n```"
    raw = parse_llm_payload(fenced, "openai", "m", 1)
    assert raw.get("change_order_number").value == "CO-014"
    data = json.loads(field_payload(GOOD_LLM_VALUES))
    data["fields"]["project_name"]["model_confidence"] = 97  # percent -> 0.97
    assert parse_llm_payload(json.dumps(data), "anthropic", "m", 1).get("project_name").model_confidence == 0.97


def test_json_schema_is_strict_and_complete():
    schema = extraction_json_schema()
    fields = schema["properties"]["fields"]
    assert schema["additionalProperties"] is False and fields["additionalProperties"] is False
    assert fields["required"] == list(FIELD_NAMES)
    one = fields["properties"]["additional_cost"]
    assert set(one["required"]) == set(one["properties"]) and one["additionalProperties"] is False


# --------------------------------------------------------------------------- provider adapters (fake SDK clients)


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _anthropic(response, **settings_kw):
    beta, plain = _FakeMessages(response), _FakeMessages(response)
    client = SimpleNamespace(beta=SimpleNamespace(messages=beta), messages=plain)
    provider = AnthropicProvider(Settings(anthropic_api_key="test", **settings_kw), client=client)
    return provider, beta, plain


def _msg(stop_reason="end_turn", blocks=None, model="claude-opus-5-5"):
    return SimpleNamespace(stop_reason=stop_reason, content=blocks or [], model=model, stop_details=None)


def test_anthropic_provider_request_and_response():
    payload = field_payload(GOOD_LLM_VALUES)
    provider, beta, _ = _anthropic(_msg(blocks=[SimpleNamespace(type="thinking", thinking=""),
                                                SimpleNamespace(type="text", text=payload)]))
    text, model = provider.complete_json("sys", "user", extraction_json_schema())
    assert text == payload and model == "claude-opus-5-5"
    kw = beta.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["output_config"]["effort"] == "medium"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]


def test_anthropic_provider_without_fallback_uses_plain_endpoint():
    provider, beta, plain = _anthropic(_msg(blocks=[SimpleNamespace(type="text", text="{}")]),
                                       anthropic_refusal_fallback=False, anthropic_effort=None)
    provider.complete_json("s", "u", {})
    assert beta.kwargs is None and "effort" not in plain.kwargs["output_config"]


def test_anthropic_provider_uses_text_after_fallback_switch():
    blocks = [SimpleNamespace(type="text", text='{"partial'), SimpleNamespace(type="fallback"),
              SimpleNamespace(type="text", text='{"ok": 1}')]
    provider, _, _ = _anthropic(_msg(blocks=blocks, model="fallback-model"))
    assert provider.complete_json("s", "u", {}) == ('{"ok": 1}', "fallback-model")


@pytest.mark.parametrize("stop_reason, error", [("refusal", LLMRefusalError), ("max_tokens", LLMOutputError)])
def test_anthropic_provider_stop_reasons(stop_reason, error):
    provider, _, _ = _anthropic(_msg(stop_reason=stop_reason, blocks=[SimpleNamespace(type="text", text="{}")]))
    with pytest.raises(error):
        provider.complete_json("s", "u", {})


def test_openai_provider():
    payload = field_payload(GOOD_LLM_VALUES)
    msg = SimpleNamespace(content=payload, refusal=None)
    completions = _FakeMessages(SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                                                model="gpt-test"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = OpenAIProvider(Settings(openai_api_key="k", openai_model="gpt-test"), client=client)
    assert provider.complete_json("s", "u", {"type": "object"}) == (payload, "gpt-test")
    assert completions.kwargs["response_format"]["json_schema"]["strict"] is True

    msg.refusal = "I can't help with that"
    with pytest.raises(LLMRefusalError):
        provider.complete_json("s", "u", {})
    with pytest.raises(LLMConfigurationError, match="OPENAI_MODEL"):
        OpenAIProvider(Settings(openai_api_key="k"), client=client)


def test_unconfigured_provider_falls_back(settings):
    cfg = Settings(llm_provider="openai", openai_api_key="k", output_dir=settings.output_dir)  # no model
    result = ChangeOrderPipeline(cfg).process_text(CO_014_TEXT)
    assert result.extraction.fallback_used and "OPENAI_MODEL" in result.extraction.fallback_reason


# --------------------------------------------------------------------------- config & CLI


def test_settings_from_env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    assert Settings.from_env(load_env_file=False).resolve_provider() == "rules"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Settings.from_env(load_env_file=False).resolve_provider() == "anthropic"
    monkeypatch.setenv("LLM_PROVIDER", "demo")
    assert Settings.from_env(load_env_file=False).resolve_provider() == "rules"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ConfigurationError):
        Settings.from_env(load_env_file=False)
    monkeypatch.setenv("LLM_PROVIDER", "rules")
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "0")
    with pytest.raises(ConfigurationError):
        Settings.from_env(load_env_file=False)


def test_cli_compact_output(samples_dir, capsys, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "rules")
    assert cli_main([str(samples_dir / "co_014_standard.pdf"), "--compact"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["change_order_number"]["value"] == "CO-014"
    assert set(data["change_order_number"]) == {"value", "confidence", "evidence", "status"}
