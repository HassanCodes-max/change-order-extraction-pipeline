"""Streamlit UI for the Change-Order Extraction Pipeline.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import PROJECT_ROOT, PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_RULES, ConfigurationError, Settings
from src.extraction.document import DocumentProcessingError, DocumentText
from src.extraction.ocr_extractor import OCRExtractor
from src.pipeline import ChangeOrderPipeline
from src.serialization import save_result, to_compact_json, to_json
from src.validation.schemas import ApprovalStatus, ExtractionResult, FieldKind, FieldSpec, FieldStatus

SAMPLES_DIR = PROJECT_ROOT / "sample_documents"
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "txt"]
HIGH, MEDIUM = 0.85, 0.60

st.set_page_config(page_title="Change Order Intelligence", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
  .app-header {border-bottom: 1px solid #e3e8ef; padding-bottom: .6rem; margin-bottom: 1rem;}
  .app-header h1 {font-size: 1.65rem; margin: 0; color: #0f2a44; font-weight: 650;}
  .app-header p {margin: .15rem 0 0; color: #5b6b7f; font-size: .95rem;}
  .chip {display: inline-block; padding: .08rem .55rem; border-radius: 999px; font-size: .75rem;
         font-weight: 600; letter-spacing: .02em; border: 1px solid transparent; white-space: nowrap;}
  .chip-stated {background: #e7f5ec; color: #17663a; border-color: #b9e2c8;}
  .chip-inferred {background: #eef2ff; color: #3730a3; border-color: #c7d2fe;}
  .chip-conflicting {background: #fff4e5; color: #9a4b00; border-color: #fed7aa;}
  .chip-invalid {background: #fdecec; color: #a11b1b; border-color: #f5c2c2;}
  .chip-missing {background: #f1f3f6; color: #5b6b7f; border-color: #dde2e9;}
  .conf-high {color: #17663a; font-weight: 650;}
  .conf-medium {color: #9a6700; font-weight: 650;}
  .conf-low {color: #a11b1b; font-weight: 650;}
  .evidence {background: #f7f9fc; border-left: 3px solid #1f4e79; padding: .45rem .7rem;
             font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .82rem;
             white-space: pre-wrap; color: #1d2b3a; border-radius: 0 4px 4px 0;}
  .muted {color: #6b7a8c; font-size: .82rem;}
  div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- helpers


@st.cache_resource(show_spinner=False)
def _base_settings() -> Settings:
    return Settings.from_env()


@st.cache_resource(show_spinner=False)
def _pipeline(provider: str | None) -> ChangeOrderPipeline:
    return ChangeOrderPipeline(_base_settings(), provider=provider)


@st.cache_resource(show_spinner=False)
def _ocr_status() -> tuple[bool, str | None]:
    ocr = OCRExtractor(_base_settings())
    return ocr.available, ocr.unavailable_reason


def _conf_class(conf: float) -> str:
    return "conf-high" if conf >= HIGH else "conf-medium" if conf >= MEDIUM else "conf-low"


def _chip(status: FieldStatus) -> str:
    return f'<span class="chip chip-{status.value}">{status.value.upper()}</span>'


def _display_value(spec: FieldSpec, value: Any) -> str:
    if value is None:
        return "—"
    if spec.kind is FieldKind.MONEY:
        return f"${value:,.2f}"
    if spec.kind is FieldKind.DAYS:
        return f"{value:+d} days" if value else "0 days (no change)"
    if spec.kind is FieldKind.DATE:
        return value.strftime("%b %d, %Y")
    if isinstance(value, ApprovalStatus):
        return value.value.replace("_", " ").title()
    return str(value)


def _provider_options(settings: Settings) -> dict[str, str | None]:
    detected = settings.resolve_provider()
    label = {PROVIDER_ANTHROPIC: "Anthropic Claude", PROVIDER_OPENAI: "OpenAI", PROVIDER_RULES: "Deterministic"}
    return {
        f"Auto — {label[detected]}": None,
        "Anthropic Claude": PROVIDER_ANTHROPIC,
        "OpenAI": PROVIDER_OPENAI,
        "Deterministic rules (demo, offline)": PROVIDER_RULES,
    }


# --------------------------------------------------------------------------- sidebar


def sidebar(settings: Settings) -> tuple[str | None, bool, bool]:
    with st.sidebar:
        st.markdown("### Extraction engine")
        options = _provider_options(settings)
        choice = st.selectbox("Engine", list(options), index=0, label_visibility="collapsed")
        provider = options[choice]
        effective = provider or settings.resolve_provider()

        if effective == PROVIDER_ANTHROPIC:
            key_state = "configured" if settings.anthropic_api_key else "not set (SDK default credentials)"
            st.caption(f"Model: `{settings.anthropic_model}` · API key: {key_state}")
        elif effective == PROVIDER_OPENAI:
            st.caption(f"Model: `{settings.openai_model or 'OPENAI_MODEL not set'}` · API key: "
                       f"{'configured' if settings.openai_api_key else 'not set'}")
        else:
            st.caption("No API key needed. Label/pattern-based extraction with the same validation "
                       "and confidence scoring.")
        st.caption("If an LLM call fails, the pipeline falls back to deterministic extraction and says so.")

        st.markdown("### Document processing")
        ocr_ok, ocr_reason = _ocr_status()
        if ocr_ok:
            st.success("Tesseract OCR available", icon="✅")
        else:
            st.warning(f"OCR unavailable — scanned PDFs cannot be read. {ocr_reason}", icon="⚠️")
        force_ocr = st.checkbox("Force OCR on all PDF pages", value=False, disabled=not ocr_ok,
                                help="Use when a PDF's embedded text layer is garbled.")
        save_copy = st.checkbox("Save JSON to outputs/ folder", value=False)

        st.markdown("### Confidence legend")
        st.markdown(
            f'<span class="conf-high">■ ≥ {HIGH:.2f} high</span><br>'
            f'<span class="conf-medium">■ {MEDIUM:.2f}–{HIGH:.2f} review suggested</span><br>'
            f'<span class="conf-low">■ &lt; {MEDIUM:.2f} verify manually</span>',
            unsafe_allow_html=True,
        )
        st.caption("Confidence is computed from evidence grounding, extractor agreement, OCR quality and "
                   "validation checks — not from the model's self-assessment alone.")
    return provider, force_ocr, save_copy


# --------------------------------------------------------------------------- input


def input_panel() -> tuple[str, Any, str] | None:
    """Return (kind, payload, name) for the selected input, or None."""
    tab_upload, tab_text, tab_sample = st.tabs(["Upload document", "Paste text", "Sample documents"])
    with tab_upload:
        upload = st.file_uploader("PDF (text or scanned), image, or .txt", type=UPLOAD_TYPES)
    with tab_text:
        pasted = st.text_area("Change-order text", height=220,
                              placeholder="Paste the change order text, e-mail or OCR output here…")
    with tab_sample:
        samples = sorted(p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt"})
        sample = st.selectbox("Synthetic sample", ["—"] + samples, index=0)

    source = st.radio("Process", ["Uploaded file", "Pasted text", "Sample document"], horizontal=True,
                      index=0 if upload else 1 if pasted.strip() else 2 if sample != "—" else 0)
    if source == "Uploaded file" and upload is not None:
        return "file", upload.getvalue(), upload.name
    if source == "Pasted text" and pasted.strip():
        return "text", pasted, "pasted_text"
    if source == "Sample document" and sample != "—":
        return "path", SAMPLES_DIR / sample, sample
    return None


def load_document(pipeline: ChangeOrderPipeline, kind: str, payload: Any, name: str, force_ocr: bool) -> DocumentText:
    """Stage 1: turn the selected input into text (text layer, OCR or raw text)."""
    if kind == "text":
        return pipeline.load_text(payload, name)
    if kind == "path":
        return pipeline.load_file(payload, force_ocr=force_ocr)
    return pipeline.load_bytes(payload, name, force_ocr=force_ocr)


def _format_source(document: DocumentText) -> str:
    if len(document.pages) == 1:
        return document.pages[0].text
    return "\n\n".join(f"──── page {p.number} ({p.method}) ────\n{p.text}" for p in document.pages)


# --------------------------------------------------------------------------- results


def results_panel(result: ExtractionResult) -> None:
    s, ex, doc = result.summary, result.extraction, result.document
    errors = sum(i.severity == "error" for i in result.validation_issues)
    warnings = sum(i.severity == "warning" for i in result.validation_issues)

    cols = st.columns(5)
    cols[0].metric("Fields extracted", f"{s.extracted} / {s.total_fields}")
    cols[1].metric("Mean confidence", f"{s.overall_confidence:.2f}", help="Mean over extracted fields")
    cols[2].metric("Conflicts", s.conflicting)
    cols[3].metric("Missing required", len(s.missing_required))
    cols[4].metric("Errors / warnings", f"{errors} / {warnings}")

    engine = {"anthropic": "Anthropic Claude", "openai": "OpenAI", "rules": "Deterministic rules"}[ex.engine]
    meta = (f"Engine: <b>{engine}</b>{f' (<code>{html.escape(ex.model)}</code>)' if ex.model else ''}"
            f"{' · cross-checked with rules' if ex.cross_checked_with_rules else ''}"
            f" · {doc.page_count} page(s), input: {doc.input_type}"
            f"{f' · OCR pages: {doc.ocr_pages} (mean {doc.ocr_mean_confidence:.0f}/100)' if doc.ocr_pages else ''}"
            f" · {ex.processing_seconds:.2f}s")
    st.markdown(f'<div class="muted">{meta}</div>', unsafe_allow_html=True)
    if ex.fallback_used:
        st.warning(f"LLM fallback: {ex.fallback_reason}", icon="⚠️")
    if s.needs_review:
        st.warning("**Needs human review** — " + "; ".join(s.review_reasons), icon="🔎")
    else:
        st.success("All checks passed — no review flags raised.", icon="✅")

    tab_fields, tab_issues, tab_json, tab_text = st.tabs(
        ["Extracted fields", f"Validation ({len(result.validation_issues)})", "JSON", "Source text"])
    with tab_fields:
        fields_table(result)
        field_details(result)
    with tab_issues:
        issues_view(result)
    with tab_json:
        json_view(result)
    with tab_text:
        st.caption(f"Text the extractors saw ({doc.character_count:,} characters).")
        st.code(st.session_state.get("source_text", ""), language=None)


def fields_table(result: ExtractionResult) -> None:
    rows = []
    for spec, field in result.field_items():
        rows.append({
            "Field": spec.label + (" *" if spec.required else ""),
            "Value": _display_value(spec, field.value) if field.value is not None
            else (f"invalid: {field.raw_value}" if field.status is FieldStatus.INVALID else "—"),
            "Status": field.status.value,
            "Confidence": field.confidence,
            "Source": field.source,
            "Evidence": (field.evidence or "").replace("\n", " ⏎ "),
        })
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=min(38 * (len(rows) + 1), 720),
        column_config={
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0,
                                                          format="%.2f", width="small"),
            "Field": st.column_config.TextColumn(width="medium"),
            "Value": st.column_config.TextColumn(width="medium"),
            "Status": st.column_config.TextColumn(width="small"),
            "Source": st.column_config.TextColumn(width="small"),
            "Evidence": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption("* required field. Source: llm, rules, llm+rules (both agree) or none.")


def field_details(result: ExtractionResult) -> None:
    st.markdown("#### Field evidence & confidence breakdown")
    for spec, field in result.field_items():
        conf = field.confidence
        header = f"{spec.label} — {_display_value(spec, field.value)}  ·  {field.status.value}  ·  {conf:.2f}"
        with st.expander(header, expanded=False):
            st.markdown(
                f'{_chip(field.status)} &nbsp; <span class="{_conf_class(conf)}">confidence {conf:.2f}</span>'
                f' &nbsp; <span class="muted">source: {field.source}</span>',
                unsafe_allow_html=True,
            )
            if field.evidence:
                st.markdown(f'<div class="evidence">{html.escape(field.evidence)}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="muted">No evidence — field not found in the document.</span>',
                            unsafe_allow_html=True)
            if field.raw_value is not None and field.raw_value != str(field.value):
                st.markdown(f"**Raw value:** `{field.raw_value}`")
            if field.alternatives:
                st.markdown("**Alternative values found:** " + ", ".join(f"`{a}`" for a in field.alternatives))
            if field.notes:
                st.markdown("**Notes:**\n" + "\n".join(f"- {n}" for n in field.notes))
            st.markdown("**How the confidence was computed:**")
            st.code("\n".join(field.confidence_factors), language=None)


def issues_view(result: ExtractionResult) -> None:
    if not result.validation_issues:
        st.info("No validation issues.")
        return
    render = {"error": st.error, "warning": st.warning, "info": st.info}
    for severity in ("error", "warning", "info"):
        for issue in (i for i in result.validation_issues if i.severity == severity):
            fields = f"  \n`{', '.join(issue.fields)}`" if issue.fields else ""
            render[severity](f"**{issue.code}** — {issue.message}{fields}")


def json_view(result: ExtractionResult) -> None:
    full, compact = to_json(result), to_compact_json(result)
    stem = Path(result.document.source_name).stem or "change_order"
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("⬇ Download full report (JSON)", full, file_name=f"{stem}_extraction.json",
                       mime="application/json", type="primary", width="stretch")
    c2.download_button("⬇ Download fields only (JSON)", compact, file_name=f"{stem}_fields.json",
                       mime="application/json", width="stretch")
    view = st.radio("Preview", ["Full report", "Fields only"], horizontal=True, label_visibility="collapsed")
    st.code(full if view == "Full report" else compact, language="json")


# --------------------------------------------------------------------------- main


def main() -> None:
    st.markdown(
        '<div class="app-header"><h1>Change Order Intelligence</h1>'
        "<p>Extract validated, evidence-backed data from construction change orders — PDFs, scans or "
        "raw text.</p></div>",
        unsafe_allow_html=True,
    )
    try:
        settings = _base_settings()
    except ConfigurationError as exc:
        st.error(f"Configuration error: {exc}. Fix your .env file and reload.")
        st.stop()

    provider, force_ocr, save_copy = sidebar(settings)
    selected = input_panel()
    clicked = st.button("Process document", type="primary", disabled=selected is None)
    if selected is None:
        st.caption("Upload a file, paste text, or choose a sample document to begin.")

    if clicked and selected is not None:
        kind, payload, name = selected
        pipeline = _pipeline(provider)
        with st.status(f"Processing **{name}**…", expanded=True) as status:
            try:
                st.write("1/3 · Reading document (text layer, OCR fallback)…")
                document = load_document(pipeline, kind, payload, name, force_ocr)
                ocr_note = f", OCR on page(s) {document.ocr_pages}" if document.ocr_pages else ""
                st.write(f"   ↳ {len(document.pages)} page(s), {len(document.text):,} characters{ocr_note}")
                st.write(f"2/3 · Extracting fields (engine: **{pipeline.provider}**)…")
                result = pipeline.process_document(document)
                st.write("3/3 · Validated: normalization, evidence grounding, cross-field checks, scoring")
                st.session_state["result"] = result
                st.session_state["source_text"] = _format_source(document)
                if save_copy:
                    path = save_result(result, settings.output_dir)
                    st.write(f"Saved to `{path}`")
                status.update(label=f"Processed **{name}** in {result.extraction.processing_seconds:.2f}s",
                              state="complete", expanded=False)
            except DocumentProcessingError as exc:
                status.update(label="Could not process the document", state="error", expanded=True)
                st.error(str(exc))
                st.session_state.pop("result", None)
            except Exception as exc:  # last-resort guard so the UI never shows a raw traceback
                status.update(label="Unexpected error", state="error", expanded=True)
                st.exception(exc)
                st.session_state.pop("result", None)

    if "result" in st.session_state:
        st.divider()
        results_panel(st.session_state["result"])


main()
