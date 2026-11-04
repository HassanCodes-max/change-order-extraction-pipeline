# Change-Order Extraction Pipeline

Turns messy construction change-order documents (text PDFs, scanned PDFs, images or raw text) into
**validated JSON**. Every field carries a value, a confidence score in [0, 1], verbatim evidence from
the source, and an explanation of how that score was computed.

```
PDF / scan / image / text  ──►  text  ──►  LLM + rule-based extraction  ──►  validation  ──►  JSON
```

* **Runs with or without an API key.** With `ANTHROPIC_API_KEY` (or OpenAI settings), Claude does the
  semantic extraction and a deterministic rule engine cross-checks it. Without a key, the rule engine
  runs alone ("demo mode") with the same validation and scoring.
* **Never invents data.** Missing fields are `null` with confidence `0.0`. Contradictory source values
  are reported as written and flagged; they are never "corrected".
* **Confidence is computed, not asserted.** Scores come from evidence grounding, agreement between the
  two extractors, OCR word confidence and cross-field checks. The LLM's self-reported confidence can
  only lower a score.

---

## 1. Problem statement

Change orders arrive as scanned forms, e-mails, AIA-style letters and spreadsheet exports, with
inconsistent labels ("CO #", "Change Order No.", "co no."), credits written as `(3,200.00)`, durations
written as "two weeks", conflicting numbers between header and body, and arithmetic that doesn't add
up. Downstream systems such as cost control and ERP need clean, typed data. Reviewers need to know
**which values to trust and why**.

## 2. Architecture

```
app.py (Streamlit UI)        src/cli.py (command line)
          │                          │
          └────────► src/pipeline.py ◄┘   ChangeOrderPipeline: load_* → process_document
                          │
   ┌──────────────────────┼────────────────────────────┬───────────────────────────┐
   ▼                      ▼                            ▼                           ▼
src/extraction/        src/intelligence/            src/validation/            src/serialization.py
 pdf_extractor.py       llm_extractor.py  (Claude/    schemas.py   (Pydantic     to_json / compact /
  PyMuPDF text layer      OpenAI, structured output,    models, field registry)  save_result
  per-page OCR fallback   retries, error mapping)     validators.py (dates,
 ocr_extractor.py       rule_extractor.py (labels,     money, days, status,
  Tesseract + adaptive    columns, AIA prose)          cross-field checks)
  denoise retry         reconcile.py (LLM vs rules)
 text_processor.py      confidence.py (scoring)
  cleaning, layout,     prompt_templates.py (prompt +
  evidence grounding      JSON schema from registry)
 document.py
```

**Workflow (`ChangeOrderPipeline.process_document`):**

1. **Ingest.** PDF pages with a text layer are read with PyMuPDF. Words are regrouped into visual rows
   so `Label      Value` columns stay together. Pages with fewer than 40 characters are rasterized
   (300 DPI) and OCR'd with Tesseract. A low-confidence first OCR pass is retried with stronger
   denoising. Raw text is cleaned (unicode, hyphenation, whitespace).
2. **Extract (two independent passes).**
   * *LLM pass* (if configured): one structured-output call. The JSON schema is generated from the
     field registry. The model returns raw values, a verbatim evidence quote, a status
     (`stated/inferred/conflicting/missing`), alternatives and a self-confidence.
   * *Rule pass* (always): label vocabulary, column layouts, values printed below labels, multi-line
     descriptions, checkboxes, and AIA prose ("The Contract Sum will be increased by…").
3. **Reconcile** per field: agreement, disagreement, recovery of fields the LLM missed, conflicts the
   LLM didn't report, and replacement of ungrounded LLM values when the rules found grounded ones.
4. **Validate.** Type normalization, evidence grounding, and a value-in-evidence check, followed by
   cross-field checks (contract arithmetic, date order, approval vs. approver, required fields).
5. **Score.** Explainable confidence per field (see §5).
6. **Serialize.** Validate against the Pydantic schema, round-trip through JSON, then output.

## 3. Technology choices

| Concern | Choice | Why |
|---|---|---|
| PDF text | **PyMuPDF** | Fast, pure-wheel install on Windows; word boxes for layout reconstruction; rasterization for OCR. |
| OCR | **Tesseract 5** via `pytesseract` | Free, CPU-only, gives per-word confidence (used in scoring). |
| LLM | **Anthropic Claude** (default `claude-opus-5-5`), **OpenAI** optional | Official SDKs; JSON-schema structured output on both; provider chosen by env vars. |
| Validation | **Pydantic v2** | Strict typed schema, generic `ExtractedField[T]`, invariants enforced at construction. |
| Retries | **tenacity** | Exponential backoff for transient errors and malformed output, on top of SDK HTTP retries. |
| UI | **Streamlit** | Upload, review and download in one lightweight app. |

There is no GPU requirement, no heavyweight ML framework, and no database. Everything runs on a normal
Windows laptop.

## 4. Extraction methodology

**Field registry.** `FIELD_SPECS` in [schemas.py](src/validation/schemas.py) defines all 18 fields once
(name, label, type, required). The prompt, JSON schema, validators and UI are generated from it.

| Field | Type | Required |
|---|---|---|
| project_name, change_order_number | text | ✔ |
| change_order_date | date | ✔ |
| description_of_changes | long text | ✔ |
| project_number, original_contract_number, contractor_name, client_name, prepared_by, approved_by | text | |
| reason_for_change | long text | |
| original_contract_amount, additional_cost, deducted_amount, revised_contract_amount | money | |
| schedule_impact_days | signed integer days | |
| requested_completion_date | date | |
| approval_status | `approved \| approved_with_conditions \| pending \| rejected \| void` | |

**LLM design** ([llm_extractor.py](src/intelligence/llm_extractor.py),
[prompt_templates.py](src/intelligence/prompt_templates.py)):

* **Structured output.** Anthropic uses `output_config.format` (JSON schema) and OpenAI uses
  `response_format: json_schema, strict: true`. The response is still re-validated with Pydantic, so
  malformed output (invalid JSON, missing fields, out-of-range confidence) triggers a retry.
* **No reformatting by the model.** The prompt asks for values *as written* plus a verbatim quote.
  Normalization stays deterministic and testable in code.
* **Injection hardening.** The prompt tells the model to treat the document as untrusted data and
  never to compute missing amounts.
* **Error policy.** Rate limits, 5xx errors, network failures and bad output are retried
  (`LLM_MAX_ATTEMPTS`). Auth errors, unknown models, rejected requests and refusals are not retried.
  In either case the pipeline **falls back to the rule engine** and records `fallback_reason`.
* **Refusal fallback (Claude).** The server-side fallback beta (`fallbacks: "default"`) is enabled by
  default, so a policy decline is re-run on a fallback model within the same call. Disable it with
  `ANTHROPIC_REFUSAL_FALLBACK=false`.
* **No truncation.** Documents over `LLM_MAX_INPUT_CHARS` are never truncated. The pipeline uses the
  rule engine instead and says why.

**Normalization** ([validators.py](src/validation/validators.py)) reports every interpretation in the
field's `notes`:

* **Dates:** ISO, `03/14/2025`, `14.03.25`, `March 3, 2025`, `3rd of March 2025`. Numeric dates where
  either reading is valid are flagged *ambiguous* and read using `DATE_ORDER` (default MDY).
  Impossible dates such as `02/30/2026` become status `invalid` with value `null`; the raw text is kept
  in `raw_value`.
* **Money:** `$`, `USD`, `EUR`, thousands separators, European `12.500,00`, `12.5k`, parenthesized or
  negative credits, and "None" (stored as 0, with a note). A spaced dash ("Cost - $500") is a
  separator, not a minus sign.
* **Days:** digits, number words ("five (5) calendar days"), weeks (×7, marked *converted*), "no
  change" (0) and reductions (negative). Months are rejected rather than guessed. Working days are
  flagged as not converted.

## 5. Confidence-scoring methodology

Implemented in [confidence.py](src/intelligence/confidence.py). For a field with a value:

```
score = base(status)                          stated 0.90 · inferred 0.65 · conflicting 0.45
      × evidence grounding                    exact 1.00 · fuzzy 0.92 · weak 0.75 · none 0.60 · not found 0.45
      × value-in-evidence check               0.75 if the quote does not contain the value
      × normalization                         ambiguous date 0.80 · unit/word conversion 0.95
      × OCR word confidence                   0.5 + 0.5 × (confidence of the value's own words / 100)
      × extractor cross-check                 disagree 0.80 · recovered-by-rules 0.85
      × cross-field validation                arithmetic mismatch 0.70 · consistent 1.05 · date order 0.80 …
      × model self-confidence (dampening)     0.75 + 0.25 × p
      + agreement bonus                       LLM and rules agree: closes 40% of the gap to 1.0
      capped at 0.99;  null value → 0.00
```

* **Grounding** ([text_processor.py](src/extraction/text_processor.py)): the evidence quote is
  searched in the document (whitespace/case/quote-insensitive), with a token-level fuzzy fallback for
  OCR noise. Evidence that cannot be found is the strongest hallucination signal. It drops the score
  below 0.5 and raises `UNGROUNDED_EVIDENCE`.
* **OCR** uses the Tesseract confidence of the specific words that make up the value, not the page
  average. A misread `'Pending` on a 92/100 page still scores low.
* Every applied factor is listed in `confidence_factors`, for example:

```json
"confidence_factors": ["base for 'stated': 0.90", "x1.00 evidence match in source: exact",
                       "x0.70 cross-field: contract arithmetic mismatch", "= 0.63"]
```

**Status semantics:** `stated` (written explicitly), `inferred` (derived; model must explain),
`conflicting` (document gives different values: `value` holds the primary one, `alternatives` the
others), `invalid` (found but not normalizable: `value=null`, `raw_value` kept), `missing` (`null`, 0.0).

The weights are hand-calibrated heuristics, not learned probabilities. See §13.

## 6. Validation methodology

* **Schema** ([schemas.py](src/validation/schemas.py)): `extra="forbid"` everywhere; confidence bounded
  to [0, 1]. Invariants are enforced at construction: `stated`/`inferred` require a value, a `null`
  value requires confidence 0, and `missing` requires `null`. Every result is round-tripped through JSON
  before it is returned.
* **Cross-field checks** (`validate_cross_fields`):
  * `original + additional − deducted = revised` (±`AMOUNT_TOLERANCE`), also accounting for
    previously approved change orders when the document lists them (AIA G701 style).
    A mismatch is an **error** and shows the expected value and the difference. Values are **not
    modified**.
  * If the revised amount is missing, the computed value is shown as *information only*. It is never
    inserted.
  * Required fields; non-positive contract sums; negative additions; changes over 50% of the contract.
  * Completion date before the CO date; implausible years.
  * "Approved" without an approver, or an approver named while the status is pending or rejected.
  * Document level: OCR quality, LLM fallback, garbled text layers, "this doesn't look like a change
    order".
* **Summary**: counts per status, `missing_required`, mean confidence, and `needs_review` with
  human-readable `review_reasons`.

## 7. Installation (Windows, CPU-only)

Requires Python 3.10+ (developed and tested on 3.14).

```powershell
git clone <your-repo-url> change-order-extraction-pipeline
cd change-order-extraction-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# OCR engine (needed only for scanned PDFs / images)
winget install UB-Mannheim.TesseractOCR
```

Tesseract is auto-detected on PATH or in `C:\Program Files\Tesseract-OCR`. Otherwise set
`TESSERACT_CMD`. On macOS use `brew install tesseract`; on Debian/Ubuntu use
`apt install tesseract-ocr`. Without Tesseract, text PDFs and raw text still work, and the app tells
you when OCR is missing.

## 8. Environment configuration

```powershell
Copy-Item .env.example .env   # then edit .env
```

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto` · `anthropic` · `openai` · `rules` (`demo`). `auto` = Anthropic if `ANTHROPIC_API_KEY` is set, else OpenAI if `OPENAI_API_KEY` and `OPENAI_MODEL` are set, else rules. |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | – / `claude-opus-5-5` | Claude credentials and model. |
| `ANTHROPIC_EFFORT` | `medium` | `low`…`max`; `none` omits the parameter for models that don't support it. |
| `ANTHROPIC_REFUSAL_FALLBACK` | `true` | Server-side refusal fallback (beta). |
| `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` | – | OpenAI or a compatible endpoint. The model must be set explicitly. |
| `LLM_MAX_ATTEMPTS`, `LLM_SDK_MAX_RETRIES`, `LLM_TIMEOUT_SECONDS` | 3, 2, 120 | Retry and timeout policy. |
| `TESSERACT_CMD`, `OCR_DPI`, `OCR_LANGUAGE`, `OCR_PSM` | auto, 300, eng, 3 | OCR. |
| `DATE_ORDER` | `MDY` | Reading of ambiguous numeric dates (`DMY` for non-US documents). |
| `MAX_PDF_PAGES`, `MAX_UPLOAD_MB`, `LLM_MAX_INPUT_CHARS` | 50, 25, 200000 | Resource limits. |

API keys are read only from the environment or `.env`. `.env` is git-ignored.

## 9. Running

```powershell
# Web UI  ->  http://localhost:8501
streamlit run app.py

# CLI: full report, compact field map, or save to outputs/
python -m src.cli sample_documents\co_014_standard.pdf
python -m src.cli sample_documents\co_031_conflicting.pdf --compact
python -m src.cli sample_documents\co_007_scanned.pdf --provider rules --save
python -m src.cli --text "Change Order No: CO-9`nProject: Elm Plaza`nAdditional Cost: $2,000" --compact

# Regenerate the synthetic samples
python -m scripts.generate_samples
```

The UI provides:

* input by upload, paste or built-in sample;
* engine selection, with live OCR availability;
* a staged processing status;
* summary metrics and a review banner;
* a field table with confidence bars;
* per-field evidence, raw value, alternatives, notes and the confidence breakdown;
* validation issues grouped by severity;
* a JSON preview and downloads (full report or fields only);
* the exact source text the extractors saw.

## 10. Sample documents and output

All synthetic and fictional ([scripts/generate_samples.py](scripts/generate_samples.py)):

| File | Exercises |
|---|---|
| `co_014_standard.pdf` | Clean two-column form, consistent arithmetic → 18/18 fields |
| `co_007_scanned.pdf` | Image-only PDF (skewed, noisy) → OCR path, per-word OCR confidence |
| `co_022_missing_fields.txt` | 11 of 18 fields absent → nulls, never guessed |
| `co_031_conflicting.pdf` | CO-031 vs CO-013, 10 vs 14 days, revised amount off by $9,000, blank approver line |
| `co_045_messy_format.txt` | Lowercase labels, dot leaders, `(3,200.00)` credit, "two weeks", invalid `02/30/2026` |
| `co_052_narrative_email.txt` | No labels: AIA-style prose in an e-mail (where the LLM matters most) |

Compact output (`--compact`) for `co_014_standard.pdf`, from the rule engine:

```json
{
  "change_order_number": {"value": "CO-014", "confidence": 0.9,
                          "evidence": "Change Order No.:   CO-014", "status": "stated"},
  "additional_cost": {"value": 12500.0, "confidence": 0.95,
                      "evidence": "Additional Cost:   $12,500.00", "status": "stated"},
  "schedule_impact_days": {"value": 5, "confidence": 0.9, "status": "stated",
                           "evidence": "Schedule Impact: Project completion extended by five (5) calendar days."}
}
```

A conflicting field from `co_031_conflicting.pdf` (full report format):

```json
"change_order_number": {
  "value": "CO-031", "confidence": 0.45, "evidence": "CHANGE ORDER No. CO-031",
  "status": "conflicting", "raw_value": "CO-031", "source": "rules", "alternatives": ["CO-013"],
  "notes": ["document states different values for this field: 'CO-031' (line 1), 'CO-013' (line 4)"],
  "confidence_factors": ["base for 'conflicting': 0.45", "x1.00 evidence match in source: exact", "= 0.45"]
}
```

The full reports are in [examples/](examples/). Each has `document`, `extraction`, `fields`,
`validation_issues` and `summary`.

## 11. Testing

```powershell
python -m pytest            # 102 tests, about 45 s (OCR tests are skipped if Tesseract is absent)
```

| File | Covers |
|---|---|
| [test_extraction.py](tests/test_extraction.py) | Text-layer PDFs, scanned-PDF OCR, noisy scans, unreadable scans, force-OCR, corrupt and encrypted PDFs, layout reconstruction, evidence grounding, rule extraction on standard, messy, conflicting, checkbox and AIA-prose inputs |
| [test_validation.py](tests/test_validation.py) | Date, currency, day and status normalization (valid, ambiguous, invalid), cross-field checks, schema invariants, confidence monotonicity and bounds, JSON round-trip |
| [test_pipeline.py](tests/test_pipeline.py) | End-to-end on every sample, missing/conflicting/invalid handling, the LLM path with fake providers (agreement, hallucination caught, recovered fields, inferred values, retry-then-succeed, fallback after failures, non-retryable errors, no truncation), Anthropic/OpenAI adapters against fake SDK clients (request shape, refusals, truncation, fallback blocks), config parsing, CLI |

**Verification status:** the LLM integration is tested against fake SDK clients that check the
request shape and the handling of every response type. It has **not** been run against the live
Anthropic or OpenAI APIs in this repository's test suite, because no keys are required or assumed. To
check it live, set a key and run `python -m src.cli sample_documents\co_052_narrative_email.txt`.

## 12. Known limitations and failure modes

| Failure mode | How it is detected | How it is handled or communicated |
|---|---|---|
| **Poor-quality scans** | Tesseract per-word and page confidence; noise heuristic | Stronger-denoise OCR retry; confidence scaled by the value's own word confidence; `OCR_USED` warning below 70/100; an unreadable scan either raises a clear error or returns text flagged with quality warnings, never silently empty fields. |
| **Incorrect OCR results** (`0`↔`O`, stray marks) | Fuzzy grounding; normalization failures; arithmetic check | Stray edge characters stripped (with a note); values that fail to parse become `invalid`; misread amounts usually break the arithmetic check. |
| **Handwritten information** | Tesseract is not a handwriting recognizer; such regions get low or no confidence | Typically missing or low confidence and flagged for review. **Not supported** as an extraction source. |
| **LLM hallucination** | Every evidence quote is located in the source; the value must appear in its quote; independent rule-engine cross-check | Ungrounded evidence: ×0.45 and `UNGROUNDED_EVIDENCE`. If the rules found a grounded value, it replaces the LLM's, and the LLM value is kept in `alternatives`. |
| **Conflicting contract values** | Rule engine collects every labeled occurrence; LLM reports `conflicting`; arithmetic check | `status: conflicting` with `alternatives`; arithmetic mismatch is an **error** with the expected value; nothing is silently fixed. |
| **Missing information** | Field absent from both extractors | `null`, confidence 0, `missing`; required fields raise `REQUIRED_FIELD_MISSING`; the revised amount is never computed into the output. |
| **Ambiguous descriptions or dates** | Month/day ambiguity; LLM `inferred` status | Ambiguous dates ×0.80 with a note (`DATE_ORDER` configurable); inferred values start at 0.65 and need a model note. |
| **Incorrect confidence estimates** | – | Weights are hand-tuned on synthetic data, **not calibrated** against labeled real documents. Treat scores as a ranking and triage signal, not as probabilities. Calibrating on a labeled set is the next step. |
| **Unusual layouts** | – | Row reconstruction suits forms and tables but interleaves side-by-side prose columns. Multi-page tables, rotated pages and nested tables are not specially handled. The rule engine is label-dependent: free-form prose needs the LLM (see `co_052` in demo mode). |
| **Not a change order** | LLM `is_change_order`, or a keyword heuristic | `NOT_A_CHANGE_ORDER` warning and `needs_review`. |
| **LLM unavailable or refused** | Typed SDK exceptions, `stop_reason` | Retries for transient errors; rule-engine fallback with `fallback_reason` in the output and the UI. |

Other limitations:

* One change order per document: multi-CO packets are not split.
* Amounts are assumed to share one currency (a non-USD currency is noted, not converted).
* Line items are not extracted individually.
* Month durations are not converted to days.
* Only English labels are supported (Tesseract language is configurable).

## 13. Publishing to GitHub

```powershell
git init
git add .
git status                 # confirm .env is NOT listed
git commit -m "Change-order extraction pipeline"
gh repo create change-order-extraction-pipeline --private --source . --push
# or: create an empty repo on github.com, then
# git remote add origin https://github.com/<you>/change-order-extraction-pipeline.git
# git branch -M main; git push -u origin main
```

## Project layout

```
app.py                     Streamlit UI
src/config.py              environment settings
src/pipeline.py            orchestration
src/serialization.py       JSON output
src/cli.py                 command line
src/extraction/            document.py, pdf_extractor.py, ocr_extractor.py, text_processor.py
src/intelligence/          llm_extractor.py, prompt_templates.py, rule_extractor.py, reconcile.py, confidence.py
src/validation/            schemas.py, validators.py
scripts/generate_samples.py  synthetic documents
sample_documents/          generated samples
examples/                  example JSON outputs
tests/                     pytest suite
outputs/                   saved reports (git-ignored)
```

Additions to the suggested structure: `rule_extractor.py` (offline mode, fallback and cross-check),
`reconcile.py` (merging policy kept separate from scoring), `document.py` (shared document model),
`serialization.py` (output separated from validation) and `cli.py` (scriptable demo).
