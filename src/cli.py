"""Command-line interface.

Examples::

    python -m src.cli sample_documents/co_014_standard.pdf
    python -m src.cli sample_documents/*.pdf --provider rules --save
    python -m src.cli --text "Change Order No: CO-9 ..." --compact
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import ConfigurationError, Settings
from src.extraction.document import DocumentProcessingError
from src.pipeline import ChangeOrderPipeline
from src.serialization import save_result, to_compact_json, to_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract structured data from change-order documents.")
    parser.add_argument("files", nargs="*", type=Path, help="PDF, image or .txt files")
    parser.add_argument("--text", help="raw text to process instead of files")
    parser.add_argument("--provider", choices=["auto", "anthropic", "openai", "rules"],
                        help="override LLM_PROVIDER")
    parser.add_argument("--force-ocr", action="store_true", help="OCR every PDF page")
    parser.add_argument("--compact", action="store_true", help="print only {field: value/confidence/evidence}")
    parser.add_argument("--save", action="store_true", help="write full JSON reports to OUTPUT_DIR")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not args.files and not args.text:
        parser.error("provide at least one file or --text")

    try:
        settings = Settings.from_env()
        provider = None if args.provider in (None, "auto") else args.provider
        pipeline = ChangeOrderPipeline(settings, provider=provider)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    exit_code = 0
    jobs = [("--text", None)] if args.text else []
    jobs += [(str(p), p) for p in args.files]
    for label, path in jobs:
        try:
            result = pipeline.process_text(args.text, "cli_text") if path is None else pipeline.process_file(
                path, force_ocr=args.force_ocr)
        except DocumentProcessingError as exc:
            print(f"[{label}] error: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        print(to_compact_json(result) if args.compact else to_json(result))
        if args.save:
            print(f"[{label}] saved to {save_result(result, settings.output_dir)}", file=sys.stderr)
        s = result.summary
        print(f"[{label}] engine={result.extraction.engine} extracted={s.extracted}/{s.total_fields} "
              f"overall_confidence={s.overall_confidence:.2f} needs_review={s.needs_review}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
