from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from script_to_doc.pipeline import PipelineConfig, run_pipeline

# Expose the FastAPI app when this file is imported by platforms like Vercel.
try:
    from api import app as fastapi_app  # type: ignore
except Exception:  # pragma: no cover - only used in serverless runtime
    fastapi_app = None

app = fastapi_app


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training Word documents from transcript text files.",
    )
    parser.add_argument("input_dir", help="Directory containing transcript .txt files")
    parser.add_argument("output_dir", help="Directory where Word documents will be saved")
    parser.add_argument(
        "--summary-ratio",
        type=float,
        default=0.2,
        help="Target ratio of sentences to keep in the summary (default: 0.2)",
    )
    parser.add_argument(
        "--summary-word-count",
        type=int,
        default=None,
        help="Optional absolute word count for the summary",
    )
    parser.add_argument(
        "--max-summary-sentences",
        type=int,
        default=10,
        help="Maximum number of sentences to keep in the summary",
    )
    parser.add_argument(
        "--min-summary-sentences",
        type=int,
        default=3,
        help="Minimum number of sentences required before summarization",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON results",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    config = PipelineConfig(
        summary_ratio=args.summary_ratio,
        summary_word_count=args.summary_word_count,
        max_summary_sentences=args.max_summary_sentences,
        min_summary_sentences=args.min_summary_sentences,
    )

    try:
        results = run_pipeline(input_dir, output_dir, config=config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        serializable = [
            {
                "transcript": str(item["transcript"]),
                "steps": item["steps"],
                "document_path": str(item["document_path"]),
            }
            for item in results
        ]
        print(json.dumps(serializable, indent=2))
    else:
        for item in results:
            print(f"Processed {item['transcript'].name} -> {item['document_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
