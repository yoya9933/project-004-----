from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_risk_modeling.data_quality import validate_annotation_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the governed annotation dataset before training.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "annotation_workbook.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / ".artifacts" / "data_quality" / "annotation_workbook.json",
    )
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--min-labeled-rows", type=int, default=30)
    args = parser.parse_args()

    if args.min_rows <= 0 or args.min_labeled_rows <= 0:
        parser.error("--min-rows and --min-labeled-rows must be positive")
    if not args.input_csv.is_file():
        parser.error(f"input CSV not found: {args.input_csv}")

    report = validate_annotation_csv(
        args.input_csv,
        output_json=args.output_json,
        min_rows=args.min_rows,
        min_labeled_rows=args.min_labeled_rows,
    )
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
