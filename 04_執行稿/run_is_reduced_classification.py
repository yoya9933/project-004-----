from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_risk_modeling.classification import run_classification  # noqa: E402
from legal_risk_modeling.cli_contract import (  # noqa: E402
    ensure_existing_csv,
    non_negative_float,
    positive_int,
    validate_temporal_split,
)
from legal_risk_modeling.paths import classification_output_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Governed is_reduced classification pipeline.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "annotation_workbook.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=classification_output_dir(PROJECT_ROOT))
    parser.add_argument("--min-labeled-rows", type=positive_int, default=30)
    parser.add_argument("--l2", type=non_negative_float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-start-year", type=int, default=2021)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--latest-check-year", type=int, default=2026)
    parser.add_argument("--use-derived-label-from-amounts", action="store_true")
    parser.add_argument("--run-id", default="classification-governed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = ensure_existing_csv(args.input_csv)
    validate_temporal_split(
        args.train_start_year,
        args.train_end_year,
        args.validation_year,
        args.test_year,
        args.latest_check_year,
    )
    result = run_classification(
        project_root=PROJECT_ROOT,
        input_csv=input_csv,
        output_dir=args.output_dir,
        min_labeled_rows=args.min_labeled_rows,
        l2=args.l2,
        random_state=args.random_state,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        validation_year=args.validation_year,
        test_year=args.test_year,
        latest_check_year=args.latest_check_year,
        use_derived_label_from_amounts=args.use_derived_label_from_amounts,
        run_id=args.run_id,
        strict_git=True,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
