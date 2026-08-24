from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_risk_modeling.classification import prepare_classification_frame  # noqa: E402
from legal_risk_modeling.cli_contract import ensure_existing_csv  # noqa: E402
from legal_risk_modeling.data_quality import (  # noqa: E402
    validate_classification_model_frame,
    validate_ratio_model_frame,
)
from legal_risk_modeling.paths import data_quality_output_dir  # noqa: E402
from legal_risk_modeling.temporal import TemporalSplitPolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model-specific data quality profiles.")
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "annotation_workbook.csv",
    )
    parser.add_argument(
        "--ratio-csv",
        type=Path,
        default=PROJECT_ROOT
        / "06_交付物"
        / "reduction_ratio_model_expanded_824"
        / "usable_ratio_model_frame.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=data_quality_output_dir(PROJECT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classification_csv = ensure_existing_csv(args.classification_csv)
    ratio_csv = ensure_existing_csv(args.ratio_csv)
    policy = TemporalSplitPolicy()

    classification_raw = pd.read_csv(classification_csv, encoding="utf-8-sig")
    classification_frame = prepare_classification_frame(classification_raw)
    classification_frame = classification_frame[classification_frame["is_reduced_label"].notna()].copy()
    classification_report = validate_classification_model_frame(classification_frame, policy=policy)

    ratio_frame = pd.read_csv(ratio_csv, encoding="utf-8-sig")
    if "target_quality" in ratio_frame.columns:
        ratio_frame = ratio_frame[ratio_frame["target_quality"] == "ok"].copy()
    ratio_report = validate_ratio_model_frame(ratio_frame, policy=policy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classification.json").write_text(
        json.dumps(classification_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ratio.json").write_text(
        json.dumps(ratio_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {"classification": classification_report, "ratio": ratio_report}
    print(json.dumps(result, ensure_ascii=False))
    failed = [name for name, report in result.items() if report["status"] != "pass"]
    if failed:
        raise SystemExit("model data quality gate failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
