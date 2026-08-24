from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_risk_modeling.classification import prepare_classification_frame, run_classification
from legal_risk_modeling.features import FEATURE_NAMES
from legal_risk_modeling.manifest import validate_manifest
from legal_risk_modeling.models import build_classification_model


def make_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year, count in [(2021, 4), (2022, 4), (2023, 4), (2024, 4), (2025, 4), (2026, 4)]:
        for index in range(count):
            reduced = index % 2
            claimed = 100000 + year + index * 100
            allowed = claimed * (0.5 if reduced else 1.0)
            rows.append(
                {
                    "JID": f"{year}-{index}",
                    "decision_year": year,
                    "is_reduced": reduced,
                    "contract_price": 1000000 + index * 5000,
                    "claimed_penalty": claimed,
                    "allowed_penalty": allowed,
                    "delay_days": 10 + index,
                    "issue_owner_fault": reduced,
                    "issue_contractor_fault": 1 - reduced,
                    "key_reason": "法院酌減違約金" if reduced else "維持約定違約金",
                }
            )
    return rows


def test_classification_uses_shared_feature_contract() -> None:
    frame = prepare_classification_frame(pd.DataFrame(make_rows()))
    assert all(feature in frame.columns for feature in FEATURE_NAMES)
    model = build_classification_model(l2=0.01, random_state=42)
    train = frame[frame["decision_year"].between(2021, 2023)]
    model.fit(train[FEATURE_NAMES], train["is_reduced_label"].astype(int))
    assert model.predict_proba(train[FEATURE_NAMES]).shape == (12, 2)


def test_classification_run_writes_governed_release_and_manifest_v2(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    input_csv = input_dir / "annotation_workbook.csv"
    pd.DataFrame(make_rows()).to_csv(input_csv, index=False, encoding="utf-8-sig")

    result = run_classification(
        project_root=tmp_path,
        input_csv=input_csv,
        output_dir=output_dir,
        min_labeled_rows=10,
        run_id="test-classification",
    )
    assert result["status"] == "trained"
    assert result["promotion_status"] in {"approved", "rejected"}
    assert (output_dir / "promotion_report.json").exists()
    release = json.loads((output_dir / "approved_release.json").read_text(encoding="utf-8"))
    assert release["status"] == "approved"
    assert release["default_model"] in {"majority_baseline", "logistic_regression_l2"}
    manifest = validate_manifest(project_root=tmp_path, manifest_path=output_dir / "manifest.json")
    assert manifest["schema_version"] == 2
    assert manifest["run_id"] == "test-classification"
    assert manifest["datasets"][0]["sha256"]
    assert manifest["artifacts"]
    assert manifest["model_spec"]["target"] == "is_reduced"
    assert manifest["temporal_split_policy"]["validation_year"] == 2024
