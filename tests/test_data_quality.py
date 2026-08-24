from __future__ import annotations

import pandas as pd

from legal_risk_modeling.data_quality import validate_annotation_frame


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"JID": "A", "decision_year": 2021, "is_reduced": 0, "contract_price": 1000, "claimed_penalty": 100, "allowed_penalty": 100, "delay_days": 5},
            {"JID": "B", "decision_year": 2022, "is_reduced": 1, "contract_price": 1200, "claimed_penalty": 120, "allowed_penalty": 60, "delay_days": 8},
            {"JID": "C", "decision_year": 2023, "is_reduced": "否", "contract_price": 900, "claimed_penalty": 90, "allowed_penalty": 90, "delay_days": 3},
            {"JID": "D", "decision_year": 2024, "is_reduced": "是", "contract_price": 2000, "claimed_penalty": 200, "allowed_penalty": 100, "delay_days": 10},
        ]
    )


def test_valid_annotation_frame_passes() -> None:
    report = validate_annotation_frame(_valid_frame(), min_rows=4, min_labeled_rows=4)
    assert report["status"] == "pass"
    assert report["stats"]["unique_jids"] == 4
    assert report["stats"]["class_counts"] == {"0": 2, "1": 2}


def test_duplicate_jid_fails() -> None:
    frame = _valid_frame()
    frame.loc[1, "JID"] = "A"
    report = validate_annotation_frame(frame, min_rows=4, min_labeled_rows=4)
    assert report["status"] == "fail"
    assert "duplicate_jid" in {item["code"] for item in report["errors"]}


def test_invalid_year_label_and_negative_value_fail() -> None:
    frame = _valid_frame()
    frame["decision_year"] = frame["decision_year"].astype(object)
    frame.loc[0, "decision_year"] = "unknown"
    frame["is_reduced"] = frame["is_reduced"].astype(object)
    frame.loc[1, "is_reduced"] = "maybe"
    frame.loc[2, "delay_days"] = -1
    report = validate_annotation_frame(frame, min_rows=4, min_labeled_rows=2)
    codes = {item["code"] for item in report["errors"]}
    assert report["status"] == "fail"
    assert "invalid_decision_year" in codes
    assert "invalid_is_reduced_label" in codes
    assert "negative_numeric_values" in codes
