from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_risk_modeling.models import build_ratio_candidates
from legal_risk_modeling.promotion import build_promotion_report, evaluate_model


def metric(model: str, split: str, mae: float, rmse: float) -> dict[str, object]:
    return {
        "model": model,
        "split": split,
        "n": 100,
        "mae": mae,
        "rmse": rmse,
        "r2": 0.0,
        "bucket_accuracy": 0.2,
    }


def test_weak_model_cannot_be_promoted() -> None:
    rows = [
        metric("mean_baseline", "validation_2024", 0.35, 0.38),
        metric("mean_baseline", "test_2025", 0.34, 0.37),
        metric("mean_baseline", "latest_2026", 0.33, 0.375),
        metric("ridge_regression_l2", "validation_2024", 0.343, 0.377),
        metric("ridge_regression_l2", "test_2025", 0.339, 0.372),
        metric("ridge_regression_l2", "latest_2026", 0.35, 0.397),
    ]
    decision = evaluate_model(rows, "ridge_regression_l2")
    assert decision.passed is False
    assert any("validation RMSE gain" in reason for reason in decision.reasons)
    assert any("test RMSE regressed" in reason for reason in decision.reasons)
    assert any("latest RMSE regressed" in reason for reason in decision.reasons)


def test_strong_model_can_be_promoted() -> None:
    rows = [
        metric("mean_baseline", "validation_2024", 0.35, 0.38),
        metric("mean_baseline", "test_2025", 0.34, 0.37),
        metric("mean_baseline", "latest_2026", 0.33, 0.375),
        metric("candidate", "validation_2024", 0.33, 0.36),
        metric("candidate", "test_2025", 0.32, 0.35),
        metric("candidate", "latest_2026", 0.31, 0.355),
    ]
    decision = evaluate_model(rows, "candidate")
    assert decision.passed is True
    assert decision.reasons == []


def test_no_passing_model_falls_back_to_mean_baseline() -> None:
    rows = [
        metric("mean_baseline", "validation_2024", 0.35, 0.38),
        metric("mean_baseline", "test_2025", 0.34, 0.37),
        metric("mean_baseline", "latest_2026", 0.33, 0.375),
        metric("candidate", "validation_2024", 0.349, 0.379),
        metric("candidate", "test_2025", 0.341, 0.371),
        metric("candidate", "latest_2026", 0.331, 0.376),
    ]
    report = build_promotion_report(rows, ["candidate"])
    assert report["approved_models"] == []
    assert report["default_model"] == "mean_baseline"


def test_shared_model_source_contains_expected_ridge_search_space() -> None:
    candidates = build_ratio_candidates(random_state=42)
    ridge_names = [name for name, _ in candidates["ridge_regression_l2"]]
    assert "ridge_alpha_0.05" in ridge_names
    assert "ridge_alpha_100" in ridge_names
    assert len(ridge_names) == 7
