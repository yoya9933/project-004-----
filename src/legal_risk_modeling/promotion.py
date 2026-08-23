from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotionPolicy:
    validation_rmse_improvement_min: float = 0.01
    validation_mae_improvement_min: float = 0.005
    test_rmse_regression_max: float = 0.0
    latest_rmse_regression_max: float = 0.0
    require_test_not_worse: bool = True
    require_latest_not_worse: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    model: str
    passed: bool
    reasons: list[str]
    metrics: dict[str, Any]
    policy: dict[str, Any]


def load_metrics_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("mae", "rmse", "r2", "bucket_accuracy"):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
        row["n"] = int(row["n"])
    return rows


def _by_model_split(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["model"]), str(row["split"])): row for row in rows}


def evaluate_model(
    rows: list[dict[str, Any]],
    model: str,
    baseline: str = "mean_baseline",
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    policy = policy or PromotionPolicy()
    indexed = _by_model_split(rows)
    reasons: list[str] = []

    required = ["validation_2024", "test_2025", "latest_2026"]
    for split in required:
        if (model, split) not in indexed or (baseline, split) not in indexed:
            reasons.append(f"missing metrics for {split}")

    if reasons:
        return PromotionDecision(model, False, reasons, {}, asdict(policy))

    val = indexed[(model, "validation_2024")]
    val_base = indexed[(baseline, "validation_2024")]
    test = indexed[(model, "test_2025")]
    test_base = indexed[(baseline, "test_2025")]
    latest = indexed[(model, "latest_2026")]
    latest_base = indexed[(baseline, "latest_2026")]

    val_rmse_gain = float(val_base["rmse"]) - float(val["rmse"])
    val_mae_gain = float(val_base["mae"]) - float(val["mae"])
    test_rmse_delta = float(test["rmse"]) - float(test_base["rmse"])
    latest_rmse_delta = float(latest["rmse"]) - float(latest_base["rmse"])

    if val_rmse_gain < policy.validation_rmse_improvement_min:
        reasons.append(
            f"validation RMSE gain {val_rmse_gain:.4f} < required {policy.validation_rmse_improvement_min:.4f}"
        )
    if val_mae_gain < policy.validation_mae_improvement_min:
        reasons.append(
            f"validation MAE gain {val_mae_gain:.4f} < required {policy.validation_mae_improvement_min:.4f}"
        )
    if policy.require_test_not_worse and test_rmse_delta > policy.test_rmse_regression_max:
        reasons.append(f"test RMSE regressed by {test_rmse_delta:.4f}")
    if policy.require_latest_not_worse and latest_rmse_delta > policy.latest_rmse_regression_max:
        reasons.append(f"latest RMSE regressed by {latest_rmse_delta:.4f}")

    metrics = {
        "validation_rmse_gain_vs_baseline": round(val_rmse_gain, 6),
        "validation_mae_gain_vs_baseline": round(val_mae_gain, 6),
        "test_rmse_delta_vs_baseline": round(test_rmse_delta, 6),
        "latest_rmse_delta_vs_baseline": round(latest_rmse_delta, 6),
        "validation": val,
        "test": test,
        "latest": latest,
    }
    return PromotionDecision(model, not reasons, reasons, metrics, asdict(policy))


def build_promotion_report(
    rows: list[dict[str, Any]],
    candidate_models: list[str] | None = None,
    policy: PromotionPolicy | None = None,
) -> dict[str, Any]:
    models = candidate_models or sorted({str(row["model"]) for row in rows if row["model"] != "mean_baseline"})
    decisions = [evaluate_model(rows, model, policy=policy) for model in models]
    approved = [decision.model for decision in decisions if decision.passed]
    return {
        "baseline": "mean_baseline",
        "approved_models": approved,
        "default_model": approved[0] if approved else "mean_baseline",
        "decisions": [asdict(decision) for decision in decisions],
    }
