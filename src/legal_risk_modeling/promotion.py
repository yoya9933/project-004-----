from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


RATIO_BASELINE = "mean_baseline"
CLASSIFICATION_BASELINE = "majority_baseline"


def ratio_promotion_policy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation_rmse_gain_min": 0.01,
        "validation_mae_gain_min": 0.005,
        "max_test_rmse_regression": 0.0,
        "max_latest_rmse_regression": 0.0,
    }


def classification_promotion_policy() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation_f1_gain_min": 0.0,
        "min_validation_f1": 0.55,
        "max_test_f1_regression": 0.02,
        "min_test_f1": 0.55,
        "latest_min_n_for_gate": 20,
        "max_latest_f1_regression": 0.05,
    }


@dataclass
class PromotionDecision:
    model: str
    passed: bool
    reasons: list[str]
    metrics: dict[str, float]


@dataclass
class ClassificationPromotionDecision:
    model: str
    passed: bool
    reasons: list[str]
    notices: list[str]
    metrics: dict[str, float]


def _find(rows: Iterable[dict[str, Any]], model: str, split: str) -> dict[str, Any]:
    for row in rows:
        if row.get("model") == model and row.get("split") == split:
            return row
    raise KeyError(f"missing metrics for model={model!r}, split={split!r}")


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"missing numeric metric {key!r} in {row}")
    return float(value)


def evaluate_model(
    rows: list[dict[str, Any]],
    model: str,
    *,
    baseline: str = RATIO_BASELINE,
    validation_split: str = "validation_2024",
    test_split: str = "test_2025",
    latest_split: str = "latest_2026",
) -> PromotionDecision:
    policy = ratio_promotion_policy()
    baseline_validation = _find(rows, baseline, validation_split)
    baseline_test = _find(rows, baseline, test_split)
    baseline_latest = _find(rows, baseline, latest_split)
    candidate_validation = _find(rows, model, validation_split)
    candidate_test = _find(rows, model, test_split)
    candidate_latest = _find(rows, model, latest_split)

    validation_rmse_gain = _number(baseline_validation, "rmse") - _number(
        candidate_validation, "rmse"
    )
    validation_mae_gain = _number(baseline_validation, "mae") - _number(
        candidate_validation, "mae"
    )
    test_rmse_delta = _number(candidate_test, "rmse") - _number(baseline_test, "rmse")
    latest_rmse_delta = _number(candidate_latest, "rmse") - _number(
        baseline_latest, "rmse"
    )

    reasons: list[str] = []
    if validation_rmse_gain < policy["validation_rmse_gain_min"]:
        reasons.append(
            f"validation RMSE gain {validation_rmse_gain:.4f} < "
            f"{policy['validation_rmse_gain_min']:.4f}"
        )
    if validation_mae_gain < policy["validation_mae_gain_min"]:
        reasons.append(
            f"validation MAE gain {validation_mae_gain:.4f} < "
            f"{policy['validation_mae_gain_min']:.4f}"
        )
    if test_rmse_delta > policy["max_test_rmse_regression"]:
        reasons.append(f"test RMSE regressed by {test_rmse_delta:.4f}")
    if latest_rmse_delta > policy["max_latest_rmse_regression"]:
        reasons.append(f"latest RMSE regressed by {latest_rmse_delta:.4f}")

    return PromotionDecision(
        model=model,
        passed=not reasons,
        reasons=reasons,
        metrics={
            "validation_rmse_gain_vs_baseline": round(validation_rmse_gain, 6),
            "validation_mae_gain_vs_baseline": round(validation_mae_gain, 6),
            "test_rmse_delta_vs_baseline": round(test_rmse_delta, 6),
            "latest_rmse_delta_vs_baseline": round(latest_rmse_delta, 6),
        },
    )


def build_promotion_report(
    rows: list[dict[str, Any]],
    candidate_models: list[str] | None = None,
    *,
    baseline: str = RATIO_BASELINE,
    validation_split: str = "validation_2024",
    test_split: str = "test_2025",
    latest_split: str = "latest_2026",
) -> dict[str, Any]:
    if candidate_models is None:
        candidate_models = sorted(
            {str(row["model"]) for row in rows if row["model"] != baseline}
        )
    decisions = [
        evaluate_model(
            rows,
            model,
            baseline=baseline,
            validation_split=validation_split,
            test_split=test_split,
            latest_split=latest_split,
        )
        for model in candidate_models
    ]
    approved = [decision.model for decision in decisions if decision.passed]
    return {
        "schema_version": 2,
        "baseline": baseline,
        "policy": ratio_promotion_policy(),
        "approved_models": approved,
        "default_model": approved[0] if approved else baseline,
        "decisions": [asdict(decision) for decision in decisions],
    }


def evaluate_classifier(
    rows: list[dict[str, Any]],
    model: str,
    *,
    baseline: str = CLASSIFICATION_BASELINE,
    validation_split: str = "validation_2024",
    test_split: str = "test_2025",
    latest_split: str = "latest_2026",
) -> ClassificationPromotionDecision:
    policy = classification_promotion_policy()
    baseline_validation = _find(rows, baseline, validation_split)
    baseline_test = _find(rows, baseline, test_split)
    baseline_latest = _find(rows, baseline, latest_split)
    candidate_validation = _find(rows, model, validation_split)
    candidate_test = _find(rows, model, test_split)
    candidate_latest = _find(rows, model, latest_split)

    validation_f1 = _number(candidate_validation, "f1")
    validation_f1_gain = validation_f1 - _number(baseline_validation, "f1")
    test_f1 = _number(candidate_test, "f1")
    test_f1_delta = test_f1 - _number(baseline_test, "f1")
    latest_f1 = _number(candidate_latest, "f1")
    latest_f1_delta = latest_f1 - _number(baseline_latest, "f1")
    latest_n = int(candidate_latest.get("n", 0))

    reasons: list[str] = []
    notices: list[str] = []
    if validation_f1 < policy["min_validation_f1"]:
        reasons.append(
            f"validation F1 {validation_f1:.4f} < {policy['min_validation_f1']:.4f}"
        )
    if validation_f1_gain < policy["validation_f1_gain_min"]:
        reasons.append(
            f"validation F1 gain {validation_f1_gain:.4f} < "
            f"{policy['validation_f1_gain_min']:.4f}"
        )
    if test_f1 < policy["min_test_f1"]:
        reasons.append(f"test F1 {test_f1:.4f} < {policy['min_test_f1']:.4f}")
    if test_f1_delta < -policy["max_test_f1_regression"]:
        reasons.append(f"test F1 regressed by {-test_f1_delta:.4f}")

    if latest_n >= policy["latest_min_n_for_gate"]:
        if latest_f1_delta < -policy["max_latest_f1_regression"]:
            reasons.append(f"latest F1 regressed by {-latest_f1_delta:.4f}")
    else:
        notices.append(
            f"latest split n={latest_n} is below gate minimum "
            f"{policy['latest_min_n_for_gate']}; latest F1 is monitoring-only"
        )

    return ClassificationPromotionDecision(
        model=model,
        passed=not reasons,
        reasons=reasons,
        notices=notices,
        metrics={
            "validation_f1": round(validation_f1, 6),
            "validation_f1_gain_vs_baseline": round(validation_f1_gain, 6),
            "test_f1": round(test_f1, 6),
            "test_f1_delta_vs_baseline": round(test_f1_delta, 6),
            "latest_f1": round(latest_f1, 6),
            "latest_f1_delta_vs_baseline": round(latest_f1_delta, 6),
            "latest_n": float(latest_n),
        },
    )


def build_classification_promotion_report(
    rows: list[dict[str, Any]],
    candidate_models: list[str] | None = None,
    *,
    baseline: str = CLASSIFICATION_BASELINE,
    validation_split: str = "validation_2024",
    test_split: str = "test_2025",
    latest_split: str = "latest_2026",
) -> dict[str, Any]:
    if candidate_models is None:
        candidate_models = ["logistic_regression_l2"]
    decisions = [
        evaluate_classifier(
            rows,
            model,
            baseline=baseline,
            validation_split=validation_split,
            test_split=test_split,
            latest_split=latest_split,
        )
        for model in candidate_models
    ]
    approved = [decision.model for decision in decisions if decision.passed]
    return {
        "schema_version": 1,
        "baseline": baseline,
        "policy": classification_promotion_policy(),
        "approved_models": approved,
        "default_model": approved[0] if approved else baseline,
        "decisions": [asdict(decision) for decision in decisions],
    }
