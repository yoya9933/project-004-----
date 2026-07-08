from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_NAMES = [
    "x_log_contract_price",
    "x_log_claimed_penalty",
    "x_claim_to_contract_ratio",
    "x_delay_days",
    "x_penalty_per_delay_day",
    "x_issue_owner_fault",
    "x_issue_contractor_fault",
    "x_issue_extension_request",
    "x_issue_actual_damage_unclear",
    "x_issue_partial_completion",
    "x_issue_used_by_owner",
    "x_ai_issue_owner_fault",
    "x_ai_issue_contractor_fault",
    "x_ai_issue_extension_request",
    "x_ai_issue_actual_damage_unclear",
    "x_ai_issue_partial_completion",
    "x_ai_issue_used_by_owner",
    "x_money_candidate_count",
    "x_delay_candidate_count",
]

MODELS = ["mean_baseline", "ridge_regression_l2", "lasso_regression_l1"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return default


def get_bucket(ratio: float) -> str:
    if ratio <= 0.05:
        return "全免或近乎全免"
    if ratio < 0.30:
        return "大幅酌減"
    if ratio < 0.70:
        return "中度酌減"
    if ratio < 0.99:
        return "小幅酌減"
    return "未酌減"


def split_rows(
    rows: list[dict[str, str]],
    train_start_year: int,
    train_end_year: int,
    validation_year: int,
    test_year: int,
    latest_check_year: int,
) -> list[tuple[str, list[dict[str, str]]]]:
    return [
        (
            f"train_{train_start_year}_{train_end_year}",
            [
                row
                for row in rows
                if train_start_year <= int(to_float(row.get("decision_year"))) <= train_end_year
            ],
        ),
        (
            f"validation_{validation_year}",
            [row for row in rows if int(to_float(row.get("decision_year"))) == validation_year],
        ),
        (
            f"test_{test_year}",
            [row for row in rows if int(to_float(row.get("decision_year"))) == test_year],
        ),
        (
            f"latest_{latest_check_year}",
            [row for row in rows if int(to_float(row.get("decision_year"))) == latest_check_year],
        ),
    ]


def get_scaler(rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = [to_float(row.get(name)) for row in rows]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        std = math.sqrt(variance)
        means[name] = mean
        stds[name] = std if std >= 1e-9 else 1.0
    return means, stds


def scaled_vector(row: dict[str, str], means: dict[str, float], stds: dict[str, float]) -> list[float]:
    return [(to_float(row.get(name)) - means[name]) / stds[name] for name in FEATURE_NAMES]


def soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def train_linear_model(
    rows: list[dict[str, str]],
    means: dict[str, float],
    stds: dict[str, float],
    model_kind: str,
    lambda_value: float,
    iterations: int,
    learning_rate: float,
) -> list[float]:
    weights = [0.0] * (len(FEATURE_NAMES) + 1)
    n = float(len(rows))
    prepared = [
        (scaled_vector(row, means, stds), to_float(row.get("remaining_ratio")))
        for row in rows
    ]
    for _ in range(iterations):
        grad = [0.0] * len(weights)
        for x_values, target in prepared:
            pred = weights[0] + sum(weights[i + 1] * x_values[i] for i in range(len(x_values)))
            error = pred - target
            grad[0] += error
            for i, value in enumerate(x_values):
                grad[i + 1] += error * value
        weights[0] -= learning_rate * (grad[0] / n)
        for j in range(1, len(weights)):
            step = weights[j] - learning_rate * (grad[j] / n)
            if model_kind == "ridge":
                step -= learning_rate * lambda_value * weights[j]
            elif model_kind == "lasso":
                step = soft_threshold(step, learning_rate * lambda_value)
            weights[j] = step
    return weights


def linear_prediction(
    row: dict[str, str],
    means: dict[str, float],
    stds: dict[str, float],
    weights: list[float],
) -> float:
    x_values = scaled_vector(row, means, stds)
    return weights[0] + sum(weights[i + 1] * x_values[i] for i in range(len(x_values)))


def clip_ratio(value: float) -> float:
    return min(1.0, max(0.0, value))


def add_predictions(
    output: list[dict[str, object]],
    rows: list[dict[str, str]],
    split_name: str,
    model_name: str,
    predict,
) -> None:
    for row in rows:
        raw_pred = float(predict(row))
        pred_ratio = clip_ratio(raw_pred)
        actual_ratio = to_float(row.get("remaining_ratio"))
        output.append(
            {
                "model": model_name,
                "split": split_name,
                "JID": row.get("JID", ""),
                "decision_year": row.get("decision_year", ""),
                "actual_remaining_ratio": round(actual_ratio, 6),
                "predicted_remaining_ratio_raw": round(raw_pred, 6),
                "predicted_remaining_ratio": round(pred_ratio, 6),
                "actual_reduction_rate": round(1.0 - actual_ratio, 6),
                "predicted_reduction_rate": round(1.0 - pred_ratio, 6),
                "actual_bucket": get_bucket(actual_ratio),
                "predicted_bucket": get_bucket(pred_ratio),
            }
        )


def metric_row(model_name: str, split_name: str, predictions: list[dict[str, object]]) -> dict[str, object]:
    n = len(predictions)
    if n == 0:
        return {
            "model": model_name,
            "split": split_name,
            "n": 0,
            "mae": "",
            "rmse": "",
            "r2": "",
            "bucket_accuracy": "",
        }

    actual_values = [float(row["actual_remaining_ratio"]) for row in predictions]
    mean_actual = sum(actual_values) / n
    abs_error = 0.0
    squared_error = 0.0
    sst = 0.0
    bucket_hits = 0
    for row in predictions:
        actual = float(row["actual_remaining_ratio"])
        pred = float(row["predicted_remaining_ratio"])
        abs_error += abs(actual - pred)
        squared_error += (actual - pred) ** 2
        sst += (actual - mean_actual) ** 2
        if row["actual_bucket"] == row["predicted_bucket"]:
            bucket_hits += 1
    r2 = "" if sst <= 0 else round(1.0 - (squared_error / sst), 4)
    return {
        "model": model_name,
        "split": split_name,
        "n": n,
        "mae": round(abs_error / n, 4),
        "rmse": round(math.sqrt(squared_error / n), 4),
        "r2": r2,
        "bucket_accuracy": round(bucket_hits / n, 4),
    }


def target_quality_counts(model_frame: Path | None, usable_rows: list[dict[str, str]]) -> dict[str, int]:
    if model_frame and model_frame.exists():
        counts: dict[str, int] = {}
        for row in read_csv(model_frame):
            key = row.get("target_quality", "") or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))
    return {"ok": len(usable_rows)}


def parse_args() -> argparse.Namespace:
    default_output = PROJECT_ROOT / "06_交付物" / "reduction_ratio_model_expanded_824"
    parser = argparse.ArgumentParser(description="Fast standard-library ratio model backtest.")
    parser.add_argument(
        "--usable-frame",
        type=Path,
        default=default_output / "usable_ratio_model_frame.csv",
    )
    parser.add_argument("--model-frame", type=Path, default=default_output / "ratio_model_frame.csv")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--train-start-year", type=int, default=2021)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--latest-check-year", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--ridge-lambda", type=float, default=0.05)
    parser.add_argument("--lasso-lambda", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usable_rows = [row for row in read_csv(args.usable_frame) if row.get("target_quality") == "ok"]
    splits = split_rows(
        usable_rows,
        args.train_start_year,
        args.train_end_year,
        args.validation_year,
        args.test_year,
        args.latest_check_year,
    )
    train_rows = splits[0][1]
    if len(train_rows) < 10:
        raise SystemExit(f"not enough training rows: {len(train_rows)}")

    means, stds = get_scaler(train_rows)
    ridge_weights = train_linear_model(
        train_rows,
        means,
        stds,
        "ridge",
        args.ridge_lambda,
        args.iterations,
        args.learning_rate,
    )
    lasso_weights = train_linear_model(
        train_rows,
        means,
        stds,
        "lasso",
        args.lasso_lambda,
        args.iterations,
        args.learning_rate,
    )
    mean_remaining_ratio = sum(to_float(row.get("remaining_ratio")) for row in train_rows) / len(train_rows)

    predictions: list[dict[str, object]] = []
    for split_name, split_data in splits:
        add_predictions(
            predictions,
            split_data,
            split_name,
            "mean_baseline",
            lambda _row: mean_remaining_ratio,
        )
        add_predictions(
            predictions,
            split_data,
            split_name,
            "ridge_regression_l2",
            lambda row: linear_prediction(row, means, stds, ridge_weights),
        )
        add_predictions(
            predictions,
            split_data,
            split_name,
            "lasso_regression_l1",
            lambda row: linear_prediction(row, means, stds, lasso_weights),
        )

    metrics: list[dict[str, object]] = []
    for model_name in MODELS:
        for split_name, _split_data in splits:
            subset = [
                row
                for row in predictions
                if row["model"] == model_name and row["split"] == split_name
            ]
            metrics.append(metric_row(model_name, split_name, subset))

    coefficients: list[dict[str, object]] = []
    for model_name, weights in [
        ("ridge_regression_l2", ridge_weights),
        ("lasso_regression_l1", lasso_weights),
    ]:
        for index, feature in enumerate(["intercept", *FEATURE_NAMES]):
            coef = weights[index]
            coefficients.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "mean": "" if feature == "intercept" else round(means[feature], 8),
                    "std": "" if feature == "intercept" else round(stds[feature], 8),
                    "coefficient_scaled": round(coef, 8),
                    "direction": "准許比例同向"
                    if coef > 0
                    else "酌減比例同向"
                    if coef < 0
                    else "接近零",
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "metrics.csv",
        metrics,
        ["model", "split", "n", "mae", "rmse", "r2", "bucket_accuracy"],
    )
    write_csv(
        args.output_dir / "predictions.csv",
        predictions,
        [
            "model",
            "split",
            "JID",
            "decision_year",
            "actual_remaining_ratio",
            "predicted_remaining_ratio_raw",
            "predicted_remaining_ratio",
            "actual_reduction_rate",
            "predicted_reduction_rate",
            "actual_bucket",
            "predicted_bucket",
        ],
    )
    write_csv(
        args.output_dir / "model_coefficients.csv",
        coefficients,
        ["model", "feature", "mean", "std", "coefficient_scaled", "direction"],
    )

    quality_counts = target_quality_counts(args.model_frame, usable_rows)
    status = {
        "status": "ok",
        "implementation": "run_reduction_ratio_model_fast.py",
        "usable_frame": str(args.usable_frame),
        "model_frame": str(args.model_frame),
        "total_rows": sum(quality_counts.values()),
        "usable_target_rows": len(usable_rows),
        "training_split": f"{args.train_start_year}-{args.train_end_year}",
        "validation_split": str(args.validation_year),
        "test_split": str(args.test_year),
        "latest_check_split": str(args.latest_check_year),
        "target_quality_counts": quality_counts,
        "model_features": FEATURE_NAMES,
        "models": MODELS,
        "outputs": {
            "metrics": str(args.output_dir / "metrics.csv"),
            "predictions": str(args.output_dir / "predictions.csv"),
            "model_coefficients": str(args.output_dir / "model_coefficients.csv"),
        },
        "note": "Fast Python backtest consumes the PowerShell-generated ratio feature frame and predicts remaining_ratio.",
    }
    (args.output_dir / "model_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 擴樣酌減比例迴歸模型回測結果",
        "",
        "## 設計",
        "",
        "- 目標變數：`remaining_ratio = allowed_penalty / claimed_penalty`。",
        f"- 時間切分：{args.train_start_year}-{args.train_end_year} 訓練、{args.validation_year} 驗證、{args.test_year} 測試、{args.latest_check_year} 最新年度檢查。",
        "- 模型：mean baseline、Ridge Regression、Lasso Regression。",
        "- 指標：MAE、RMSE、R2、分段命中率。",
        "- 本次使用 `run_reduction_ratio_model_fast.py` 讀取已產生的特徵框，加速大樣本回測。",
        "",
        "## 筆數",
        "",
        f"- 全部擴樣候選：{sum(quality_counts.values())}",
        f"- 可用比例目標：{len(usable_rows)}",
        "",
        "## 輸出",
        "",
        "- `ratio_model_frame.csv`",
        "- `usable_ratio_model_frame.csv`",
        "- `target_quality_summary.csv`",
        "- `metrics.csv`",
        "- `predictions.csv`",
        "- `model_coefficients.csv`",
        "- `model_status.json`",
        "",
        "## 注意",
        "",
        "- 本次擴樣仍是 AI 假設選值，尚未完成逐案人工查核。",
        "- 結果適合作為樣本數敏感度分析，不應直接當成正式模型成效。",
    ]
    (args.output_dir / "迴歸模型狀態.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"usable rows: {len(usable_rows)}")
    print(f"metrics: {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
