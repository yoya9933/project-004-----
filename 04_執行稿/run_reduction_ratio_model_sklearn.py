from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def make_candidates(random_state: int) -> dict[str, list[tuple[str, object]]]:
    return {
        "mean_baseline": [
            ("mean_baseline", DummyRegressor(strategy="mean")),
        ],
        "ridge_regression_l2": [
            (
                f"ridge_alpha_{alpha:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", Ridge(alpha=alpha, random_state=random_state)),
                    ]
                ),
            )
            for alpha in [0.001, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0]
        ],
        "lasso_regression_l1": [
            (
                f"lasso_alpha_{alpha:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", Lasso(alpha=alpha, max_iter=10000, random_state=random_state)),
                    ]
                ),
            )
            for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
        ],
        "elastic_net": [
            (
                f"elastic_alpha_{alpha:g}_l1_{l1_ratio:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            ElasticNet(
                                alpha=alpha,
                                l1_ratio=l1_ratio,
                                max_iter=10000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            )
            for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05]
            for l1_ratio in [0.2, 0.5, 0.8]
        ],
        "random_forest": [
            (
                f"rf_depth_{depth}_leaf_{leaf}",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
            for depth in [3, 5, 8, None]
            for leaf in [5, 10, 20]
        ],
        "extra_trees": [
            (
                f"extra_depth_{depth}_leaf_{leaf}",
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
            for depth in [3, 5, 8, None]
            for leaf in [5, 10, 20]
        ],
        "gradient_boosting": [
            (
                f"gbr_depth_{depth}_lr_{learning_rate:g}_leaf_{leaf}",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=learning_rate,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=random_state,
                ),
            )
            for depth in [1, 2, 3]
            for learning_rate in [0.01, 0.03, 0.05]
            for leaf in [5, 10, 20]
        ],
        "hist_gradient_boosting": [
            (
                f"hist_lr_{learning_rate:g}_leaf_{leaf}_l2_{l2:g}",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=learning_rate,
                    min_samples_leaf=leaf,
                    l2_regularization=l2,
                    random_state=random_state,
                ),
            )
            for learning_rate in [0.01, 0.03, 0.05]
            for leaf in [10, 20, 30]
            for l2 in [0.0, 0.1, 1.0]
        ],
    }


def split_df(
    df: pd.DataFrame,
    train_start_year: int,
    train_end_year: int,
    validation_year: int,
    test_year: int,
    latest_check_year: int,
) -> list[tuple[str, pd.DataFrame]]:
    return [
        (
            f"train_{train_start_year}_{train_end_year}",
            df[(df["decision_year"] >= train_start_year) & (df["decision_year"] <= train_end_year)],
        ),
        (f"validation_{validation_year}", df[df["decision_year"] == validation_year]),
        (f"test_{test_year}", df[df["decision_year"] == test_year]),
        (f"latest_{latest_check_year}", df[df["decision_year"] == latest_check_year]),
    ]


def metric_row(model_name: str, split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    if len(y_true) == 0:
        return {
            "model": model_name,
            "split": split_name,
            "n": 0,
            "mae": "",
            "rmse": "",
            "r2": "",
            "bucket_accuracy": "",
        }
    clipped = np.clip(y_pred, 0.0, 1.0)
    actual_buckets = [get_bucket(float(value)) for value in y_true]
    pred_buckets = [get_bucket(float(value)) for value in clipped]
    return {
        "model": model_name,
        "split": split_name,
        "n": int(len(y_true)),
        "mae": round(float(mean_absolute_error(y_true, clipped)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, clipped))), 4),
        "r2": round(float(r2_score(y_true, clipped)), 4) if len(y_true) > 1 else "",
        "bucket_accuracy": round(float(np.mean(np.array(actual_buckets) == np.array(pred_buckets))), 4),
    }


def prediction_rows(
    model_name: str,
    split_name: str,
    split: pd.DataFrame,
    raw_pred: np.ndarray,
) -> list[dict[str, object]]:
    clipped = np.clip(raw_pred, 0.0, 1.0)
    rows: list[dict[str, object]] = []
    for (_, row), raw, pred in zip(split.iterrows(), raw_pred, clipped):
        actual = float(row["remaining_ratio"])
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "JID": row.get("JID", ""),
                "decision_year": int(row["decision_year"]),
                "actual_remaining_ratio": round(actual, 6),
                "predicted_remaining_ratio_raw": round(float(raw), 6),
                "predicted_remaining_ratio": round(float(pred), 6),
                "actual_reduction_rate": round(1.0 - actual, 6),
                "predicted_reduction_rate": round(1.0 - float(pred), 6),
                "actual_bucket": get_bucket(actual),
                "predicted_bucket": get_bucket(float(pred)),
            }
        )
    return rows


def select_best_by_validation(
    family: str,
    candidates: list[tuple[str, object]],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[str, object, list[dict[str, object]]]:
    x_train = train[FEATURE_NAMES]
    y_train = train["remaining_ratio"].astype(float).to_numpy()
    x_validation = validation[FEATURE_NAMES]
    y_validation = validation["remaining_ratio"].astype(float).to_numpy()
    search_rows: list[dict[str, object]] = []
    best_name = ""
    best_model = None
    best_rmse = float("inf")

    for candidate_name, model in candidates:
        model.fit(x_train, y_train)
        pred = np.clip(model.predict(x_validation), 0.0, 1.0)
        rmse = float(np.sqrt(mean_squared_error(y_validation, pred)))
        mae = float(mean_absolute_error(y_validation, pred))
        r2 = float(r2_score(y_validation, pred)) if len(y_validation) > 1 else float("nan")
        search_rows.append(
            {
                "family": family,
                "candidate": candidate_name,
                "validation_mae": round(mae, 6),
                "validation_rmse": round(rmse, 6),
                "validation_r2": round(r2, 6) if not np.isnan(r2) else "",
            }
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = candidate_name
            best_model = model

    assert best_model is not None
    return best_name, best_model, search_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sklearn ratio model backtest with expanded samples.")
    parser.add_argument(
        "--usable-frame",
        type=Path,
        default=PROJECT_ROOT
        / "06_交付物"
        / "reduction_ratio_model_expanded_824"
        / "usable_ratio_model_frame.csv",
    )
    parser.add_argument(
        "--model-frame",
        type=Path,
        default=PROJECT_ROOT
        / "06_交付物"
        / "reduction_ratio_model_expanded_824"
        / "ratio_model_frame.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "reduction_ratio_model_expanded_824_sklearn",
    )
    parser.add_argument("--train-start-year", type=int, default=2021)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--latest-check-year", type=int, default=2026)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.usable_frame)
    df = df[df["target_quality"] == "ok"].copy()
    df["decision_year"] = df["decision_year"].astype(int)
    df["remaining_ratio"] = df["remaining_ratio"].astype(float)
    for name in FEATURE_NAMES:
        df[name] = pd.to_numeric(df[name], errors="coerce").fillna(0.0)

    splits = split_df(
        df,
        args.train_start_year,
        args.train_end_year,
        args.validation_year,
        args.test_year,
        args.latest_check_year,
    )
    split_map = dict(splits)
    train = split_map[f"train_{args.train_start_year}_{args.train_end_year}"]
    validation = split_map[f"validation_{args.validation_year}"]

    candidates = make_candidates(args.random_state)
    selected_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}
    for family, family_candidates in candidates.items():
        best_candidate, best_model, family_search = select_best_by_validation(
            family, family_candidates, train, validation
        )
        fitted_models[family] = best_model
        search_rows.extend(family_search)
        selected_rows.append({"model": family, "selected_candidate": best_candidate})

    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for model_name, model in fitted_models.items():
        for split_name, split in splits:
            x_split = split[FEATURE_NAMES]
            y_split = split["remaining_ratio"].astype(float).to_numpy()
            raw_pred = model.predict(x_split) if len(split) else np.array([])
            metrics.append(metric_row(model_name, split_name, y_split, raw_pred))
            predictions.extend(prediction_rows(model_name, split_name, split, raw_pred))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(
        args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selected_rows).to_csv(
        args.output_dir / "selected_models.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(search_rows).to_csv(
        args.output_dir / "validation_search.csv", index=False, encoding="utf-8-sig"
    )

    quality_counts = (
        pd.read_csv(args.model_frame)["target_quality"].value_counts().sort_index().to_dict()
        if args.model_frame.exists()
        else {"ok": len(df)}
    )
    status = {
        "status": "ok",
        "implementation": "run_reduction_ratio_model_sklearn.py",
        "usable_frame": str(args.usable_frame),
        "model_frame": str(args.model_frame),
        "total_rows": int(sum(quality_counts.values())),
        "usable_target_rows": int(len(df)),
        "training_split": f"{args.train_start_year}-{args.train_end_year}",
        "validation_split": str(args.validation_year),
        "test_split": str(args.test_year),
        "latest_check_split": str(args.latest_check_year),
        "target_quality_counts": {str(key): int(value) for key, value in quality_counts.items()},
        "feature_names": FEATURE_NAMES,
        "models": list(fitted_models),
        "selection_policy": "For each model family, select the candidate with the lowest validation RMSE, trained on 2021-2023 only.",
        "outputs": {
            "metrics": str(args.output_dir / "metrics.csv"),
            "predictions": str(args.output_dir / "predictions.csv"),
            "selected_models": str(args.output_dir / "selected_models.csv"),
            "validation_search": str(args.output_dir / "validation_search.csv"),
        },
        "note": "Predictions are clipped to 0-1 before metrics, matching the existing ratio-model evaluation behavior.",
    }
    (args.output_dir / "model_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 擴樣 sklearn 酌減比例模型回測結果",
        "",
        "## 設計",
        "",
        "- 目標變數：`remaining_ratio = allowed_penalty / claimed_penalty`。",
        f"- 時間切分：{args.train_start_year}-{args.train_end_year} 訓練、{args.validation_year} 驗證、{args.test_year} 測試、{args.latest_check_year} 最新年度檢查。",
        "- 特徵：沿用主線比例模型的 19 個特徵。",
        "- 模型：mean baseline、Ridge、Lasso、ElasticNet、RandomForest、ExtraTrees、GradientBoosting、HistGradientBoosting。",
        "- 選模：各模型家族以 2024 驗證集 RMSE 最低者為代表，訓練仍只使用 2021-2023。",
        "",
        "## 輸出",
        "",
        "- `metrics.csv`",
        "- `predictions.csv`",
        "- `selected_models.csv`",
        "- `validation_search.csv`",
        "- `model_status.json`",
        "",
        "## 注意",
        "",
        "- 擴樣資料仍是 AI 假設選值，尚未完成逐案人工查核。",
        "- 本結果適合作為樣本數與模型家族敏感度分析，不應直接當成正式人工標註模型成效。",
    ]
    (args.output_dir / "迴歸模型狀態.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"usable rows: {len(df)}")
    print(f"metrics: {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
