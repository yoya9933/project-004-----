from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_risk_modeling.cli_contract import ensure_existing_csv  # noqa: E402
from legal_risk_modeling.features import FEATURE_NAMES, normalize_feature_frame  # noqa: E402
from legal_risk_modeling.manifest import build_manifest, write_manifest  # noqa: E402
from legal_risk_modeling.models import build_ratio_candidates, model_spec  # noqa: E402
from legal_risk_modeling.paths import ratio_output_dir  # noqa: E402
from legal_risk_modeling.promotion import (  # noqa: E402
    build_promotion_report,
    ratio_promotion_policy,
)
from legal_risk_modeling.temporal import TemporalSplitPolicy  # noqa: E402


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


def metric_row(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
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


def select_best_by_training_cv(
    family: str,
    candidates: list[tuple[str, object]],
    train: pd.DataFrame,
    policy: TemporalSplitPolicy,
) -> tuple[str, object, list[dict[str, object]]]:
    folds = policy.rolling_cv_splits(train)
    if not folds:
        raise ValueError("training period must contain at least two non-empty years for rolling CV")

    search_rows: list[dict[str, object]] = []
    best_name = ""
    best_template = None
    best_rmse = float("inf")

    for candidate_name, template in candidates:
        fold_rmse: list[float] = []
        fold_mae: list[float] = []
        for _, fit_frame, validation_frame in folds:
            model = clone(template)
            model.fit(
                fit_frame[FEATURE_NAMES],
                fit_frame["remaining_ratio"].astype(float).to_numpy(),
            )
            pred = np.clip(model.predict(validation_frame[FEATURE_NAMES]), 0.0, 1.0)
            y_validation = validation_frame["remaining_ratio"].astype(float).to_numpy()
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_validation, pred))))
            fold_mae.append(float(mean_absolute_error(y_validation, pred)))

        mean_rmse = float(np.mean(fold_rmse))
        mean_mae = float(np.mean(fold_mae))
        search_rows.append(
            {
                "family": family,
                "candidate": candidate_name,
                "cv_folds": len(folds),
                "cv_mean_mae": round(mean_mae, 6),
                "cv_mean_rmse": round(mean_rmse, 6),
                "cv_fold_rmse": "|".join(f"{value:.6f}" for value in fold_rmse),
            }
        )
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_name = candidate_name
            best_template = template

    if best_template is None:
        raise RuntimeError(f"no candidate selected for {family}")
    selected = clone(best_template)
    selected.fit(train[FEATURE_NAMES], train["remaining_ratio"].astype(float).to_numpy())
    return best_name, selected, search_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Governed sklearn ratio-model backtest.")
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
    parser.add_argument("--output-dir", type=Path, default=ratio_output_dir(PROJECT_ROOT))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-start-year", type=int, default=2021)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--latest-check-year", type=int, default=2026)
    parser.add_argument("--run-id", default="ratio-model-governed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usable_frame = ensure_existing_csv(args.usable_frame)
    model_frame = ensure_existing_csv(args.model_frame)
    policy = TemporalSplitPolicy(
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        validation_year=args.validation_year,
        test_year=args.test_year,
        latest_check_year=args.latest_check_year,
    )

    df = pd.read_csv(usable_frame, encoding="utf-8-sig")
    if "target_quality" in df.columns:
        df = df[df["target_quality"] == "ok"].copy()
    df["decision_year"] = pd.to_numeric(df["decision_year"], errors="raise").astype(int)
    df["remaining_ratio"] = pd.to_numeric(df["remaining_ratio"], errors="raise").astype(float)
    df = normalize_feature_frame(df)

    splits = policy.split_frame(df)
    split_map = dict(splits)
    train = split_map[policy.train_name]
    validation = split_map[policy.validation_name]
    if train.empty or validation.empty:
        raise ValueError("training and promotion-validation splits must be non-empty")

    selected_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}
    for family, family_candidates in build_ratio_candidates(args.random_state).items():
        best_candidate, best_model, family_search = select_best_by_training_cv(
            family, family_candidates, train, policy
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
    for stale_name in ["validation_search.csv", "manifest.json"]:
        (args.output_dir / stale_name).unlink(missing_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    predictions_path = args.output_dir / "predictions.csv"
    selected_path = args.output_dir / "selected_models.csv"
    search_path = args.output_dir / "tuning_search.csv"
    promotion_path = args.output_dir / "promotion_report.json"
    release_path = args.output_dir / "approved_release.json"

    pd.DataFrame(metrics).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(predictions_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(search_rows).to_csv(search_path, index=False, encoding="utf-8-sig")

    promotion = build_promotion_report(
        metrics,
        validation_split=policy.validation_name,
        test_split=policy.test_name,
        latest_split=policy.latest_name,
    )
    promotion_path.write_text(json.dumps(promotion, ensure_ascii=False, indent=2), encoding="utf-8")
    release = {
        "schema_version": 2,
        "status": "approved",
        "default_model": promotion["default_model"],
        "approved_models": promotion["approved_models"],
        "baseline": promotion["baseline"],
        "predictions_artifact": predictions_path.relative_to(PROJECT_ROOT).as_posix(),
        "metrics_artifact": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
        "promotion_report": promotion_path.relative_to(PROJECT_ROOT).as_posix(),
        "note": "UI may only expose baseline plus models listed in approved_models.",
    }
    release_path.write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        dataset_paths=[usable_frame, model_frame],
        artifact_paths=[
            metrics_path,
            predictions_path,
            selected_path,
            search_path,
            promotion_path,
            release_path,
        ],
        model_spec=model_spec(),
        run_id=args.run_id,
        invocation={
            "random_state": args.random_state,
            "usable_frame": usable_frame.relative_to(PROJECT_ROOT).as_posix(),
            "model_frame": model_frame.relative_to(PROJECT_ROOT).as_posix(),
        },
        temporal_split_policy=policy.as_dict(),
        promotion_policy=ratio_promotion_policy(),
        strict_git=True,
    )
    write_manifest(args.output_dir / "manifest.json", manifest)

    print(
        json.dumps(
            {
                "default_model": promotion["default_model"],
                "approved_models": promotion["approved_models"],
                "tuning": "training-period rolling CV",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
