from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from legal_risk_modeling.promotion import build_promotion_report  # noqa: E402


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


def split_df(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    year = pd.to_numeric(df["decision_year"], errors="coerce")
    return [
        ("train_2021_2023", df[(year >= 2021) & (year <= 2023)].copy()),
        ("validation_2024", df[year == 2024].copy()),
        ("test_2025", df[year == 2025].copy()),
        ("latest_2026", df[year == 2026].copy()),
    ]


def metric_row(model_name: str, split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    if len(y_true) == 0:
        return {"model": model_name, "split": split_name, "n": 0, "mae": "", "rmse": "", "r2": "", "bucket_accuracy": ""}
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


def prediction_rows(model_name: str, split_name: str, split: pd.DataFrame, raw_pred: np.ndarray) -> list[dict[str, object]]:
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

    if best_model is None:
        raise RuntimeError(f"no candidate selected for {family}")
    return best_name, best_model, search_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Governed sklearn ratio-model backtest.")
    parser.add_argument(
        "--usable-frame",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "reduction_ratio_model_expanded_824" / "usable_ratio_model_frame.csv",
    )
    parser.add_argument(
        "--model-frame",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "reduction_ratio_model_expanded_824" / "ratio_model_frame.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ratio_output_dir(PROJECT_ROOT),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--run-id", default="ratio-model-governed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usable_frame = ensure_existing_csv(args.usable_frame)
    model_frame = ensure_existing_csv(args.model_frame) if args.model_frame.exists() else None
    df = pd.read_csv(usable_frame)
    if "target_quality" in df.columns:
        df = df[df["target_quality"] == "ok"].copy()
    df["decision_year"] = pd.to_numeric(df["decision_year"], errors="raise").astype(int)
    df["remaining_ratio"] = pd.to_numeric(df["remaining_ratio"], errors="raise").astype(float)
    df = normalize_feature_frame(df)

    splits = split_df(df)
    split_map = dict(splits)
    train = split_map["train_2021_2023"]
    validation = split_map["validation_2024"]
    if train.empty or validation.empty:
        raise ValueError("training and validation splits must be non-empty")

    selected_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}
    for family, family_candidates in build_ratio_candidates(args.random_state).items():
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
    metrics_path = args.output_dir / "metrics.csv"
    predictions_path = args.output_dir / "predictions.csv"
    selected_path = args.output_dir / "selected_models.csv"
    search_path = args.output_dir / "validation_search.csv"
    promotion_path = args.output_dir / "promotion_report.json"
    release_path = args.output_dir / "approved_release.json"

    pd.DataFrame(metrics).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions).to_csv(predictions_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(search_rows).to_csv(search_path, index=False, encoding="utf-8-sig")

    promotion = build_promotion_report(metrics)
    promotion_path.write_text(json.dumps(promotion, ensure_ascii=False, indent=2), encoding="utf-8")
    release = {
        "status": "approved",
        "default_model": promotion["default_model"],
        "approved_models": promotion["approved_models"],
        "baseline": promotion["baseline"],
        "predictions_artifact": str(predictions_path.relative_to(PROJECT_ROOT)),
        "metrics_artifact": str(metrics_path.relative_to(PROJECT_ROOT)),
        "promotion_report": str(promotion_path.relative_to(PROJECT_ROOT)),
        "note": "UI may only expose baseline plus models listed in approved_models.",
    }
    release_path.write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_paths = [usable_frame, model_frame] if model_frame is not None else [usable_frame]
    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        dataset_paths=dataset_paths,
        artifact_paths=[metrics_path, predictions_path, selected_path, search_path, promotion_path, release_path],
        model_spec=model_spec(),
        run_id=args.run_id,
    )
    write_manifest(args.output_dir / "manifest.json", manifest)

    print(json.dumps({"default_model": promotion["default_model"], "approved_models": promotion["approved_models"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
