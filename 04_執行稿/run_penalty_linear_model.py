from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "06_交付物" / "analysis_report_version" / "penalty_cases_cleaned.csv"
DEFAULT_FALLBACK_INPUT_CSV = PROJECT_ROOT / "06_交付物" / "penalty_cases.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "06_交付物" / "advanced_linear_model"

DEFAULT_FEATURES = [
    "delay_days",
    "penalty_to_contract",
    "issue_actual_damage_unclear",
    "issue_partial_completion",
    "issue_owner_fault",
    "issue_extension_request",
    "issue_used_by_owner",
]

RANDOM_STATE = 42


def read_input(path: Path | None) -> tuple[pd.DataFrame, Path]:
    if path and path.exists():
        return pd.read_csv(path, encoding="utf-8-sig"), path
    if DEFAULT_INPUT_CSV.exists():
        return pd.read_csv(DEFAULT_INPUT_CSV, encoding="utf-8-sig"), DEFAULT_INPUT_CSV
    return pd.read_csv(DEFAULT_FALLBACK_INPUT_CSV, encoding="utf-8-sig"), DEFAULT_FALLBACK_INPUT_CSV


def to_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def ensure_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for field in ["contract_price", "claimed_penalty", "allowed_penalty", "delay_days"]:
        if field in df.columns:
            df[field] = to_number(df[field])
    if {"contract_price", "claimed_penalty", "allowed_penalty"}.issubset(df.columns):
        complete = (
            df["contract_price"].notna()
            & df["contract_price"].gt(0)
            & df["claimed_penalty"].notna()
            & df["claimed_penalty"].gt(0)
            & df["allowed_penalty"].notna()
            & df["allowed_penalty"].ge(0)
        )
        df.loc[complete, "penalty_to_contract"] = (
            df.loc[complete, "claimed_penalty"] / df.loc[complete, "contract_price"]
        )
        df.loc[complete, "remaining_ratio"] = (
            df.loc[complete, "allowed_penalty"] / df.loc[complete, "claimed_penalty"]
        )
        df.loc[complete, "reduction_rate"] = 1 - df.loc[complete, "remaining_ratio"]
        df.loc[complete, "is_reduced"] = (
            df.loc[complete, "allowed_penalty"] < df.loc[complete, "claimed_penalty"]
        ).astype(int)
    return df


def select_model_frame(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing_features = [feature for feature in features if feature not in df.columns]
    for feature in missing_features:
        df[feature] = pd.NA
    if target not in df.columns:
        df[target] = pd.NA

    model_df = df[features + [target, "case_id"]].copy()
    for col in features + [target]:
        model_df[col] = to_number(model_df[col])

    used_features = [feature for feature in features if model_df[feature].notna().any()]
    skipped_features = [feature for feature in features if feature not in used_features]
    if not used_features:
        return model_df.iloc[0:0], used_features, skipped_features

    return model_df.dropna(subset=used_features + [target]), used_features, skipped_features


def stop_report(output_dir: Path, reason: str, details: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_status.json").write_text(
        json.dumps({"status": "skipped", "reason": reason, **details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 線性模型輔助分析狀態",
        "",
        f"- 狀態：skipped",
        f"- 原因：{reason}",
        "",
        "## 詳細資訊",
        "",
        "```json",
        json.dumps(details, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 建議",
        "",
        "- 先補齊 `contract_price`、`delay_days`、`claimed_penalty`、`allowed_penalty`。",
        "- 目標欄位 `is_reduced` 需要同時有 0 與 1。",
        "- 建議至少每個類別各 2 筆以上，再做交叉驗證。",
    ]
    (output_dir / "線性模型輔助分析狀態.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model(
    model_df: pd.DataFrame,
    features: list[str],
    target: str,
    output_dir: Path,
    skipped_features: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    X = model_df[features]
    y = model_df[target].astype(int)

    class_counts = y.value_counts().sort_index()
    min_class_count = int(class_counts.min())
    if len(class_counts) < 2:
        stop_report(
            output_dir,
            "target_has_single_class",
            {
                "n_rows": int(len(model_df)),
                "class_counts": class_counts.to_dict(),
                "used_features": features,
                "skipped_blank_features": skipped_features,
            },
        )
        return
    if min_class_count < 2:
        stop_report(
            output_dir,
            "too_few_rows_per_class_for_cross_validation",
            {
                "n_rows": int(len(model_df)),
                "class_counts": class_counts.to_dict(),
                "used_features": features,
                "skipped_blank_features": skipped_features,
            },
        )
        return

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=make_scorer(balanced_accuracy_score),
    )
    model.fit(X, y)

    coefficients = pd.DataFrame(
        {
            "feature": features,
            "coefficient": model.named_steps["clf"].coef_[0],
        }
    ).sort_values("coefficient", ascending=False)
    coefficients["direction"] = coefficients["coefficient"].map(
        lambda value: "酌減同向" if value > 0 else ("不酌減同向" if value < 0 else "接近零")
    )
    coefficients.to_csv(
        output_dir / "logistic_regression_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "status": "ok",
        "model": "StandardScaler + LogisticRegression(class_weight='balanced')",
        "target": target,
        "features": features,
        "skipped_blank_features": skipped_features,
        "n_rows": int(len(model_df)),
        "class_counts": {str(k): int(v) for k, v in class_counts.to_dict().items()},
        "cv": f"StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={RANDOM_STATE})",
        "balanced_accuracy_mean": float(scores.mean()),
        "balanced_accuracy_std": float(scores.std()),
    }
    (output_dir / "model_status.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 線性模型輔助分析",
        "",
        "## 模型定位",
        "",
        "本模型用於觀察本樣本中哪些欄位與法院是否酌減較有關聯，不作為判決預測或法律因果推論。",
        "",
        "## 驗證結果",
        "",
        f"- 使用列數：{metrics['n_rows']}",
        f"- 類別分布：{metrics['class_counts']}",
        f"- 交叉驗證：{metrics['cv']}",
        f"- 平均 balanced accuracy：{metrics['balanced_accuracy_mean']:.3f}",
        f"- balanced accuracy 標準差：{metrics['balanced_accuracy_std']:.3f}",
        "",
        "## 係數表",
        "",
        coefficients.to_markdown(index=False),
        "",
        "## 解讀限制",
        "",
        "- 正係數代表該欄位在本樣本中與 `is_reduced=1` 同向，不代表法院必然因此酌減。",
        "- 樣本數小時，係數可能對單一案件非常敏感。",
        "- 應搭配判決理由摘要與人工查核結果一起解讀。",
    ]
    (output_dir / "線性模型輔助分析.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the advanced linear model version.")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", default="is_reduced")
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, input_path = read_input(args.input_csv)
    prepared = ensure_derived_fields(raw)
    model_df, used_features, skipped_features = select_model_frame(
        prepared, args.features, args.target
    )

    if model_df.empty:
        stop_report(
            args.output_dir,
            "no_complete_model_rows",
            {
                "input_csv": str(input_path),
                "target": args.target,
                "requested_features": args.features,
                "used_features": used_features,
                "skipped_blank_features": skipped_features,
                "raw_rows": int(len(raw)),
            },
        )
        print(f"Skipped model: no complete rows. See {args.output_dir}")
        return

    run_model(model_df, used_features, args.target, args.output_dir, skipped_features)
    print(f"Wrote model outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
