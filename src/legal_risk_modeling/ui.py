from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .paths import resolve_ratio_release_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_LABELS = {
    "mean_baseline": "Mean baseline",
    "ridge_regression_l2": "Ridge",
    "lasso_regression_l1": "Lasso",
    "elastic_net": "ElasticNet",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "HistGradientBoosting",
}


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_release_bundle() -> dict[str, Any]:
    release_dir = resolve_ratio_release_dir(PROJECT_ROOT)
    release_path = release_dir / "approved_release.json"
    promotion_path = release_dir / "promotion_report.json"
    manifest_path = release_dir / "manifest.json"
    metrics_path = release_dir / "metrics.csv"
    predictions_path = release_dir / "predictions.csv"

    required = [release_path, promotion_path, metrics_path, predictions_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            "Governed model artifacts are missing: "
            f"{joined}. Run 04_執行稿/run_reduction_ratio_model_sklearn.py offline first."
        )

    release = load_json(str(release_path))
    promotion = load_json(str(promotion_path))
    metrics = load_csv(str(metrics_path))
    predictions = load_csv(str(predictions_path))
    manifest = load_json(str(manifest_path)) if manifest_path.exists() else None

    baseline = str(release.get("baseline") or "mean_baseline")
    approved = [str(model) for model in release.get("approved_models", [])]
    allowed_models = list(dict.fromkeys([baseline, *approved]))
    default_model = str(release.get("default_model") or baseline)

    if default_model not in allowed_models:
        raise ValueError(
            f"approved_release.json is invalid: default_model={default_model!r} is not approved"
        )

    metric_models = set(metrics["model"].astype(str))
    prediction_models = set(predictions["model"].astype(str))
    missing_metric_models = [model for model in allowed_models if model not in metric_models]
    missing_prediction_models = [model for model in allowed_models if model not in prediction_models]
    if missing_metric_models or missing_prediction_models:
        raise ValueError(
            "approved release references missing artifacts: "
            f"metrics={missing_metric_models}, predictions={missing_prediction_models}"
        )

    return {
        "release": release,
        "promotion": promotion,
        "metrics": metrics,
        "predictions": predictions,
        "manifest": manifest,
        "release_dir": release_dir,
        "allowed_models": allowed_models,
        "default_model": default_model,
    }


def render_governance_status(bundle: dict[str, Any]) -> None:
    release = bundle["release"]
    promotion = bundle["promotion"]
    manifest = bundle["manifest"]

    st.subheader("Model governance")
    col1, col2, col3 = st.columns(3)
    col1.metric("UI default", model_label(bundle["default_model"]))
    col2.metric("Approved learned models", len(release.get("approved_models", [])))
    col3.metric("Promotion decisions", len(promotion.get("decisions", [])))

    if not release.get("approved_models"):
        st.warning("目前沒有 learned model 通過 promotion gate；UI 因此自動退回 Mean baseline。")
    else:
        st.success("UI 只允許 baseline 與已通過 promotion gate 的模型。")

    if manifest is None:
        st.warning(
            "目前這批 legacy artifacts 尚無 manifest。重新執行 governed offline training 後，"
            "每次 run 會記錄 Git SHA、資料 SHA256、artifact SHA256、Python/platform 與 package versions。"
        )
    else:
        with st.expander("Run manifest", expanded=False):
            st.write(f"Run ID: `{manifest.get('run_id', '—')}`")
            st.write(f"Git SHA: `{manifest.get('git_sha') or 'unavailable'}`")
            st.write(f"Created UTC: `{manifest.get('created_at_utc', '—')}`")
            st.write(f"Dataset files: {len(manifest.get('datasets', []))}")
            st.write(f"Tracked packages: {len(manifest.get('packages', {}))}")
            st.json(manifest, expanded=False)


def render_promotion_table(bundle: dict[str, Any]) -> None:
    decisions = bundle["promotion"].get("decisions", [])
    if not decisions:
        st.info("尚無 promotion decision 明細。")
        return

    rows = []
    for decision in decisions:
        metrics = decision.get("metrics", {})
        rows.append(
            {
                "model": model_label(str(decision.get("model"))),
                "passed": bool(decision.get("passed")),
                "validation_rmse_gain": metrics.get("validation_rmse_gain_vs_baseline"),
                "validation_mae_gain": metrics.get("validation_mae_gain_vs_baseline"),
                "test_rmse_delta": metrics.get("test_rmse_delta_vs_baseline"),
                "latest_rmse_delta": metrics.get("latest_rmse_delta_vs_baseline"),
                "reason": "；".join(decision.get("reasons", [])),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_metrics(bundle: dict[str, Any], selected_model: str) -> None:
    metrics = bundle["metrics"].copy()
    baseline = str(bundle["release"].get("baseline") or "mean_baseline")
    selected = metrics[metrics["model"].astype(str).isin({baseline, selected_model})].copy()
    selected["model"] = selected["model"].astype(str).map(model_label)
    st.subheader("Backtest metrics")
    st.caption(
        "2021–2023 為 train；2024 validation；2025 test；2026 latest check。"
        "Learned model 未通過 gate 時不會成為 UI default。"
    )
    st.dataframe(selected, use_container_width=True, hide_index=True)


def render_case_predictions(bundle: dict[str, Any], selected_model: str) -> None:
    predictions = bundle["predictions"].copy()
    model_rows = predictions[predictions["model"].astype(str) == selected_model].copy()
    if model_rows.empty:
        st.info("目前 approved artifact 沒有此模型的 prediction rows。")
        return

    st.subheader("Approved artifact predictions")
    jid_options = model_rows["JID"].dropna().astype(str).drop_duplicates().tolist()
    query = st.text_input("搜尋 JID", placeholder="輸入完整或部分 JID")
    if query:
        jid_options = [jid for jid in jid_options if query.lower() in jid.lower()]

    if not jid_options:
        st.info("沒有符合條件的 JID。")
        return

    selected_jid = st.selectbox("案件", jid_options)
    case_rows = model_rows[model_rows["JID"].astype(str) == selected_jid].copy()
    if case_rows.empty:
        return

    row = case_rows.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", model_label(selected_model))
    col2.metric("Split", str(row.get("split", "—")))
    col3.metric("Actual remaining", f"{float(row['actual_remaining_ratio']):.1%}")
    col4.metric("Predicted remaining", f"{float(row['predicted_remaining_ratio']):.1%}")

    display_columns = [
        column
        for column in [
            "JID",
            "decision_year",
            "split",
            "actual_remaining_ratio",
            "predicted_remaining_ratio",
            "actual_reduction_rate",
            "predicted_reduction_rate",
            "actual_bucket",
            "predicted_bucket",
        ]
        if column in case_rows.columns
    ]
    st.dataframe(case_rows[display_columns], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="工程違約金模型治理儀表板",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("工程違約金風險模型")
    st.caption(
        "此 App 不訓練模型。所有模型與 predictions 必須先經離線 pipeline、promotion gate，"
        "再由 approved_release.json 允許 UI 載入。"
    )

    try:
        bundle = load_release_bundle()
    except (FileNotFoundError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        st.code("python 04_執行稿/run_reduction_ratio_model_sklearn.py")
        st.stop()

    allowed_models = bundle["allowed_models"]
    default_model = bundle["default_model"]
    selected_model = st.sidebar.selectbox(
        "Approved model",
        options=allowed_models,
        index=allowed_models.index(default_model),
        format_func=model_label,
    )

    render_governance_status(bundle)
    tab_predictions, tab_metrics, tab_gate = st.tabs(["Predictions", "Backtest", "Promotion gate"])
    with tab_predictions:
        render_case_predictions(bundle, selected_model)
    with tab_metrics:
        render_metrics(bundle, selected_model)
    with tab_gate:
        render_promotion_table(bundle)

    st.info("本工具只展示回測與模型治理結果，不構成法律意見，也不能取代法院或人工專業判斷。")
