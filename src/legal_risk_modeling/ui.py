from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .manifest import validate_manifest
from .paths import (
    artifact_root,
    classification_output_dir,
    data_quality_output_dir,
    rag_benchmark_output_dir,
    resolve_ratio_release_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HUMAN_GOLD_GATE_MIN_QUERIES = 20

MODEL_LABELS = {
    "mean_baseline": "Mean baseline",
    "ridge_regression_l2": "Ridge",
    "lasso_regression_l1": "Lasso",
    "elastic_net": "ElasticNet",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "HistGradientBoosting",
    "majority_baseline": "Majority baseline",
    "keyword_rule_baseline": "Keyword-rule baseline",
    "logistic_regression_l2": "Logistic regression",
}


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_json(str(path))
    except (OSError, json.JSONDecodeError):
        return None


def _quality_status(reports: list[dict[str, Any] | None]) -> str:
    present = [report for report in reports if report is not None]
    if any(str(report.get("status", "")).lower() == "fail" for report in present):
        return "FAIL"
    if len(present) != len(reports):
        return "NOT AVAILABLE"
    if all(str(report.get("status", "")).lower() == "pass" for report in present):
        return "PASS"
    return "UNKNOWN"


def _format_metric(value: object, *, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def load_release_bundle() -> dict[str, Any]:
    release_dir = resolve_ratio_release_dir(PROJECT_ROOT)
    release_path = release_dir / "approved_release.json"
    promotion_path = release_dir / "promotion_report.json"
    manifest_path = release_dir / "manifest.json"
    metrics_path = release_dir / "metrics.csv"
    predictions_path = release_dir / "predictions.csv"

    required = [release_path, promotion_path, manifest_path, metrics_path, predictions_path]
    missing = [path for path in required if not path.is_file()]
    if missing:
        joined = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(
            "Governed model artifacts are missing: "
            f"{joined}. Run 04_執行稿/run_reduction_ratio_model_sklearn.py offline first."
        )

    manifest = validate_manifest(project_root=PROJECT_ROOT, manifest_path=manifest_path)
    release = load_json(str(release_path))
    promotion = load_json(str(promotion_path))
    metrics = load_csv(str(metrics_path))
    predictions = load_csv(str(predictions_path))

    baseline = str(release.get("baseline") or "mean_baseline")
    approved = [str(model) for model in release.get("approved_models", [])]
    allowed_models = list(dict.fromkeys([baseline, *approved]))
    default_model = str(release.get("default_model") or baseline)

    if release.get("status") != "approved":
        raise ValueError("approved_release.json must have status='approved'")
    if default_model not in allowed_models:
        raise ValueError(
            f"approved_release.json is invalid: default_model={default_model!r} is not approved"
        )
    if promotion.get("default_model") != default_model:
        raise ValueError("release and promotion report disagree on default_model")
    if promotion.get("approved_models", []) != release.get("approved_models", []):
        raise ValueError("release and promotion report disagree on approved_models")

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


def load_dashboard_snapshot(bundle: dict[str, Any]) -> dict[str, Any]:
    classification_dir = classification_output_dir(PROJECT_ROOT)
    rag_dir = rag_benchmark_output_dir(PROJECT_ROOT)
    quality_dir = data_quality_output_dir(PROJECT_ROOT)

    classification_status = _load_optional_json(classification_dir / "model_status.json")
    classification_release = _load_optional_json(classification_dir / "approved_release.json")
    rag_proxy = _load_optional_json(rag_dir / "metrics.json")
    rag_human = _load_optional_json(rag_dir / "human_metrics.json")

    quality_reports = [
        _load_optional_json(quality_dir / "annotation_workbook.json"),
        _load_optional_json(quality_dir / "classification.json"),
        _load_optional_json(quality_dir / "ratio.json"),
    ]

    manifest_git_sha = str((bundle["manifest"].get("git") or {}).get("sha") or "")
    ci_snapshot_verified = False
    history_dir = artifact_root(PROJECT_ROOT) / "benchmark_history"
    if manifest_git_sha and history_dir.is_dir():
        for snapshot_path in history_dir.glob("*.json"):
            snapshot = _load_optional_json(snapshot_path)
            if snapshot and str(snapshot.get("git_sha") or "") == manifest_git_sha:
                ci_snapshot_verified = True
                break

    ratio_approved = [str(model) for model in bundle["release"].get("approved_models", [])]
    ratio_decisions = bundle["promotion"].get("decisions", [])
    ratio_passed = sum(bool(decision.get("passed")) for decision in ratio_decisions)

    classification_default = None
    classification_candidate = None
    classification_promotion = None
    if classification_release:
        classification_default = classification_release.get("default_model")
    if classification_status:
        classification_candidate = classification_status.get("model")
        classification_promotion = classification_status.get("promotion_status")

    proxy_k = int((rag_proxy or {}).get("k") or 3)
    human_k = int((rag_human or {}).get("k") or proxy_k)
    eligible_queries = int((rag_human or {}).get("eligible_query_count") or 0)
    approved_judgments = int((rag_human or {}).get("approved_judgment_count") or 0)
    human_gate_active = eligible_queries >= HUMAN_GOLD_GATE_MIN_QUERIES

    return {
        "system": {
            "manifest": "PASS",
            "data_quality": _quality_status(quality_reports),
            "ci_evidence": "VERIFIED" if ci_snapshot_verified else "NOT BUNDLED",
        },
        "ratio": {
            "default_model": bundle["default_model"],
            "approved_models": ratio_approved,
            "passed_decisions": ratio_passed,
            "decision_count": len(ratio_decisions),
        },
        "classification": {
            "available": bool(classification_status and classification_release),
            "default_model": classification_default,
            "candidate": classification_candidate,
            "promotion_status": classification_promotion,
        },
        "rag": {
            "available": bool(rag_proxy and rag_human),
            "proxy_k": proxy_k,
            "proxy_mrr": (rag_proxy or {}).get(f"mrr_at_{proxy_k}"),
            "proxy_hit_rate": (rag_proxy or {}).get(f"hit_rate_at_{proxy_k}"),
            "human_k": human_k,
            "human_ndcg": (rag_human or {}).get(f"ndcg_at_{human_k}"),
            "eligible_queries": eligible_queries,
            "approved_judgments": approved_judgments,
            "gate_min_queries": HUMAN_GOLD_GATE_MIN_QUERIES,
            "gate_active": human_gate_active,
        },
    }


def render_demo_dashboard(bundle: dict[str, Any], snapshot: dict[str, Any]) -> None:
    st.subheader("Governance dashboard")
    st.caption("一頁確認 release integrity、model promotion 與 RAG evaluation readiness。")

    top_left, top_right = st.columns(2)
    with top_left:
        with st.container(border=True):
            st.markdown("#### System status")
            col1, col2, col3 = st.columns(3)
            col1.metric("Manifest", snapshot["system"]["manifest"])
            col2.metric("Data Quality", snapshot["system"]["data_quality"])
            col3.metric("CI evidence", snapshot["system"]["ci_evidence"])
            st.caption(
                "CI evidence 代表 artifact 內含與 Manifest Git SHA 對得上的 benchmark snapshot；"
                "不是現場連線查詢 GitHub 狀態。"
            )

    with top_right:
        with st.container(border=True):
            st.markdown("#### Ratio")
            ratio = snapshot["ratio"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Default", model_label(str(ratio["default_model"])))
            col2.metric("Approved learned", len(ratio["approved_models"]))
            col3.metric(
                "Promotion",
                f"{ratio['passed_decisions']}/{ratio['decision_count']} passed",
            )
            if ratio["approved_models"]:
                st.success(
                    "Approved: "
                    + ", ".join(model_label(model) for model in ratio["approved_models"])
                )
            else:
                st.warning("Learned models rejected；正式 release 維持 Mean baseline。")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        with st.container(border=True):
            st.markdown("#### Classification")
            classification = snapshot["classification"]
            if not classification["available"]:
                st.info("Classification governed artifacts 尚未 bundled。")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Default",
                    model_label(str(classification["default_model"])),
                )
                col2.metric(
                    "Candidate",
                    model_label(str(classification["candidate"])),
                )
                col3.metric(
                    "Promotion",
                    str(classification["promotion_status"] or "unknown").upper(),
                )
                if str(classification["promotion_status"]).lower() == "approved":
                    st.success("Candidate 已通過 classification promotion gate。")
                else:
                    st.warning("Candidate 未通過 promotion；正式 release 使用 baseline。")

    with bottom_right:
        with st.container(border=True):
            st.markdown("#### RAG")
            rag = snapshot["rag"]
            if not rag["available"]:
                st.info("RAG benchmark artifacts 尚未 bundled。")
            else:
                col1, col2 = st.columns(2)
                col1.metric(
                    f"Proxy MRR@{rag['proxy_k']}",
                    _format_metric(rag["proxy_mrr"]),
                )
                col2.metric(
                    f"Proxy HitRate@{rag['proxy_k']}",
                    _format_metric(rag["proxy_hit_rate"]),
                )
                col3, col4 = st.columns(2)
                human_ndcg = (
                    _format_metric(rag["human_ndcg"])
                    if rag["eligible_queries"] > 0
                    else "—"
                )
                col3.metric(f"Human nDCG@{rag['human_k']}", human_ndcg)
                col4.metric("Human gate", "ACTIVE" if rag["gate_active"] else "INACTIVE")

                progress = min(
                    rag["eligible_queries"] / max(rag["gate_min_queries"], 1),
                    1.0,
                )
                st.progress(progress)
                st.caption(
                    f"Human Gold progress: {rag['eligible_queries']}/{rag['gate_min_queries']} "
                    f"eligible queries；approved judgments: {rag['approved_judgments']}。"
                )

    st.caption(
        f"Governed release Git SHA: `{(bundle['manifest'].get('git') or {}).get('sha') or 'unavailable'}`"
    )


def render_governance_status(bundle: dict[str, Any]) -> None:
    release = bundle["release"]
    promotion = bundle["promotion"]
    manifest = bundle["manifest"]

    st.subheader("Ratio governance detail")
    col1, col2, col3 = st.columns(3)
    col1.metric("UI default", model_label(bundle["default_model"]))
    col2.metric("Approved learned models", len(release.get("approved_models", [])))
    col3.metric("Promotion decisions", len(promotion.get("decisions", [])))

    if not release.get("approved_models"):
        st.warning("目前沒有 learned model 通過 promotion gate；UI 因此自動退回 Mean baseline。")
    else:
        st.success("UI 只允許 baseline 與已通過 promotion gate 的模型。")

    git = manifest.get("git") or {}
    runtime = manifest.get("runtime") or {}
    with st.expander("Run manifest", expanded=False):
        st.write(f"Run ID: `{manifest.get('run_id', '—')}`")
        st.write(f"Git SHA: `{git.get('sha') or 'unavailable'}`")
        st.write(f"Git dirty: `{git.get('dirty')}`")
        st.write(f"Created UTC: `{manifest.get('created_at_utc', '—')}`")
        st.write(f"Dataset files: {len(manifest.get('datasets', []))}")
        st.write(f"Tracked packages: {len(runtime.get('packages', {}))}")
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
        "Hyperparameter tuning 僅使用 training period 內的 rolling CV；"
        "validation/test/latest 保留給 promotion 與監控。"
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
        "再由 approved_release.json + manifest integrity validation 允許 UI 載入。"
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

    dashboard_snapshot = load_dashboard_snapshot(bundle)
    render_demo_dashboard(bundle, dashboard_snapshot)
    st.divider()

    render_governance_status(bundle)
    tab_predictions, tab_metrics, tab_gate = st.tabs(["Predictions", "Backtest", "Promotion gate"])
    with tab_predictions:
        render_case_predictions(bundle, selected_model)
    with tab_metrics:
        render_metrics(bundle, selected_model)
    with tab_gate:
        render_promotion_table(bundle)

    st.info("本工具只展示回測與模型治理結果，不構成法律意見，也不能取代法院或人工專業判斷。")
