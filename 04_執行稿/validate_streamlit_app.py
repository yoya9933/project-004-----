from __future__ import annotations

import csv
import importlib.util
import json
import py_compile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "streamlit_app.py"
README_PATH = PROJECT_ROOT / "README.md"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
ANNOTATION_PATH = PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "annotation_workbook.csv"
SIMILAR_CASES_PATH = PROJECT_ROOT / "06_交付物" / "rag_model_explanation" / "similar_case_evidence.csv"
FEATURE_ANALYSIS_PATH = PROJECT_ROOT / "06_交付物" / "120_判決主要特徵值總表.csv"
OUTPUT_DIR = PROJECT_ROOT / "05_測試與驗證"


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate() -> dict[str, Any]:
    failures: list[str] = []

    require(APP_PATH.exists(), f"Missing root Streamlit app: {APP_PATH}", failures)
    require(README_PATH.exists(), f"Missing root README: {README_PATH}", failures)
    require(REQUIREMENTS_PATH.exists(), f"Missing requirements file: {REQUIREMENTS_PATH}", failures)
    require(ANNOTATION_PATH.exists(), f"Missing annotation workbook: {ANNOTATION_PATH}", failures)
    require(SIMILAR_CASES_PATH.exists(), f"Missing similar case evidence: {SIMILAR_CASES_PATH}", failures)
    require(
        FEATURE_ANALYSIS_PATH.exists(),
        f"Missing feature analysis CSV: {FEATURE_ANALYSIS_PATH}",
        failures,
    )

    compile_ok = False
    if APP_PATH.exists():
        try:
            py_compile.compile(str(APP_PATH), doraise=True)
            compile_ok = True
        except py_compile.PyCompileError as exc:
            failures.append(f"streamlit_app.py does not compile: {exc.msg}")

    source = APP_PATH.read_text(encoding="utf-8") if APP_PATH.exists() else ""
    for required_text in [
        "streamlit as st",
        "ANNOTATION_PATH",
        "run_live_training",
        "merge_training_results",
        "is_case_trained",
        "render_untrained_model_notice",
        "color: var(--tool-ink);",
        '[data-testid="stToolbar"]',
        '[data-testid="stHeader"]',
        '[data-testid="stAppViewContainer"]',
        '[data-testid="stSidebar"]',
        '[data-testid="stMetric"]',
        '--tool-page: #f4f6f5;',
        '--tool-sidebar: #ffffff;',
        'background: var(--tool-page) !important;',
        "#MainMenu",
        "尚未現場訓練，模型比較會在訓練後顯示",
        "現場訓練模型",
        "render_sidebar",
        "案件儀表板",
        "資料總覽",
        "FEATURE_ANALYSIS_PATH",
        "prepare_feature_correlation_frame",
        "compute_feature_correlation",
        "render_feature_correlation",
        "特徵相關性",
        "限制說明",
        "RAG 相似案例",
        "重要特徵",
    ]:
        require(required_text in source, f"streamlit_app.py missing text: {required_text}", failures)

    for removed_text in [
        'st.selectbox("風險"',
        '"高風險"',
        'class="risk-badge',
        '"風險": item.get("riskLevel")',
        "**風險分布**",
        "**風險規則**",
        "酌減區間",
        "區間：",
        "actualBucket",
        "ridgePredictedBucket",
        "lassoPredictedBucket",
        "reduction_bucket",
        "def reduction_bucket",
    ]:
        require(removed_text not in source, f"streamlit_app.py still contains removed UI/data field: {removed_text}", failures)

    requirements = REQUIREMENTS_PATH.read_text(encoding="utf-8") if REQUIREMENTS_PATH.exists() else ""
    requirements_lower = requirements.lower()
    require("streamlit" in requirements_lower, "requirements.txt does not include streamlit", failures)
    require("pandas" in requirements_lower, "requirements.txt does not include pandas", failures)
    require("scikit-learn" in requirements_lower, "requirements.txt does not include scikit-learn", failures)

    readme = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    for required_text in ["streamlit run streamlit_app.py", "requirements.txt", "annotation_workbook.csv", "現場訓練"]:
        require(required_text in readme, f"README.md missing text: {required_text}", failures)
    require("酌減區間" not in readme, "README.md still mentions removed bucket metric", failures)

    annotation_rows = load_csv(ANNOTATION_PATH) if ANNOTATION_PATH.exists() else []
    similar_rows = load_csv(SIMILAR_CASES_PATH) if SIMILAR_CASES_PATH.exists() else []
    similar_by_jid: dict[str, list[dict[str, str]]] = {}
    for row in similar_rows:
        similar_by_jid.setdefault(row.get("query_JID", ""), []).append(row)

    require(len(annotation_rows) == 120, f"annotation row count != 120: {len(annotation_rows)}", failures)
    require(len(similar_rows) == 600, f"similar evidence count != 600: {len(similar_rows)}", failures)

    missing_similar = 0
    for item in annotation_rows:
        jid = item.get("JID", "")
        if len(similar_by_jid.get(jid, [])) != 5:
            missing_similar += 1
    require(missing_similar == 0, f"Cases missing 5 similar cases: {missing_similar}", failures)

    streamlit_available = importlib.util.find_spec("streamlit") is not None
    pandas_available = importlib.util.find_spec("pandas") is not None
    sklearn_available = importlib.util.find_spec("sklearn") is not None

    live_training_ok = False
    live_training_case_count = 0
    live_training_ratio_count = 0
    live_feature_contribution_count = 0
    missing_models = 0
    case_label_includes_jid = False
    app_module: Any = None
    if APP_PATH.exists() and streamlit_available and pandas_available and sklearn_available:
        try:
            spec = importlib.util.spec_from_file_location("streamlit_app_validation_target", APP_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module spec from {APP_PATH}")
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)
            live_result = app_module.run_live_training()
            live_training_case_count = len(live_result["cases"])
            live_training_ratio_count = live_result["metadata"].get("usableRatioRows", 0)
            live_feature_contribution_count = live_result["metadata"].get("featureContributionCount", 0)
            required_models = {"logistic_regression_l2", "ridge_regression_l2", "lasso_regression_l1"}
            for item in live_result["cases"]:
                jid = str(item.get("jid", ""))
                if not required_models.issubset(set(live_result["contributionsByJid"].get(jid, {}).keys())):
                    missing_models += 1
            label_map = app_module.case_label_map(live_result["cases"])
            case_label_includes_jid = all(
                str(item.get("jid", "")) in str(label_map.get(str(item.get("jid", "")), ""))
                for item in live_result["cases"]
            )
            live_training_ok = live_training_case_count == 120 and live_training_ratio_count >= 100
        except Exception as exc:  # noqa: BLE001
            failures.append(f"live training failed: {exc}")
    require(live_training_ok, "live training did not produce expected case predictions", failures)
    require(missing_models == 0, f"Cases missing live contribution models: {missing_models}", failures)
    require(case_label_includes_jid, "Case select labels must include JID to distinguish duplicate titles", failures)

    feature_analysis_ok = False
    feature_analysis_rows = 0
    feature_analysis_reduction_rate_rows = 0
    feature_matrix_shape: list[int] = []
    constant_feature_is_nan = False
    if FEATURE_ANALYSIS_PATH.exists() and app_module is not None:
        try:
            correlation_rows = load_csv(FEATURE_ANALYSIS_PATH)
            correlation_frame = app_module.prepare_feature_correlation_frame(
                correlation_rows
            )
            correlation_result = app_module.compute_feature_correlation(
                correlation_frame
            )
            feature_analysis_rows = len(correlation_frame)
            feature_analysis_reduction_rate_rows = int(
                correlation_frame["酌減率"].notna().sum()
            )
            feature_matrix_shape = list(correlation_result["matrix"].shape)
            constant_value = correlation_result["matrix"].loc[
                "部分完成／部分驗收", "業主可歸責"
            ]
            constant_feature_is_nan = bool(app_module.pd.isna(constant_value))
            feature_analysis_ok = (
                feature_analysis_rows == 120
                and feature_matrix_shape == [6, 6]
                and len(correlation_result["is_reduced"]) == 6
                and len(correlation_result["reduction_rate"]) == 6
                and constant_feature_is_nan
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"feature correlation validation failed: {exc}")
    require(feature_analysis_ok, "feature correlation analysis validation failed", failures)

    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "checks": {
            "appPath": str(APP_PATH.relative_to(PROJECT_ROOT)),
            "readmePath": str(README_PATH.relative_to(PROJECT_ROOT)),
            "compileOk": compile_ok,
            "streamlitInstalled": streamlit_available,
            "pandasInstalled": pandas_available,
            "sklearnInstalled": sklearn_available,
            "liveTrainingOk": live_training_ok,
            "liveTrainingCaseCount": live_training_case_count,
            "liveTrainingUsableRatioRows": live_training_ratio_count,
            "caseCount": len(annotation_rows),
            "similarCaseCount": len(similar_rows),
            "featureContributionCount": live_feature_contribution_count,
            "missingSimilar": missing_similar,
            "missingContributionModels": missing_models,
            "caseLabelIncludesJid": case_label_includes_jid,
            "featureAnalysisOk": feature_analysis_ok,
            "featureAnalysisRows": feature_analysis_rows,
            "featureAnalysisReductionRateRows": feature_analysis_reduction_rate_rows,
            "featureMatrixShape": feature_matrix_shape,
            "constantFeatureIsNan": constant_feature_is_nan,
        },
    }


def write_report(status: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "streamlit_app_validation.json"
    md_path = OUTPUT_DIR / "streamlit_app_validation.md"
    json_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Streamlit 風險評估工具驗證",
        "",
        f"- 狀態：{status['status']}",
        f"- App 位置：`{status['checks']['appPath']}`",
        f"- README 位置：`{status['checks']['readmePath']}`",
        f"- Python 編譯：{status['checks']['compileOk']}",
        f"- 本機已安裝 Streamlit：{status['checks']['streamlitInstalled']}",
        f"- 本機已安裝 pandas：{status['checks']['pandasInstalled']}",
        f"- 本機已安裝 scikit-learn：{status['checks']['sklearnInstalled']}",
        f"- 現場訓練可執行：{status['checks']['liveTrainingOk']}",
        f"- 現場訓練案件數：{status['checks']['liveTrainingCaseCount']}",
        f"- 現場訓練可用比例樣本數：{status['checks']['liveTrainingUsableRatioRows']}",
        f"- 案件數：{status['checks']['caseCount']}",
        f"- 相似案例數：{status['checks']['similarCaseCount']}",
        f"- 特徵貢獻數：{status['checks']['featureContributionCount']}",
        f"- 缺相似案例案件數：{status['checks']['missingSimilar']}",
        f"- 缺特徵模型案件數：{status['checks']['missingContributionModels']}",
    ]
    if status["failures"]:
        lines.extend(["", "## 失敗項目", ""])
        lines.extend(f"- {failure}" for failure in status["failures"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    status = validate()
    write_report(status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
