from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_CSV = PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "annotation_workbook.csv"
DEFAULT_RAG_INDEX_CSV = PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "rag_case_index.csv"
DEFAULT_SIMILAR_CASES_CSV = PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "rag_similar_cases.csv"
DEFAULT_CLASSIFICATION_DIR = PROJECT_ROOT / "06_交付物" / "is_reduced_classification"
DEFAULT_RATIO_DIR = PROJECT_ROOT / "06_交付物" / "reduction_ratio_model"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "06_交付物" / "rag_model_explanation"

CLASSIFICATION_MODEL = "logistic_regression_l2"
RATIO_MAIN_MODEL = "ridge_regression_l2"
RATIO_SECONDARY_MODEL = "lasso_regression_l1"
RATIO_BASELINE_MODEL = "mean_baseline"

FEATURE_LABELS = {
    "x_log_contract_price": "契約總價(log)",
    "x_log_claimed_penalty": "主張違約金(log)",
    "x_claim_to_contract_ratio": "主張違約金/契約總價",
    "x_delay_days": "逾期天數",
    "x_penalty_per_delay_day": "每日違約金",
    "x_issue_owner_fault": "業主可歸責爭點",
    "x_issue_contractor_fault": "承包商可歸責爭點",
    "x_issue_extension_request": "展延申請爭點",
    "x_issue_actual_damage_unclear": "實際損害不明爭點",
    "x_issue_partial_completion": "部分完工爭點",
    "x_issue_used_by_owner": "業主已使用成果爭點",
    "x_ai_issue_owner_fault": "AI 提示：業主可歸責",
    "x_ai_issue_contractor_fault": "AI 提示：承包商可歸責",
    "x_ai_issue_extension_request": "AI 提示：展延申請",
    "x_ai_issue_actual_damage_unclear": "AI 提示：實際損害不明",
    "x_ai_issue_partial_completion": "AI 提示：部分完工",
    "x_ai_issue_used_by_owner": "AI 提示：業主已使用成果",
    "x_money_candidate_count": "AI 候選金額數",
    "x_delay_candidate_count": "AI 候選天數數",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace("，", "").replace("%", "")
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def to_int_text(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return str(int(number))


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def compact(text: str, limit: int = 120) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows if row.get(key, "")}


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key, "")].append(row)
    return groups


def prediction_map(rows: list[dict[str, str]], model: str, key: str = "JID") -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows if row.get("model") == model and row.get(key)}


def coefficient_rows(rows: list[dict[str, str]], model: str | None = None) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("feature") == "intercept":
            continue
        if model is not None and row.get("model") != model:
            continue
        filtered.append(row)
    return filtered


def feature_value(row: dict[str, str] | None, feature: str) -> float:
    if not row:
        return 0.0
    return to_float(row.get(feature)) or 0.0


def build_contributions(
    jid: str,
    model_family: str,
    model: str,
    feature_row: dict[str, str] | None,
    coeff_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coeff in coeff_rows:
        feature = coeff.get("feature", "")
        value = feature_value(feature_row, feature)
        mean = to_float(coeff.get("mean")) or 0.0
        std = to_float(coeff.get("std")) or 1.0
        if std == 0:
            std = 1.0
        coefficient = to_float(coeff.get("coefficient_scaled")) or 0.0
        standardized = (value - mean) / std
        contribution = standardized * coefficient
        if model_family == "classification":
            interpretation = "提高酌減機率" if contribution > 0 else "降低酌減機率" if contribution < 0 else "影響接近零"
        else:
            interpretation = "提高准許比例/降低酌減幅度" if contribution > 0 else "降低准許比例/提高酌減幅度" if contribution < 0 else "影響接近零"
        rows.append(
            {
                "JID": jid,
                "model_family": model_family,
                "model": model,
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "feature_value": round(value, 8),
                "mean": round(mean, 8),
                "std": round(std, 8),
                "coefficient_scaled": round(coefficient, 8),
                "standardized_value": round(standardized, 8),
                "contribution": round(contribution, 8),
                "contribution_abs": round(abs(contribution), 8),
                "interpretation": interpretation,
            }
        )
    rows.sort(key=lambda item: float(item["contribution_abs"]), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank_by_abs"] = index
    return rows


def summarize_contributions(rows: list[dict[str, Any]], sign: str, limit: int = 3) -> str:
    if sign == "positive":
        selected = [row for row in rows if float(row["contribution"]) > 0]
        selected.sort(key=lambda row: float(row["contribution"]), reverse=True)
    else:
        selected = [row for row in rows if float(row["contribution"]) < 0]
        selected.sort(key=lambda row: abs(float(row["contribution"])), reverse=True)
    parts = []
    for row in selected[:limit]:
        value = fmt(row["feature_value"], 3)
        contribution = fmt(row["contribution"], 3)
        parts.append(f"{row['feature_label']}({contribution}; 值={value})")
    return "；".join(parts)


def abs_error(actual: Any, predicted: Any) -> str:
    actual_number = to_float(actual)
    predicted_number = to_float(predicted)
    if actual_number is None or predicted_number is None:
        return ""
    return fmt(abs(actual_number - predicted_number), 4)


def is_correct(actual: Any, predicted: Any) -> str:
    actual_text = str(actual or "").strip()
    predicted_text = str(predicted or "").strip()
    if not actual_text or not predicted_text:
        return ""
    return "1" if actual_text == predicted_text else "0"


def top_similar_digest(similar_rows: list[dict[str, str]], limit: int = 5) -> str:
    parts = []
    for row in similar_rows[:limit]:
        parts.append(
            "#{rank} {jid}({year}, {court}, score={score})".format(
                rank=row.get("similar_rank", ""),
                jid=row.get("similar_JID", ""),
                year=row.get("similar_decision_year", ""),
                court=row.get("similar_court", ""),
                score=row.get("similarity_score", ""),
            )
        )
    return "；".join(parts)


def shared_terms_digest(similar_rows: list[dict[str, str]], limit: int = 12) -> str:
    ordered_terms: list[str] = []
    seen: set[str] = set()
    for row in similar_rows:
        for term in str(row.get("shared_terms", "")).split(";"):
            term = term.strip()
            if not term or term in seen:
                continue
            seen.add(term)
            ordered_terms.append(term)
    return ";".join(ordered_terms[:limit])


def snippet_digest(similar_rows: list[dict[str, str]], limit: int = 2) -> str:
    parts = []
    for row in similar_rows[:limit]:
        snippet = row.get("similar_reduction_snippet") or row.get("similar_delay_snippet") or ""
        parts.append(f"#{row.get('similar_rank')} {row.get('similar_JID')}: {compact(snippet, 120)}")
    return "；".join(parts)


def similar_label_summary(
    similar_rows: list[dict[str, str]],
    annotation_by_jid: dict[str, dict[str, str]],
) -> dict[str, str]:
    labeled_count = 0
    reduced_count = 0
    ratios: list[float] = []
    for row in similar_rows:
        annotation = annotation_by_jid.get(row.get("similar_JID", ""))
        if not annotation:
            continue
        reduced = to_int_text(annotation.get("is_reduced"))
        if reduced:
            labeled_count += 1
            if reduced == "1":
                reduced_count += 1
        ratio = to_float(annotation.get("remaining_ratio"))
        if ratio is not None and 0 <= ratio <= 1:
            ratios.append(ratio)
    avg_ratio = sum(ratios) / len(ratios) if ratios else None
    return {
        "similar_labeled_count": str(labeled_count),
        "similar_labeled_reduced_count": str(reduced_count),
        "similar_avg_remaining_ratio": fmt(avg_ratio, 4) if avg_ratio is not None else "",
    }


def make_case_markdown(summary_rows: list[dict[str, Any]], evidence_by_query: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# RAG 相似案例與模型解釋卡",
        "",
        "本文件由 `build_rag_model_explanation_pack.py` 產生。每張卡整合案件標註、分類模型預測、比例模型預測、模型特徵貢獻與 RAG 相似案例。",
        "",
        "> 注意：目前標註為 AI 假設版，所有金額、比例、法院結論與關鍵爭點仍需回到原判決全文人工查核。",
        "",
    ]
    split_labels = {
        "train_2021_2023": "訓練集 2021-2023",
        "validation_2024": "驗證集 2024",
        "test_2025": "測試集 2025",
        "latest_2026": "最新年度檢查 2026",
    }
    for row in summary_rows:
        jid = str(row.get("JID", ""))
        title = str(row.get("JTITLE", ""))
        split = str(row.get("split", ""))
        lines.extend(
            [
                f"## {jid}｜{title}",
                "",
                f"- 切分：{split_labels.get(split, split)}；法院：{row.get('court', '')}；年度：{row.get('decision_year', '')}。",
                f"- 是否酌減模型：實際 `{row.get('actual_is_reduced', '')}`，預測 `{row.get('predicted_is_reduced', '')}`，酌減機率 `{row.get('reduction_probability', '')}`，是否命中 `{row.get('classification_correct', '')}`。",
                f"- 酌減比例模型：實際 remaining_ratio `{row.get('actual_remaining_ratio', '')}`，Ridge 預測 `{row.get('ridge_predicted_remaining_ratio', '')}`，Lasso 預測 `{row.get('lasso_predicted_remaining_ratio', '')}`，mean baseline `{row.get('mean_predicted_remaining_ratio', '')}`。",
                f"- 分類模型主要推向酌減：{row.get('top_classification_toward_reduction', '') or '無明顯正向貢獻'}。",
                f"- 分類模型主要推向未酌減：{row.get('top_classification_toward_no_reduction', '') or '無明顯負向貢獻'}。",
                f"- Ridge 主要推向更大酌減：{row.get('top_ratio_toward_more_reduction_ridge', '') or '無明顯負向貢獻'}。",
                f"- Ridge 主要推向較少酌減：{row.get('top_ratio_toward_less_reduction_ridge', '') or '無明顯正向貢獻'}。",
                "- RAG 相似案例：",
            ]
        )
        for evidence in evidence_by_query.get(jid, [])[:5]:
            lines.append(
                "  - #{rank} `{similar_jid}`，score `{score}`，{court}，{title}；共同詞：{terms}；片段：{snippet}".format(
                    rank=evidence.get("similar_rank", ""),
                    similar_jid=evidence.get("similar_JID", ""),
                    score=evidence.get("similarity_score", ""),
                    court=evidence.get("similar_court", ""),
                    title=evidence.get("similar_title", ""),
                    terms=compact(str(evidence.get("shared_terms", "")), 80),
                    snippet=compact(str(evidence.get("similar_reduction_snippet") or evidence.get("similar_delay_snippet") or ""), 120),
                )
            )
        lines.extend(
            [
                "- 查核提醒：相似案例只提示可比較的法律事實與爭點，不可直接套用結論；模型特徵貢獻只反映目前資料與模型的統計方向。",
                "",
            ]
        )
    return "\n".join(lines)


def make_workflow_markdown(status: dict[str, Any]) -> str:
    return f"""# RAG 相似案例檢索與模型解釋流程

## 目的

本流程把既有 RAG 相似案例檢索結果、`is_reduced` 分類模型、`remaining_ratio` 比例模型與模型係數整合成案件級解釋包。它的定位是輔助人工閱讀、風險辨識與報告展示，不是自動判決或法律意見。

## 一次重跑命令

```powershell
python .\\04_執行稿\\build_rag_model_explanation_pack.py
```

## 輸入

- 標註資料：`06_交付物/ai_rag_annotation/annotation_workbook.csv`
- RAG 索引：`06_交付物/ai_rag_annotation/rag_case_index.csv`
- 相似案例：`06_交付物/ai_rag_annotation/rag_similar_cases.csv`
- 分類模型：`06_交付物/is_reduced_classification/`
- 比例模型：`06_交付物/reduction_ratio_model/`

## 輸出

- `case_explanation_summary.csv`：每案一列，整合模型預測、特徵貢獻摘要、前 5 件相似案例與查核提示。
- `similar_case_evidence.csv`：每組 query/similar 一列，保留 600 列相似案例證據，並附上 query 的模型預測。
- `model_feature_contributions.csv`：每案、每模型、每特徵的標準化值、係數與貢獻值，可用來追溯模型解釋。
- `case_explanation_cards.md`：120 件案件的可讀式解釋卡。
- `rag_model_explanation_workflow.md`：本流程說明。
- `explanation_status.json`：機器可讀的輸出狀態與列數。

## 解釋流程

1. 先用 `case_explanation_summary.csv` 找到目標案件，確認其時間切分、實際標註、分類模型酌減機率與比例模型預測。
2. 查看 `top_classification_toward_reduction` 與 `top_classification_toward_no_reduction`，判斷分類模型主要被哪些事實型特徵推動。
3. 查看 Ridge 比例模型的 `top_ratio_toward_more_reduction_ridge` 與 `top_ratio_toward_less_reduction_ridge`，判斷模型預測准許比例時的主要方向。
4. 進入 `similar_case_evidence.csv`，閱讀前 5 件相似案例的共同詞、相似度、法院、年度與片段。
5. 回到原判決全文查核金額角色、法院最終結論、展延/歸責/損害等爭點；相似案例不得直接套用結論。
6. 報告時同時呈現 baseline 與模型結果，避免只挑模型分數或只挑相似案例支持單一結論。

## 目前狀態

- 案件解釋列數：{status["case_count"]}
- 相似案例證據列數：{status["similar_case_evidence_count"]}
- 特徵貢獻列數：{status["feature_contribution_count"]}
- 主要分類模型：`{CLASSIFICATION_MODEL}`
- 主要比例模型：`{RATIO_MAIN_MODEL}`，並保留 `{RATIO_SECONDARY_MODEL}` 與 `{RATIO_BASELINE_MODEL}` 比較。

## 限制

- 目前正式欄位為 AI 假設版標註，尚未等同逐案人工查核。
- RAG 相似度主要來自詞彙與爭點重疊，代表閱讀優先順序，不代表法律上案情完全相同。
- 線性模型的特徵貢獻是目前資料與模型下的統計方向，不應被解讀為法院必然採納的法律因果。
- 比例模型在 2025 測試集尚未優於 mean baseline，因此比例預測更適合作為流程展示與後續改良基準。
"""


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    annotation_rows = read_csv(args.annotation_csv)
    rag_index_rows = read_csv(args.rag_index_csv)
    similar_rows = read_csv(args.similar_cases_csv)
    class_predictions = read_csv(args.classification_dir / "predictions.csv")
    class_coefficients = read_csv(args.classification_dir / "model_coefficients.csv")
    class_features = read_csv(args.classification_dir / "feature_matrix.csv")
    ratio_predictions = read_csv(args.ratio_dir / "predictions.csv")
    ratio_coefficients = read_csv(args.ratio_dir / "model_coefficients.csv")
    ratio_features = read_csv(args.ratio_dir / "ratio_model_frame.csv")

    annotation_by_jid = index_by(annotation_rows, "JID")
    rag_index_by_jid = index_by(rag_index_rows, "JID")
    similar_by_query = group_by(similar_rows, "query_JID")
    for grouped_rows in similar_by_query.values():
        grouped_rows.sort(key=lambda row: int(to_float(row.get("similar_rank")) or 9999))

    class_feature_by_jid = index_by(class_features, "JID")
    ratio_feature_by_jid = index_by(ratio_features, "JID")
    class_pred_by_jid = prediction_map(class_predictions, CLASSIFICATION_MODEL)
    ratio_mean_pred_by_jid = prediction_map(ratio_predictions, RATIO_BASELINE_MODEL)
    ratio_ridge_pred_by_jid = prediction_map(ratio_predictions, RATIO_MAIN_MODEL)
    ratio_lasso_pred_by_jid = prediction_map(ratio_predictions, RATIO_SECONDARY_MODEL)

    class_coeff_rows = coefficient_rows(class_coefficients)
    ridge_coeff_rows = coefficient_rows(ratio_coefficients, RATIO_MAIN_MODEL)
    lasso_coeff_rows = coefficient_rows(ratio_coefficients, RATIO_SECONDARY_MODEL)

    all_contributions: list[dict[str, Any]] = []
    contributions_by_case_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for annotation in annotation_rows:
        jid = annotation.get("JID", "")
        class_contribs = build_contributions(
            jid, "classification", CLASSIFICATION_MODEL, class_feature_by_jid.get(jid), class_coeff_rows
        )
        ridge_contribs = build_contributions(
            jid, "ratio", RATIO_MAIN_MODEL, ratio_feature_by_jid.get(jid), ridge_coeff_rows
        )
        lasso_contribs = build_contributions(
            jid, "ratio", RATIO_SECONDARY_MODEL, ratio_feature_by_jid.get(jid), lasso_coeff_rows
        )
        contributions_by_case_model[(jid, CLASSIFICATION_MODEL)] = class_contribs
        contributions_by_case_model[(jid, RATIO_MAIN_MODEL)] = ridge_contribs
        contributions_by_case_model[(jid, RATIO_SECONDARY_MODEL)] = lasso_contribs
        all_contributions.extend(class_contribs)
        all_contributions.extend(ridge_contribs)
        all_contributions.extend(lasso_contribs)

    evidence_rows: list[dict[str, Any]] = []
    for row in similar_rows:
        query_jid = row.get("query_JID", "")
        query_annotation = annotation_by_jid.get(query_jid, {})
        class_pred = class_pred_by_jid.get(query_jid, {})
        ratio_pred = ratio_ridge_pred_by_jid.get(query_jid, {})
        similar_annotation = annotation_by_jid.get(row.get("similar_JID", ""), {})
        evidence_rows.append(
            {
                "query_JID": query_jid,
                "query_split": class_pred.get("split") or ratio_pred.get("split", ""),
                "query_title": row.get("query_title", ""),
                "query_actual_is_reduced": query_annotation.get("is_reduced", ""),
                "query_reduction_probability": class_pred.get("probability", ""),
                "query_predicted_is_reduced": class_pred.get("predicted", ""),
                "query_actual_remaining_ratio": query_annotation.get("remaining_ratio", ""),
                "query_ridge_predicted_remaining_ratio": ratio_pred.get("predicted_remaining_ratio", ""),
                "similar_rank": row.get("similar_rank", ""),
                "similar_JID": row.get("similar_JID", ""),
                "similarity_score": row.get("similarity_score", ""),
                "similar_decision_year": row.get("similar_decision_year", ""),
                "similar_court": row.get("similar_court", ""),
                "similar_title": row.get("similar_title", ""),
                "shared_terms": row.get("shared_terms", ""),
                "similar_matched_strong_reduction_terms": row.get("similar_matched_strong_reduction_terms", ""),
                "similar_available_in_annotation": "1" if similar_annotation else "0",
                "similar_is_reduced": similar_annotation.get("is_reduced", ""),
                "similar_remaining_ratio": similar_annotation.get("remaining_ratio", ""),
                "similar_key_reason": similar_annotation.get("key_reason", ""),
                "similar_json_file": row.get("similar_json_file", ""),
                "similar_reduction_snippet": row.get("similar_reduction_snippet", ""),
                "similar_delay_snippet": row.get("similar_delay_snippet", ""),
                "human_review_focus": "比對共同詞與片段後，回到 query 與 similar 的原判決全文確認金額、結論與爭點；不得直接套用相似案例結論。",
            }
        )

    evidence_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_query[str(row.get("query_JID", ""))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for annotation in annotation_rows:
        jid = annotation.get("JID", "")
        class_pred = class_pred_by_jid.get(jid, {})
        mean_pred = ratio_mean_pred_by_jid.get(jid, {})
        ridge_pred = ratio_ridge_pred_by_jid.get(jid, {})
        lasso_pred = ratio_lasso_pred_by_jid.get(jid, {})
        sim_rows = similar_by_query.get(jid, [])
        similar_summary = similar_label_summary(sim_rows, annotation_by_jid)
        class_contribs = contributions_by_case_model.get((jid, CLASSIFICATION_MODEL), [])
        ridge_contribs = contributions_by_case_model.get((jid, RATIO_MAIN_MODEL), [])
        rag_row = rag_index_by_jid.get(jid, {})
        summary_rows.append(
            {
                "JID": jid,
                "decision_year": annotation.get("decision_year", ""),
                "split": class_pred.get("split") or ridge_pred.get("split", ""),
                "court": annotation.get("court", ""),
                "JTITLE": annotation.get("JTITLE", ""),
                "annotation_priority": annotation.get("annotation_priority", ""),
                "manual_checked": annotation.get("manual_checked", ""),
                "relevance_score": annotation.get("relevance_score") or rag_row.get("relevance_score", ""),
                "actual_is_reduced": annotation.get("is_reduced", ""),
                "predicted_is_reduced": class_pred.get("predicted", ""),
                "reduction_probability": class_pred.get("probability", ""),
                "classification_correct": is_correct(annotation.get("is_reduced"), class_pred.get("predicted")),
                "actual_remaining_ratio": annotation.get("remaining_ratio", ""),
                "actual_reduction_rate": annotation.get("reduction_rate", ""),
                "actual_bucket": ridge_pred.get("actual_bucket", ""),
                "mean_predicted_remaining_ratio": mean_pred.get("predicted_remaining_ratio", ""),
                "ridge_predicted_remaining_ratio": ridge_pred.get("predicted_remaining_ratio", ""),
                "ridge_predicted_reduction_rate": ridge_pred.get("predicted_reduction_rate", ""),
                "ridge_predicted_bucket": ridge_pred.get("predicted_bucket", ""),
                "ridge_abs_error": abs_error(ridge_pred.get("actual_remaining_ratio"), ridge_pred.get("predicted_remaining_ratio")),
                "lasso_predicted_remaining_ratio": lasso_pred.get("predicted_remaining_ratio", ""),
                "lasso_predicted_reduction_rate": lasso_pred.get("predicted_reduction_rate", ""),
                "lasso_predicted_bucket": lasso_pred.get("predicted_bucket", ""),
                "lasso_abs_error": abs_error(lasso_pred.get("actual_remaining_ratio"), lasso_pred.get("predicted_remaining_ratio")),
                "mean_abs_error": abs_error(mean_pred.get("actual_remaining_ratio"), mean_pred.get("predicted_remaining_ratio")),
                "top_classification_toward_reduction": summarize_contributions(class_contribs, "positive"),
                "top_classification_toward_no_reduction": summarize_contributions(class_contribs, "negative"),
                "top_ratio_toward_less_reduction_ridge": summarize_contributions(ridge_contribs, "positive"),
                "top_ratio_toward_more_reduction_ridge": summarize_contributions(ridge_contribs, "negative"),
                "similar_case_count": str(len(sim_rows)),
                **similar_summary,
                "top_similar_cases": top_similar_digest(sim_rows),
                "similar_shared_terms": shared_terms_digest(sim_rows),
                "similar_snippet_digest": snippet_digest(sim_rows),
                "source_json_file": annotation.get("json_file", ""),
                "explanation_note": "本列整合 AI 假設版標註、模型預測、模型特徵貢獻與 RAG 相似案例；正式報告前需回到原判決全文人工查核。",
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "case_explanation_summary.csv"
    evidence_path = output_dir / "similar_case_evidence.csv"
    contributions_path = output_dir / "model_feature_contributions.csv"
    cards_path = output_dir / "case_explanation_cards.md"
    workflow_path = output_dir / "rag_model_explanation_workflow.md"
    status_path = output_dir / "explanation_status.json"

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    evidence_fields = list(evidence_rows[0].keys()) if evidence_rows else []
    contribution_fields = [
        "JID",
        "model_family",
        "model",
        "feature",
        "feature_label",
        "feature_value",
        "mean",
        "std",
        "coefficient_scaled",
        "standardized_value",
        "contribution",
        "contribution_abs",
        "rank_by_abs",
        "interpretation",
    ]

    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(evidence_path, evidence_rows, evidence_fields)
    write_csv(contributions_path, all_contributions, contribution_fields)

    status = {
        "status": "ok",
        "case_count": len(summary_rows),
        "similar_case_evidence_count": len(evidence_rows),
        "feature_contribution_count": len(all_contributions),
        "classification_model": CLASSIFICATION_MODEL,
        "ratio_main_model": RATIO_MAIN_MODEL,
        "ratio_secondary_model": RATIO_SECONDARY_MODEL,
        "ratio_baseline_model": RATIO_BASELINE_MODEL,
        "inputs": {
            "annotation_csv": str(args.annotation_csv),
            "rag_index_csv": str(args.rag_index_csv),
            "similar_cases_csv": str(args.similar_cases_csv),
            "classification_dir": str(args.classification_dir),
            "ratio_dir": str(args.ratio_dir),
        },
        "outputs": {
            "case_explanation_summary": str(summary_path),
            "similar_case_evidence": str(evidence_path),
            "model_feature_contributions": str(contributions_path),
            "case_explanation_cards": str(cards_path),
            "rag_model_explanation_workflow": str(workflow_path),
            "explanation_status": str(status_path),
        },
        "notes": [
            "Current labels are AI-assumed and require manual verification before formal claims.",
            "RAG similarity is lexical/evidence-oriented and should not be treated as legal identity.",
            "Feature contributions explain model mechanics, not legal causation.",
        ],
    }

    write_text(cards_path, make_case_markdown(summary_rows, evidence_by_query))
    write_text(workflow_path, make_workflow_markdown(status))
    write_text(status_path, json.dumps(status, ensure_ascii=False, indent=2))

    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG + model explanation package.")
    parser.add_argument("--annotation-csv", type=Path, default=DEFAULT_ANNOTATION_CSV)
    parser.add_argument("--rag-index-csv", type=Path, default=DEFAULT_RAG_INDEX_CSV)
    parser.add_argument("--similar-cases-csv", type=Path, default=DEFAULT_SIMILAR_CASES_CSV)
    parser.add_argument("--classification-dir", type=Path, default=DEFAULT_CLASSIFICATION_DIR)
    parser.add_argument("--ratio-dir", type=Path, default=DEFAULT_RATIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    status = build_pack(parse_args())
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
