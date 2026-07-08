# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_PATH = PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "annotation_workbook.csv"
SIMILAR_CASES_PATH = PROJECT_ROOT / "06_交付物" / "rag_model_explanation" / "similar_case_evidence.csv"

SPLIT_ORDER = {
    "train_2021_2023": 1,
    "validation_2024": 2,
    "test_2025": 3,
    "latest_2026": 4,
}
SPLIT_LABELS = {
    "train_2021_2023": "訓練集 2021-2023",
    "validation_2024": "驗證集 2024",
    "test_2025": "測試集 2025",
    "latest_2026": "最新年度 2026",
}
CONTRIBUTION_MODELS = {
    "logistic_regression_l2": "分類 Logistic",
    "ridge_regression_l2": "比例 Ridge",
    "lasso_regression_l1": "比例 Lasso",
}
HIDDEN_FEATURE_LABELS = {"每日違約金"}
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
ISSUE_FIELDS = [
    "owner_fault",
    "contractor_fault",
    "extension_request",
    "actual_damage_unclear",
    "partial_completion",
    "used_by_owner",
]
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


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def pct(value: Any, digits: int = 1) -> str:
    if not is_number(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    if not is_number(value):
        return "—"
    return f"{float(value):.{digits}f}"


def yes_no(value: Any) -> str:
    if value == 1:
        return "是"
    if value == 0:
        return "否"
    return "—"


def safe(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def html_block(markup: str) -> str:
    return "\n".join(line.strip() for line in dedent(markup).strip().splitlines())


def render_html(markup: str) -> None:
    body = html_block(markup)
    if hasattr(st, "html"):
        st.html(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def clamp_ratio(value: Any) -> float:
    if not is_number(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def is_hidden_feature(row: dict[str, Any]) -> bool:
    return str(row.get("label") or "").strip() in HIDDEN_FEATURE_LABELS


def visible_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not is_hidden_feature(row)]


def remove_hidden_feature_terms(value: Any) -> str:
    parts = [part.strip() for part in str(value or "").split("；") if part.strip()]
    visible_parts = [
        part
        for part in parts
        if not any(part.startswith(hidden_label) for hidden_label in HIDDEN_FEATURE_LABELS)
    ]
    return "；".join(visible_parts) or "—"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float_or_none(value: Any) -> float | None:
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
    if not math.isfinite(number):
        return None
    return number


def to_int_or_none(value: Any) -> int | None:
    number = to_float_or_none(value)
    if number is None:
        return None
    return int(number)


def parse_label(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y", "是", "有", "酌減", "已酌減"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "否", "無", "未酌減", "不酌減"}:
        return 0
    number = to_float_or_none(text)
    if number is None:
        return None
    return int(number != 0)


def flag(value: Any) -> int:
    label = parse_label(value)
    return 0 if label is None else int(label)


def delimited_count(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    parts = [part for part in re.split(r"[;；]", text) if part.strip()]
    return len(parts) if parts else 1


def candidate_count(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    pipe_count = text.count("｜")
    if pipe_count > 0:
        return pipe_count
    return delimited_count(text)


def log_or_zero(value: float | None) -> float:
    if value is None or value <= 0:
        return 0.0
    return math.log(value)


def split_for_year(year: int | None) -> str:
    if year is None:
        return ""
    if 2021 <= year <= 2023:
        return "train_2021_2023"
    if year == 2024:
        return "validation_2024"
    if year == 2025:
        return "test_2025"
    if year == 2026:
        return "latest_2026"
    return f"year_{year}"


def target_quality(claimed_penalty: float | None, allowed_penalty: float | None, ratio: float | None) -> str:
    if claimed_penalty is None or allowed_penalty is None:
        return "missing_claimed_or_allowed_penalty"
    if claimed_penalty <= 0:
        return "claimed_penalty_nonpositive"
    if allowed_penalty < 0:
        return "allowed_penalty_negative"
    if ratio is None or ratio < 0 or ratio > 1:
        return "remaining_ratio_out_of_0_1_range"
    return "ok"


def compact_text(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def live_risk_level(probability: float | None, predicted_reduction_rate: float | None) -> str:
    prob = probability if probability is not None else -1.0
    rate = predicted_reduction_rate if predicted_reduction_rate is not None else -1.0
    if prob >= 0.70 or rate >= 0.50:
        return "高"
    if prob >= 0.45 or rate >= 0.25:
        return "中"
    if probability is None and predicted_reduction_rate is None:
        return "未訓練"
    return "低"


def live_risk_reason(probability: float | None, predicted_reduction_rate: float | None) -> str:
    if probability is None and predicted_reduction_rate is None:
        return "尚未現場訓練模型。請按左側「現場訓練模型」後查看模型結果。"
    parts: list[str] = []
    if probability is not None:
        parts.append(f"酌減機率 {probability:.1%}")
    if predicted_reduction_rate is not None:
        parts.append(f"預測酌減率 {predicted_reduction_rate:.1%}")
    return "，".join(parts)


def actual_ratios(row: dict[str, Any]) -> tuple[float | None, float | None]:
    claimed = to_float_or_none(row.get("claimed_penalty"))
    allowed = to_float_or_none(row.get("allowed_penalty"))
    if claimed is not None and allowed is not None and claimed > 0:
        remaining = allowed / claimed
        return remaining, 1.0 - remaining
    remaining = to_float_or_none(row.get("remaining_ratio"))
    reduction = to_float_or_none(row.get("reduction_rate"))
    if reduction is None and remaining is not None:
        reduction = 1.0 - remaining
    return remaining, reduction


def format_base_case(row: dict[str, str]) -> dict[str, Any]:
    year = to_int_or_none(row.get("decision_year"))
    split = split_for_year(year)
    remaining, reduction = actual_ratios(row)
    return {
        "jid": row.get("JID", ""),
        "year": year,
        "split": split,
        "splitLabel": split_label(split),
        "court": row.get("court", ""),
        "title": row.get("JTITLE", ""),
        "priority": to_int_or_none(row.get("annotation_priority")),
        "manualChecked": row.get("manual_checked", ""),
        "relevanceScore": to_float_or_none(row.get("relevance_score")),
        "actualIsReduced": parse_label(row.get("is_reduced")),
        "predictedIsReduced": None,
        "classificationCorrect": None,
        "reductionProbability": None,
        "actualRemainingRatio": remaining,
        "actualReductionRate": reduction,
        "meanPredictedRemainingRatio": None,
        "ridgePredictedRemainingRatio": None,
        "ridgePredictedReductionRate": None,
        "ridgeAbsError": None,
        "lassoPredictedRemainingRatio": None,
        "lassoPredictedReductionRate": None,
        "lassoAbsError": None,
        "meanAbsError": None,
        "topClassificationTowardReduction": "",
        "topClassificationTowardNoReduction": "",
        "topRatioTowardLessReductionRidge": "",
        "topRatioTowardMoreReductionRidge": "",
        "similarCaseCount": 0,
        "similarLabeledCount": 0,
        "similarLabeledReducedCount": 0,
        "similarAvgRemainingRatio": None,
        "similarSharedTerms": "",
        "sourceJsonFile": row.get("json_file", ""),
        "riskLevel": "未訓練",
        "riskReason": live_risk_reason(None, None),
    }


def format_similar(row: dict[str, str]) -> dict[str, Any]:
    return {
        "queryJid": row.get("query_JID", ""),
        "rank": to_int_or_none(row.get("similar_rank")),
        "jid": row.get("similar_JID", ""),
        "score": to_float_or_none(row.get("similarity_score")),
        "year": to_int_or_none(row.get("similar_decision_year")),
        "court": row.get("similar_court", ""),
        "title": row.get("similar_title", ""),
        "sharedTerms": row.get("shared_terms", ""),
        "strongTerms": row.get("similar_matched_strong_reduction_terms", ""),
        "availableInAnnotation": row.get("similar_available_in_annotation") == "1",
        "isReduced": parse_label(row.get("similar_is_reduced")),
        "remainingRatio": to_float_or_none(row.get("similar_remaining_ratio")),
        "keyReason": row.get("similar_key_reason", ""),
        "jsonFile": row.get("similar_json_file", ""),
        "reductionSnippet": compact_text(row.get("similar_reduction_snippet"), 260),
        "delaySnippet": compact_text(row.get("similar_delay_snippet"), 220),
    }


@st.cache_data(show_spinner=False)
def load_base_payload() -> dict[str, Any]:
    annotation_rows = read_csv_rows(ANNOTATION_PATH)
    cases = [format_base_case(row) for row in annotation_rows]
    similar_rows = read_csv_rows(SIMILAR_CASES_PATH) if SIMILAR_CASES_PATH.exists() else []
    similar_by_jid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in similar_rows:
        item = format_similar(row)
        if item["queryJid"]:
            similar_by_jid[item["queryJid"]].append(item)
    for items in similar_by_jid.values():
        items.sort(key=lambda item: item.get("rank") or 999)

    for case in cases:
        case["similarCaseCount"] = len(similar_by_jid.get(case["jid"], []))

    risk_counts = Counter(case["riskLevel"] for case in cases)
    split_counts = Counter(case["split"] for case in cases)
    year_counts = Counter(str(case["year"]) for case in cases if case.get("year"))
    return {
        "metadata": {
            "generatedFrom": str(ANNOTATION_PATH.relative_to(PROJECT_ROOT)),
            "caseCount": len(cases),
            "similarCaseCount": len(similar_rows),
            "featureContributionCount": 0,
            "missingRidgePrediction": len(cases),
            "missingReductionProbability": len(cases),
            "trainedInApp": False,
            "labeledRows": sum(1 for case in cases if case["actualIsReduced"] is not None),
            "usableRatioRows": sum(1 for case in cases if case["actualRemainingRatio"] is not None),
            "riskCounts": dict(sorted(risk_counts.items())),
            "splitCounts": dict(sorted(split_counts.items())),
            "yearCounts": dict(sorted(year_counts.items())),
            "modelLabels": {
                "classification": "Logistic Regression",
                "ratioMain": "Ridge Regression",
                "ratioSecondary": "Lasso Regression",
                "baseline": "Mean baseline",
            },
            "riskRule": {
                "高": "酌減機率 >= 70% 或預測酌減率 >= 50%",
                "中": "酌減機率 >= 45% 或預測酌減率 >= 25%",
                "低": "未達高/中門檻",
            },
            "notice": "資料來自 annotation_workbook.csv；模型數字需按「現場訓練模型」後才會由 Streamlit 重新訓練產生。",
        },
        "cases": cases,
        "similarCasesByJid": dict(similar_by_jid),
        "contributionsByJid": {},
    }


def build_feature_frame(annotation_rows: list[dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in annotation_rows:
        year = to_int_or_none(row.get("decision_year"))
        contract_price = to_float_or_none(row.get("contract_price"))
        delay_days = to_float_or_none(row.get("delay_days"))
        claimed_penalty = to_float_or_none(row.get("claimed_penalty"))
        allowed_penalty = to_float_or_none(row.get("allowed_penalty"))
        remaining, reduction = actual_ratios(row)
        claim_to_contract = (
            claimed_penalty / contract_price
            if claimed_penalty is not None and contract_price is not None and contract_price > 0
            else 0.0
        )
        penalty_per_delay_day = (
            claimed_penalty / delay_days
            if claimed_penalty is not None and delay_days is not None and delay_days > 0
            else 0.0
        )
        feature_row: dict[str, Any] = {
            "JID": row.get("JID", ""),
            "decision_year": year,
            "split": split_for_year(year),
            "is_reduced_label": parse_label(row.get("is_reduced")),
            "remaining_ratio": remaining,
            "reduction_rate": reduction,
            "target_quality": target_quality(claimed_penalty, allowed_penalty, remaining),
            "x_log_contract_price": log_or_zero(contract_price),
            "x_log_claimed_penalty": log_or_zero(claimed_penalty),
            "x_claim_to_contract_ratio": claim_to_contract,
            "x_delay_days": delay_days or 0.0,
            "x_penalty_per_delay_day": penalty_per_delay_day,
            "x_money_candidate_count": candidate_count(row.get("ai_contract_price_candidates"))
            + candidate_count(row.get("ai_claimed_penalty_candidates"))
            + candidate_count(row.get("ai_allowed_penalty_candidates")),
            "x_delay_candidate_count": candidate_count(row.get("ai_delay_days_candidates")),
        }
        for issue in ISSUE_FIELDS:
            feature_row[f"x_issue_{issue}"] = flag(row.get(f"issue_{issue}"))
            feature_row[f"x_ai_issue_{issue}"] = flag(row.get(f"ai_suggest_issue_{issue}"))
        rows.append(feature_row)

    frame = pd.DataFrame(rows)
    for feature in FEATURE_NAMES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
    return frame


def case_splits(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("train_2021_2023", frame[(frame["decision_year"] >= 2021) & (frame["decision_year"] <= 2023)]),
        ("validation_2024", frame[frame["decision_year"] == 2024]),
        ("test_2025", frame[frame["decision_year"] == 2025]),
        ("latest_2026", frame[frame["decision_year"] == 2026]),
    ]


def classify_predictions(frame: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], Pipeline]:
    labeled = frame[frame["is_reduced_label"].notna()].copy()
    train = labeled[(labeled["decision_year"] >= 2021) & (labeled["decision_year"] <= 2023)]
    if len(labeled) < 30:
        raise ValueError(f"可用分類標籤不足：{len(labeled)}")
    if train["is_reduced_label"].nunique() < 2:
        raise ValueError("訓練切分缺少二元分類標籤")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs", C=100.0)),
        ]
    )
    model.fit(train[FEATURE_NAMES], train["is_reduced_label"].astype(int))

    predictions: dict[str, dict[str, Any]] = {}
    for split_name, split_frame in case_splits(labeled):
        if split_frame.empty:
            continue
        probabilities = model.predict_proba(split_frame[FEATURE_NAMES])[:, 1]
        for (_, row), probability in zip(split_frame.iterrows(), probabilities):
            predicted = int(probability >= 0.5)
            actual = int(row["is_reduced_label"])
            predictions[str(row["JID"])] = {
                "split": split_name,
                "actual": actual,
                "predicted": predicted,
                "probability": float(probability),
                "correct": int(predicted == actual),
            }
    return predictions, model


def ratio_predictions(frame: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], Pipeline, Pipeline, float, int]:
    usable = frame[frame["target_quality"] == "ok"].copy()
    train = usable[(usable["decision_year"] >= 2021) & (usable["decision_year"] <= 2023)]
    if len(usable) < 30:
        raise ValueError(f"可用比例目標不足：{len(usable)}")
    if len(train) < 10:
        raise ValueError(f"訓練切分比例目標不足：{len(train)}")

    ridge_model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=0.05))])
    lasso_model = Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=10000))])
    ridge_model.fit(train[FEATURE_NAMES], train["remaining_ratio"])
    lasso_model.fit(train[FEATURE_NAMES], train["remaining_ratio"])
    mean_ratio = float(train["remaining_ratio"].mean())

    predictions: dict[str, dict[str, Any]] = {}
    for split_name, split_frame in case_splits(usable):
        if split_frame.empty:
            continue
        ridge_raw = ridge_model.predict(split_frame[FEATURE_NAMES])
        lasso_raw = lasso_model.predict(split_frame[FEATURE_NAMES])
        for row_index, (_, row) in enumerate(split_frame.iterrows()):
            actual = float(row["remaining_ratio"])
            ridge = float(max(0.0, min(1.0, ridge_raw[row_index])))
            lasso = float(max(0.0, min(1.0, lasso_raw[row_index])))
            mean = float(max(0.0, min(1.0, mean_ratio)))
            predictions[str(row["JID"])] = {
                "split": split_name,
                "actualRemainingRatio": actual,
                "actualReductionRate": 1.0 - actual,
                "meanPredictedRemainingRatio": mean,
                "ridgePredictedRemainingRatio": ridge,
                "ridgePredictedReductionRate": 1.0 - ridge,
                "ridgeAbsError": abs(actual - ridge),
                "lassoPredictedRemainingRatio": lasso,
                "lassoPredictedReductionRate": 1.0 - lasso,
                "lassoAbsError": abs(actual - lasso),
                "meanAbsError": abs(actual - mean),
            }
    return predictions, ridge_model, lasso_model, mean_ratio, len(usable)


def contribution_rows(
    frame: pd.DataFrame,
    pipeline: Pipeline,
    model_family: str,
    model_name: str,
) -> dict[str, list[dict[str, Any]]]:
    scaler: StandardScaler = pipeline.named_steps["scale"]
    estimator = pipeline.named_steps["model"]
    coef_values = estimator.coef_[0] if getattr(estimator, "coef_", []).ndim == 2 else estimator.coef_
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, row in frame.iterrows():
        values = row[FEATURE_NAMES].astype(float).to_numpy()
        standardized = (values - scaler.mean_) / scaler.scale_
        rows: list[dict[str, Any]] = []
        for index, feature in enumerate(FEATURE_NAMES):
            contribution = float(standardized[index] * coef_values[index])
            if model_family == "classification":
                interpretation = "提高酌減機率" if contribution > 0 else "降低酌減機率" if contribution < 0 else "影響接近零"
            else:
                interpretation = (
                    "提高准許比例/降低酌減幅度"
                    if contribution > 0
                    else "降低准許比例/提高酌減幅度"
                    if contribution < 0
                    else "影響接近零"
                )
            rows.append(
                {
                    "jid": row["JID"],
                    "modelFamily": model_family,
                    "model": model_name,
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature),
                    "value": round(float(values[index]), 6),
                    "coefficient": round(float(coef_values[index]), 6),
                    "standardizedValue": round(float(standardized[index]), 6),
                    "contribution": round(contribution, 6),
                    "absContribution": round(abs(contribution), 6),
                    "interpretation": interpretation,
                }
            )
        rows.sort(key=lambda item: item["absContribution"], reverse=True)
        for rank, item in enumerate(rows, start=1):
            item["rank"] = rank
        grouped[str(row["JID"])] = rows[:8]
    return grouped


def summarize_contributions(rows: list[dict[str, Any]], sign: str, limit: int = 3) -> str:
    if sign == "positive":
        selected = [row for row in rows if float(row.get("contribution") or 0.0) > 0]
        selected.sort(key=lambda row: float(row.get("contribution") or 0.0), reverse=True)
    else:
        selected = [row for row in rows if float(row.get("contribution") or 0.0) < 0]
        selected.sort(key=lambda row: abs(float(row.get("contribution") or 0.0)), reverse=True)
    parts = []
    for row in selected[:limit]:
        parts.append(f"{row['label']}({num(row.get('contribution'))}; 值={num(row.get('value'))})")
    return "；".join(parts)


def train_models(annotation_rows: list[dict[str, str]]) -> dict[str, Any]:
    frame = build_feature_frame(annotation_rows)
    class_predictions, class_model = classify_predictions(frame)
    ratio_pred, ridge_model, lasso_model, mean_ratio, usable_ratio_rows = ratio_predictions(frame)

    class_contribs = contribution_rows(frame, class_model, "classification", "logistic_regression_l2")
    ridge_contribs = contribution_rows(frame, ridge_model, "ratio", "ridge_regression_l2")
    lasso_contribs = contribution_rows(frame, lasso_model, "ratio", "lasso_regression_l1")
    contributions_by_jid: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for jid, rows in class_contribs.items():
        contributions_by_jid[jid]["logistic_regression_l2"] = rows
    for jid, rows in ridge_contribs.items():
        contributions_by_jid[jid]["ridge_regression_l2"] = rows
    for jid, rows in lasso_contribs.items():
        contributions_by_jid[jid]["lasso_regression_l1"] = rows

    labeled_rows = int(frame["is_reduced_label"].notna().sum())
    return {
        "classificationPredictions": class_predictions,
        "ratioPredictions": ratio_pred,
        "contributionsByJid": dict(contributions_by_jid),
        "metadata": {
            "trainedInApp": True,
            "labeledRows": labeled_rows,
            "usableRatioRows": usable_ratio_rows,
            "meanTrainingRemainingRatio": mean_ratio,
            "featureContributionCount": sum(
                len(rows) for by_model in contributions_by_jid.values() for rows in by_model.values()
            ),
        },
    }


def merge_training_results(base_payload: dict[str, Any], training_result: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    class_predictions = training_result["classificationPredictions"]
    ratio_pred = training_result["ratioPredictions"]
    contributions_by_jid = training_result["contributionsByJid"]

    for item in payload["cases"]:
        jid = str(item.get("jid"))
        class_pred = class_predictions.get(jid, {})
        ratio = ratio_pred.get(jid, {})
        if class_pred:
            item["split"] = class_pred.get("split", item.get("split"))
            item["splitLabel"] = split_label(item["split"])
            item["predictedIsReduced"] = class_pred["predicted"]
            item["reductionProbability"] = class_pred["probability"]
            item["classificationCorrect"] = class_pred["correct"]
        if ratio:
            item.update(ratio)

        model_rows = contributions_by_jid.get(jid, {})
        class_rows = model_rows.get("logistic_regression_l2", [])
        ridge_rows = model_rows.get("ridge_regression_l2", [])
        item["topClassificationTowardReduction"] = summarize_contributions(class_rows, "positive")
        item["topClassificationTowardNoReduction"] = summarize_contributions(class_rows, "negative")
        item["topRatioTowardLessReductionRidge"] = summarize_contributions(ridge_rows, "positive")
        item["topRatioTowardMoreReductionRidge"] = summarize_contributions(ridge_rows, "negative")

        probability = item.get("reductionProbability")
        ridge_rate = item.get("ridgePredictedReductionRate")
        item["riskLevel"] = live_risk_level(probability, ridge_rate)
        item["riskReason"] = live_risk_reason(probability, ridge_rate)

    metadata = payload["metadata"]
    metadata.update(training_result["metadata"])
    metadata["generatedFrom"] = f"{ANNOTATION_PATH.relative_to(PROJECT_ROOT)} + Streamlit 現場訓練"
    metadata["riskCounts"] = dict(sorted(Counter(case["riskLevel"] for case in payload["cases"]).items()))
    metadata["missingRidgePrediction"] = sum(1 for case in payload["cases"] if case["ridgePredictedRemainingRatio"] is None)
    metadata["missingReductionProbability"] = sum(1 for case in payload["cases"] if case["reductionProbability"] is None)
    metadata["notice"] = "模型數字由本次 Streamlit 按鈕現場訓練產生；資料仍為 AI 假設版標註，不是法律意見。"
    payload["contributionsByJid"] = contributions_by_jid
    return payload


def run_live_training() -> dict[str, Any]:
    base_payload = load_base_payload()
    annotation_rows = read_csv_rows(ANNOTATION_PATH)
    training_result = train_models(annotation_rows)
    return merge_training_results(base_payload, training_result)


def install_style() -> None:
    render_html(
        """
        <style>
          :root {
            --tool-bg: #eef0ef;
            --tool-surface: #ffffff;
            --tool-soft: #f7f8f7;
            --tool-ink: #202124;
            --tool-muted: #646760;
            --tool-line: #cfd5d1;
            --tool-accent: #176b5d;
            --tool-accent-soft: #d9eee8;
          }
          [data-testid="stToolbar"],
          [data-testid="stDecoration"],
          [data-testid="stStatusWidget"],
          #MainMenu {
            display: none !important;
            visibility: hidden !important;
          }
          [data-testid="stHeader"] {
            height: 0 !important;
            min-height: 0 !important;
            background: transparent !important;
          }
          .block-container {
            padding-top: 0.75rem;
            padding-bottom: 2rem;
          }
          .tool-title {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            padding: 1rem 1.1rem;
            border: 1px solid var(--tool-line);
            border-radius: 8px;
            background: var(--tool-surface);
            color: var(--tool-ink);
            min-width: 0;
            overflow-wrap: anywhere;
          }
          .tool-title h1 {
            margin: 0;
            color: var(--tool-ink);
            font-size: 1.65rem;
            line-height: 1.2;
            letter-spacing: 0;
          }
          .eyebrow {
            margin: 0 0 0.25rem 0;
            color: var(--tool-accent);
            font-size: 0.82rem;
            font-weight: 800;
          }
          .muted {
            color: var(--tool-muted);
            font-size: 0.9rem;
            line-height: 1.6;
            overflow-wrap: anywhere;
          }
          .panel {
            border: 1px solid var(--tool-line);
            border-radius: 8px;
            padding: 1rem;
            background: var(--tool-surface);
            color: var(--tool-ink);
            min-width: 0;
            overflow-wrap: anywhere;
          }
          .panel h3 {
            margin: 0 0 0.75rem 0;
            color: var(--tool-ink);
            font-size: 1.05rem;
            letter-spacing: 0;
          }
          .bar-row {
            display: grid;
            grid-template-columns: minmax(6rem, 8rem) minmax(0, 1fr) minmax(3.8rem, 4.5rem);
            gap: 0.65rem;
            align-items: center;
            margin: 0.55rem 0;
            color: var(--tool-ink);
            font-size: 0.93rem;
          }
          .bar-row > span {
            min-width: 0;
            overflow-wrap: anywhere;
          }
          .bar-label {
            font-weight: 800;
          }
          .bar-track {
            height: 0.8rem;
            overflow: hidden;
            border-radius: 999px;
            background: #e8e3d7;
          }
          .bar-fill {
            height: 100%;
            border-radius: inherit;
            background: var(--tool-accent);
          }
          .detail-box {
            margin-top: 0.9rem;
            padding-top: 0.8rem;
            border-top: 1px solid var(--tool-line);
            color: var(--tool-muted);
            font-size: 0.9rem;
            line-height: 1.7;
          }
          .detail-box strong {
            color: var(--tool-ink);
          }
          .similar-card {
            display: grid;
            grid-template-columns: 4rem minmax(0, 1fr);
            gap: 0.8rem;
            border-top: 1px solid var(--tool-line);
            padding: 0.9rem 0 0.2rem;
            color: var(--tool-ink);
            min-width: 0;
          }
          .similar-score {
            display: grid;
            place-items: center;
            width: 3.3rem;
            height: 3.3rem;
            border-radius: 8px;
            background: var(--tool-accent-soft);
            color: var(--tool-accent);
            font-weight: 900;
          }
          .similar-title {
            margin: 0;
            color: var(--tool-ink);
            font-weight: 900;
            line-height: 1.35;
          }
          .similar-meta,
          .similar-terms,
          .similar-snippet {
            margin: 0.22rem 0 0;
            font-size: 0.9rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
          }
          .similar-meta,
          .similar-terms {
            color: var(--tool-muted);
          }
          .feature-positive {
            color: var(--risk-high);
            font-weight: 800;
          }
          .feature-negative {
            color: var(--risk-low);
            font-weight: 800;
          }
          .feature-table {
            display: grid;
            gap: 0.35rem;
            margin-top: 0.6rem;
          }
          .feature-grid {
            display: grid;
            grid-template-columns: minmax(9rem, 1.15fr) minmax(4.5rem, 0.45fr) minmax(4.5rem, 0.45fr) minmax(10rem, 1fr);
            gap: 0.65rem;
            align-items: center;
            padding: 0.62rem 0;
            border-top: 1px solid var(--tool-line);
            color: var(--tool-ink);
            font-size: 0.9rem;
          }
          .feature-grid > span {
            min-width: 0;
            overflow-wrap: anywhere;
          }
          .feature-header {
            border-top: 0;
            padding-top: 0;
            color: var(--tool-muted);
            font-size: 0.78rem;
            font-weight: 900;
          }
          .feature-label {
            font-weight: 850;
          }
          .feature-number {
            text-align: right;
            font-variant-numeric: tabular-nums;
          }
          .notice {
            border: 1px solid #eadba9;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #fff8e5;
            color: #5c4a20;
            line-height: 1.7;
            overflow-wrap: anywhere;
          }
          @media (max-width: 900px) {
            .feature-grid {
              grid-template-columns: minmax(0, 1fr) minmax(4.5rem, 0.45fr) minmax(4.5rem, 0.45fr);
            }
            .feature-grid .feature-interpretation {
              grid-column: 1 / -1;
            }
          }
          @media (max-width: 700px) {
            .tool-title,
            .bar-row,
            .similar-card {
              display: block;
            }
            .feature-header {
              display: none;
            }
            .feature-grid {
              display: block;
            }
            .feature-grid > span {
              display: block;
              margin: 0.2rem 0;
            }
            .feature-number {
              text-align: left;
            }
            .bar-track,
            .similar-score {
              margin-top: 0.5rem;
            }
          }
        </style>
        """
    )


def ensure_session_defaults() -> None:
    defaults = {
        "search_query": "",
        "year_filter": "全部年度",
        "split_filter": "全部切分",
        "contribution_model": "logistic_regression_l2",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset(name: str) -> None:
    st.session_state.search_query = ""
    st.session_state.year_filter = "全部年度"
    st.session_state.split_filter = "全部切分"
    if name == "test2025":
        st.session_state.year_filter = "2025"
        st.session_state.split_filter = "test_2025"
    elif name == "latest2026":
        st.session_state.year_filter = "2026"
        st.session_state.split_filter = "latest_2026"


def split_label(value: str) -> str:
    if value == "全部切分":
        return value
    return SPLIT_LABELS.get(value, value)


def filter_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyword = str(st.session_state.search_query).strip().lower()
    year = "" if st.session_state.year_filter == "全部年度" else st.session_state.year_filter
    split = "" if st.session_state.split_filter == "全部切分" else st.session_state.split_filter

    rows: list[dict[str, Any]] = []
    for item in cases:
        haystack = f"{item.get('jid', '')} {item.get('title', '')} {item.get('court', '')}".lower()
        if keyword and keyword not in haystack:
            continue
        if year and str(item.get("year")) != year:
            continue
        if split and item.get("split") != split:
            continue
        rows.append(item)

    return sorted(
        rows,
        key=lambda item: (
            -int(item.get("year") or 0),
            int(item.get("priority") or 999999),
        ),
    )


def preferred_jid(cases: list[dict[str, Any]]) -> str | None:
    for predicate in [
        lambda item: item.get("split") == "test_2025",
        lambda item: True,
    ]:
        match = next((item for item in cases if predicate(item)), None)
        if match:
            return str(match.get("jid"))
    return None


def case_label_map(cases: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in cases:
        jid = str(item.get("jid"))
        labels[jid] = (
            f"{item.get('title') or '未命名案件'}"
            f"｜{item.get('year') or '—'}"
        )
    return labels


def render_training_controls() -> None:
    with st.sidebar:
        st.markdown("**現場訓練**")
        if st.button("現場訓練模型", type="primary", use_container_width=True):
            with st.spinner("正在從 annotation_workbook.csv 產生特徵並訓練模型..."):
                try:
                    st.session_state.live_training_payload = run_live_training()
                except Exception as exc:  # noqa: BLE001
                    st.session_state.live_training_payload = None
                    st.error(f"訓練失敗：{exc}")
        trained_payload = st.session_state.get("live_training_payload")
        if trained_payload:
            metadata = trained_payload["metadata"]
            st.success(
                f"已完成：{metadata.get('labeledRows', '—')} 筆分類標籤，"
                f"{metadata.get('usableRatioRows', '—')} 筆比例目標。"
            )
        else:
            st.info("尚未訓練；模型數字會維持空白。")


def render_sidebar(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    cases: list[dict[str, Any]] = payload["cases"]
    metadata = payload["metadata"]
    years = sorted({str(item.get("year")) for item in cases if item.get("year")})
    splits = sorted(
        {str(item.get("split")) for item in cases if item.get("split")},
        key=lambda value: SPLIT_ORDER.get(value, 99),
    )

    with st.sidebar:
        st.caption("工程違約金")
        st.title("評估工具")
        st.caption(f"{metadata.get('caseCount', len(cases))} 件案件")

        st.markdown("**展示快捷篩選**")
        col1, col2, col3 = st.columns(3)
        col1.button("2025 測試", use_container_width=True, on_click=apply_preset, args=("test2025",))
        col2.button("2026 最新", use_container_width=True, on_click=apply_preset, args=("latest2026",))
        col3.button("清除", use_container_width=True, on_click=apply_preset, args=("clear",))

    filtered = filter_cases(cases)
    labels = case_label_map(filtered)
    selected_jid = None

    with st.sidebar:
        if filtered:
            jids = [str(item.get("jid")) for item in filtered]
            if st.session_state.get("selected_jid") not in jids:
                st.session_state.selected_jid = preferred_jid(filtered)
            selected_jid = st.selectbox(
                "案件",
                jids,
                key="selected_jid",
                format_func=lambda jid: labels.get(jid, jid),
            )
        else:
            st.warning("沒有符合條件的案件")

        st.text_input("搜尋", placeholder="案號、案名、法院", key="search_query")
        st.selectbox("年度", ["全部年度", *years], key="year_filter")
        st.selectbox("切分", ["全部切分", *splits], key="split_filter", format_func=split_label)

        st.markdown(f"**符合條件：{len(filtered)} 件**")
        if not filtered:
            return filtered, None

        download_data = json.dumps(filtered, ensure_ascii=False, indent=2)
        st.download_button(
            "下載目前篩選 JSON",
            data=download_data,
            file_name="risk_tool_filtered_cases.json",
            mime="application/json",
            use_container_width=True,
        )

    selected = next((item for item in filtered if str(item.get("jid")) == selected_jid), None)
    return filtered, selected


def render_header(item: dict[str, Any]) -> None:
    render_html(
        f"""
        <div class="tool-title">
          <div>
            <p class="eyebrow">{safe(item.get("year"))}｜{safe(item.get("splitLabel"))}｜{safe(item.get("court"))}</p>
            <h1>{safe(item.get("title") or "未命名案件")}</h1>
            <p class="muted">{safe(item.get("jid"))}</p>
          </div>
        </div>
        """
    )


def render_metrics(item: dict[str, Any]) -> None:
    cols = st.columns(3)
    cols[0].metric("酌減機率", pct(item.get("reductionProbability")), help="Logistic Regression")
    cols[1].metric("預測准許比例", pct(item.get("ridgePredictedRemainingRatio")), help="Ridge Regression")
    cols[2].metric("預測酌減率", pct(item.get("ridgePredictedReductionRate")), help="1 - remaining_ratio")
    st.caption(item.get("riskReason") or "—")


def is_case_trained(item: dict[str, Any]) -> bool:
    return is_number(item.get("reductionProbability")) or is_number(item.get("ridgePredictedRemainingRatio"))


def render_untrained_model_notice() -> None:
    render_html(
        """
        <div class="panel">
          <h3>尚未現場訓練</h3>
          <p class="muted">
            尚未現場訓練，模型比較會在訓練後顯示。請先按左側「現場訓練模型」，
            app 會從 annotation_workbook.csv 產生特徵並重新訓練 Logistic、Ridge 與 Lasso。
          </p>
        </div>
        """
    )


def render_model_compare(item: dict[str, Any]) -> None:
    rows = [
        ("Mean baseline", item.get("meanPredictedRemainingRatio")),
        ("Ridge", item.get("ridgePredictedRemainingRatio")),
        ("Lasso", item.get("lassoPredictedRemainingRatio")),
    ]
    parts = ['<div class="panel"><h3>模型比較</h3>']
    for label, ratio in rows:
        width = max(2.0, clamp_ratio(ratio) * 100) if is_number(ratio) else 0.0
        parts.append(
            f"""
            <div class="bar-row">
              <span class="bar-label">{safe(label)}</span>
              <span class="bar-track"><span class="bar-fill" style="width:{width:.1f}%"></span></span>
              <span>{pct(ratio)}</span>
            </div>
            """
        )
    detail = f"""
      <div class="detail-box">
        <strong>AI 假設版回測對照</strong><br>
        是否酌減：{yes_no(item.get("actualIsReduced"))}；
        分類預測：{yes_no(item.get("predictedIsReduced"))}；
        命中：{yes_no(item.get("classificationCorrect"))}<br>
        實際准許比例：{pct(item.get("actualRemainingRatio"))}；
        實際酌減率：{pct(item.get("actualReductionRate"))}<br>
        Ridge 誤差：{num(item.get("ridgeAbsError"))}；
        Lasso 誤差：{num(item.get("lassoAbsError"))}；
        Mean baseline 誤差：{num(item.get("meanAbsError"))}
      </div>
    """
    parts.append(detail + "</div>")
    render_html("\n".join(parts))


def render_feature_summary(item: dict[str, Any]) -> None:
    render_html(
        f"""
        <div class="panel">
          <h3>模型方向摘要</h3>
          <p class="muted"><strong>提高酌減機率：</strong>{safe(remove_hidden_feature_terms(item.get("topClassificationTowardReduction")))}</p>
          <p class="muted"><strong>降低酌減機率：</strong>{safe(remove_hidden_feature_terms(item.get("topClassificationTowardNoReduction")))}</p>
          <p class="muted"><strong>Ridge 較少酌減：</strong>{safe(remove_hidden_feature_terms(item.get("topRatioTowardLessReductionRidge")))}</p>
          <p class="muted"><strong>Ridge 較多酌減：</strong>{safe(remove_hidden_feature_terms(item.get("topRatioTowardMoreReductionRidge")))}</p>
        </div>
        """
    )


def render_features(item: dict[str, Any], contributions: dict[str, Any]) -> None:
    st.subheader("重要特徵")
    model = st.radio(
        "模型切換",
        list(CONTRIBUTION_MODELS.keys()),
        key="contribution_model",
        format_func=lambda value: CONTRIBUTION_MODELS[value],
        horizontal=True,
    )
    rows = visible_feature_rows(contributions.get(str(item.get("jid")), {}).get(model, []))
    if not rows:
        st.info("這個模型沒有可顯示的特徵貢獻。")
        return

    visible_rows = [
        """
        <div class="feature-grid feature-header">
          <span>特徵</span>
          <span class="feature-number">數值</span>
          <span class="feature-number">貢獻</span>
          <span class="feature-interpretation">解讀</span>
        </div>
        """
    ]

    for row in rows[:12]:
        contribution = row.get("contribution")
        contribution_text = num(contribution)
        if is_number(contribution):
            cls = "feature-positive" if float(contribution) >= 0 else "feature-negative"
            contribution_text = f'<span class="{cls}">{contribution_text}</span>'
        visible_rows.append(
            f"""
            <div class="feature-grid">
              <span class="feature-label">{safe(row.get("label"))}</span>
              <span class="feature-number">{num(row.get("value"))}</span>
              <span class="feature-number">{contribution_text}</span>
              <span class="feature-interpretation">{safe(row.get("interpretation"))}</span>
            </div>
            """
        )

    st.markdown("貢獻值來自標準化特徵 × 模型係數，僅解釋目前回測模型方向。")
    st.caption("正負方向依不同模型目標解讀；請搭配人工查核。")
    render_html(f'<div class="feature-table">{"".join(visible_rows)}</div>')


def render_similar_cases(item: dict[str, Any], similar_by_jid: dict[str, Any]) -> None:
    rows = similar_by_jid.get(str(item.get("jid")), [])
    st.subheader(f"RAG 相似案例（{len(rows)} 件）")
    if not rows:
        st.info("沒有相似案例。")
        return

    for row in rows:
        ratio_text = "無比例標註" if not is_number(row.get("remainingRatio")) else f"准許比例 {pct(row.get('remainingRatio'))}"
        snippet = row.get("reductionSnippet") or row.get("delaySnippet") or "無片段"
        render_html(
            f"""
            <div class="similar-card">
              <div class="similar-score">{num(row.get("score"), 2)}</div>
              <div>
                <p class="similar-title">#{safe(row.get("rank"))} {safe(row.get("title") or "未命名案件")}</p>
                <p class="similar-meta">{safe(row.get("jid"))}｜{safe(row.get("year"))}｜{safe(row.get("court"))}｜{safe(ratio_text)}</p>
                <p class="similar-terms">{safe(row.get("sharedTerms") or "無共同詞")}</p>
                <p class="similar-snippet">{safe(snippet)}</p>
              </div>
            </div>
            """
        )


def render_overview(payload: dict[str, Any], filtered: list[dict[str, Any]]) -> None:
    metadata = payload["metadata"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("全部案件", metadata.get("caseCount", len(payload["cases"])))
    col2.metric("目前篩選", len(filtered))
    col3.metric("相似案例", metadata.get("similarCaseCount", "—"))
    col4.metric("特徵貢獻", metadata.get("featureContributionCount", "—"))

    split_counts = metadata.get("splitCounts", {})
    year_counts = metadata.get("yearCounts", {})
    left, right = st.columns(2)
    with left:
        st.markdown("**年度分布**")
        st.json(year_counts, expanded=False)
    with right:
        st.markdown("**時間切分**")
        st.json({SPLIT_LABELS.get(k, k): v for k, v in split_counts.items()}, expanded=False)


def render_notice(payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    render_html(
        f"""
        <div class="notice">
          <strong>限制說明</strong><br>
          {safe(metadata.get("notice") or "本工具僅供展示與回測解釋。")}<br>
          模型成果應定位為爭點分類與酌減可能性回測，不能宣稱可取代法院判斷或直接預測個案結果。
        </div>
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="工程違約金風險評估工具",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    install_style()
    ensure_session_defaults()

    try:
        base_payload = load_base_payload()
    except Exception as exc:  # noqa: BLE001
        st.error(f"資料載入失敗：{exc}")
        st.stop()

    render_training_controls()
    payload = st.session_state.get("live_training_payload") or base_payload

    filtered, selected = render_sidebar(payload)
    if not selected:
        st.title("工程違約金風險評估工具")
        st.info("請調整左側篩選條件。")
        render_notice(payload)
        return

    render_header(selected)
    render_metrics(selected)

    tab_case, tab_data, tab_limits = st.tabs(["案件儀表板", "資料總覽", "限制說明"])
    with tab_case:
        if is_case_trained(selected):
            left, right = st.columns([0.95, 1.05])
            with left:
                render_model_compare(selected)
            with right:
                render_feature_summary(selected)
            render_features(selected, payload["contributionsByJid"])
        else:
            render_untrained_model_notice()
        render_similar_cases(selected, payload["similarCasesByJid"])

    with tab_data:
        render_overview(payload, filtered)
        rows = [
            {
                "案號": item.get("jid"),
                "案名": item.get("title"),
                "年度": item.get("year"),
                "切分": item.get("splitLabel"),
                "法院": item.get("court"),
                "酌減機率": pct(item.get("reductionProbability")),
                "預測准許比例": pct(item.get("ridgePredictedRemainingRatio")),
            }
            for item in filtered
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab_limits:
        render_notice(payload)
        st.markdown("**人工查核重點**")
        st.write("- 確認金額是否為契約總價、主張違約金、法院准許違約金或其他款項。")
        st.write("- 確認法院最終結論、展延、歸責、損害與使用收益等爭點。")
        st.write("- RAG 相似案例只作閱讀優先順序，不直接套用法律結論。")


if __name__ == "__main__":
    main()
