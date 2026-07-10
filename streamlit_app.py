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
from typing import Any, TypeGuard

import pandas as pd
import streamlit as st
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_PATH = (
    PROJECT_ROOT
    / "06_交付物"
    / "ai_rag_annotation_expanded_824"
    / "annotation_workbook_ai_assumed.csv"
)
SIMILAR_CASES_PATH = (
    PROJECT_ROOT
    / "06_交付物"
    / "ai_rag_annotation_expanded_824"
    / "rag_similar_cases.csv"
)

FORMAL_FEATURE_COLUMNS = {
    "正式特徵_業主可歸責": "業主可歸責",
    "正式特徵_承包商可歸責": "承包商可歸責",
    "正式特徵_展延免計工期爭議": "展延／免計工期爭議",
    "正式特徵_實際損害不明偏低": "實際損害不明／偏低",
    "正式特徵_部分完成部分驗收": "部分完成／部分驗收",
    "正式特徵_業主已使用受益": "業主已使用／受益",
}
ANNOTATION_FEATURE_COLUMNS = {
    "issue_owner_fault": "業主可歸責",
    "issue_contractor_fault": "承包商可歸責",
    "issue_extension_request": "展延／免計工期爭議",
    "issue_actual_damage_unclear": "實際損害不明／偏低",
    "issue_partial_completion": "部分完成／部分驗收",
    "issue_used_by_owner": "業主已使用／受益",
}
FEATURE_ANALYSIS_REQUIRED_COLUMNS = [
    *FORMAL_FEATURE_COLUMNS,
    "是否酌減",
    "酌減率",
    "主張違約金",
    "法院准許違約金",
]

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
HIT_FILTER_ALL = "全部"
HIT_FILTER_CORRECT = "命中"
HIT_FILTER_WRONG = "未命中"
HIT_FILTER_OPTIONS = [HIT_FILTER_ALL, HIT_FILTER_CORRECT, HIT_FILTER_WRONG]
CONTRIBUTION_MODELS = {
    "logistic_regression_l2": "分類 Logistic",
    "ridge_regression_l2": "比例 Ridge",
    "lasso_regression_l1": "比例 Lasso",
}
RATIO_MODEL_OPTIONS = {
    "ridge_regression_l2": {
        "label": "Ridge",
        "remainingField": "ridgePredictedRemainingRatio",
        "errorField": "ridgeAbsError",
        "contributionModel": "ridge_regression_l2",
    },
    "lasso_regression_l1": {
        "label": "Lasso",
        "remainingField": "lassoPredictedRemainingRatio",
        "errorField": "lassoAbsError",
        "contributionModel": "lasso_regression_l1",
    },
    "mean_baseline": {
        "label": "Mean baseline",
        "remainingField": "meanPredictedRemainingRatio",
        "errorField": "meanAbsError",
        "contributionModel": "",
    },
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


def is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def pct(value: Any, digits: int = 1) -> str:
    if not is_number(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    if not is_number(value):
        return "—"
    return f"{float(value):.{digits}f}"


def pp(value: Any, digits: int = 1) -> str:
    if not is_number(value):
        return "—"
    return f"{abs(float(value)) * 100:.{digits}f} 個百分點"


def money(value: Any) -> str:
    if not is_number(value):
        return "—"
    return f"NT$ {float(value):,.0f}"


def money_gap(value: Any) -> str:
    if not is_number(value):
        return "—"
    number = float(value)
    if abs(number) < 0.5:
        return "相同"
    direction = "高估" if number > 0 else "低估"
    return f"{direction} {money(abs(number))}"


def yes_no(value: Any) -> str:
    if value == 1:
        return "是"
    if value == 0:
        return "否"
    return "—"


def ratio_model_key(value: Any = None) -> str:
    key = str(value or st.session_state.get("ratio_model") or "ridge_regression_l2")
    return key if key in RATIO_MODEL_OPTIONS else "ridge_regression_l2"


def ratio_model_label(value: Any) -> str:
    key = ratio_model_key(value)
    return str(RATIO_MODEL_OPTIONS[key]["label"])


def ratio_model_remaining_ratio(item: dict[str, Any], model_key: Any = None) -> float | None:
    key = ratio_model_key(model_key)
    field = str(RATIO_MODEL_OPTIONS[key]["remainingField"])
    value = item.get(field)
    return float(value) if is_number(value) else None


def ratio_model_reduction_rate(item: dict[str, Any], model_key: Any = None) -> float | None:
    remaining = ratio_model_remaining_ratio(item, model_key)
    return 1.0 - remaining if remaining is not None else None


def ratio_model_allowed_penalty(item: dict[str, Any], model_key: Any = None) -> float | None:
    claimed_penalty = item.get("claimedPenalty")
    remaining = ratio_model_remaining_ratio(item, model_key)
    if not is_number(claimed_penalty) or remaining is None:
        return None
    return float(claimed_penalty) * remaining


def ratio_model_abs_error(item: dict[str, Any], model_key: Any = None) -> float | None:
    key = ratio_model_key(model_key)
    field = str(RATIO_MODEL_OPTIONS[key]["errorField"])
    value = item.get(field)
    return float(value) if is_number(value) else None


def ratio_model_contribution_model(model_key: Any = None) -> str:
    key = ratio_model_key(model_key)
    return str(RATIO_MODEL_OPTIONS[key]["contributionModel"])


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
    text = "" if value is None else str(value).strip().lower()
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


def prepare_feature_correlation_frame(
    rows: list[dict[str, str]],
) -> pd.DataFrame:
    if not rows:
        raise ValueError("正式特徵資料沒有任何案件")
    missing = [
        name for name in FEATURE_ANALYSIS_REQUIRED_COLUMNS if name not in rows[0]
    ]
    if missing:
        raise ValueError(f"正式特徵資料缺少必要欄位：{', '.join(missing)}")

    records: list[dict[str, float | None]] = []
    for row in rows:
        record = {
            label: (
                float(parsed)
                if (parsed := parse_label(row.get(source))) is not None
                else None
            )
            for source, label in FORMAL_FEATURE_COLUMNS.items()
        }
        is_reduced = parse_label(row.get("是否酌減"))
        record["是否酌減"] = float(is_reduced) if is_reduced is not None else None

        claimed = to_float_or_none(row.get("主張違約金"))
        allowed = to_float_or_none(row.get("法院准許違約金"))
        reduction_rate = to_float_or_none(row.get("酌減率"))
        valid_amounts = (
            claimed is not None
            and claimed > 0
            and allowed is not None
            and 0 <= allowed <= claimed
        )
        record["酌減率"] = (
            reduction_rate
            if valid_amounts
            and reduction_rate is not None
            and 0 <= reduction_rate <= 1
            else None
        )
        records.append(record)

    return pd.DataFrame(records, dtype="float64")


def prepare_annotation_feature_correlation_frame(
    rows: list[dict[str, str]],
) -> pd.DataFrame:
    if not rows:
        raise ValueError("AI 假設標註資料沒有任何案件")
    missing = [
        name for name in [*ANNOTATION_FEATURE_COLUMNS, "is_reduced"] if name not in rows[0]
    ]
    if missing:
        raise ValueError(f"AI 假設標註資料缺少必要欄位：{', '.join(missing)}")

    records: list[dict[str, float | None]] = []
    for row in rows:
        record = {
            label: (
                float(parsed)
                if (parsed := parse_label(row.get(source))) is not None
                else None
            )
            for source, label in ANNOTATION_FEATURE_COLUMNS.items()
        }
        is_reduced = parse_label(row.get("is_reduced"))
        record["是否酌減"] = float(is_reduced) if is_reduced is not None else None

        claimed = to_float_or_none(row.get("claimed_penalty"))
        allowed = to_float_or_none(row.get("allowed_penalty"))
        _, reduction_rate = actual_ratios(row)
        valid_amounts = (
            claimed is not None
            and claimed > 0
            and allowed is not None
            and 0 <= allowed <= claimed
        )
        record["酌減率"] = (
            reduction_rate
            if valid_amounts
            and reduction_rate is not None
            and 0 <= reduction_rate <= 1
            else None
        )
        records.append(record)

    return pd.DataFrame(records, dtype="float64")


def correlation_status(value: Any) -> str:
    if not is_number(value):
        return "無變異，無法計算"
    magnitude = abs(float(value))
    if magnitude >= 0.7:
        strength = "強"
    elif magnitude >= 0.4:
        strength = "中"
    elif magnitude >= 0.2:
        strength = "弱"
    else:
        strength = "極弱"
    direction = (
        "正相關"
        if float(value) > 0
        else "負相關"
        if float(value) < 0
        else "無線性相關"
    )
    return f"{strength}{direction}"


def target_correlation_table(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FORMAL_FEATURE_COLUMNS.values():
        pair = frame[[feature, target]].dropna()
        can_compute = (
            len(pair) >= 2
            and pair[feature].nunique() >= 2
            and pair[target].nunique() >= 2
        )
        correlation = (
            float(pair[feature].corr(pair[target])) if can_compute else math.nan
        )
        rows.append(
            {
                "特徵": feature,
                "相關係數": correlation,
                "有效樣本": len(pair),
                "判讀": correlation_status(correlation),
            }
        )
    return pd.DataFrame(rows)


def compute_feature_correlation(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    features = list(FORMAL_FEATURE_COLUMNS.values())
    counts = pd.DataFrame(
        [
            {
                "特徵": feature,
                "0 件數": int((frame[feature] == 0).sum()),
                "1 件數": int((frame[feature] == 1).sum()),
                "缺值": int(frame[feature].isna().sum()),
            }
            for feature in features
        ]
    )
    return {
        "counts": counts,
        "matrix": frame[features].corr(method="pearson"),
        "is_reduced": target_correlation_table(frame, "是否酌減"),
        "reduction_rate": target_correlation_table(frame, "酌減率"),
    }


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


def label_is_reduced(value: Any) -> str:
    parsed = parse_label(value)
    if parsed == 1:
        return "有酌減"
    if parsed == 0:
        return "未酌減"
    return "—"


def classification_feedback(item: dict[str, Any]) -> tuple[str, str, str]:
    actual = parse_label(item.get("actualIsReduced"))
    predicted = parse_label(item.get("predictedIsReduced"))
    if actual is None or predicted is None:
        return (
            "分類尚未評分",
            "feedback-neutral",
            "缺少分類預測或原判決是否酌減標註。",
        )
    if actual == predicted:
        return (
            "方向命中",
            "feedback-good",
            f"模型判斷{label_is_reduced(predicted)}，原判決也是{label_is_reduced(actual)}。",
        )
    return (
        "方向未命中",
        "feedback-bad",
        f"模型判斷{label_is_reduced(predicted)}，但原判決是{label_is_reduced(actual)}。",
    )


def ratio_feedback(error: Any) -> tuple[str, str, str]:
    if not is_number(error):
        return (
            "比例尚未評分",
            "feedback-neutral",
            "缺少比例預測或原判決准許比例。",
        )
    ratio_error = float(error)
    if ratio_error <= 0.10:
        return (
            "比例精準",
            "feedback-good",
            "准許比例差距在 10 個百分點內。",
        )
    if ratio_error <= 0.20:
        return (
            "比例可參考",
            "feedback-watch",
            "准許比例差距在 20 個百分點內。",
        )
    return (
        "比例偏差大",
        "feedback-bad",
        "准許比例差距超過 20 個百分點。",
    )


def overall_prediction_feedback(item: dict[str, Any], model_key: Any = None) -> tuple[str, str, str]:
    correct = parse_label(item.get("classificationCorrect"))
    error = ratio_model_abs_error(item, model_key)
    if correct is None and error is None:
        return (
            "尚未產生回饋",
            "feedback-neutral",
            "請先在左側執行現場訓練，才會顯示這筆案例的即時回饋。",
        )
    if correct == 1 and is_number(error) and float(error) <= 0.10:
        return (
            "方向與比例都接近",
            "feedback-good",
            "這筆案例的是否酌減方向命中，准許比例也貼近原判決。",
        )
    if correct == 1 and is_number(error) and float(error) <= 0.20:
        return (
            "方向命中，比例可參考",
            "feedback-watch",
            "模型抓到是否酌減方向，但准許比例仍有可見差距。",
        )
    if correct == 1:
        return (
            "方向命中，比例需修正",
            "feedback-watch",
            "是否酌減方向正確，但准許比例偏離原判決較多。",
        )
    if correct == 0 and is_number(error) and float(error) <= 0.20:
        return (
            "比例接近，但方向未命中",
            "feedback-watch",
            "准許比例差距不大，但是否酌減的二元判斷沒有對上原判決。",
        )
    return (
        "本案預測偏離原判決",
        "feedback-bad",
        "方向或比例至少一項偏離，這筆案例應回到判決理由與特徵標註檢查。",
    )


def average(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def performance_summary_rows(cases: list[dict[str, Any]], model_key: Any = None) -> list[dict[str, Any]]:
    ratio_key = ratio_model_key(model_key)
    model_label = ratio_model_label(ratio_key)
    split_keys = sorted(
        {str(item.get("split")) for item in cases if item.get("split")},
        key=lambda value: SPLIT_ORDER.get(value, 99),
    )
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [
        (
            split_key,
            split_label(split_key),
            [item for item in cases if item.get("split") == split_key],
        )
        for split_key in split_keys
    ]
    groups.append(("overall", "整體", cases))

    rows: list[dict[str, Any]] = []
    for split_key, label, split_cases in groups:
        class_samples = [
            item for item in split_cases if parse_label(item.get("actualIsReduced")) is not None
        ]
        scored_class_samples = [
            item
            for item in class_samples
            if parse_label(item.get("predictedIsReduced")) is not None
        ]
        correct_count = sum(
            1
            for item in scored_class_samples
            if parse_label(item.get("actualIsReduced")) == parse_label(item.get("predictedIsReduced"))
        )
        predicted_reduced = [
            item for item in scored_class_samples if parse_label(item.get("predictedIsReduced")) == 1
        ]
        true_positive = sum(
            1 for item in predicted_reduced if parse_label(item.get("actualIsReduced")) == 1
        )
        ratio_samples = [
            item for item in split_cases if item.get("targetQuality") == "ok"
        ]
        ratio_errors = [
            error
            for item in ratio_samples
            if is_number(error := ratio_model_abs_error(item, ratio_key))
        ]
        accuracy = (
            correct_count / len(scored_class_samples)
            if scored_class_samples
            else None
        )
        precision = true_positive / len(predicted_reduced) if predicted_reduced else None
        mae = average([float(error) for error in ratio_errors])
        within_10pp = (
            sum(1 for error in ratio_errors if float(error) <= 0.10) / len(ratio_errors)
            if ratio_errors
            else None
        )
        within_20pp = (
            sum(1 for error in ratio_errors if float(error) <= 0.20) / len(ratio_errors)
            if ratio_errors
            else None
        )
        rows.append(
            {
                "split": split_key,
                "資料切分": label,
                "全部案件": len(split_cases),
                "分類樣本(n)": len(class_samples),
                "比例樣本(n)": len(ratio_samples),
                "分類命中率": pct(accuracy),
                "酌減 precision": pct(precision),
                f"{model_label} 平均比例誤差": pp(mae),
                "10pp內比例": pct(within_10pp),
                "20pp內比例": pct(within_20pp),
            }
        )
    return rows


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
    claimed_penalty = to_float_or_none(row.get("claimed_penalty"))
    allowed_penalty = to_float_or_none(row.get("allowed_penalty"))
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
        "claimedPenalty": claimed_penalty,
        "allowedPenalty": allowed_penalty,
        "targetQuality": target_quality(claimed_penalty, allowed_penalty, remaining),
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
            "notice": "資料來自 annotation_workbook_ai_assumed.csv；模型數字需按「現場訓練模型」後才會由 Streamlit 重新訓練產生。",
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
    coef_values: Any = getattr(estimator, "coef_", None)
    if coef_values is None:
        raise ValueError(f"模型 {model_name} 不含 coef_，無法計算特徵貢獻")
    if getattr(coef_values, "ndim", 1) == 2:
        coef_values = coef_values[0]
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
            --tool-page: #f4f6f5;
            --tool-sidebar: #ffffff;
            --tool-surface: #ffffff;
            --tool-soft: #f7f8f7;
            --tool-ink: #1f2933;
            --tool-muted: #5c6670;
            --tool-line: #cfd5d1;
            --tool-accent: #176b5d;
            --tool-accent-soft: #d9eee8;
            --risk-high: #b42318;
            --risk-low: #147a3f;
          }
          html,
          body,
          .stApp,
          [data-testid="stAppViewContainer"],
          [data-testid="stAppViewContainer"] > .main {
            background: var(--tool-page) !important;
            color: var(--tool-ink) !important;
          }
          [data-testid="stSidebar"],
          [data-testid="stSidebar"] > div {
            background: var(--tool-sidebar) !important;
            color: var(--tool-ink) !important;
            border-right: 1px solid var(--tool-line);
          }
          [data-testid="stSidebar"] *,
          [data-testid="stMetric"],
          [data-testid="stMetric"] * {
            color: var(--tool-ink) !important;
          }
          [data-testid="stMetric"] {
            background: transparent !important;
          }
          .stMarkdown,
          .stMarkdown p,
          .stMarkdown li,
          .stMarkdown div,
          .stCaption,
          label,
          p,
          h1,
          h2,
          h3,
          h4 {
            color: var(--tool-ink);
          }
          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] input,
          textarea {
            background: #ffffff !important;
            color: var(--tool-ink) !important;
            border-color: var(--tool-line) !important;
          }
          button[data-baseweb="tab"] {
            color: var(--tool-muted) !important;
          }
          button[data-baseweb="tab"][aria-selected="true"] {
            color: #c2272d !important;
          }
          .stButton button,
          .stDownloadButton button {
            border-color: var(--tool-line) !important;
            background: #ffffff !important;
            color: var(--tool-ink) !important;
          }
          .stButton button[kind="primary"],
          .stButton button[kind="primary"] *,
          [data-testid="stSidebar"] .stButton button[kind="primary"],
          [data-testid="stSidebar"] .stButton button[kind="primary"] * {
            border-color: var(--tool-accent) !important;
            background: var(--tool-accent) !important;
            color: #ffffff !important;
          }
          .block-container {
            padding-top: 2.75rem;
            padding-bottom: 2rem;
          }
          .tool-title {
            display: block;
            padding: 1rem 1.1rem;
            border: 1px solid var(--tool-line);
            border-radius: 8px;
            background: var(--tool-surface);
            color: var(--tool-ink);
            min-width: 0;
            overflow-wrap: anywhere;
          }
          .case-heading {
            min-width: 0;
          }
          .tool-title h1 {
            margin: 0;
            color: var(--tool-ink);
            font-size: 1.65rem;
            line-height: 1.2;
            letter-spacing: 0;
          }
          .case-meta-line {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem 0.75rem;
            align-items: center;
            margin-top: 0.55rem;
            min-width: 0;
            color: var(--tool-muted);
            font-size: 0.88rem;
            line-height: 1.45;
          }
          .case-id-label {
            display: inline;
            color: var(--tool-muted);
            font-weight: 900;
          }
          .case-id-value {
            display: inline;
            color: var(--tool-ink);
            font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
            font-size: 0.84rem;
            font-weight: 800;
            overflow-wrap: anywhere;
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
          .prediction-feedback {
            margin: 0.15rem 0 0.95rem;
            padding: 0.15rem 0 0.15rem 0.85rem;
            border-left: 4px solid var(--tool-muted);
            color: var(--tool-ink);
          }
          .prediction-feedback h4 {
            margin: 0.1rem 0 0.25rem;
            font-size: 1.02rem;
            letter-spacing: 0;
          }
          .prediction-feedback p {
            margin: 0;
            color: var(--tool-muted);
            line-height: 1.55;
          }
          .feedback-kicker {
            color: var(--tool-accent) !important;
            font-size: 0.78rem;
            font-weight: 900;
          }
          .feedback-good {
            border-left-color: var(--risk-low);
          }
          .feedback-watch {
            border-left-color: #a16207;
          }
          .feedback-bad {
            border-left-color: var(--risk-high);
          }
          .feedback-neutral {
            border-left-color: var(--tool-muted);
          }
          .judgment-grid,
          .feedback-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0 0.9rem;
            margin-top: 0.8rem;
            border-top: 1px solid var(--tool-line);
          }
          .judgment-grid > div,
          .feedback-grid > div {
            min-width: 0;
            padding: 0.72rem 0 0.2rem;
            border-bottom: 1px solid var(--tool-line);
          }
          .judgment-label,
          .feedback-label {
            display: block;
            color: var(--tool-muted);
            font-size: 0.78rem;
            font-weight: 850;
          }
          .judgment-value,
          .feedback-value {
            display: block;
            color: var(--tool-ink);
            font-size: 0.98rem;
            font-weight: 900;
            line-height: 1.35;
            overflow-wrap: anywhere;
          }
          .feedback-text {
            margin: 0.2rem 0 0;
            color: var(--tool-muted);
            font-size: 0.88rem;
            line-height: 1.55;
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
            .judgment-grid,
            .feedback-grid {
              grid-template-columns: minmax(0, 1fr);
            }
            .feature-grid {
              grid-template-columns: minmax(0, 1fr) minmax(4.5rem, 0.45fr) minmax(4.5rem, 0.45fr);
            }
            .feature-grid .feature-interpretation {
              grid-column: 1 / -1;
            }
          }
          @media (max-width: 700px) {
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
        "hit_filter": HIT_FILTER_ALL,
        "ratio_model": "ridge_regression_l2",
        "contribution_model": "logistic_regression_l2",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def apply_preset(name: str) -> None:
    st.session_state.search_query = ""
    st.session_state.year_filter = "全部年度"
    st.session_state.split_filter = "全部切分"
    st.session_state.hit_filter = HIT_FILTER_ALL
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


def matches_hit_filter(item: dict[str, Any], hit_filter: str) -> bool:
    correct = parse_label(item.get("classificationCorrect"))
    if hit_filter == HIT_FILTER_CORRECT:
        return correct == 1
    if hit_filter == HIT_FILTER_WRONG:
        return correct == 0
    return True


def filter_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyword = str(st.session_state.search_query).strip().lower()
    year = "" if st.session_state.year_filter == "全部年度" else st.session_state.year_filter
    split = "" if st.session_state.split_filter == "全部切分" else st.session_state.split_filter
    hit_filter = str(st.session_state.hit_filter)

    rows: list[dict[str, Any]] = []
    for item in cases:
        haystack = f"{item.get('jid', '')} {item.get('title', '')} {item.get('court', '')}".lower()
        if keyword and keyword not in haystack:
            continue
        if year and str(item.get("year")) != year:
            continue
        if split and item.get("split") != split:
            continue
        if not matches_hit_filter(item, hit_filter):
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
        jid = str(item.get("jid") or "")
        year = item.get("year") or "-"
        title = item.get("title") or "未命名案件"
        court = item.get("court") or "未知法院"
        labels[jid] = (
            f"{year} | {title}"
            f" | {court}"
            f" | {jid}"
        )
    return labels


def render_training_controls() -> None:
    with st.sidebar:
        st.markdown("**現場訓練**")
        if st.button("現場訓練模型", type="primary", use_container_width=True):
            with st.spinner("正在從 annotation_workbook_ai_assumed.csv 產生特徵並訓練模型..."):
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

        st.markdown("**模型選擇**")
        st.selectbox(
            "比例模型",
            list(RATIO_MODEL_OPTIONS.keys()),
            key="ratio_model",
            format_func=ratio_model_label,
            help="控制右側預測准許比例、預測酌減率與比例模型重要特徵。",
        )

    filtered = filter_cases(cases)
    labels = case_label_map(filtered)
    selected_jid = None

    with st.sidebar:
        if filtered:
            jids = [str(item.get("jid")) for item in filtered]
            if st.session_state.get("selected_jid") not in jids:
                st.session_state.selected_jid = preferred_jid(filtered) or jids[0]

            def format_case_label(jid: Any) -> str:
                jid_text = str(jid)
                return labels.get(jid_text, jid_text)

            selected_jid = st.selectbox(
                "案件",
                jids,
                key="selected_jid",
                format_func=format_case_label,
            )
        else:
            st.warning("沒有符合條件的案件")

        st.text_input("搜尋", placeholder="案號、案名、法院", key="search_query")
        st.selectbox("年度", ["全部年度", *years], key="year_filter")
        st.selectbox("切分", ["全部切分", *splits], key="split_filter", format_func=split_label)
        st.selectbox("分類命中", HIT_FILTER_OPTIONS, key="hit_filter")

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
          <div class="case-heading">
            <p class="eyebrow">{safe(item.get("year"))}｜{safe(item.get("splitLabel"))}｜{safe(item.get("court"))}</p>
            <h1>{safe(item.get("title") or "未命名案件")}</h1>
            <div class="case-meta-line">
              <span><span class="case-id-label">案號</span> <span class="case-id-value">{safe(item.get("jid"))}</span></span>
            </div>
          </div>
        </div>
        """
    )


def render_metrics(item: dict[str, Any]) -> None:
    model_key = ratio_model_key()
    model_label = ratio_model_label(model_key)
    predicted_remaining = ratio_model_remaining_ratio(item, model_key)
    predicted_reduction = ratio_model_reduction_rate(item, model_key)
    cols = st.columns(3)
    cols[0].metric("酌減機率", pct(item.get("reductionProbability")), help="Logistic Regression")
    cols[1].metric(f"{model_label} 准許比例", pct(predicted_remaining), help=model_label)
    cols[2].metric(f"{model_label} 酌減率", pct(predicted_reduction), help="1 - selected remaining_ratio")
    st.caption(live_risk_reason(item.get("reductionProbability"), predicted_reduction))


def is_case_trained(item: dict[str, Any]) -> bool:
    return is_number(item.get("reductionProbability")) or is_number(item.get("ridgePredictedRemainingRatio"))


def render_untrained_model_notice() -> None:
    render_html(
        """
        <div class="panel">
          <h3>尚未現場訓練</h3>
          <p class="muted">
            尚未現場訓練，模型比較會在訓練後顯示。請先按左側「現場訓練模型」，
            app 會從 annotation_workbook_ai_assumed.csv 產生特徵並重新訓練 Logistic、Ridge 與 Lasso。
          </p>
        </div>
        """
    )


def render_model_compare(item: dict[str, Any]) -> None:
    model_key = ratio_model_key()
    model_label = ratio_model_label(model_key)
    predicted_remaining = ratio_model_remaining_ratio(item, model_key)
    predicted_reduction = ratio_model_reduction_rate(item, model_key)
    predicted_allowed = ratio_model_allowed_penalty(item, model_key)
    amount_gap = (
        predicted_allowed - float(item["allowedPenalty"])
        if predicted_allowed is not None and is_number(item.get("allowedPenalty"))
        else None
    )
    selected_error = ratio_model_abs_error(item, model_key)
    overall_label, overall_class, overall_detail = overall_prediction_feedback(item, model_key)
    class_label, class_class, class_detail = classification_feedback(item)
    ratio_label, ratio_class, ratio_detail = ratio_feedback(selected_error)
    parts = ['<div class="panel"><h3>預測示範回饋</h3>']
    detail = f"""
      <div class="prediction-feedback {safe(overall_class)}">
        <p class="feedback-kicker">即時回饋</p>
        <h4>{safe(overall_label)}</h4>
        <p>{safe(overall_detail)}</p>
      </div>
      <div class="judgment-grid">
        <div>
          <span class="judgment-label">原判決是否酌減</span>
          <span class="judgment-value">{safe(label_is_reduced(item.get("actualIsReduced")))}</span>
        </div>
        <div>
          <span class="judgment-label">模型分類預測</span>
          <span class="judgment-value">{safe(label_is_reduced(item.get("predictedIsReduced")))}</span>
        </div>
        <div>
          <span class="judgment-label">主張違約金</span>
          <span class="judgment-value">{money(item.get("claimedPenalty"))}</span>
        </div>
        <div>
          <span class="judgment-label">法院准許違約金</span>
          <span class="judgment-value">{money(item.get("allowedPenalty"))}</span>
        </div>
        <div>
          <span class="judgment-label">原判決准許比例</span>
          <span class="judgment-value">{pct(item.get("actualRemainingRatio"))}</span>
        </div>
        <div>
          <span class="judgment-label">{safe(model_label)} 預測准許比例</span>
          <span class="judgment-value">{pct(predicted_remaining)}</span>
        </div>
        <div>
          <span class="judgment-label">{safe(model_label)} 預測准許金額</span>
          <span class="judgment-value">{money(predicted_allowed)}</span>
        </div>
        <div>
          <span class="judgment-label">與法院准許金額差距</span>
          <span class="judgment-value">{money_gap(amount_gap)}</span>
        </div>
      </div>
      <div class="feedback-grid">
        <div class="prediction-feedback {safe(class_class)}">
          <span class="feedback-label">是否酌減方向</span>
          <span class="feedback-value">{safe(class_label)}</span>
          <p class="feedback-text">{safe(class_detail)}</p>
        </div>
        <div class="prediction-feedback {safe(ratio_class)}">
          <span class="feedback-label">准許比例能力</span>
          <span class="feedback-value">{safe(ratio_label)}，差距 {pp(selected_error)}</span>
          <p class="feedback-text">{safe(ratio_detail)} {safe(model_label)} 預測酌減率 {pct(predicted_reduction)}。</p>
        </div>
      </div>
      <div class="detail-box">
        <strong>AI 假設版回測對照</strong><br>
        目前選用比例模型：{safe(model_label)}；絕對誤差：{num(selected_error)}<br>
        實際酌減率：{pct(item.get("actualReductionRate"))}；
        {safe(model_label)} 預測酌減率：{pct(predicted_reduction)}<br>
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
    model = ratio_model_contribution_model()
    model_label = ratio_model_label(ratio_model_key())
    st.caption(f"目前側欄比例模型：{model_label}")
    if not model:
        st.info("Mean baseline 使用訓練集平均准許比例，沒有個案特徵係數可顯示。")
        return
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


def render_performance_summary(payload: dict[str, Any]) -> None:
    cases: list[dict[str, Any]] = payload["cases"]
    model_key = ratio_model_key()
    model_label = ratio_model_label(model_key)
    rows = performance_summary_rows(cases, model_key)
    by_split = {row["split"]: row for row in rows}
    overall = by_split.get("overall", {})
    train = by_split.get("train_2021_2023", {})
    validation = by_split.get("validation_2024", {})

    st.markdown("### 整體回測能力")
    st.caption(
        "訓練集 2021-2023 是 in-sample；2024 驗證、2025 測試與 2026 最新年度是 out-of-time 回測。"
        "分類命中率是是否酌減方向的 accuracy；酌減 precision 只看模型預測為酌減的案件；"
        f"{model_label} 平均比例誤差是准許比例 MAE。"
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("訓練分類樣本", train.get("分類樣本(n)", "—"))
    col2.metric("驗證分類樣本", validation.get("分類樣本(n)", "—"))
    col3.metric("整體分類命中率", overall.get("分類命中率", "—"))
    col4.metric(f"{model_label} 整體 MAE", overall.get(f"{model_label} 平均比例誤差", "—"))

    display_rows = [{key: value for key, value in row.items() if key != "split"} for row in rows]
    st.dataframe(display_rows, use_container_width=True, hide_index=True)


def correlation_cell_style(value: Any) -> str:
    if not is_number(value):
        return "background-color: #f1f5f4; color: #64748b;"
    bounded = max(-1.0, min(1.0, float(value)))
    intensity = abs(bounded)
    if bounded >= 0:
        red = 220
        green = 252 - round(60 * intensity)
        blue = 231 - round(45 * intensity)
    else:
        red = 219 - round(35 * intensity)
        green = 234 - round(45 * intensity)
        blue = 254
    return f"background-color: rgb({red}, {green}, {blue}); color: #17201d;"


def format_correlation_value(value: Any) -> str:
    return f"{float(value):.3f}" if is_number(value) else "—"


def render_target_correlation(title: str, table: pd.DataFrame) -> None:
    st.markdown(f"### {title}")
    display = table.copy()
    display["相關係數"] = display["相關係數"].map(format_correlation_value)
    st.dataframe(display, use_container_width=True, hide_index=True)

    chart = table.dropna(subset=["相關係數"])[["特徵", "相關係數"]]
    if chart.empty:
        st.info("所有特徵皆因無變異或有效樣本不足而無法繪圖。")
    else:
        st.bar_chart(
            chart,
            x="特徵",
            y="相關係數",
            horizontal=True,
            use_container_width=True,
        )


def render_feature_correlation() -> None:
    st.subheader("六項正式特徵相關性")
    st.caption(
        "二元特徵間的 Pearson 相關等同 Phi coefficient；"
        "二元特徵與酌減率的 Pearson 相關等同 point-biserial correlation。"
    )
    try:
        rows = read_csv_rows(ANNOTATION_PATH)
        frame = prepare_annotation_feature_correlation_frame(rows)
        result = compute_feature_correlation(frame)
    except (FileNotFoundError, ValueError) as exc:
        st.warning(f"無法建立相關性分析：{exc}")
        return

    valid_is_reduced = int(frame["是否酌減"].notna().sum())
    valid_reduction_rate = int(frame["酌減率"].notna().sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("案件數", len(frame))
    col2.metric("是否酌減有效樣本", valid_is_reduced)
    col3.metric("酌減率有效樣本", valid_reduction_rate)

    st.markdown("### 特徵分布")
    st.dataframe(result["counts"], use_container_width=True, hide_index=True)

    st.markdown("### 六項特徵相關矩陣")
    matrix_style = (
        result["matrix"]
        .style.map(correlation_cell_style)
        .format(format_correlation_value)
    )
    st.dataframe(matrix_style, use_container_width=True)
    st.caption(
        "資料來源為 824 件 AI 假設版高相關案件；若某特徵沒有 0/1 變異，"
        "其相關係數會顯示為「—」，不是零相關。"
    )

    left, right = st.columns(2)
    with left:
        render_target_correlation("與是否酌減的相關性", result["is_reduced"])
    with right:
        render_target_correlation("與酌減率的相關性", result["reduction_rate"])

    st.info(
        "相關係數只表示線性共同變動，不代表法律因果；"
        "目前資料仍包含 AI 假設標註，正式結論需回到判決全文查核。"
    )


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

    tab_case, tab_data, tab_correlation, tab_limits = st.tabs(
        ["案件儀表板", "資料總覽", "特徵相關性", "限制說明"]
    )
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
        render_performance_summary(payload)
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

    with tab_correlation:
        render_feature_correlation()

    with tab_limits:
        render_notice(payload)
        st.markdown("**人工查核重點**")
        st.write("- 確認金額是否為契約總價、主張違約金、法院准許違約金或其他款項。")
        st.write("- 確認法院最終結論、展延、歸責、損害與使用收益等爭點。")
        st.write("- RAG 相似案例只作閱讀優先順序，不直接套用法律結論。")


if __name__ == "__main__":
    main()

#http://localhost:8522 | streamlit run streamlit_app.py
