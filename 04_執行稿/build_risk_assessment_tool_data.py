from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLANATION_DIR = PROJECT_ROOT / "06_交付物" / "rag_model_explanation"
OUTPUT_DIR = PROJECT_ROOT / "06_交付物" / "risk_assessment_tool"
DATA_DIR = OUTPUT_DIR / "data"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def round_or_none(value: Any, digits: int = 4) -> float | None:
    number = to_float(value)
    if number is None:
        return None
    return round(number, digits)


def compact(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def risk_level(probability: float | None, predicted_reduction_rate: float | None) -> str:
    prob = probability if probability is not None else -1.0
    rate = predicted_reduction_rate if predicted_reduction_rate is not None else -1.0
    if prob >= 0.70 or rate >= 0.50:
        return "高"
    if prob >= 0.45 or rate >= 0.25:
        return "中"
    return "低"


def risk_reason(probability: float | None, predicted_reduction_rate: float | None) -> str:
    if probability is None and predicted_reduction_rate is None:
        return "分類與比例模型資料不足"
    parts: list[str] = []
    if probability is not None:
        parts.append(f"酌減機率 {probability:.1%}")
    if predicted_reduction_rate is not None:
        parts.append(f"預測酌減率 {predicted_reduction_rate:.1%}")
    return "，".join(parts)


def split_label(split: str) -> str:
    labels = {
        "train_2021_2023": "訓練集 2021-2023",
        "validation_2024": "驗證集 2024",
        "test_2025": "測試集 2025",
        "latest_2026": "最新年度檢查 2026",
    }
    return labels.get(split, split)


def format_case(row: dict[str, str]) -> dict[str, Any]:
    probability = round_or_none(row.get("reduction_probability"), 6)
    ridge_rate = round_or_none(row.get("ridge_predicted_reduction_rate"), 6)
    level = risk_level(probability, ridge_rate)
    actual_is_reduced = to_int(row.get("actual_is_reduced"))
    predicted_is_reduced = to_int(row.get("predicted_is_reduced"))
    return {
        "jid": row.get("JID", ""),
        "year": to_int(row.get("decision_year")),
        "split": row.get("split", ""),
        "splitLabel": split_label(row.get("split", "")),
        "court": row.get("court", ""),
        "title": row.get("JTITLE", ""),
        "priority": to_int(row.get("annotation_priority")),
        "manualChecked": row.get("manual_checked", ""),
        "relevanceScore": round_or_none(row.get("relevance_score"), 2),
        "actualIsReduced": actual_is_reduced,
        "predictedIsReduced": predicted_is_reduced,
        "classificationCorrect": to_int(row.get("classification_correct")),
        "reductionProbability": probability,
        "actualRemainingRatio": round_or_none(row.get("actual_remaining_ratio"), 6),
        "actualReductionRate": round_or_none(row.get("actual_reduction_rate"), 6),
        "actualBucket": row.get("actual_bucket", ""),
        "meanPredictedRemainingRatio": round_or_none(row.get("mean_predicted_remaining_ratio"), 6),
        "ridgePredictedRemainingRatio": round_or_none(row.get("ridge_predicted_remaining_ratio"), 6),
        "ridgePredictedReductionRate": ridge_rate,
        "ridgePredictedBucket": row.get("ridge_predicted_bucket", ""),
        "ridgeAbsError": round_or_none(row.get("ridge_abs_error"), 6),
        "lassoPredictedRemainingRatio": round_or_none(row.get("lasso_predicted_remaining_ratio"), 6),
        "lassoPredictedReductionRate": round_or_none(row.get("lasso_predicted_reduction_rate"), 6),
        "lassoPredictedBucket": row.get("lasso_predicted_bucket", ""),
        "lassoAbsError": round_or_none(row.get("lasso_abs_error"), 6),
        "meanAbsError": round_or_none(row.get("mean_abs_error"), 6),
        "topClassificationTowardReduction": row.get("top_classification_toward_reduction", ""),
        "topClassificationTowardNoReduction": row.get("top_classification_toward_no_reduction", ""),
        "topRatioTowardLessReductionRidge": row.get("top_ratio_toward_less_reduction_ridge", ""),
        "topRatioTowardMoreReductionRidge": row.get("top_ratio_toward_more_reduction_ridge", ""),
        "similarCaseCount": to_int(row.get("similar_case_count")) or 0,
        "similarLabeledCount": to_int(row.get("similar_labeled_count")) or 0,
        "similarLabeledReducedCount": to_int(row.get("similar_labeled_reduced_count")) or 0,
        "similarAvgRemainingRatio": round_or_none(row.get("similar_avg_remaining_ratio"), 6),
        "similarSharedTerms": row.get("similar_shared_terms", ""),
        "sourceJsonFile": row.get("source_json_file", ""),
        "riskLevel": level,
        "riskReason": risk_reason(probability, ridge_rate),
    }


def format_similar(row: dict[str, str]) -> dict[str, Any]:
    return {
        "queryJid": row.get("query_JID", ""),
        "rank": to_int(row.get("similar_rank")),
        "jid": row.get("similar_JID", ""),
        "score": round_or_none(row.get("similarity_score"), 4),
        "year": to_int(row.get("similar_decision_year")),
        "court": row.get("similar_court", ""),
        "title": row.get("similar_title", ""),
        "sharedTerms": row.get("shared_terms", ""),
        "strongTerms": row.get("similar_matched_strong_reduction_terms", ""),
        "availableInAnnotation": row.get("similar_available_in_annotation") == "1",
        "isReduced": to_int(row.get("similar_is_reduced")),
        "remainingRatio": round_or_none(row.get("similar_remaining_ratio"), 6),
        "keyReason": row.get("similar_key_reason", ""),
        "jsonFile": row.get("similar_json_file", ""),
        "reductionSnippet": compact(row.get("similar_reduction_snippet"), 260),
        "delaySnippet": compact(row.get("similar_delay_snippet"), 220),
    }


def format_contribution(row: dict[str, str]) -> dict[str, Any]:
    return {
        "jid": row.get("JID", ""),
        "modelFamily": row.get("model_family", ""),
        "model": row.get("model", ""),
        "feature": row.get("feature", ""),
        "label": row.get("feature_label", ""),
        "value": round_or_none(row.get("feature_value"), 6),
        "coefficient": round_or_none(row.get("coefficient_scaled"), 6),
        "standardizedValue": round_or_none(row.get("standardized_value"), 6),
        "contribution": round_or_none(row.get("contribution"), 6),
        "absContribution": round_or_none(row.get("contribution_abs"), 6),
        "rank": to_int(row.get("rank_by_abs")),
        "interpretation": row.get("interpretation", ""),
    }


def grouped_top_contributions(rows: list[dict[str, str]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rank = to_int(row.get("rank_by_abs"))
        if rank is None or rank > 8:
            continue
        item = format_contribution(row)
        grouped[item["jid"]][item["model"]].append(item)
    for by_model in grouped.values():
        for items in by_model.values():
            items.sort(key=lambda item: item.get("rank") or 999)
    return {jid: dict(by_model) for jid, by_model in grouped.items()}


def build_payload() -> dict[str, Any]:
    summary_rows = read_csv(EXPLANATION_DIR / "case_explanation_summary.csv")
    similar_rows = read_csv(EXPLANATION_DIR / "similar_case_evidence.csv")
    contribution_rows = read_csv(EXPLANATION_DIR / "model_feature_contributions.csv")

    cases = [format_case(row) for row in summary_rows]
    similar_by_jid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in similar_rows:
        item = format_similar(row)
        similar_by_jid[item["queryJid"]].append(item)
    for items in similar_by_jid.values():
        items.sort(key=lambda item: item.get("rank") or 999)

    risk_counts = Counter(case["riskLevel"] for case in cases)
    split_counts = Counter(case["split"] for case in cases)
    year_counts = Counter(str(case["year"]) for case in cases)
    missing_ridge = sum(1 for case in cases if case["ridgePredictedRemainingRatio"] is None)
    missing_probability = sum(1 for case in cases if case["reductionProbability"] is None)

    return {
        "metadata": {
            "generatedFrom": "06_交付物/rag_model_explanation",
            "caseCount": len(cases),
            "similarCaseCount": len(similar_rows),
            "featureContributionCount": len(contribution_rows),
            "missingRidgePrediction": missing_ridge,
            "missingReductionProbability": missing_probability,
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
            "notice": "AI 假設版標註與回測展示，不是法律意見；正式結論需回到原判決全文人工查核。",
        },
        "cases": cases,
        "similarCasesByJid": dict(similar_by_jid),
        "contributionsByJid": grouped_top_contributions(contribution_rows),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    data_path = DATA_DIR / "risk_tool_data.json"
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    status = {
        "status": "ok",
        "output": str(data_path),
        "caseCount": payload["metadata"]["caseCount"],
        "similarCaseCount": payload["metadata"]["similarCaseCount"],
        "featureContributionCount": payload["metadata"]["featureContributionCount"],
        "missingRidgePrediction": payload["metadata"]["missingRidgePrediction"],
        "missingReductionProbability": payload["metadata"]["missingReductionProbability"],
    }
    (DATA_DIR / "risk_tool_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
