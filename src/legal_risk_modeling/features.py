from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

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

SPLITS = {
    "train_2021_2023": (2021, 2023),
    "validation_2024": (2024, 2024),
    "test_2025": (2025, 2025),
    "latest_2026": (2026, 2026),
}


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _flag(value: Any) -> int:
    if value in (1, "1", True, "true", "True", "yes", "是"):
        return 1
    return 0


def _candidate_count(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return len([part for part in text.replace("；", ";").split(";") if part.strip()])


def _log_or_zero(value: float | None) -> float:
    return math.log1p(value) if value is not None and value > 0 else 0.0


def build_feature_row(row: Mapping[str, Any]) -> dict[str, Any]:
    contract_price = _to_float(row.get("contract_price"))
    delay_days = _to_float(row.get("delay_days"))
    claimed_penalty = _to_float(row.get("claimed_penalty"))
    allowed_penalty = _to_float(row.get("allowed_penalty"))

    remaining_ratio = None
    if claimed_penalty is not None and claimed_penalty > 0 and allowed_penalty is not None:
        remaining_ratio = allowed_penalty / claimed_penalty

    result: dict[str, Any] = {
        "JID": row.get("JID", ""),
        "decision_year": pd.to_numeric(row.get("decision_year"), errors="coerce"),
        "remaining_ratio": remaining_ratio,
        "x_log_contract_price": _log_or_zero(contract_price),
        "x_log_claimed_penalty": _log_or_zero(claimed_penalty),
        "x_claim_to_contract_ratio": (
            claimed_penalty / contract_price
            if claimed_penalty is not None and contract_price is not None and contract_price > 0
            else 0.0
        ),
        "x_delay_days": delay_days or 0.0,
        "x_penalty_per_delay_day": (
            claimed_penalty / delay_days
            if claimed_penalty is not None and delay_days is not None and delay_days > 0
            else 0.0
        ),
        "x_money_candidate_count": (
            _candidate_count(row.get("ai_contract_price_candidates"))
            + _candidate_count(row.get("ai_claimed_penalty_candidates"))
            + _candidate_count(row.get("ai_allowed_penalty_candidates"))
        ),
        "x_delay_candidate_count": _candidate_count(row.get("ai_delay_days_candidates")),
    }
    for issue in ISSUE_FIELDS:
        result[f"x_issue_{issue}"] = _flag(row.get(f"issue_{issue}"))
        result[f"x_ai_issue_{issue}"] = _flag(row.get(f"ai_suggest_issue_{issue}"))
    return result


def build_feature_frame(rows: list[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(build_feature_row(row) for row in rows)
    for feature in FEATURE_NAMES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
    return frame


def normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    missing = [feature for feature in FEATURE_NAMES if feature not in result.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    for feature in FEATURE_NAMES:
        result[feature] = pd.to_numeric(result[feature], errors="coerce").fillna(0.0)
    return result


def case_splits(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    year = pd.to_numeric(frame["decision_year"], errors="coerce")
    return [
        (name, frame[(year >= start) & (year <= end)].copy())
        for name, (start, end) in SPLITS.items()
    ]
