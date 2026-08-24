from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .features import FEATURE_NAMES
from .temporal import TemporalSplitPolicy

REQUIRED_ANNOTATION_COLUMNS = ("JID", "decision_year", "is_reduced")
NONNEGATIVE_NUMERIC_COLUMNS = (
    "contract_price",
    "claimed_penalty",
    "allowed_penalty",
    "delay_days",
)
_TRUE_LABELS = {"1", "true", "t", "yes", "y", "是", "有", "酌減", "已酌減"}
_FALSE_LABELS = {"0", "false", "f", "no", "n", "否", "無", "未酌減", "不酌減"}


def _blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _binary_label(value: Any) -> int | None:
    if _blank(value):
        return None
    text = str(value).strip().lower()
    if text in _TRUE_LABELS:
        return 1
    if text in _FALSE_LABELS:
        return 0
    try:
        number = float(text.replace(",", "").replace("，", "").replace("%", ""))
    except ValueError:
        return None
    if not pd.notna(number) or number not in (0.0, 1.0):
        return None
    return int(number)


def _numeric(series: pd.Series) -> pd.Series:
    text = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(text, errors="coerce")


def validate_annotation_frame(
    frame: pd.DataFrame,
    *,
    min_rows: int = 30,
    min_labeled_rows: int = 30,
    min_year: int = 1900,
    max_year: int = 2100,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    row_count = int(len(frame))
    missing = [column for column in REQUIRED_ANNOTATION_COLUMNS if column not in frame.columns]
    if missing:
        errors.append({"code": "missing_required_columns", "columns": missing})
        return {
            "schema_version": 1,
            "profile": "annotation_workbook",
            "status": "fail",
            "row_count": row_count,
            "required_columns": list(REQUIRED_ANNOTATION_COLUMNS),
            "errors": errors,
            "warnings": warnings,
            "stats": {},
        }

    if row_count < min_rows:
        errors.append({"code": "not_enough_rows", "actual": row_count, "minimum": int(min_rows)})

    jid = frame["JID"].astype("string").fillna("").str.strip()
    blank_jids = int((jid == "").sum())
    if blank_jids:
        errors.append({"code": "blank_jid", "count": blank_jids})
    duplicate_mask = jid.ne("") & jid.duplicated(keep=False)
    duplicate_jids = sorted(set(jid[duplicate_mask].tolist()))
    if duplicate_jids:
        errors.append({"code": "duplicate_jid", "count": len(duplicate_jids), "examples": duplicate_jids[:10]})

    year_raw = frame["decision_year"]
    years = _numeric(year_raw)
    invalid_years = (~year_raw.map(_blank)) & years.isna()
    if invalid_years.any():
        errors.append({"code": "invalid_decision_year", "count": int(invalid_years.sum())})
    out_of_range = years.notna() & ((years < min_year) | (years > max_year))
    if out_of_range.any():
        errors.append({"code": "decision_year_out_of_range", "count": int(out_of_range.sum())})

    raw_labels = frame["is_reduced"]
    labels = raw_labels.map(_binary_label)
    invalid_labels = (~raw_labels.map(_blank)) & labels.isna()
    if invalid_labels.any():
        examples = sorted(set(raw_labels[invalid_labels].astype(str).tolist()))
        errors.append({"code": "invalid_is_reduced_label", "count": int(invalid_labels.sum()), "examples": examples[:10]})
    labeled = labels.dropna().astype(int)
    if len(labeled) < min_labeled_rows:
        errors.append({"code": "not_enough_labeled_rows", "actual": int(len(labeled)), "minimum": int(min_labeled_rows)})
    class_counts = {str(k): int(v) for k, v in labeled.value_counts().sort_index().items()}
    if labeled.nunique() < 2:
        errors.append({"code": "label_has_single_class", "class_counts": class_counts})

    numeric_stats: dict[str, Any] = {}
    for column in NONNEGATIVE_NUMERIC_COLUMNS:
        if column not in frame.columns:
            warnings.append({"code": "optional_numeric_column_missing", "column": column})
            continue
        values = _numeric(frame[column])
        nonblank = ~frame[column].map(_blank)
        unparseable = int((nonblank & values.isna()).sum())
        negatives = int((values < 0).fillna(False).sum())
        if unparseable:
            warnings.append({"code": "unparseable_numeric_values", "column": column, "count": unparseable})
        if negatives:
            errors.append({"code": "negative_numeric_values", "column": column, "count": negatives})
        numeric_stats[column] = {
            "non_null_numeric": int(values.notna().sum()),
            "negative": negatives,
            "unparseable_nonblank": unparseable,
        }

    if "claimed_penalty" in frame.columns and "allowed_penalty" in frame.columns:
        claimed = _numeric(frame["claimed_penalty"])
        allowed = _numeric(frame["allowed_penalty"])
        above_claim = claimed.notna() & allowed.notna() & (allowed > claimed)
        if above_claim.any():
            warnings.append({"code": "allowed_penalty_above_claimed", "count": int(above_claim.sum())})

    return {
        "schema_version": 1,
        "profile": "annotation_workbook",
        "status": "fail" if errors else "pass",
        "row_count": row_count,
        "required_columns": list(REQUIRED_ANNOTATION_COLUMNS),
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "unique_jids": int(jid[jid.ne("")].nunique()),
            "blank_jids": blank_jids,
            "decision_year_min": int(years.min()) if years.notna().any() else None,
            "decision_year_max": int(years.max()) if years.notna().any() else None,
            "labeled_rows": int(len(labeled)),
            "class_counts": class_counts,
            "numeric_columns": numeric_stats,
        },
    }


def _validate_model_frame(
    frame: pd.DataFrame,
    *,
    profile: str,
    target_column: str,
    policy: TemporalSplitPolicy,
    binary_target: bool,
    target_range: tuple[float, float] | None,
    max_feature_missing_rate: float,
    min_split_rows: dict[str, int],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    required = ["JID", "decision_year", target_column, *FEATURE_NAMES]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return {
            "schema_version": 2,
            "profile": profile,
            "status": "fail",
            "row_count": int(len(frame)),
            "errors": [{"code": "missing_required_columns", "columns": missing}],
            "warnings": [],
            "stats": {},
        }

    work = frame.copy()
    jid = work["JID"].astype("string").fillna("").str.strip()
    if jid.eq("").any():
        errors.append({"code": "blank_jid", "count": int(jid.eq("").sum())})
    duplicate = jid.ne("") & jid.duplicated(keep=False)
    if duplicate.any():
        errors.append({"code": "duplicate_jid", "count": int(duplicate.sum())})

    years = pd.to_numeric(work["decision_year"], errors="coerce")
    if years.isna().any():
        errors.append({"code": "invalid_decision_year", "count": int(years.isna().sum())})
    work["decision_year"] = years

    target = pd.to_numeric(work[target_column], errors="coerce")
    invalid_target = (~work[target_column].map(_blank)) & target.isna()
    if invalid_target.any():
        errors.append({"code": "invalid_target", "count": int(invalid_target.sum())})
    missing_target = target.isna()
    if missing_target.any():
        errors.append({"code": "missing_target", "count": int(missing_target.sum())})
    if binary_target:
        invalid_binary = target.notna() & ~target.isin([0.0, 1.0])
        if invalid_binary.any():
            errors.append({"code": "invalid_binary_target", "count": int(invalid_binary.sum())})
    if target_range is not None:
        low, high = target_range
        out_of_range = target.notna() & ((target < low) | (target > high))
        if out_of_range.any():
            errors.append(
                {
                    "code": "target_out_of_range",
                    "count": int(out_of_range.sum()),
                    "minimum": low,
                    "maximum": high,
                }
            )

    feature_stats: dict[str, Any] = {}
    for feature in FEATURE_NAMES:
        numeric = pd.to_numeric(work[feature], errors="coerce")
        missing_rate = float(numeric.isna().mean()) if len(work) else 1.0
        feature_stats[feature] = {"missing_rate": round(missing_rate, 6)}
        if missing_rate > max_feature_missing_rate:
            errors.append(
                {
                    "code": "feature_missing_rate_too_high",
                    "column": feature,
                    "actual": round(missing_rate, 6),
                    "maximum": max_feature_missing_rate,
                }
            )

    split_stats: dict[str, Any] = {}
    split_map = dict(policy.split_frame(work))
    requirements = {
        policy.train_name: min_split_rows.get("train", 10),
        policy.validation_name: min_split_rows.get("validation", 3),
        policy.test_name: min_split_rows.get("test", 3),
        policy.latest_name: min_split_rows.get("latest", 1),
    }
    for split_name, minimum in requirements.items():
        split = split_map[split_name]
        split_stats[split_name] = {"rows": int(len(split)), "minimum": int(minimum)}
        if len(split) < minimum:
            errors.append(
                {
                    "code": "split_too_small",
                    "split": split_name,
                    "actual": int(len(split)),
                    "minimum": int(minimum),
                }
            )

    train = split_map[policy.train_name]
    if binary_target and not train.empty:
        train_target = pd.to_numeric(train[target_column], errors="coerce").dropna().astype(int)
        counts = {str(key): int(value) for key, value in train_target.value_counts().sort_index().items()}
        split_stats[policy.train_name]["class_counts"] = counts
        if train_target.nunique() < 2:
            errors.append({"code": "training_target_single_class", "class_counts": counts})

    return {
        "schema_version": 2,
        "profile": profile,
        "status": "fail" if errors else "pass",
        "row_count": int(len(work)),
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "feature_missing": feature_stats,
            "splits": split_stats,
            "target_non_null": int(target.notna().sum()),
        },
        "temporal_split_policy": policy.as_dict(),
    }


def validate_classification_model_frame(
    frame: pd.DataFrame,
    *,
    policy: TemporalSplitPolicy | None = None,
    max_feature_missing_rate: float = 0.25,
) -> dict[str, Any]:
    return _validate_model_frame(
        frame,
        profile="classification_model_frame",
        target_column="is_reduced_label",
        policy=policy or TemporalSplitPolicy(),
        binary_target=True,
        target_range=(0.0, 1.0),
        max_feature_missing_rate=max_feature_missing_rate,
        min_split_rows={"train": 10, "validation": 3, "test": 3, "latest": 1},
    )


def validate_ratio_model_frame(
    frame: pd.DataFrame,
    *,
    policy: TemporalSplitPolicy | None = None,
    max_feature_missing_rate: float = 0.25,
) -> dict[str, Any]:
    return _validate_model_frame(
        frame,
        profile="ratio_model_frame",
        target_column="remaining_ratio",
        policy=policy or TemporalSplitPolicy(),
        binary_target=False,
        target_range=(0.0, 1.0),
        max_feature_missing_rate=max_feature_missing_rate,
        min_split_rows={"train": 10, "validation": 3, "test": 3, "latest": 1},
    )


def validate_annotation_csv(
    input_csv: Path,
    *,
    output_json: Path | None = None,
    min_rows: int = 30,
    min_labeled_rows: int = 30,
) -> dict[str, Any]:
    frame = pd.read_csv(input_csv, encoding="utf-8-sig")
    report = validate_annotation_frame(frame, min_rows=min_rows, min_labeled_rows=min_labeled_rows)
    report["input_csv"] = str(input_csv)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
