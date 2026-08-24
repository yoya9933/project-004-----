from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .features import FEATURE_NAMES, build_feature_frame
from .manifest import build_manifest, write_manifest
from .models import build_classification_model, classification_model_spec
from .promotion import (
    build_classification_promotion_report,
    classification_promotion_policy,
)
from .temporal import TemporalSplitPolicy

_TEXT_FIELDS = [
    "JTITLE",
    "key_reason",
    "source_snippet_penalty",
    "source_snippet_delay",
    "source_snippet_reduction",
    "source_snippets_combined",
    "ai_evidence_issue_owner_fault",
    "ai_evidence_issue_contractor_fault",
    "ai_evidence_issue_extension_request",
    "ai_evidence_issue_actual_damage_unclear",
    "ai_evidence_issue_partial_completion",
    "ai_evidence_issue_used_by_owner",
]


def _binary_label(value: Any) -> int | None:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y", "是", "有", "酌減", "已酌減"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "否", "無", "未酌減", "不酌減"}:
        return 0
    try:
        number = float(text.replace(",", "").replace("，", "").replace("%", ""))
    except ValueError:
        return None
    return int(number != 0) if np.isfinite(number) else None


def _number(value: Any) -> float | None:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return None
    text = str(value).strip().replace(",", "").replace("，", "").replace("%", "")
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def _keyword_rule(row: Mapping[str, Any]) -> int:
    parts = []
    for field in _TEXT_FIELDS:
        value = row.get(field)
        if value is not None and not pd.isna(value):
            parts.append(str(value))
    text = "\n".join(parts)
    has_252 = bool(re.search(r"民法第\s*(252|２５２|二百五十二)\s*條|第\s*(252|２５２)\s*條", text))
    has_discretion = any(term in text for term in ("酌減", "核減", "酌予"))
    has_over_high = any(term in text for term in ("過高", "顯非合理", "顯屬過高"))
    return int(has_252 or has_discretion or has_over_high)


def prepare_classification_frame(
    raw: pd.DataFrame,
    *,
    use_derived_label_from_amounts: bool = False,
) -> pd.DataFrame:
    raw = raw.reset_index(drop=True).copy()
    features = build_feature_frame(raw.to_dict(orient="records"))

    labels: list[int | None] = []
    label_sources: list[str] = []
    derived_labels: list[int | None] = []
    keyword_flags: list[int] = []

    for row in raw.to_dict(orient="records"):
        label = _binary_label(row.get("is_reduced"))
        label_source = "manual_is_reduced" if label is not None else ""
        claimed = _number(row.get("claimed_penalty"))
        allowed = _number(row.get("allowed_penalty"))
        derived = None
        if claimed is not None and claimed > 0 and allowed is not None and allowed >= 0:
            derived = int(allowed < claimed)
            if label is None and use_derived_label_from_amounts:
                label = derived
                label_source = "derived_from_manual_amounts"

        labels.append(label)
        label_sources.append(label_source)
        derived_labels.append(derived)
        keyword_flags.append(_keyword_rule(row))

    features["is_reduced_label"] = pd.array(labels, dtype="Int64")
    features["label_source"] = label_sources
    features["derived_is_reduced_from_amounts"] = pd.array(derived_labels, dtype="Int64")
    features["x_keyword_rule_predict_reduced"] = keyword_flags

    for column in [
        "annotation_status",
        "annotation_priority",
        "court",
        "JTITLE",
        "JCASE",
        "JNO",
        "JDATE",
        "json_file",
    ]:
        if column in raw.columns:
            features[column] = raw[column]
    return features


def _metrics(model: str, split: str, actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    n = int(len(actual))
    if n == 0:
        return {
            "model": model,
            "split": split,
            "n": 0,
            "positives": 0,
            "negatives": 0,
            "accuracy": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "roc_auc": "",
        }
    predicted = (probability >= 0.5).astype(int)
    unique = np.unique(actual)
    auc: float | str = ""
    if len(unique) == 2:
        auc = round(float(roc_auc_score(actual, probability)), 4)
    return {
        "model": model,
        "split": split,
        "n": n,
        "positives": int(np.sum(actual == 1)),
        "negatives": int(np.sum(actual == 0)),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "precision": round(float(precision_score(actual, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
        "roc_auc": auc,
    }


def _prediction_rows(
    model: str,
    split_name: str,
    split: pd.DataFrame,
    probability: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (_, row), prob in zip(split.iterrows(), probability):
        rows.append(
            {
                "model": model,
                "split": split_name,
                "JID": row.get("JID", ""),
                "decision_year": int(row["decision_year"]),
                "actual": int(row["is_reduced_label"]),
                "predicted": int(prob >= 0.5),
                "probability": round(float(prob), 6),
            }
        )
    return rows


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def run_classification(
    *,
    project_root: Path,
    input_csv: Path,
    output_dir: Path,
    min_labeled_rows: int = 30,
    l2: float = 0.01,
    random_state: int = 42,
    train_start_year: int = 2021,
    train_end_year: int = 2023,
    validation_year: int = 2024,
    test_year: int = 2025,
    latest_check_year: int = 2026,
    use_derived_label_from_amounts: bool = False,
    run_id: str = "classification-governed",
    strict_git: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    input_csv = input_csv.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if l2 <= 0:
        raise ValueError("l2 must be positive")

    policy = TemporalSplitPolicy(
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        validation_year=validation_year,
        test_year=test_year,
        latest_check_year=latest_check_year,
    )
    for stale_name in [
        "metrics.csv",
        "predictions.csv",
        "model_coefficients.csv",
        "model_status.json",
        "promotion_report.json",
        "approved_release.json",
        "manifest.json",
    ]:
        (output_dir / stale_name).unlink(missing_ok=True)

    raw = pd.read_csv(input_csv, encoding="utf-8-sig")
    frame = prepare_classification_frame(
        raw,
        use_derived_label_from_amounts=use_derived_label_from_amounts,
    )
    labeled = frame[frame["is_reduced_label"].notna()].copy()
    labeled["is_reduced_label"] = labeled["is_reduced_label"].astype(int)

    feature_path = output_dir / "feature_matrix.csv"
    labeled_path = output_dir / "labeled_feature_matrix.csv"
    status_path = output_dir / "model_status.json"
    frame.to_csv(feature_path, index=False, encoding="utf-8-sig")
    labeled.to_csv(labeled_path, index=False, encoding="utf-8-sig")

    counts = labeled["is_reduced_label"].value_counts().to_dict()
    class_counts = {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}
    base_status: dict[str, Any] = {
        "run_id": run_id,
        "input_csv": _relative(input_csv, project_root),
        "total_rows": int(len(frame)),
        "labeled_rows": int(len(labeled)),
        "class_counts": class_counts,
        "model_features": FEATURE_NAMES,
        "split_policy": policy.as_dict(),
        "outputs": {
            "feature_matrix": _relative(feature_path, project_root),
            "labeled_feature_matrix": _relative(labeled_path, project_root),
        },
        "note": "Only official/manual labels are used by default; AI suggestions remain features.",
    }

    invocation = {
        "min_labeled_rows": min_labeled_rows,
        "l2": l2,
        "random_state": random_state,
        "use_derived_label_from_amounts": use_derived_label_from_amounts,
    }

    def finish(
        status: dict[str, Any],
        artifacts: list[Path],
        *,
        promotion_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = build_manifest(
            project_root=project_root,
            dataset_paths=[input_csv],
            artifact_paths=[*artifacts, status_path],
            model_spec=classification_model_spec(l2),
            run_id=run_id,
            invocation=invocation,
            temporal_split_policy=policy.as_dict(),
            promotion_policy=promotion_policy,
            strict_git=strict_git,
        )
        write_manifest(output_dir / "manifest.json", manifest)
        return status

    if len(labeled) < min_labeled_rows:
        return finish(
            {**base_status, "status": "skipped", "reason": "not_enough_official_is_reduced_labels"},
            [feature_path, labeled_path],
        )
    if class_counts["0"] == 0 or class_counts["1"] == 0:
        return finish(
            {**base_status, "status": "skipped", "reason": "target_has_single_class"},
            [feature_path, labeled_path],
        )

    splits = policy.split_frame(labeled)
    split_map = dict(splits)
    train = split_map[policy.train_name]
    train_counts = train["is_reduced_label"].value_counts().to_dict()
    if len(train) < 10 or 0 not in train_counts or 1 not in train_counts:
        return finish(
            {
                **base_status,
                "status": "skipped",
                "reason": "not_enough_two_class_training_rows_in_train_split",
            },
            [feature_path, labeled_path],
        )

    classifier = build_classification_model(l2=l2, random_state=random_state)
    classifier.fit(train[FEATURE_NAMES], train["is_reduced_label"].astype(int).to_numpy())
    majority_probability = float(train["is_reduced_label"].mean())

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for split_name, split in splits:
        actual = split["is_reduced_label"].astype(int).to_numpy()
        probabilities = {
            "majority_baseline": np.full(len(split), majority_probability, dtype=float),
            "keyword_rule_baseline": split["x_keyword_rule_predict_reduced"].astype(float).to_numpy(),
            "logistic_regression_l2": (
                classifier.predict_proba(split[FEATURE_NAMES])[:, 1]
                if len(split)
                else np.array([], dtype=float)
            ),
        }
        for model_name, probability in probabilities.items():
            metrics_rows.append(_metrics(model_name, split_name, actual, probability))
            prediction_rows.extend(_prediction_rows(model_name, split_name, split, probability))

    metrics_path = output_dir / "metrics.csv"
    predictions_path = output_dir / "predictions.csv"
    coefficients_path = output_dir / "model_coefficients.csv"
    promotion_path = output_dir / "promotion_report.json"
    release_path = output_dir / "approved_release.json"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False, encoding="utf-8-sig")

    scaler = classifier.named_steps["scale"]
    estimator = classifier.named_steps["model"]
    coefficient_rows = [
        {
            "feature": "intercept",
            "mean": "",
            "std": "",
            "coefficient_scaled": round(float(estimator.intercept_[0]), 8),
        }
    ]
    coefficient_rows.extend(
        {
            "feature": feature,
            "mean": round(float(scaler.mean_[index]), 8),
            "std": round(float(scaler.scale_[index]), 8),
            "coefficient_scaled": round(float(estimator.coef_[0][index]), 8),
        }
        for index, feature in enumerate(FEATURE_NAMES)
    )
    pd.DataFrame(coefficient_rows).to_csv(coefficients_path, index=False, encoding="utf-8-sig")

    promotion = build_classification_promotion_report(
        metrics_rows,
        validation_split=policy.validation_name,
        test_split=policy.test_name,
        latest_split=policy.latest_name,
    )
    promotion_path.write_text(json.dumps(promotion, ensure_ascii=False, indent=2), encoding="utf-8")
    release = {
        "schema_version": 1,
        "status": "approved",
        "default_model": promotion["default_model"],
        "approved_models": promotion["approved_models"],
        "baseline": promotion["baseline"],
        "predictions_artifact": _relative(predictions_path, project_root),
        "metrics_artifact": _relative(metrics_path, project_root),
        "promotion_report": _relative(promotion_path, project_root),
    }
    release_path.write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")

    learned_approved = "logistic_regression_l2" in promotion["approved_models"]
    status = {
        **base_status,
        "status": "trained",
        "model": "logistic_regression_l2",
        "baselines": ["majority_baseline", "keyword_rule_baseline"],
        "l2": l2,
        "promotion_status": "approved" if learned_approved else "rejected",
        "release_default_model": promotion["default_model"],
        "outputs": {
            **base_status["outputs"],
            "metrics": _relative(metrics_path, project_root),
            "predictions": _relative(predictions_path, project_root),
            "model_coefficients": _relative(coefficients_path, project_root),
            "promotion_report": _relative(promotion_path, project_root),
            "approved_release": _relative(release_path, project_root),
        },
    }
    return finish(
        status,
        [
            feature_path,
            labeled_path,
            metrics_path,
            predictions_path,
            coefficients_path,
            promotion_path,
            release_path,
        ],
        promotion_policy=classification_promotion_policy(),
    )
