from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import cross_val_score
from tpot import TPOTClassifier
from tpot.config.get_configspace import get_search_space


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_CSV = PROJECT_ROOT / "05_測試與驗證" / "test_issue_features.csv"
SUMMARY_CSV = PROJECT_ROOT / "03_研究與分析" / "tpot_issue_strict_linear_summary.csv"
DETAILS_JSON = PROJECT_ROOT / "03_研究與分析" / "tpot_issue_strict_linear_details.json"

RANDOM_STATE = 42
MAX_TIME_MINS = 0.25
MAX_EVAL_TIME_MINS = 0.1
LINEAR_CLASSIFIERS = [
    "LogisticRegression",
    "LinearSVC",
    "SGDClassifier",
    "LinearDiscriminantAnalysis",
]


def model_name(model) -> str:
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1].__class__.__name__
    return model.__class__.__name__


def best_score_value(tpot: TPOTClassifier) -> float | None:
    selected = getattr(tpot, "selected_best_score", None)
    if selected is None:
        return None
    for key in ("balanced_accuracy_score", "balanced_accuracy"):
        if hasattr(selected, "get") and key in selected:
            return float(selected.get(key))
    if hasattr(selected, "iloc"):
        numeric = pd.to_numeric(selected, errors="coerce").dropna()
        if not numeric.empty:
            return float(numeric.iloc[0])
    return None


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = pd.read_csv(INPUT_CSV)
    issue_cols = [col for col in df.columns if col.startswith("issue_")]

    title_features = pd.get_dummies(df[["title"]], columns=["title"], prefix="title")
    issue_features = df[issue_cols].astype(int)

    rows: list[dict] = []
    details: dict[str, dict] = {
        "input_csv": str(INPUT_CSV),
        "n_rows": int(len(df)),
        "issue_columns": issue_cols,
        "settings": {
            "search_space": "strict_linear_classifiers",
            "candidate_classifiers": LINEAR_CLASSIFIERS,
            "scorer": "balanced_accuracy",
            "max_time_mins_per_target": MAX_TIME_MINS,
            "max_eval_time_mins": MAX_EVAL_TIME_MINS,
            "random_state": RANDOM_STATE,
        },
        "targets": {},
    }

    for target in issue_cols:
        y = df[target].astype(int)
        counts = y.value_counts().sort_index()
        positives = int(counts.get(1, 0))
        negatives = int(counts.get(0, 0))

        row = {
            "target": target,
            "n_rows": len(df),
            "positive": positives,
            "negative": negatives,
            "cv": None,
            "tpot_best_balanced_accuracy": None,
            "cv_balanced_accuracy_mean": None,
            "cv_balanced_accuracy_std": None,
            "linear_model_type": None,
            "model": None,
            "status": "ok",
        }

        if y.nunique() < 2:
            row["status"] = "skipped_single_class"
            rows.append(row)
            details["targets"][target] = row
            print(f"[skip] {target}: only one class")
            continue

        min_class_count = int(counts.min())
        cv = max(2, min(3, min_class_count))
        row["cv"] = cv

        X = pd.concat(
            [title_features, issue_features.drop(columns=[target])],
            axis=1,
        ).astype(float)

        print(
            f"[fit] {target}: positive={positives}, negative={negatives}, "
            f"features={X.shape[1]}, cv={cv}"
        )

        try:
            search_space = get_search_space(
                LINEAR_CLASSIFIERS,
                n_classes=2,
                n_samples=len(X),
                n_features=X.shape[1],
                random_state=RANDOM_STATE,
            )
            tpot = TPOTClassifier(
                search_space=search_space,
                scorers=["balanced_accuracy"],
                cv=cv,
                max_time_mins=MAX_TIME_MINS,
                max_eval_time_mins=MAX_EVAL_TIME_MINS,
                n_jobs=1,
                random_state=RANDOM_STATE,
                verbose=0,
            )
            tpot.fit(X, y)

            model = tpot.fitted_pipeline_
            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="balanced_accuracy",
            )

            row["tpot_best_balanced_accuracy"] = best_score_value(tpot)
            row["cv_balanced_accuracy_mean"] = float(cv_scores.mean())
            row["cv_balanced_accuracy_std"] = float(cv_scores.std())
            row["linear_model_type"] = model_name(model)
            row["model"] = repr(model)

            details["targets"][target] = {
                **row,
                "feature_columns": list(X.columns),
            }

            print(
                f"[ok] {target}: model={row['linear_model_type']}, "
                f"cv_bal_acc={row['cv_balanced_accuracy_mean']:.3f}"
            )

        except Exception as exc:
            row["status"] = "error"
            row["model"] = f"{exc.__class__.__name__}: {exc}"
            details["targets"][target] = row
            print(f"[error] {target}: {exc.__class__.__name__}: {exc}")

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    DETAILS_JSON.write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(summary[[
        "target",
        "status",
        "positive",
        "negative",
        "cv_balanced_accuracy_mean",
        "linear_model_type",
    ]].to_string(index=False))
    print()
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DETAILS_JSON}")


if __name__ == "__main__":
    main()
