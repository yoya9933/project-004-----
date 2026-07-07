from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import cross_val_score
from tpot import TPOTClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_CSV = PROJECT_ROOT / "05_測試與驗證" / "test_issue_features.csv"
SUMMARY_CSV = PROJECT_ROOT / "03_研究與分析" / "tpot_issue_model_summary.csv"
DETAILS_JSON = PROJECT_ROOT / "03_研究與分析" / "tpot_issue_model_details.json"

RANDOM_STATE = 42
MAX_TIME_MINS = 0.3
MAX_EVAL_TIME_MINS = 0.1


def final_estimator_name(pipeline) -> str:
    if hasattr(pipeline, "steps") and pipeline.steps:
        return pipeline.steps[-1][1].__class__.__name__
    return pipeline.__class__.__name__


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

    df = pd.read_csv(INPUT_CSV)
    issue_cols = [col for col in df.columns if col.startswith("issue_")]

    base_features = pd.get_dummies(df[["title"]], columns=["title"], prefix="title")
    issue_features = df[issue_cols].astype(int)

    rows: list[dict] = []
    details: dict[str, dict] = {
        "input_csv": str(INPUT_CSV),
        "n_rows": int(len(df)),
        "issue_columns": issue_cols,
        "settings": {
            "search_space": "linear",
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
            "final_estimator": None,
            "pipeline": None,
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
            [base_features, issue_features.drop(columns=[target])],
            axis=1,
        ).astype(float)

        print(
            f"[fit] {target}: positive={positives}, negative={negatives}, "
            f"features={X.shape[1]}, cv={cv}"
        )

        try:
            tpot = TPOTClassifier(
                search_space="linear",
                scorers=["balanced_accuracy"],
                cv=cv,
                max_time_mins=MAX_TIME_MINS,
                max_eval_time_mins=MAX_EVAL_TIME_MINS,
                n_jobs=1,
                random_state=RANDOM_STATE,
                verbose=0,
            )
            tpot.fit(X, y)

            pipeline = tpot.fitted_pipeline_
            cv_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring="balanced_accuracy",
            )

            row["tpot_best_balanced_accuracy"] = best_score_value(tpot)
            row["cv_balanced_accuracy_mean"] = float(cv_scores.mean())
            row["cv_balanced_accuracy_std"] = float(cv_scores.std())
            row["final_estimator"] = final_estimator_name(pipeline)
            row["pipeline"] = repr(pipeline)

            details["targets"][target] = {
                **row,
                "feature_columns": list(X.columns),
            }

            print(
                f"[ok] {target}: estimator={row['final_estimator']}, "
                f"cv_bal_acc={row['cv_balanced_accuracy_mean']:.3f}"
            )

        except Exception as exc:
            row["status"] = "error"
            row["pipeline"] = f"{exc.__class__.__name__}: {exc}"
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
        "final_estimator",
    ]].to_string(index=False))
    print()
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DETAILS_JSON}")


if __name__ == "__main__":
    main()
