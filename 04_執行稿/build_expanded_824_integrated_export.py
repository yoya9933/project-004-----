from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_PATH = (
    PROJECT_ROOT
    / "06_交付物"
    / "reduction_ratio_model_expanded_824"
    / "ratio_model_frame.csv"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "06_交付物"
    / "reduction_ratio_model_expanded_824_sklearn"
    / "predictions.csv"
)
FINAL_AMOUNTS_PATH = (
    PROJECT_ROOT / "06_交付物" / "final_judgment_amounts" / "final_judgment_amounts.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "06_交付物"
    / "reduction_ratio_model_expanded_824_sklearn"
    / "integrated_cases_824.csv"
)

MODEL_ORDER = [
    "mean_baseline",
    "ridge_regression_l2",
    "lasso_regression_l1",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def predicted_amount(claimed_penalty: str, predicted_ratio: str) -> str:
    if not claimed_penalty.strip() or not predicted_ratio.strip():
        return ""
    return f"{float(claimed_penalty) * float(predicted_ratio):.6f}"


def main() -> None:
    frame_rows = read_csv(FRAME_PATH)
    prediction_rows = read_csv(PREDICTIONS_PATH)

    predictions: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in prediction_rows:
        predictions[row["JID"]][row["model"]] = row

    final_amounts: dict[str, dict[str, str]] = {}
    if FINAL_AMOUNTS_PATH.exists():
        final_amounts = {row["JID"]: row for row in read_csv(FINAL_AMOUNTS_PATH)}

    base_columns = list(frame_rows[0])
    final_columns = [
        "final_judgment_extraction_status",
        "final_judgment_amount",
        "principal_award_total",
        "final_judgment_amount_text",
    ]
    model_columns = [
        f"{model}_{suffix}"
        for model in MODEL_ORDER
        for suffix in (
            "predicted_remaining_ratio",
            "predicted_reduction_rate",
            "predicted_allowed_penalty",
        )
    ]

    output_rows: list[dict[str, str]] = []
    for frame in frame_rows:
        jid = frame["JID"]
        output = dict(frame)
        final = final_amounts.get(jid, {})
        output.update(
            {
                "final_judgment_extraction_status": final.get("extraction_status", ""),
                "final_judgment_amount": final.get("final_judgment_amount", ""),
                "principal_award_total": final.get("principal_award_total", ""),
                "final_judgment_amount_text": final.get("final_judgment_amount_text", ""),
            }
        )

        for model in MODEL_ORDER:
            prediction = predictions.get(jid, {}).get(model, {})
            ratio = prediction.get("predicted_remaining_ratio", "")
            output[f"{model}_predicted_remaining_ratio"] = ratio
            output[f"{model}_predicted_reduction_rate"] = prediction.get(
                "predicted_reduction_rate", ""
            )
            output[f"{model}_predicted_allowed_penalty"] = predicted_amount(
                frame.get("claimed_penalty", ""), ratio
            )
        output_rows.append(output)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=base_columns + final_columns + model_columns)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"output={OUTPUT_PATH}")
    print(f"rows={len(output_rows)}")
    print(f"columns={len(base_columns) + len(final_columns) + len(model_columns)}")
    print(f"cases_with_predictions={len(predictions)}")
    print(f"cases_with_final_judgment_record={len(final_amounts)}")


if __name__ == "__main__":
    main()
