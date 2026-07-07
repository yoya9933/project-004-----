from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISSUE_CSV = PROJECT_ROOT / "05_測試與驗證" / "test_issue_features.csv"
DEFAULT_MONEY_CSV = PROJECT_ROOT / "03_研究與分析" / "money_feature_amounts_wide.csv"
DEFAULT_JSON_DIR = PROJECT_ROOT / "05_測試與驗證" / "test"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "06_交付物" / "penalty_cases.csv"
DEFAULT_DATA_DICTIONARY = PROJECT_ROOT / "06_交付物" / "penalty_cases_data_dictionary.csv"

MANUAL_NUMERIC_FIELDS = [
    "contract_price",
    "delay_days",
    "penalty_rate_per_day",
    "claimed_penalty",
    "allowed_penalty",
]

DERIVED_FIELDS = [
    "penalty_to_contract",
    "remaining_ratio",
    "reduction_rate",
    "is_reduced",
]

RECOMMENDED_ISSUE_FIELDS = [
    "issue_actual_damage_unclear",
    "issue_partial_completion",
    "issue_owner_fault",
    "issue_contractor_fault",
    "issue_extension_request",
    "issue_used_by_owner",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_jid(jid: str) -> dict[str, Any]:
    parts = str(jid).split(",")
    padded = parts + [""] * (6 - len(parts))
    court_code, roc_year, case_type, case_number, decision_date_raw, sequence = padded[:6]
    decision_date = ""
    decision_year = ""
    if re.fullmatch(r"\d{8}", decision_date_raw or ""):
        decision_date = (
            f"{decision_date_raw[:4]}-{decision_date_raw[4:6]}-{decision_date_raw[6:8]}"
        )
        decision_year = decision_date_raw[:4]

    case_no = ""
    if roc_year and case_type and case_number:
        case_no = f"{roc_year}年度{case_type}字第{case_number}號"

    return {
        "case_id": str(jid).replace(",", "_"),
        "court_code": court_code,
        "case_year_roc": roc_year,
        "decision_year": decision_year,
        "case_type": case_type,
        "case_number": case_number,
        "case_sequence": sequence,
        "decision_date": decision_date,
        "case_no": case_no,
    }


def infer_court_name(jfull: str) -> str:
    if not isinstance(jfull, str) or not jfull.strip():
        return ""
    first_line = jfull.splitlines()[0].strip()
    first_line = re.sub(r"\s+", "", first_line)
    first_line = re.sub(r"(民事|刑事|行政)?判決.*$", "", first_line)
    return first_line


def load_json_info(json_dir: Path, file_name: str) -> dict[str, str]:
    path = json_dir / file_name
    if not path.exists():
        return {"court": "", "json_file": "", "judgment_text_available": "0"}

    data = json.loads(path.read_text(encoding="utf-8"))
    court = infer_court_name(str(data.get("JFULL", "")))
    return {
        "court": court,
        "json_file": str(path.relative_to(PROJECT_ROOT)),
        "judgment_text_available": "1" if data.get("JFULL") else "0",
    }


def load_money_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    money = read_csv(path)
    keep_cols = [
        col
        for col in [
            "jid",
            "issue_delay_or_penalty",
            "issue_payment",
            "issue_acceptance_or_settlement",
        ]
        if col in money.columns
    ]
    if not keep_cols:
        return pd.DataFrame()
    money = money[keep_cols].copy()
    rename_map = {
        "issue_delay_or_penalty": "amount_candidates_delay_or_penalty",
        "issue_payment": "amount_candidates_payment",
        "issue_acceptance_or_settlement": "amount_candidates_acceptance_or_settlement",
    }
    return money.rename(columns=rename_map)


def build_rows(issue_df: pd.DataFrame, json_dir: Path, money_df: pd.DataFrame) -> pd.DataFrame:
    issue_cols = [col for col in issue_df.columns if col.startswith("issue_")]
    rows: list[dict[str, Any]] = []

    money_by_jid: dict[str, dict[str, Any]] = {}
    if not money_df.empty and "jid" in money_df.columns:
        money_by_jid = money_df.set_index("jid").to_dict(orient="index")

    for _, row in issue_df.iterrows():
        jid = str(row["jid"])
        file_name = str(row.get("file", ""))
        record: dict[str, Any] = {
            **parse_jid(jid),
            **load_json_info(json_dir, file_name),
            "jid": jid,
            "source_file": file_name,
            "title": row.get("title", ""),
            "project_type": "",
            "key_reason": "",
            "manual_note": "",
            "annotation_status": "needs_manual_review",
        }

        for field in MANUAL_NUMERIC_FIELDS + DERIVED_FIELDS:
            record[field] = pd.NA

        for field in RECOMMENDED_ISSUE_FIELDS:
            record[field] = pd.NA

        for col in issue_cols:
            record[col] = row[col]

        for col, value in money_by_jid.get(jid, {}).items():
            record[col] = value

        rows.append(record)

    preferred_order = [
        "case_id",
        "court",
        "court_code",
        "case_year_roc",
        "decision_year",
        "case_no",
        "case_type",
        "case_number",
        "case_sequence",
        "decision_date",
        "title",
        "project_type",
        *MANUAL_NUMERIC_FIELDS,
        *DERIVED_FIELDS,
        *RECOMMENDED_ISSUE_FIELDS,
        *issue_cols,
        "key_reason",
        "manual_note",
        "annotation_status",
        "jid",
        "source_file",
        "json_file",
        "judgment_text_available",
        "amount_candidates_delay_or_penalty",
        "amount_candidates_payment",
        "amount_candidates_acceptance_or_settlement",
    ]

    df = pd.DataFrame(rows)
    ordered = [col for col in preferred_order if col in df.columns]
    extras = [col for col in df.columns if col not in ordered]
    return df[ordered + extras]


def write_data_dictionary(path: Path, issue_cols: list[str]) -> None:
    rows = [
        ("case_id", "文字", "自訂案件代號，由 jid 轉換而來"),
        ("court", "文字", "法院名稱，優先由 JSON 判決全文第一行推定"),
        ("case_year_roc", "文字", "案號年度，民國年"),
        ("decision_year", "文字", "裁判西元年"),
        ("case_no", "文字", "案號文字"),
        ("project_type", "類別", "人工標註：建築、道路、橋梁、機電、水利等"),
        ("contract_price", "數字", "人工標註：契約總價"),
        ("delay_days", "數字", "人工標註：法院認定逾期天數"),
        ("penalty_rate_per_day", "數字", "人工標註：每日違約金比例，例如 0.001"),
        ("claimed_penalty", "數字", "人工標註：業主主張或扣罰違約金"),
        ("allowed_penalty", "數字", "人工標註：法院最後准許違約金"),
        ("penalty_to_contract", "數字", "衍生欄位：claimed_penalty / contract_price"),
        ("remaining_ratio", "數字", "衍生欄位：allowed_penalty / claimed_penalty"),
        ("reduction_rate", "數字", "衍生欄位：1 - remaining_ratio"),
        ("is_reduced", "0/1", "衍生欄位：allowed_penalty < claimed_penalty"),
        ("key_reason", "文字", "人工標註：法院核心理由摘要"),
        ("amount_candidates_delay_or_penalty", "文字", "既有金額候選，僅供人工標註參考"),
    ]
    rows.extend((field, "0/1", "建議新增人工爭點標註欄位") for field in RECOMMENDED_ISSUE_FIELDS)
    rows.extend((field, "0/1", "既有 TPOT 爭點標註欄位") for field in issue_cols)
    dictionary = pd.DataFrame(rows, columns=["field", "type", "description"])
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a penalty_cases.csv annotation starter from issue labels."
    )
    parser.add_argument("--issue-csv", type=Path, default=DEFAULT_ISSUE_CSV)
    parser.add_argument("--money-csv", type=Path, default=DEFAULT_MONEY_CSV)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--data-dictionary", type=Path, default=DEFAULT_DATA_DICTIONARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issue_df = read_csv(args.issue_csv)
    money_df = load_money_candidates(args.money_csv)
    output = build_rows(issue_df, args.json_dir, money_df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    issue_cols = [col for col in issue_df.columns if col.startswith("issue_")]
    write_data_dictionary(args.data_dictionary, issue_cols)

    print(f"Wrote {len(output)} rows to {args.output_csv}")
    print(f"Wrote data dictionary to {args.data_dictionary}")


if __name__ == "__main__":
    main()
