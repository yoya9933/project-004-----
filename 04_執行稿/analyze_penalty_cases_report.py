from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "06_交付物" / "penalty_cases.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "06_交付物" / "analysis_report_version"

NUMERIC_FIELDS = [
    "contract_price",
    "delay_days",
    "penalty_rate_per_day",
    "claimed_penalty",
    "allowed_penalty",
]

REQUIRED_FOR_RATIOS = ["contract_price", "claimed_penalty", "allowed_penalty"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def to_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    for field in NUMERIC_FIELDS:
        if field not in df.columns:
            df[field] = pd.NA
        df[field] = to_number(df[field])

    complete = (
        df["contract_price"].notna()
        & df["contract_price"].gt(0)
        & df["claimed_penalty"].notna()
        & df["claimed_penalty"].gt(0)
        & df["allowed_penalty"].notna()
        & df["allowed_penalty"].ge(0)
    )

    df.loc[complete, "penalty_to_contract"] = (
        df.loc[complete, "claimed_penalty"] / df.loc[complete, "contract_price"]
    )
    df.loc[complete, "remaining_ratio"] = (
        df.loc[complete, "allowed_penalty"] / df.loc[complete, "claimed_penalty"]
    )
    df.loc[complete, "reduction_rate"] = 1 - df.loc[complete, "remaining_ratio"]
    df.loc[complete, "is_reduced"] = (
        df.loc[complete, "allowed_penalty"] < df.loc[complete, "claimed_penalty"]
    ).astype(int)

    return df, complete


def configure_plot_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def write_missingness(df: pd.DataFrame, output_dir: Path, complete: pd.Series) -> pd.DataFrame:
    rows = []
    for field in NUMERIC_FIELDS:
        missing = int(df[field].isna().sum())
        rows.append(
            {
                "field": field,
                "missing_rows": missing,
                "filled_rows": int(df[field].notna().sum()),
                "missing_rate": missing / len(df) if len(df) else math.nan,
            }
        )
    rows.append(
        {
            "field": "complete_ratio_fields",
            "missing_rows": int((~complete).sum()),
            "filled_rows": int(complete.sum()),
            "missing_rate": int((~complete).sum()) / len(df) if len(df) else math.nan,
        }
    )
    missingness = pd.DataFrame(rows)
    missingness.to_csv(output_dir / "penalty_cases_missingness.csv", index=False, encoding="utf-8-sig")
    return missingness


def write_group_summary(df: pd.DataFrame, output_dir: Path, complete: pd.Series) -> pd.DataFrame:
    working = df.loc[complete].copy()
    if working.empty:
        summary = pd.DataFrame(
            columns=[
                "penalty_ratio_group",
                "cases",
                "reduced_cases",
                "reduction_case_rate",
                "avg_remaining_ratio",
                "avg_reduction_rate",
            ]
        )
    else:
        bins = [0, 0.05, 0.10, 0.20, float("inf")]
        labels = ["0-5%", "5-10%", "10-20%", "20%以上"]
        working["penalty_ratio_group"] = pd.cut(
            working["penalty_to_contract"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )
        summary = (
            working.groupby("penalty_ratio_group", observed=False)
            .agg(
                cases=("case_id", "count"),
                reduced_cases=("is_reduced", "sum"),
                reduction_case_rate=("is_reduced", "mean"),
                avg_remaining_ratio=("remaining_ratio", "mean"),
                avg_reduction_rate=("reduction_rate", "mean"),
            )
            .reset_index()
        )
        for col in ["reduction_case_rate", "avg_remaining_ratio", "avg_reduction_rate"]:
            summary[col] = summary[col] * 100

    summary.to_csv(output_dir / "penalty_ratio_group_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def issue_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if not col.startswith("issue_"):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isin([0, 1]).any():
            cols.append(col)
            df[col] = numeric
    return cols


def write_issue_summary(df: pd.DataFrame, output_dir: Path, complete: pd.Series) -> pd.DataFrame:
    cols = issue_columns(df)
    rows = []
    working = df.loc[complete].copy()
    for col in cols:
        present = working[pd.to_numeric(working[col], errors="coerce").eq(1)]
        absent = working[pd.to_numeric(working[col], errors="coerce").eq(0)]
        rows.append(
            {
                "issue": col,
                "present_cases": len(present),
                "present_reduction_case_rate": present["is_reduced"].mean() * 100
                if len(present)
                else pd.NA,
                "present_avg_remaining_ratio": present["remaining_ratio"].mean() * 100
                if len(present)
                else pd.NA,
                "absent_cases": len(absent),
                "absent_reduction_case_rate": absent["is_reduced"].mean() * 100
                if len(absent)
                else pd.NA,
                "absent_avg_remaining_ratio": absent["remaining_ratio"].mean() * 100
                if len(absent)
                else pd.NA,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "issue_reduction_comparison.csv", index=False, encoding="utf-8-sig")
    return summary


def write_scatter(df: pd.DataFrame, output_dir: Path, complete: pd.Series) -> Path | None:
    working = df.loc[complete & df["delay_days"].notna()].copy()
    if working.empty:
        return None

    configure_plot_fonts()
    output_path = output_dir / "delay_days_vs_remaining_ratio.png"
    sizes = (working["penalty_to_contract"] * 1000).clip(lower=30, upper=400)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        working["delay_days"],
        working["remaining_ratio"] * 100,
        s=sizes,
        alpha=0.72,
        edgecolors="#333333",
        linewidths=0.4,
    )
    for _, row in working.iterrows():
        plt.text(
            row["delay_days"],
            row["remaining_ratio"] * 100,
            str(row.get("case_id", "")),
            fontsize=7,
            alpha=0.85,
        )
    plt.xlabel("逾期天數")
    plt.ylabel("酌減後比例（%）")
    plt.title("逾期天數與法院酌減後比例之關係")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.1f}%"


def markdown_table(df: pd.DataFrame, columns: Iterable[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_尚無可輸出資料。_"
    limited = df.loc[:, list(columns)].head(max_rows).copy()
    return limited.to_markdown(index=False)


def write_markdown_report(
    df: pd.DataFrame,
    output_dir: Path,
    complete: pd.Series,
    missingness: pd.DataFrame,
    group_summary: pd.DataFrame,
    issue_summary: pd.DataFrame,
    scatter_path: Path | None,
) -> Path:
    complete_count = int(complete.sum())
    reduced_count = int(df.loc[complete, "is_reduced"].sum()) if complete_count else 0
    avg_remaining = df.loc[complete, "remaining_ratio"].mean() * 100 if complete_count else pd.NA
    avg_reduction = df.loc[complete, "reduction_rate"].mean() * 100 if complete_count else pd.NA

    lines = [
        "# 工程逾期違約金酌減因素探索分析",
        "",
        "## 定位",
        "",
        "本報告定位為判決資料標註、探索性分析與線性模型前置資料整理，不作為法院判決預測。",
        "",
        "## 資料狀態",
        "",
        f"- 總列數：{len(df)}",
        f"- 已具備契約總價、主張違約金、法院准許金額的列數：{complete_count}",
        f"- 已可判斷酌減列數：{complete_count}",
        f"- 酌減件數：{reduced_count}",
        f"- 平均酌減後比例：{percent(avg_remaining)}",
        f"- 平均酌減幅度：{percent(avg_reduction)}",
        "",
        "## 缺值檢查",
        "",
        markdown_table(missingness, ["field", "missing_rows", "filled_rows", "missing_rate"]),
        "",
        "## 違約金占契約總價分組",
        "",
        markdown_table(
            group_summary,
            [
                "penalty_ratio_group",
                "cases",
                "reduced_cases",
                "reduction_case_rate",
                "avg_remaining_ratio",
                "avg_reduction_rate",
            ],
        ),
        "",
        "## 爭點比較",
        "",
        markdown_table(
            issue_summary,
            [
                "issue",
                "present_cases",
                "present_reduction_case_rate",
                "present_avg_remaining_ratio",
                "absent_cases",
                "absent_reduction_case_rate",
            ],
        ),
        "",
        "## 圖表",
        "",
        f"- 散佈圖：`{scatter_path.name}`" if scatter_path else "- 散佈圖：尚未產生，需先補齊 `delay_days` 與金額欄位。",
        "",
        "## 研究限制",
        "",
        "- 樣本數有限，且判決事實差異大。",
        "- 金額候選清單只能輔助人工標註，不能直接視為契約總價、主張違約金或法院准許金額。",
        "- 線性模型與係數僅代表本樣本中的統計關聯，不代表法律因果關係。",
        "- 若 `contract_price`、`claimed_penalty`、`allowed_penalty` 或 `delay_days` 未補齊，圖表與統計表會保持空白或只輸出缺值檢查。",
    ]
    report_path = output_dir / "工程逾期違約金酌減因素探索分析.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the report-version penalty analysis outputs.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csv(args.input_csv)
    cleaned, complete = prepare_dataframe(raw)
    cleaned.to_csv(args.output_dir / "penalty_cases_cleaned.csv", index=False, encoding="utf-8-sig")

    missingness = write_missingness(cleaned, args.output_dir, complete)
    group_summary = write_group_summary(cleaned, args.output_dir, complete)
    issue_summary = write_issue_summary(cleaned, args.output_dir, complete)
    scatter_path = write_scatter(cleaned, args.output_dir, complete)
    report_path = write_markdown_report(
        cleaned,
        args.output_dir,
        complete,
        missingness,
        group_summary,
        issue_summary,
        scatter_path,
    )

    print(f"Wrote cleaned data to {args.output_dir / 'penalty_cases_cleaned.csv'}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
