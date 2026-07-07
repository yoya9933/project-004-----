from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_ROOT = (
    PROJECT_ROOT
    / "02_輸入資料"
    / "法律課程資料庫-20260707T080337Z-3-001"
    / "法律課程資料庫"
)
DEFAULT_INDEX_CSV = DATABASE_ROOT / "2021-2026判決總表.csv"
DEFAULT_JSON_ROOT = DATABASE_ROOT / "依年份分類"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "06_交付物" / "keyword_screening"


@dataclass(frozen=True)
class KeywordPattern:
    label: str
    pattern: re.Pattern[str]


PENALTY_PATTERNS = [
    KeywordPattern("違約金", re.compile("違約金")),
    KeywordPattern("逾期違約金", re.compile("逾期違約金")),
]

DELAY_PATTERNS = [
    KeywordPattern("逾期", re.compile("逾期")),
    KeywordPattern("遲延", re.compile("遲延")),
    KeywordPattern("工期", re.compile("工期")),
    KeywordPattern("展延", re.compile("展延")),
    KeywordPattern("展期", re.compile("展期")),
    KeywordPattern("完工期限", re.compile("完工期限")),
]

REDUCTION_PATTERNS = [
    KeywordPattern("酌減", re.compile("酌減")),
    KeywordPattern("過高", re.compile("過高")),
    KeywordPattern("相當", re.compile("相當")),
    KeywordPattern("民法第252條", re.compile(r"民法第\s*252\s*條")),
    KeywordPattern("民法第２５２條", re.compile("民法第\\s*２５２\\s*條")),
    KeywordPattern("民法第二百五十二條", re.compile("民法第?二百五十二條")),
    KeywordPattern("第252條", re.compile(r"第\s*252\s*條")),
]

STRONG_REDUCTION_PATTERNS = [
    KeywordPattern("酌減", re.compile("酌減")),
    KeywordPattern("過高", re.compile("過高")),
    KeywordPattern("民法第252條", re.compile(r"民法第\s*252\s*條")),
    KeywordPattern("民法第２５２條", re.compile("民法第\\s*２５２\\s*條")),
    KeywordPattern("民法第二百五十二條", re.compile("民法第?二百五十二條")),
    KeywordPattern("第252條", re.compile(r"第\s*252\s*條")),
]


def read_index(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def read_judgment_text(json_path: Path) -> tuple[str, bool, str]:
    if not json_path.exists():
        return "", False, "json_missing"
    try:
        data = json.loads(json_path.read_text(encoding="utf-8-sig", errors="replace"))
    except json.JSONDecodeError:
        return "", False, "json_decode_error"
    text = data.get("JFULL", "")
    return str(text or ""), bool(text), "ok"


def normalize_for_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def match_patterns(text: str, patterns: Iterable[KeywordPattern]) -> tuple[int, list[str]]:
    total = 0
    labels: list[str] = []
    for keyword in patterns:
        count = len(keyword.pattern.findall(text))
        if count:
            total += count
            labels.append(keyword.label)
    return total, labels


def first_match_span(text: str, patterns: Iterable[KeywordPattern]) -> tuple[int, int] | None:
    spans: list[tuple[int, int]] = []
    for keyword in patterns:
        match = keyword.pattern.search(text)
        if match:
            spans.append(match.span())
    if not spans:
        return None
    return min(spans, key=lambda span: span[0])


def extract_snippet(text: str, patterns: Iterable[KeywordPattern], radius: int = 90) -> str:
    span = first_match_span(text, patterns)
    if not span:
        return ""
    start = max(span[0] - radius, 0)
    end = min(span[1] + radius, len(text))
    return normalize_for_snippet(text[start:end])


def parse_decision_year(jdate: object, month: object) -> str:
    raw_date = str(jdate or "")
    if re.fullmatch(r"\d{8}", raw_date):
        return raw_date[:4]
    raw_month = str(month or "")
    if re.fullmatch(r"\d{6}", raw_month):
        return raw_month[:4]
    return ""


def build_screening_frame(index_df: pd.DataFrame, json_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for _, row in index_df.iterrows():
        src_path = Path(str(row.get("src_path", "")))
        json_path = json_root / src_path
        text, has_text, read_status = read_judgment_text(json_path)
        title = str(row.get("JTITLE", "") or "")
        searchable = f"{title}\n{text}"

        penalty_count, penalty_terms = match_patterns(searchable, PENALTY_PATTERNS)
        delay_count, delay_terms = match_patterns(searchable, DELAY_PATTERNS)
        reduction_count, reduction_terms = match_patterns(searchable, REDUCTION_PATTERNS)
        strong_reduction_count, strong_reduction_terms = match_patterns(
            searchable, STRONG_REDUCTION_PATTERNS
        )

        has_penalty = penalty_count > 0
        has_delay = delay_count > 0
        has_strong_reduction = strong_reduction_count > 0
        overdue_candidate = has_penalty and has_delay
        high_relevance = overdue_candidate and has_strong_reduction

        title_penalty_count, _ = match_patterns(title, PENALTY_PATTERNS)
        title_delay_count, _ = match_patterns(title, DELAY_PATTERNS)
        title_reduction_count, _ = match_patterns(title, REDUCTION_PATTERNS)

        first_layer_score = penalty_count + delay_count
        second_layer_score = reduction_count
        strong_reduction_score = strong_reduction_count
        relevance_score = (
            first_layer_score
            + 2 * second_layer_score
            + 4 * strong_reduction_score
            + 3 * title_penalty_count
            + title_delay_count
            + title_reduction_count
        )

        first_layer_terms = sorted(set(penalty_terms + delay_terms))
        records.append(
            {
                "JID": row.get("JID", ""),
                "month": row.get("month", ""),
                "decision_year": parse_decision_year(row.get("JDATE", ""), row.get("month", "")),
                "court": row.get("court", ""),
                "JYEAR": row.get("JYEAR", ""),
                "JCASE": row.get("JCASE", ""),
                "JNO": row.get("JNO", ""),
                "JDATE": row.get("JDATE", ""),
                "JTITLE": title,
                "JFULL_len": row.get("JFULL_len", ""),
                "工程次數": row.get("工程次數", ""),
                "src_path": str(src_path).replace("\\", "/"),
                "json_file": str(json_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if json_path.exists()
                else "",
                "read_status": read_status,
                "judgment_text_available": int(has_text),
                "penalty_keyword_count": penalty_count,
                "delay_keyword_count": delay_count,
                "reduction_keyword_count": reduction_count,
                "strong_reduction_keyword_count": strong_reduction_count,
                "first_layer_score": first_layer_score,
                "second_layer_score": second_layer_score,
                "strong_reduction_score": strong_reduction_score,
                "relevance_score": relevance_score,
                "matched_penalty_terms": ";".join(sorted(set(penalty_terms))),
                "matched_delay_terms": ";".join(sorted(set(delay_terms))),
                "matched_reduction_terms": ";".join(sorted(set(reduction_terms))),
                "matched_strong_reduction_terms": ";".join(
                    sorted(set(strong_reduction_terms))
                ),
                "matched_first_layer_terms": ";".join(first_layer_terms),
                "is_overdue_penalty_candidate": int(overdue_candidate),
                "is_penalty_reduction_high_relevance": int(high_relevance),
                "first_layer_snippet": extract_snippet(
                    searchable, [*PENALTY_PATTERNS, *DELAY_PATTERNS]
                ),
                "reduction_snippet": extract_snippet(searchable, REDUCTION_PATTERNS),
            }
        )

    return pd.DataFrame(records)


def stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int,
    strata_col: str = "decision_year",
) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()

    rng_seed = random_state
    samples: list[pd.DataFrame] = []
    remaining_indices: set[int] = set(df.index)
    grouped = list(df.groupby(strata_col, dropna=False))

    for value, group in grouped:
        proportion = len(group) / len(df)
        n = max(1, math.floor(sample_size * proportion))
        n = min(n, len(group))
        part = group.sample(n=n, random_state=rng_seed + len(samples))
        samples.append(part)
        remaining_indices -= set(part.index)

    sampled = pd.concat(samples, ignore_index=False) if samples else pd.DataFrame()
    if len(sampled) < sample_size and remaining_indices:
        remaining = df.loc[sorted(remaining_indices)]
        fill = remaining.sample(
            n=min(sample_size - len(sampled), len(remaining)),
            random_state=random_state + 999,
        )
        sampled = pd.concat([sampled, fill], ignore_index=False)

    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state + 1999)

    return sampled.copy()


def write_outputs(
    screening: pd.DataFrame,
    output_dir: Path,
    sample_size: int,
    random_state: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    overdue = screening[screening["is_overdue_penalty_candidate"].eq(1)].copy()
    high = screening[screening["is_penalty_reduction_high_relevance"].eq(1)].copy()

    sort_cols = ["decision_year", "month", "court", "JID"]
    overdue = overdue.sort_values(sort_cols, kind="stable")
    high = high.sort_values(["relevance_score", *sort_cols], ascending=[False, True, True, True, True])
    sample = stratified_sample(high, sample_size=sample_size, random_state=random_state)
    sample = sample.sort_values(["decision_year", "month", "relevance_score"], ascending=[True, True, False])

    sample = sample.copy()
    sample.insert(0, "annotation_status", "needs_manual_review")
    sample.insert(1, "annotation_priority", range(1, len(sample) + 1))

    paths = {
        "all_screening": output_dir / "all_keyword_screening.csv",
        "overdue_candidates": output_dir / "overdue_penalty_candidates.csv",
        "high_relevance": output_dir / "penalty_reduction_high_relevance.csv",
        "annotation_sample": output_dir / "penalty_reduction_annotation_sample.csv",
        "summary": output_dir / "keyword_screening_summary.md",
    }

    screening.to_csv(paths["all_screening"], index=False, encoding="utf-8-sig")
    overdue.to_csv(paths["overdue_candidates"], index=False, encoding="utf-8-sig")
    high.to_csv(paths["high_relevance"], index=False, encoding="utf-8-sig")
    sample.to_csv(paths["annotation_sample"], index=False, encoding="utf-8-sig")
    write_summary(paths["summary"], screening, overdue, high, sample, sample_size)

    return paths


def year_counts(df: pd.DataFrame) -> str:
    if df.empty:
        return "無"
    counts = df.groupby("decision_year").size().reset_index(name="count")
    return "\n".join(f"- {row.decision_year}: {row['count']}" for _, row in counts.iterrows())


def write_summary(
    path: Path,
    screening: pd.DataFrame,
    overdue: pd.DataFrame,
    high: pd.DataFrame,
    sample: pd.DataFrame,
    requested_sample_size: int,
) -> None:
    lines = [
        "# 關鍵詞篩選摘要",
        "",
        "## 篩選規則",
        "",
        "- 母體：2021-2026 年工程相關判決 JSON 全文。",
        "- 第一層逾期違約金候選池：同時命中違約金語彙與逾期/工期/展延語彙。",
        "- 第二層違約金酌減高度相關池：第一層候選池中，再命中酌減、民法第252條、過高等較強酌減語彙；相當保留為輔助命中詞，但不單獨使案件進入高度相關池。",
        "- 抽樣精標清單：自第二層高度相關池依裁判年度分層抽樣。",
        "",
        "## 輸出筆數",
        "",
        f"- 母體總筆數：{len(screening)}",
        f"- 逾期違約金候選池：{len(overdue)}",
        f"- 違約金酌減高度相關池：{len(high)}",
        f"- 抽樣精標清單：{len(sample)} / 目標 {requested_sample_size}",
        "",
        "## 高度相關池年度分布",
        "",
        year_counts(high),
        "",
        "## 抽樣精標清單年度分布",
        "",
        year_counts(sample),
        "",
        "## 注意事項",
        "",
        "- 本篩選只建立候選池，不代表案件必然以逾期違約金酌減為核心爭點。",
        "- `相當` 屬於較寬鬆語彙，本腳本保留其命中紀錄，但高度相關池不接受只命中 `相當` 的案件。",
        "- 金額、逾期天數、酌減比例與法院核心理由不得只依關鍵詞自動判定，仍需回到判決全文查核。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build keyword candidate pools for overdue penalty reduction cases."
    )
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_csv)
    screening = build_screening_frame(index_df, args.json_root)
    paths = write_outputs(
        screening,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    print(f"母體總筆數: {len(screening)}")
    print(
        "逾期違約金候選池: "
        f"{int(screening['is_overdue_penalty_candidate'].sum())}"
    )
    print(
        "違約金酌減高度相關池: "
        f"{int(screening['is_penalty_reduction_high_relevance'].sum())}"
    )
    for label, output_path in paths.items():
        print(f"{label}: {output_path}")


if __name__ == "__main__":
    main()
