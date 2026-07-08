from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_CSV = (
    PROJECT_ROOT
    / "06_交付物"
    / "keyword_screening"
    / "penalty_reduction_annotation_sample.csv"
)
DEFAULT_POOL_CSV = (
    PROJECT_ROOT
    / "06_交付物"
    / "keyword_screening"
    / "penalty_reduction_high_relevance.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "06_交付物" / "ai_rag_annotation"

STRONG_REDUCTION_PATTERN = re.compile(
    r"酌減|過高|民法第\s*252\s*條|民法第\s*２５２\s*條|民法第?二百五十二條|第\s*252\s*條"
)
PENALTY_PATTERN = re.compile(r"逾期違約金|違約金")
DELAY_PATTERN = re.compile(r"展延工期|工期展延|展延|展期|逾期|遲延|工期|完工期限")
AMOUNT_PATTERN = re.compile(
    r"(?:新[臺台]幣|NT\$|新台幣)?\s*([0-9][0-9,，]{2,})\s*(?:元|圓)"
)
DAY_PATTERN = re.compile(r"([0-9]{1,5})\s*(?:日|天)")

CONTRACT_TERMS = ["契約總價", "契約價金", "契約金額", "工程總價", "總價", "工程款"]
CLAIMED_PENALTY_TERMS = ["主張", "請求", "扣罰", "逾期違約金", "違約金", "沒收"]
ALLOWED_PENALTY_TERMS = ["准許", "應給付", "酌減為", "核減為", "減為", "判命", "得請求"]
DELAY_TERMS = ["逾期", "遲延", "工期", "展延", "展期", "完工期限"]

ISSUE_PATTERNS = {
    "issue_owner_fault": [
        "業主可歸責",
        "定作人可歸責",
        "機關可歸責",
        "可歸責於被告",
        "變更設計",
        "未協力",
        "遲延交付",
        "遲延審查",
        "停工",
    ],
    "issue_contractor_fault": [
        "承攬人可歸責",
        "承包商可歸責",
        "廠商可歸責",
        "可歸責於原告",
        "未依約",
        "施工遲延",
        "逾期完工",
    ],
    "issue_extension_request": [
        "展延",
        "展期",
        "工期展延",
        "展延工期",
        "申請展延",
        "核定展延",
    ],
    "issue_actual_damage_unclear": [
        "實際損害",
        "損害甚微",
        "未舉證損害",
        "無損害",
        "損害額",
        "所受損害",
    ],
    "issue_partial_completion": [
        "部分完工",
        "已完成",
        "完工",
        "驗收",
        "竣工",
        "結算",
    ],
    "issue_used_by_owner": [
        "已使用",
        "實際使用",
        "開始使用",
        "啟用",
        "受領",
        "占有使用",
    ],
}

ISSUE_DESCRIPTIONS = {
    "issue_owner_fault": "是否涉及業主/機關/定作人可歸責、變更設計、未協力或遲延審查等因素",
    "issue_contractor_fault": "是否涉及承攬人/承包商/廠商可歸責或施工遲延等因素",
    "issue_extension_request": "是否涉及展延、展期、工期展延或核定展延",
    "issue_actual_damage_unclear": "是否涉及實際損害、損害甚微、未舉證損害或損害額不明",
    "issue_partial_completion": "是否涉及部分完工、完工、竣工、驗收或結算",
    "issue_used_by_owner": "是否涉及業主已使用、實際使用、啟用、受領或占有使用",
}

RAG_VOCAB = [
    "逾期違約金",
    "違約金",
    "酌減",
    "過高",
    "民法第252條",
    "工期",
    "展延",
    "展期",
    "逾期",
    "遲延",
    "完工期限",
    "變更設計",
    "追加工程",
    "契約變更",
    "部分完工",
    "已使用",
    "驗收",
    "實際損害",
    "損害甚微",
    "履約保證金",
    "保留款",
    "公共工程",
    "政府採購",
    "承攬",
    "工程款",
    "可歸責",
]

MANUAL_FIELDS = [
    "contract_price",
    "delay_days",
    "claimed_penalty",
    "allowed_penalty",
    "is_reduced",
    "remaining_ratio",
    "reduction_rate",
    "issue_owner_fault",
    "issue_contractor_fault",
    "issue_extension_request",
    "issue_actual_damage_unclear",
    "issue_partial_completion",
    "issue_used_by_owner",
    "key_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_judgment_text(row: dict[str, str]) -> str:
    json_file = row.get("json_file", "")
    candidates: list[Path] = []
    if json_file:
        candidates.append(PROJECT_ROOT / json_file)
    src_path = row.get("src_path", "")
    if src_path:
        candidates.append(
            PROJECT_ROOT
            / "02_輸入資料"
            / "法律課程資料庫-20260707T080337Z-3-001"
            / "法律課程資料庫"
            / "依年份分類"
            / src_path
        )

    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
        except json.JSONDecodeError:
            continue
        return str(data.get("JFULL", "") or "")
    return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def snippet_around_pattern(text: str, pattern: re.Pattern[str], radius: int = 180) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return normalize_text(text[start:end])


def snippets_around_all(text: str, patterns: Iterable[re.Pattern[str]], radius: int = 160, limit: int = 4) -> str:
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
    if not spans:
        return ""
    spans = sorted(spans, key=lambda item: item[0])
    snippets: list[str] = []
    used_starts: list[int] = []
    for start_pos, end_pos in spans:
        if any(abs(start_pos - used) < radius for used in used_starts):
            continue
        start = max(0, start_pos - radius)
        end = min(len(text), end_pos + radius)
        snippets.append(normalize_text(text[start:end]))
        used_starts.append(start_pos)
        if len(snippets) >= limit:
            break
    return " || ".join(snippets)


def compact_snippet(value: str, max_len: int = 420) -> str:
    value = normalize_text(value)
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def amount_candidates(text: str, terms: list[str], limit: int = 12) -> str:
    candidates: list[str] = []
    seen: set[tuple[str, str]] = set()
    for match in AMOUNT_PATTERN.finditer(text):
        start = max(0, match.start() - 45)
        end = min(len(text), match.end() + 45)
        context = normalize_text(text[start:end])
        if not any(term in context for term in terms):
            continue
        amount = match.group(1).replace(",", "").replace("，", "")
        key = (amount, context)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(f"{amount}｜{context}")
        if len(candidates) >= limit:
            break
    return " ; ".join(candidates)


def day_candidates(text: str, limit: int = 12) -> str:
    candidates: list[str] = []
    seen: set[str] = set()
    for match in DAY_PATTERN.finditer(text):
        start = max(0, match.start() - 35)
        end = min(len(text), match.end() + 35)
        context = normalize_text(text[start:end])
        if not any(term in context for term in DELAY_TERMS):
            continue
        value = match.group(1)
        item = f"{value}｜{context}"
        if item in seen:
            continue
        seen.add(item)
        candidates.append(item)
        if len(candidates) >= limit:
            break
    return " ; ".join(candidates)


def issue_suggestion(text: str, key: str) -> tuple[str, str]:
    patterns = ISSUE_PATTERNS[key]
    hits = [term for term in patterns if term in text]
    if not hits:
        return "", ""
    first = hits[0]
    idx = text.find(first)
    start = max(0, idx - 80)
    end = min(len(text), idx + len(first) + 120)
    return "1", compact_snippet(text[start:end], 260)


def feature_counter(row: dict[str, str], text: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for field in [
        "matched_penalty_terms",
        "matched_delay_terms",
        "matched_strong_reduction_terms",
        "matched_reduction_terms",
        "JTITLE",
        "court",
    ]:
        for token in re.split(r"[;\s,，、]+", row.get(field, "")):
            token = token.strip()
            if token:
                counter[token] += 2
    for term in RAG_VOCAB:
        count = text.count(term)
        if count:
            counter[term] += min(8, count)
    for issue, terms in ISSUE_PATTERNS.items():
        hits = sum(1 for term in terms if term in text)
        if hits:
            counter[issue] += hits
    return counter


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def shared_terms(a: Counter[str], b: Counter[str], limit: int = 12) -> str:
    common = [(term, a[term] + b[term]) for term in set(a) & set(b)]
    common.sort(key=lambda item: (-item[1], item[0]))
    return ";".join(term for term, _ in common[:limit])


def build_case_features(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    features: dict[str, dict[str, object]] = {}
    for row in rows:
        text = read_judgment_text(row)
        features[row["JID"]] = {
            "row": row,
            "text": text,
            "counter": feature_counter(row, text),
            "penalty_snippet": compact_snippet(snippet_around_pattern(text, PENALTY_PATTERN)),
            "delay_snippet": compact_snippet(snippet_around_pattern(text, DELAY_PATTERN)),
            "reduction_snippet": compact_snippet(snippet_around_pattern(text, STRONG_REDUCTION_PATTERN)),
        }
    return features


def build_annotation_rows(sample_rows: list[dict[str, str]], features: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in sample_rows:
        text = str(features[row["JID"]]["text"])
        record: dict[str, object] = {
            "annotation_status": "needs_manual_review",
            "annotation_priority": row.get("annotation_priority", ""),
            "manual_checked": "",
            "JID": row.get("JID", ""),
            "decision_year": row.get("decision_year", ""),
            "month": row.get("month", ""),
            "court": row.get("court", ""),
            "JTITLE": row.get("JTITLE", ""),
            "JCASE": row.get("JCASE", ""),
            "JNO": row.get("JNO", ""),
            "JDATE": row.get("JDATE", ""),
            "json_file": row.get("json_file", ""),
            "source_snippet_penalty": features[row["JID"]]["penalty_snippet"],
            "source_snippet_delay": features[row["JID"]]["delay_snippet"],
            "source_snippet_reduction": features[row["JID"]]["reduction_snippet"],
            "source_snippets_combined": snippets_around_all(
                text, [PENALTY_PATTERN, DELAY_PATTERN, STRONG_REDUCTION_PATTERN]
            ),
            "ai_contract_price_candidates": amount_candidates(text, CONTRACT_TERMS),
            "ai_claimed_penalty_candidates": amount_candidates(text, CLAIMED_PENALTY_TERMS),
            "ai_allowed_penalty_candidates": amount_candidates(text, ALLOWED_PENALTY_TERMS),
            "ai_delay_days_candidates": day_candidates(text),
            "matched_penalty_terms": row.get("matched_penalty_terms", ""),
            "matched_delay_terms": row.get("matched_delay_terms", ""),
            "matched_strong_reduction_terms": row.get("matched_strong_reduction_terms", ""),
            "relevance_score": row.get("relevance_score", ""),
            "manual_notes": "",
        }
        for field in MANUAL_FIELDS:
            record[field] = ""
        for issue in ISSUE_PATTERNS:
            suggest, evidence = issue_suggestion(text, issue)
            record[f"ai_suggest_{issue}"] = suggest
            record[f"ai_evidence_{issue}"] = evidence
        output.append(record)
    return output


def build_rag_index_rows(pool_rows: list[dict[str, str]], features: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in pool_rows:
        feature = features[row["JID"]]
        output.append(
            {
                "JID": row.get("JID", ""),
                "decision_year": row.get("decision_year", ""),
                "month": row.get("month", ""),
                "court": row.get("court", ""),
                "JTITLE": row.get("JTITLE", ""),
                "json_file": row.get("json_file", ""),
                "matched_first_layer_terms": row.get("matched_first_layer_terms", ""),
                "matched_strong_reduction_terms": row.get("matched_strong_reduction_terms", ""),
                "relevance_score": row.get("relevance_score", ""),
                "penalty_snippet": feature["penalty_snippet"],
                "delay_snippet": feature["delay_snippet"],
                "reduction_snippet": feature["reduction_snippet"],
            }
        )
    return output


def build_reason_summary_rows(
    pool_rows: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
    features: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    sample_jids = {row["JID"] for row in sample_rows}
    rows: list[dict[str, object]] = []
    for issue, description in ISSUE_DESCRIPTIONS.items():
        pool_hits: list[tuple[str, str]] = []
        sample_hits: list[tuple[str, str]] = []
        for row in pool_rows:
            jid = row["JID"]
            text = str(features[jid]["text"])
            suggest, evidence = issue_suggestion(text, issue)
            if suggest != "1":
                continue
            pool_hits.append((jid, evidence))
            if jid in sample_jids:
                sample_hits.append((jid, evidence))

        examples = pool_hits[:5]
        rows.append(
            {
                "reason_field": issue,
                "description": description,
                "pool_hit_count": len(pool_hits),
                "pool_hit_rate": round(len(pool_hits) / len(pool_rows), 4) if pool_rows else 0,
                "sample_hit_count": len(sample_hits),
                "sample_hit_rate": round(len(sample_hits) / len(sample_rows), 4) if sample_rows else 0,
                "example_jids": ";".join(jid for jid, _ in examples),
                "example_evidence": " || ".join(evidence for _, evidence in examples),
            }
        )
    return rows


def build_similar_cases(
    sample_rows: list[dict[str, str]],
    pool_rows: list[dict[str, str]],
    features: dict[str, dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    pool_by_jid = {row["JID"]: row for row in pool_rows}
    output: list[dict[str, object]] = []
    for query in sample_rows:
        qid = query["JID"]
        q_counter = features[qid]["counter"]
        scored: list[tuple[float, str]] = []
        for candidate in pool_rows:
            cid = candidate["JID"]
            if cid == qid:
                continue
            score = cosine_similarity(q_counter, features[cid]["counter"])
            if score > 0:
                scored.append((score, cid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for rank, (score, cid) in enumerate(scored[:top_k], start=1):
            c_row = pool_by_jid[cid]
            c_feature = features[cid]
            output.append(
                {
                    "query_JID": qid,
                    "query_title": query.get("JTITLE", ""),
                    "similar_rank": rank,
                    "similar_JID": cid,
                    "similarity_score": round(score, 4),
                    "similar_decision_year": c_row.get("decision_year", ""),
                    "similar_court": c_row.get("court", ""),
                    "similar_title": c_row.get("JTITLE", ""),
                    "shared_terms": shared_terms(q_counter, c_feature["counter"]),
                    "similar_matched_strong_reduction_terms": c_row.get(
                        "matched_strong_reduction_terms", ""
                    ),
                    "similar_json_file": c_row.get("json_file", ""),
                    "similar_reduction_snippet": c_feature["reduction_snippet"],
                    "similar_delay_snippet": c_feature["delay_snippet"],
                }
            )
    return output


def write_prompt_template(path: Path) -> None:
    content = """# AI 輔助標註提示詞模板

## 使用原則

你是工程判決資料標註助理。請只根據提供的判決原文片段與相似案例摘要，提出「候選標註」，不得宣稱已完成法律判斷。所有金額、比例、法院結論與關鍵爭點都必須回到原判決全文人工查核。

## 任務

請從輸入案件資料中初步萃取下列欄位，若片段不足以判定，請填 `unknown`，並說明需要回查的原文位置或理由。

請輸出 JSON：

```json
{
  "contract_price": {"value": "unknown", "evidence": "", "confidence": "low"},
  "delay_days": {"value": "unknown", "evidence": "", "confidence": "low"},
  "claimed_penalty": {"value": "unknown", "evidence": "", "confidence": "low"},
  "allowed_penalty": {"value": "unknown", "evidence": "", "confidence": "low"},
  "is_reduced": {"value": "unknown", "evidence": "", "confidence": "low"},
  "remaining_ratio": {"value": "unknown", "formula": "allowed_penalty / claimed_penalty", "confidence": "low"},
  "reduction_rate": {"value": "unknown", "formula": "1 - remaining_ratio", "confidence": "low"},
  "issue_owner_fault": {"value": "unknown", "evidence": ""},
  "issue_contractor_fault": {"value": "unknown", "evidence": ""},
  "issue_extension_request": {"value": "unknown", "evidence": ""},
  "issue_actual_damage_unclear": {"value": "unknown", "evidence": ""},
  "issue_partial_completion": {"value": "unknown", "evidence": ""},
  "issue_used_by_owner": {"value": "unknown", "evidence": ""},
  "key_reason": {"value": "", "evidence": ""},
  "manual_review_warnings": []
}
```

## 判斷規則

- 不可把當事人主張直接當成法院認定。
- 不可把請求金額、契約總價、已付款、保留款、利息、訴訟費用混成違約金。
- `allowed_penalty` 必須是法院最後准許或酌減後認定的違約金。
- `is_reduced` 必須比較 `claimed_penalty` 與 `allowed_penalty`，不能只因文中出現「酌減」就判定。
- 若只看到摘要或片段，請保守填 `unknown`。
"""
    path.write_text(content, encoding="utf-8")


def write_human_review_guide(path: Path) -> None:
    content = """# AI/RAG 輔助標註人工查核說明

## 建議流程

1. 先開啟 `annotation_workbook.csv`，依 `annotation_priority` 順序處理。
2. 參考 `source_snippet_penalty`、`source_snippet_delay`、`source_snippet_reduction` 找到判決相關段落。
3. 參考 `ai_*_candidates`，但不得直接複製為正式標註。
4. 回到 `json_file` 指向的原判決全文，確認金額角色與法院最後判斷。
5. 填入正式欄位：`contract_price`、`delay_days`、`claimed_penalty`、`allowed_penalty`、`is_reduced`、`remaining_ratio`、`reduction_rate`、爭點欄位與 `key_reason`。
6. 使用 `rag_similar_cases.csv` 查看相似案例，輔助理解常見酌減理由，但不要把相似案例結論套用到本案。

## 必查事項

- 金額是否為契約總價、請求違約金、法院准許違約金，還是利息/訴訟費/保留款/工程款。
- 法院是否真的酌減，或只是引用當事人請求酌減的主張。
- 判決是否為上訴審、發回、部分廢棄，避免把前審或當事人主張當成最終結果。
- 同案不同審級若要進模型，後續回測切分需避免資料洩漏。

## 欄位填寫

- `is_reduced`：0/1。若法院准許違約金小於主張違約金，填 1；否則填 0。無法確認留空。
- `remaining_ratio`：`allowed_penalty / claimed_penalty`。
- `reduction_rate`：`1 - remaining_ratio`。
- 爭點欄位：0/1。只在法院理由或重要事實中明確涉及時填 1。
"""
    path.write_text(content, encoding="utf-8")


def write_prompt_packets(
    path: Path,
    annotation_rows: list[dict[str, object]],
    similar_rows: list[dict[str, object]],
) -> None:
    by_query: dict[str, list[dict[str, object]]] = {}
    for row in similar_rows:
        by_query.setdefault(str(row["query_JID"]), []).append(row)

    with path.open("w", encoding="utf-8") as handle:
        for row in annotation_rows:
            jid = str(row["JID"])
            packet = {
                "custom_id": jid,
                "task": "工程逾期違約金酌減候選標註",
                "case": {
                    key: row.get(key, "")
                    for key in [
                        "JID",
                        "decision_year",
                        "court",
                        "JTITLE",
                        "json_file",
                        "source_snippet_penalty",
                        "source_snippet_delay",
                        "source_snippet_reduction",
                        "ai_contract_price_candidates",
                        "ai_claimed_penalty_candidates",
                        "ai_allowed_penalty_candidates",
                        "ai_delay_days_candidates",
                    ]
                },
                "similar_cases": by_query.get(jid, [])[:3],
                "instruction": "請依提示詞模板輸出 JSON 候選標註；不足判定時填 unknown，並列出人工回查警示。",
            }
            handle.write(json.dumps(packet, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    annotation_rows: list[dict[str, object]],
    similar_rows: list[dict[str, object]],
    rag_index_rows: list[dict[str, object]],
    reason_rows: list[dict[str, object]],
    top_k: int,
) -> None:
    year_counts = Counter(str(row.get("decision_year", "")) for row in annotation_rows)
    year_lines = "\n".join(f"- {year}: {year_counts[year]}" for year in sorted(year_counts))
    content = f"""# AI/RAG 輔助標註包摘要

## 產出內容

- 標註工作表：`annotation_workbook.csv`
- RAG 候選索引：`rag_case_index.csv`
- 相似案例表：`rag_similar_cases.csv`
- 常見理由語彙摘要：`reason_pattern_summary.csv`
- AI 提示詞模板：`人工智慧輔助標註提示詞模板.md`
- 每案提示詞封包：`case_prompt_packets.jsonl`
- 人工查核說明：`人工智慧與檢索增強生成輔助標註人工查核說明.md`

## 筆數

- 待精標案件：{len(annotation_rows)}
- RAG 索引案件：{len(rag_index_rows)}
- 相似案例列數：{len(similar_rows)}，每案最多 {top_k} 件。
- 常見理由分類：{len(reason_rows)} 類。

## 待精標案件年度分布

{year_lines}

## 使用限制

- 本階段只產出候選標註、證據片段與相似案例，不直接完成正式標註。
- `ai_*_candidates` 與 `ai_suggest_*` 均需人工回查原判決全文。
- 相似案例用詞彙特徵計算，適合輔助閱讀與提示，不代表法律上案件相同。
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI/RAG assisted annotation package.")
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    parser.add_argument("--pool-csv", type=Path, default=DEFAULT_POOL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_rows = read_csv(args.sample_csv)
    pool_rows = read_csv(args.pool_csv)
    all_rows_by_jid = {row["JID"]: row for row in pool_rows}
    for row in sample_rows:
        all_rows_by_jid.setdefault(row["JID"], row)
    all_rows = list(all_rows_by_jid.values())

    features = build_case_features(all_rows)
    annotation_rows = build_annotation_rows(sample_rows, features)
    rag_index_rows = build_rag_index_rows(pool_rows, features)
    reason_rows = build_reason_summary_rows(pool_rows, sample_rows, features)
    similar_rows = build_similar_cases(sample_rows, pool_rows, features, args.top_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_fields = [
        "annotation_status",
        "annotation_priority",
        "manual_checked",
        "JID",
        "decision_year",
        "month",
        "court",
        "JTITLE",
        "JCASE",
        "JNO",
        "JDATE",
        "json_file",
        *MANUAL_FIELDS,
        "ai_contract_price_candidates",
        "ai_delay_days_candidates",
        "ai_claimed_penalty_candidates",
        "ai_allowed_penalty_candidates",
        *[f"ai_suggest_{issue}" for issue in ISSUE_PATTERNS],
        *[f"ai_evidence_{issue}" for issue in ISSUE_PATTERNS],
        "source_snippet_penalty",
        "source_snippet_delay",
        "source_snippet_reduction",
        "source_snippets_combined",
        "matched_penalty_terms",
        "matched_delay_terms",
        "matched_strong_reduction_terms",
        "relevance_score",
        "manual_notes",
    ]
    write_csv(args.output_dir / "annotation_workbook.csv", annotation_rows, annotation_fields)
    write_csv(
        args.output_dir / "rag_case_index.csv",
        rag_index_rows,
        [
            "JID",
            "decision_year",
            "month",
            "court",
            "JTITLE",
            "json_file",
            "matched_first_layer_terms",
            "matched_strong_reduction_terms",
            "relevance_score",
            "penalty_snippet",
            "delay_snippet",
            "reduction_snippet",
        ],
    )
    write_csv(
        args.output_dir / "rag_similar_cases.csv",
        similar_rows,
        [
            "query_JID",
            "query_title",
            "similar_rank",
            "similar_JID",
            "similarity_score",
            "similar_decision_year",
            "similar_court",
            "similar_title",
            "shared_terms",
            "similar_matched_strong_reduction_terms",
            "similar_json_file",
            "similar_reduction_snippet",
            "similar_delay_snippet",
        ],
    )
    write_csv(
        args.output_dir / "reason_pattern_summary.csv",
        reason_rows,
        [
            "reason_field",
            "description",
            "pool_hit_count",
            "pool_hit_rate",
            "sample_hit_count",
            "sample_hit_rate",
            "example_jids",
            "example_evidence",
        ],
    )
    write_prompt_template(args.output_dir / "人工智慧輔助標註提示詞模板.md")
    write_human_review_guide(args.output_dir / "人工智慧與檢索增強生成輔助標註人工查核說明.md")
    write_prompt_packets(args.output_dir / "case_prompt_packets.jsonl", annotation_rows, similar_rows)
    write_summary(
        args.output_dir / "人工智慧與檢索增強生成輔助標註包摘要.md",
        annotation_rows,
        similar_rows,
        rag_index_rows,
        reason_rows,
        args.top_k,
    )

    print(f"annotation_workbook rows: {len(annotation_rows)}")
    print(f"rag_case_index rows: {len(rag_index_rows)}")
    print(f"rag_similar_cases rows: {len(similar_rows)}")
    print(f"reason_pattern_summary rows: {len(reason_rows)}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
