from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match:
            return match
    raise ValueError(f"missing required column; expected one of {candidates}")


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _label(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def _metadata(metadata: pd.DataFrame) -> dict[str, dict[str, str]]:
    jid_col = _column(metadata, "JID", "jid")
    outcome_col = _column(metadata, "is_reduced", "label_is_reduced")
    basis_col = None
    for candidate in ("legal_basis", "label_legal_basis"):
        try:
            basis_col = _column(metadata, candidate)
            break
        except ValueError:
            continue

    result: dict[str, dict[str, str]] = {}
    for _, row in metadata.iterrows():
        jid = _text(row[jid_col])
        if not jid:
            continue
        result[jid] = {
            "outcome": _label(row[outcome_col]),
            "basis": _text(row[basis_col]) if basis_col else "",
        }
    return result


def build_structured_gold(metadata: pd.DataFrame) -> dict[str, set[str]]:
    cases = _metadata(metadata)
    gold: dict[str, set[str]] = {}
    for query_jid, query in cases.items():
        if not query["outcome"]:
            continue
        relevant: set[str] = set()
        for candidate_jid, candidate in cases.items():
            if candidate_jid == query_jid or candidate["outcome"] != query["outcome"]:
                continue
            if query["basis"] and candidate["basis"] != query["basis"]:
                continue
            relevant.add(candidate_jid)
        if relevant:
            gold[query_jid] = relevant
    return gold


def evaluate_retrieval(
    retrieval: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    k: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if k <= 0:
        raise ValueError("k must be positive")

    query_col = _column(retrieval, "query_JID", "query_jid", "jid")
    similar_col = _column(retrieval, "similar_JID", "similar_jid")
    try:
        rank_col = _column(retrieval, "similar_rank", "rank")
    except ValueError:
        rank_col = None

    work = retrieval.copy()
    work["__query"] = work[query_col].map(_text)
    work["__similar"] = work[similar_col].map(_text)
    if rank_col:
        work["__rank"] = pd.to_numeric(work[rank_col], errors="coerce").fillna(10**9)
        work = work.sort_values(["__query", "__rank"], kind="stable")

    gold = build_structured_gold(metadata)
    per_query: list[dict[str, Any]] = []
    golden_rows: list[dict[str, str]] = []

    for query_jid in sorted(gold):
        relevant = gold[query_jid]
        retrieved = (
            work.loc[work["__query"] == query_jid, "__similar"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(k)
            .tolist()
        )
        hits = [jid for jid in retrieved if jid in relevant]
        first_hit_rank = next(
            (index for index, jid in enumerate(retrieved, start=1) if jid in relevant),
            None,
        )
        per_query.append(
            {
                "query_jid": query_jid,
                "relevant_count": len(relevant),
                "retrieved_count": len(retrieved),
                "hit_count": len(hits),
                f"precision_at_{k}": len(hits) / k,
                f"recall_at_{k}": len(hits) / len(relevant),
                f"hit_rate_at_{k}": 1.0 if hits else 0.0,
                f"reciprocal_rank_at_{k}": 1.0 / first_hit_rank if first_hit_rank else 0.0,
                "retrieved_jids": "|".join(retrieved),
                "hit_jids": "|".join(hits),
            }
        )
        golden_rows.append(
            {
                "query_jid": query_jid,
                "relevant_jids": "|".join(sorted(relevant)),
                "gold_definition": "same is_reduced and, when present, same legal_basis",
            }
        )

    per_query_df = pd.DataFrame(per_query)
    golden_df = pd.DataFrame(golden_rows)
    if per_query_df.empty:
        metrics = {
            "schema_version": 1,
            "gold_definition": "structured_proxy_v1",
            "k": k,
            "query_count": 0,
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"hit_rate_at_{k}": 0.0,
            f"mrr_at_{k}": 0.0,
        }
    else:
        metrics = {
            "schema_version": 1,
            "gold_definition": "structured_proxy_v1",
            "k": k,
            "query_count": int(len(per_query_df)),
            f"precision_at_{k}": float(per_query_df[f"precision_at_{k}"].mean()),
            f"recall_at_{k}": float(per_query_df[f"recall_at_{k}"].mean()),
            f"hit_rate_at_{k}": float(per_query_df[f"hit_rate_at_{k}"].mean()),
            f"mrr_at_{k}": float(per_query_df[f"reciprocal_rank_at_{k}"].mean()),
        }
    return metrics, per_query_df, golden_df


def write_benchmark(
    retrieval_path: Path,
    metadata_path: Path,
    output_dir: Path,
    *,
    k: int = 3,
) -> dict[str, Any]:
    retrieval = pd.read_csv(retrieval_path, encoding="utf-8-sig")
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    metrics, per_query, golden = evaluate_retrieval(retrieval, metadata, k=k)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    per_query.to_csv(output_dir / "per_query.csv", index=False, encoding="utf-8-sig")
    golden.to_csv(output_dir / "golden_queries.csv", index=False, encoding="utf-8-sig")
    return metrics
