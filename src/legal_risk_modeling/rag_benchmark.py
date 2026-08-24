from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .rag_human_gold import eligible_human_gold, load_human_gold


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


def _ranked_retrieval(retrieval: pd.DataFrame) -> pd.DataFrame:
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
    return work


def build_structured_gold(
    metadata: pd.DataFrame,
    *,
    min_basis_peers: int = 3,
) -> dict[str, set[str]]:
    if min_basis_peers <= 0:
        raise ValueError("min_basis_peers must be positive")

    cases = _metadata(metadata)
    gold: dict[str, set[str]] = {}
    for query_jid, query in cases.items():
        if not query["outcome"]:
            continue

        outcome_peers = {
            candidate_jid
            for candidate_jid, candidate in cases.items()
            if candidate_jid != query_jid and candidate["outcome"] == query["outcome"]
        }
        if not outcome_peers:
            continue

        relevant = outcome_peers
        if query["basis"]:
            basis_peers = {
                candidate_jid
                for candidate_jid in outcome_peers
                if cases[candidate_jid]["basis"] == query["basis"]
            }
            if len(basis_peers) >= min_basis_peers:
                relevant = basis_peers

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
    work = _ranked_retrieval(retrieval)
    gold = build_structured_gold(metadata, min_basis_peers=k)
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
                "gold_definition": (
                    "same is_reduced; narrow to same legal_basis only when at least k peers exist"
                ),
            }
        )

    per_query_df = pd.DataFrame(per_query)
    golden_df = pd.DataFrame(golden_rows)
    if per_query_df.empty:
        metrics = {
            "schema_version": 2,
            "gold_definition": "structured_proxy_v2",
            "k": k,
            "query_count": 0,
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"hit_rate_at_{k}": 0.0,
            f"mrr_at_{k}": 0.0,
        }
    else:
        metrics = {
            "schema_version": 2,
            "gold_definition": "structured_proxy_v2",
            "k": k,
            "query_count": int(len(per_query_df)),
            f"precision_at_{k}": float(per_query_df[f"precision_at_{k}"].mean()),
            f"recall_at_{k}": float(per_query_df[f"recall_at_{k}"].mean()),
            f"hit_rate_at_{k}": float(per_query_df[f"hit_rate_at_{k}"].mean()),
            f"mrr_at_{k}": float(per_query_df[f"reciprocal_rank_at_{k}"].mean()),
        }
    return metrics, per_query_df, golden_df


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def evaluate_human_gold(
    retrieval: pd.DataFrame,
    human_gold: pd.DataFrame,
    *,
    k: int = 3,
    min_judgments_per_query: int = 5,
    min_relevant_per_query: int = 1,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if k <= 0:
        raise ValueError("k must be positive")
    eligible, eligibility = eligible_human_gold(
        human_gold,
        min_judgments_per_query=min_judgments_per_query,
        min_relevant_per_query=min_relevant_per_query,
    )
    work = _ranked_retrieval(retrieval)
    per_query: list[dict[str, Any]] = []

    for query_jid, judgments in eligible.groupby("query_jid", sort=True):
        grade_by_jid = {
            str(row["candidate_jid"]).strip(): int(row["relevance_grade"])
            for _, row in judgments.iterrows()
        }
        retrieved = (
            work.loc[work["__query"] == str(query_jid), "__similar"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(k)
            .tolist()
        )
        grades = [grade_by_jid.get(jid, 0) for jid in retrieved]
        relevant = {jid for jid, grade in grade_by_jid.items() if grade > 0}
        hits = [jid for jid in retrieved if grade_by_jid.get(jid, 0) > 0]
        first_hit_rank = next(
            (index for index, jid in enumerate(retrieved, start=1) if grade_by_jid.get(jid, 0) > 0),
            None,
        )
        ideal_grades = sorted(grade_by_jid.values(), reverse=True)[:k]
        ideal_dcg = _dcg(ideal_grades)
        ndcg = _dcg(grades) / ideal_dcg if ideal_dcg > 0 else 0.0
        per_query.append(
            {
                "query_jid": str(query_jid),
                "judged_candidates": int(len(grade_by_jid)),
                "relevant_count": int(len(relevant)),
                "retrieved_count": int(len(retrieved)),
                "hit_count": int(len(hits)),
                f"precision_at_{k}": len(hits) / k,
                f"recall_at_{k}": len(hits) / len(relevant),
                f"hit_rate_at_{k}": 1.0 if hits else 0.0,
                f"reciprocal_rank_at_{k}": 1.0 / first_hit_rank if first_hit_rank else 0.0,
                f"ndcg_at_{k}": ndcg,
                "retrieved_jids": "|".join(retrieved),
                "retrieved_grades": "|".join(str(grade) for grade in grades),
            }
        )

    per_query_df = pd.DataFrame(per_query)
    base = {
        "schema_version": 2,
        "gold_definition": "human_graded_v1",
        "relevance_grades": [0, 1, 2, 3],
        "k": k,
        "query_count": int(len(per_query_df)),
        "judgment_count": int(len(eligible)),
        "approved_query_count": eligibility["approved_query_count"],
        "approved_judgment_count": eligibility["approved_judgment_count"],
        "eligible_query_count": eligibility["eligible_query_count"],
        "ineligible_query_count": eligibility["ineligible_query_count"],
        "eligibility_policy": {
            "min_judgments_per_query": eligibility["min_judgments_per_query"],
            "min_relevant_per_query": eligibility["min_relevant_per_query"],
        },
    }
    if per_query_df.empty:
        return {
            **base,
            "status": "insufficient_reviewed_queries",
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"hit_rate_at_{k}": 0.0,
            f"mrr_at_{k}": 0.0,
            f"ndcg_at_{k}": 0.0,
        }, per_query_df

    return {
        **base,
        "status": "evaluated",
        f"precision_at_{k}": float(per_query_df[f"precision_at_{k}"].mean()),
        f"recall_at_{k}": float(per_query_df[f"recall_at_{k}"].mean()),
        f"hit_rate_at_{k}": float(per_query_df[f"hit_rate_at_{k}"].mean()),
        f"mrr_at_{k}": float(per_query_df[f"reciprocal_rank_at_{k}"].mean()),
        f"ndcg_at_{k}": float(per_query_df[f"ndcg_at_{k}"].mean()),
    }, per_query_df


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


def write_human_benchmark(
    retrieval_path: Path,
    human_gold_path: Path,
    output_dir: Path,
    *,
    k: int = 3,
    min_judgments_per_query: int = 5,
    min_relevant_per_query: int = 1,
) -> dict[str, Any]:
    retrieval = pd.read_csv(retrieval_path, encoding="utf-8-sig")
    human_gold = load_human_gold(human_gold_path)
    metrics, per_query = evaluate_human_gold(
        retrieval,
        human_gold,
        k=k,
        min_judgments_per_query=min_judgments_per_query,
        min_relevant_per_query=min_relevant_per_query,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "human_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    per_query.to_csv(output_dir / "human_per_query.csv", index=False, encoding="utf-8-sig")
    return metrics
