from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

HUMAN_GOLD_COLUMNS = [
    "query_jid",
    "candidate_jid",
    "relevance_grade",
    "reviewer",
    "review_status",
    "notes",
    "seed_source",
]
REVIEW_STATUSES = {"review_required", "approved", "rejected"}


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match:
            return match
    raise ValueError(f"missing required column; expected one of {candidates}")


def empty_human_gold() -> pd.DataFrame:
    return pd.DataFrame(columns=HUMAN_GOLD_COLUMNS)


def load_human_gold(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in HUMAN_GOLD_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"human RAG gold is missing columns: {missing}")
    return frame[HUMAN_GOLD_COLUMNS].copy()


def validate_human_gold(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [column for column in HUMAN_GOLD_COLUMNS if column not in frame.columns]
    errors: list[dict[str, Any]] = []
    if missing:
        errors.append({"code": "missing_columns", "columns": missing})
        return {"status": "fail", "errors": errors, "approved_judgments": 0, "approved_queries": 0}

    work = frame.copy()
    work["query_jid"] = work["query_jid"].astype("string").fillna("").str.strip()
    work["candidate_jid"] = work["candidate_jid"].astype("string").fillna("").str.strip()
    work["review_status"] = work["review_status"].astype("string").fillna("").str.strip()
    work["reviewer"] = work["reviewer"].astype("string").fillna("").str.strip()

    populated = work["query_jid"].ne("") | work["candidate_jid"].ne("")
    work = work[populated].copy()
    blank_key = work["query_jid"].eq("") | work["candidate_jid"].eq("")
    if blank_key.any():
        errors.append({"code": "blank_query_or_candidate", "count": int(blank_key.sum())})
    same_case = work["query_jid"].eq(work["candidate_jid"]) & work["query_jid"].ne("")
    if same_case.any():
        errors.append({"code": "self_relevance_pair", "count": int(same_case.sum())})
    duplicate = work.duplicated(["query_jid", "candidate_jid"], keep=False)
    if duplicate.any():
        errors.append({"code": "duplicate_pair", "count": int(duplicate.sum())})

    invalid_status = work["review_status"].ne("") & ~work["review_status"].isin(REVIEW_STATUSES)
    if invalid_status.any():
        errors.append({"code": "invalid_review_status", "count": int(invalid_status.sum())})

    approved = work[work["review_status"].eq("approved")].copy()
    grade = pd.to_numeric(approved["relevance_grade"], errors="coerce")
    invalid_grade = grade.isna() | ~grade.isin([0, 1, 2, 3])
    if invalid_grade.any():
        errors.append({"code": "invalid_approved_relevance_grade", "count": int(invalid_grade.sum())})
    missing_reviewer = approved["reviewer"].eq("")
    if missing_reviewer.any():
        errors.append({"code": "approved_without_reviewer", "count": int(missing_reviewer.sum())})

    return {
        "schema_version": 1,
        "status": "fail" if errors else "pass",
        "errors": errors,
        "rows": int(len(work)),
        "approved_judgments": int(len(approved)),
        "approved_queries": int(approved["query_jid"].nunique()),
    }


def approved_human_gold(frame: pd.DataFrame) -> pd.DataFrame:
    report = validate_human_gold(frame)
    if report["status"] != "pass":
        raise ValueError(f"invalid human RAG gold: {report['errors']}")
    approved = frame[frame["review_status"].astype(str).str.strip().eq("approved")].copy()
    if approved.empty:
        return approved
    approved["relevance_grade"] = pd.to_numeric(approved["relevance_grade"], errors="raise").astype(int)
    return approved


def seed_review_queue(
    retrieval: pd.DataFrame,
    existing_gold: pd.DataFrame | None = None,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    query_col = _column(retrieval, "query_JID", "query_jid", "jid")
    candidate_col = _column(retrieval, "similar_JID", "similar_jid")
    try:
        rank_col = _column(retrieval, "similar_rank", "rank")
    except ValueError:
        rank_col = None

    existing = empty_human_gold() if existing_gold is None else existing_gold[HUMAN_GOLD_COLUMNS].copy()
    report = validate_human_gold(existing)
    if report["status"] != "pass":
        raise ValueError(f"existing human RAG gold is invalid: {report['errors']}")

    keys = {
        (str(row["query_jid"]).strip(), str(row["candidate_jid"]).strip())
        for _, row in existing.iterrows()
        if str(row["query_jid"]).strip() and str(row["candidate_jid"]).strip()
    }
    work = retrieval.copy()
    work["__query"] = work[query_col].astype("string").fillna("").str.strip()
    work["__candidate"] = work[candidate_col].astype("string").fillna("").str.strip()
    if rank_col:
        work["__rank"] = pd.to_numeric(work[rank_col], errors="coerce").fillna(10**9)
        work = work.sort_values(["__query", "__rank"], kind="stable")

    new_rows: list[dict[str, Any]] = []
    for query_jid, group in work.groupby("__query", sort=True):
        if not query_jid:
            continue
        candidates = group["__candidate"].drop_duplicates().head(top_n).tolist()
        for candidate_jid in candidates:
            if not candidate_jid or candidate_jid == query_jid:
                continue
            key = (str(query_jid), str(candidate_jid))
            if key in keys:
                continue
            keys.add(key)
            new_rows.append(
                {
                    "query_jid": query_jid,
                    "candidate_jid": candidate_jid,
                    "relevance_grade": "",
                    "reviewer": "",
                    "review_status": "review_required",
                    "notes": "",
                    "seed_source": "retriever_top_n",
                }
            )

    combined = pd.concat([existing, pd.DataFrame(new_rows, columns=HUMAN_GOLD_COLUMNS)], ignore_index=True)
    return combined[HUMAN_GOLD_COLUMNS]


def write_review_queue(
    retrieval_path: Path,
    gold_path: Path,
    output_path: Path,
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    retrieval = pd.read_csv(retrieval_path, encoding="utf-8-sig")
    existing = load_human_gold(gold_path)
    queue = seed_review_queue(retrieval, existing, top_n=top_n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output_path, index=False, encoding="utf-8-sig")
    report = validate_human_gold(queue)
    report["output_rows"] = int(len(queue))
    report["review_required"] = int(
        queue["review_status"].astype(str).str.strip().eq("review_required").sum()
    )
    return report
