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
REVIEW_QUEUE_CONTEXT_COLUMNS = [
    "query_title",
    "retrieval_rank",
    "similarity_score",
    "candidate_year",
    "candidate_court",
    "candidate_title",
    "candidate_reduction_snippet",
    "candidate_delay_snippet",
]
REVIEW_STATUSES = {"review_required", "approved", "rejected"}


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match:
            return match
    raise ValueError(f"missing required column; expected one of {candidates}")


def _optional_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    try:
        return _column(frame, *candidates)
    except ValueError:
        return None


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


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
    approved["query_jid"] = approved["query_jid"].astype("string").str.strip()
    approved["candidate_jid"] = approved["candidate_jid"].astype("string").str.strip()
    approved["relevance_grade"] = pd.to_numeric(
        approved["relevance_grade"], errors="raise"
    ).astype(int)
    return approved


def human_gold_eligibility(
    frame: pd.DataFrame,
    *,
    min_judgments_per_query: int = 5,
    min_relevant_per_query: int = 1,
) -> dict[str, Any]:
    if min_judgments_per_query <= 0:
        raise ValueError("min_judgments_per_query must be positive")
    if min_relevant_per_query <= 0:
        raise ValueError("min_relevant_per_query must be positive")

    approved = approved_human_gold(frame)
    eligible_query_jids: list[str] = []
    insufficient_judgments = 0
    insufficient_relevant = 0
    per_query: list[dict[str, Any]] = []

    for query_jid, group in approved.groupby("query_jid", sort=True):
        judgment_count = int(len(group))
        relevant_count = int((group["relevance_grade"] > 0).sum())
        reasons: list[str] = []
        if judgment_count < min_judgments_per_query:
            reasons.append("not_enough_judgments")
            insufficient_judgments += 1
        if relevant_count < min_relevant_per_query:
            reasons.append("not_enough_relevant_judgments")
            insufficient_relevant += 1
        eligible = not reasons
        if eligible:
            eligible_query_jids.append(str(query_jid))
        per_query.append(
            {
                "query_jid": str(query_jid),
                "judgment_count": judgment_count,
                "relevant_count": relevant_count,
                "eligible": eligible,
                "reasons": reasons,
            }
        )

    return {
        "schema_version": 1,
        "min_judgments_per_query": min_judgments_per_query,
        "min_relevant_per_query": min_relevant_per_query,
        "approved_judgment_count": int(len(approved)),
        "approved_query_count": int(approved["query_jid"].nunique()) if not approved.empty else 0,
        "eligible_query_count": len(eligible_query_jids),
        "ineligible_query_count": len(per_query) - len(eligible_query_jids),
        "insufficient_judgments_query_count": insufficient_judgments,
        "insufficient_relevant_query_count": insufficient_relevant,
        "eligible_query_jids": eligible_query_jids,
        "per_query": per_query,
    }


def eligible_human_gold(
    frame: pd.DataFrame,
    *,
    min_judgments_per_query: int = 5,
    min_relevant_per_query: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    approved = approved_human_gold(frame)
    eligibility = human_gold_eligibility(
        frame,
        min_judgments_per_query=min_judgments_per_query,
        min_relevant_per_query=min_relevant_per_query,
    )
    eligible_ids = set(eligibility["eligible_query_jids"])
    eligible = approved[approved["query_jid"].isin(eligible_ids)].copy()
    return eligible, eligibility


def _retrieval_context(retrieval: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    query_col = _column(retrieval, "query_JID", "query_jid", "jid")
    candidate_col = _column(retrieval, "similar_JID", "similar_jid")
    columns = {
        "query_title": _optional_column(retrieval, "query_title"),
        "retrieval_rank": _optional_column(retrieval, "similar_rank", "rank"),
        "similarity_score": _optional_column(retrieval, "similarity_score"),
        "candidate_year": _optional_column(retrieval, "similar_decision_year", "candidate_year"),
        "candidate_court": _optional_column(retrieval, "similar_court", "candidate_court"),
        "candidate_title": _optional_column(retrieval, "similar_title", "candidate_title"),
        "candidate_reduction_snippet": _optional_column(
            retrieval, "similar_reduction_snippet", "candidate_reduction_snippet"
        ),
        "candidate_delay_snippet": _optional_column(
            retrieval, "similar_delay_snippet", "candidate_delay_snippet"
        ),
    }
    result: dict[tuple[str, str], dict[str, str]] = {}
    for _, row in retrieval.iterrows():
        query_jid = _text(row[query_col])
        candidate_jid = _text(row[candidate_col])
        if not query_jid or not candidate_jid:
            continue
        key = (query_jid, candidate_jid)
        if key in result:
            continue
        result[key] = {
            output_column: _text(row[source_column]) if source_column else ""
            for output_column, source_column in columns.items()
        }
    return result


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

    combined = pd.concat(
        [existing, pd.DataFrame(new_rows, columns=HUMAN_GOLD_COLUMNS)], ignore_index=True
    )
    context = _retrieval_context(retrieval)
    for context_column in REVIEW_QUEUE_CONTEXT_COLUMNS:
        combined[context_column] = [
            context.get(
                (str(row["query_jid"]).strip(), str(row["candidate_jid"]).strip()), {}
            ).get(context_column, "")
            for _, row in combined.iterrows()
        ]
    return combined[[*HUMAN_GOLD_COLUMNS, *REVIEW_QUEUE_CONTEXT_COLUMNS]]


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
    eligibility = human_gold_eligibility(queue)
    report["output_rows"] = int(len(queue))
    report["review_required"] = int(
        queue["review_status"].astype(str).str.strip().eq("review_required").sum()
    )
    report["eligible_queries"] = eligibility["eligible_query_count"]
    report["eligibility_policy"] = {
        "min_judgments_per_query": eligibility["min_judgments_per_query"],
        "min_relevant_per_query": eligibility["min_relevant_per_query"],
    }
    return report
