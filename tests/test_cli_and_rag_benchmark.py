from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from legal_risk_modeling.cli_contract import ensure_existing_csv, validate_temporal_split
from legal_risk_modeling.rag_benchmark import (
    build_structured_gold,
    evaluate_human_gold,
    evaluate_retrieval,
)
from legal_risk_modeling.rag_human_gold import HUMAN_GOLD_COLUMNS, seed_review_queue, validate_human_gold


def test_missing_csv_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CSV file does not exist"):
        ensure_existing_csv(tmp_path / "missing.csv")


def test_temporal_split_contract() -> None:
    validate_temporal_split(2021, 2023, 2024, 2025, 2026)
    with pytest.raises(ValueError, match="invalid temporal split"):
        validate_temporal_split(2021, 2024, 2024, 2025, 2026)


def test_sparse_legal_basis_falls_back_to_outcome_peers() -> None:
    metadata = pd.DataFrame(
        [
            {"JID": "a", "is_reduced": 1, "legal_basis": "unique"},
            {"JID": "b", "is_reduced": 1, "legal_basis": "252"},
            {"JID": "c", "is_reduced": 1, "legal_basis": "252"},
            {"JID": "d", "is_reduced": 0, "legal_basis": "252"},
        ]
    )
    gold = build_structured_gold(metadata, min_basis_peers=3)
    assert gold["a"] == {"b", "c"}


def test_rag_benchmark_metrics() -> None:
    metadata = pd.DataFrame(
        [
            {"JID": "a", "is_reduced": 1, "legal_basis": "252"},
            {"JID": "b", "is_reduced": 1, "legal_basis": "252"},
            {"JID": "c", "is_reduced": 0, "legal_basis": "252"},
            {"JID": "d", "is_reduced": 1, "legal_basis": "252"},
        ]
    )
    retrieval = pd.DataFrame(
        [
            {"query_JID": "a", "similar_rank": 1, "similar_JID": "b"},
            {"query_JID": "a", "similar_rank": 2, "similar_JID": "c"},
            {"query_JID": "b", "similar_rank": 1, "similar_JID": "c"},
            {"query_JID": "b", "similar_rank": 2, "similar_JID": "a"},
            {"query_JID": "d", "similar_rank": 1, "similar_JID": "a"},
        ]
    )
    metrics, per_query, golden = evaluate_retrieval(retrieval, metadata, k=2)
    assert metrics["schema_version"] == 2
    assert metrics["query_count"] == 3
    assert metrics["hit_rate_at_2"] == 1.0
    assert round(metrics["mrr_at_2"], 6) == round((1.0 + 0.5 + 1.0) / 3, 6)
    assert len(per_query) == 3
    assert len(golden) == 3


def test_human_gold_requires_complete_graded_judgments_before_evaluation() -> None:
    retrieval = pd.DataFrame(
        [
            {"query_JID": "a", "similar_rank": 1, "similar_JID": "b"},
            {"query_JID": "a", "similar_rank": 2, "similar_JID": "c"},
            {"query_JID": "a", "similar_rank": 3, "similar_JID": "d"},
        ]
    )
    gold = pd.DataFrame(
        [
            {"query_jid": "a", "candidate_jid": "b", "relevance_grade": 3, "reviewer": "R1", "review_status": "approved", "notes": "", "seed_source": "manual"},
            {"query_jid": "a", "candidate_jid": "c", "relevance_grade": 1, "reviewer": "R1", "review_status": "approved", "notes": "", "seed_source": "manual"},
            {"query_jid": "a", "candidate_jid": "d", "relevance_grade": 0, "reviewer": "R1", "review_status": "approved", "notes": "", "seed_source": "manual"},
            {"query_jid": "a", "candidate_jid": "e", "relevance_grade": 0, "reviewer": "R1", "review_status": "approved", "notes": "", "seed_source": "manual"},
            {"query_jid": "a", "candidate_jid": "f", "relevance_grade": 0, "reviewer": "R1", "review_status": "approved", "notes": "", "seed_source": "manual"},
        ],
        columns=HUMAN_GOLD_COLUMNS,
    )
    assert validate_human_gold(gold)["status"] == "pass"
    metrics, per_query = evaluate_human_gold(retrieval, gold, k=2)
    assert metrics["gold_definition"] == "human_graded_v1"
    assert metrics["query_count"] == 1
    assert metrics["eligible_query_count"] == 1
    assert metrics["eligibility_policy"]["min_judgments_per_query"] == 5
    assert metrics["ndcg_at_2"] == 1.0
    assert len(per_query) == 1


def test_human_gold_excludes_incomplete_or_all_irrelevant_queries() -> None:
    retrieval = pd.DataFrame(
        [{"query_JID": "a", "similar_rank": 1, "similar_JID": "b"}]
    )
    rows = [
        {"query_jid": "a", "candidate_jid": candidate, "relevance_grade": 0, "reviewer": "R1", "review_status": "approved", "notes": "", "seed_source": "manual"}
        for candidate in ["b", "c", "d", "e", "f"]
    ]
    gold = pd.DataFrame(rows, columns=HUMAN_GOLD_COLUMNS)
    metrics, per_query = evaluate_human_gold(retrieval, gold, k=1)
    assert metrics["approved_query_count"] == 1
    assert metrics["eligible_query_count"] == 0
    assert metrics["query_count"] == 0
    assert metrics["status"] == "insufficient_reviewed_queries"
    assert per_query.empty


def test_review_queue_does_not_fabricate_human_labels_and_includes_context() -> None:
    retrieval = pd.DataFrame(
        [
            {
                "query_JID": "a",
                "query_title": "Query title",
                "similar_rank": 1,
                "similar_JID": "b",
                "similarity_score": 0.9,
                "similar_decision_year": 2025,
                "similar_court": "Court",
                "similar_title": "Candidate title",
                "similar_reduction_snippet": "Reduction evidence",
                "similar_delay_snippet": "Delay evidence",
            }
        ]
    )
    queue = seed_review_queue(retrieval, pd.DataFrame(columns=HUMAN_GOLD_COLUMNS), top_n=5)
    assert len(queue) == 1
    assert queue.iloc[0]["review_status"] == "review_required"
    assert str(queue.iloc[0]["reviewer"]) == ""
    assert str(queue.iloc[0]["relevance_grade"]) == ""
    assert queue.iloc[0]["query_title"] == "Query title"
    assert queue.iloc[0]["candidate_title"] == "Candidate title"
    assert queue.iloc[0]["candidate_reduction_snippet"] == "Reduction evidence"
