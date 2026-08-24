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


def test_human_gold_uses_only_approved_graded_judgments_and_ndcg() -> None:
    retrieval = pd.DataFrame(
        [
            {"query_JID": "a", "similar_rank": 1, "similar_JID": "b"},
            {"query_JID": "a", "similar_rank": 2, "similar_JID": "c"},
        ]
    )
    gold = pd.DataFrame(
        [
            {"query_jid": "a", "candidate_jid": "b", "relevance_grade": 3, "reviewer": "R1", "review_status": "approved", "notes": "", "seed_source": "manual"},
            {"query_jid": "a", "candidate_jid": "c", "relevance_grade": 1, "reviewer": "R1", "review_status": "approved", "notes": "", "seed_source": "manual"},
            {"query_jid": "a", "candidate_jid": "d", "relevance_grade": "", "reviewer": "", "review_status": "review_required", "notes": "", "seed_source": "retriever_top_n"},
        ],
        columns=HUMAN_GOLD_COLUMNS,
    )
    assert validate_human_gold(gold)["status"] == "pass"
    metrics, per_query = evaluate_human_gold(retrieval, gold, k=2)
    assert metrics["gold_definition"] == "human_graded_v1"
    assert metrics["query_count"] == 1
    assert metrics["ndcg_at_2"] == 1.0
    assert len(per_query) == 1


def test_review_queue_does_not_fabricate_human_labels() -> None:
    retrieval = pd.DataFrame(
        [{"query_JID": "a", "similar_rank": 1, "similar_JID": "b"}]
    )
    queue = seed_review_queue(retrieval, pd.DataFrame(columns=HUMAN_GOLD_COLUMNS), top_n=5)
    assert len(queue) == 1
    assert queue.iloc[0]["review_status"] == "review_required"
    assert str(queue.iloc[0]["reviewer"]) == ""
    assert str(queue.iloc[0]["relevance_grade"]) == ""
