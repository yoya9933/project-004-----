from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from legal_risk_modeling.cli_contract import ensure_existing_csv, validate_temporal_split
from legal_risk_modeling.rag_benchmark import evaluate_retrieval


def test_missing_csv_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CSV file does not exist"):
        ensure_existing_csv(tmp_path / "missing.csv")


def test_temporal_split_contract() -> None:
    validate_temporal_split(2021, 2023, 2024, 2025, 2026)
    with pytest.raises(ValueError, match="invalid temporal split"):
        validate_temporal_split(2021, 2024, 2024, 2025, 2026)


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
    assert metrics["query_count"] == 3
    assert metrics["hit_rate_at_2"] == 1.0
    assert round(metrics["mrr_at_2"], 6) == round((1.0 + 0.5 + 1.0) / 3, 6)
    assert len(per_query) == 3
    assert len(golden) == 3
