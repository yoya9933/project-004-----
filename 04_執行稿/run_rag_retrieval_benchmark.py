from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_risk_modeling.cli_contract import ensure_existing_csv, positive_int  # noqa: E402
from legal_risk_modeling.paths import rag_benchmark_output_dir  # noqa: E402
from legal_risk_modeling.rag_benchmark import write_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval with a structured relevance proxy.")
    parser.add_argument(
        "--retrieval-csv",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "rag_similar_cases.csv",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "annotation_workbook.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=rag_benchmark_output_dir(PROJECT_ROOT))
    parser.add_argument("--k", type=positive_int, default=3)
    parser.add_argument("--min-queries", type=positive_int, default=20)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_csv = ensure_existing_csv(args.retrieval_csv)
    metadata_csv = ensure_existing_csv(args.metadata_csv)
    if not 0.0 <= args.min_hit_rate <= 1.0:
        raise SystemExit("--min-hit-rate must be between 0 and 1")
    if not 0.0 <= args.min_mrr <= 1.0:
        raise SystemExit("--min-mrr must be between 0 and 1")

    metrics = write_benchmark(
        retrieval_csv,
        metadata_csv,
        args.output_dir,
        k=args.k,
    )
    hit_key = f"hit_rate_at_{args.k}"
    mrr_key = f"mrr_at_{args.k}"
    failures = []
    if metrics["query_count"] < args.min_queries:
        failures.append(f"query_count {metrics['query_count']} < {args.min_queries}")
    if metrics[hit_key] < args.min_hit_rate:
        failures.append(f"{hit_key} {metrics[hit_key]:.4f} < {args.min_hit_rate:.4f}")
    if metrics[mrr_key] < args.min_mrr:
        failures.append(f"{mrr_key} {metrics[mrr_key]:.4f} < {args.min_mrr:.4f}")
    print(json.dumps(metrics, ensure_ascii=False))
    if failures:
        raise SystemExit("RAG benchmark gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
