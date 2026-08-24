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
from legal_risk_modeling.rag_benchmark import write_benchmark, write_human_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval against proxy and reviewed human gold.")
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
    parser.add_argument(
        "--human-gold-csv",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "human_rag_gold.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=rag_benchmark_output_dir(PROJECT_ROOT))
    parser.add_argument("--k", type=positive_int, default=3)
    parser.add_argument("--min-queries", type=positive_int, default=20)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument("--min-human-queries", type=positive_int, default=20)
    parser.add_argument("--min-human-judgments-per-query", type=positive_int, default=5)
    parser.add_argument("--min-human-relevant-per-query", type=positive_int, default=1)
    parser.add_argument("--min-human-ndcg", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_csv = ensure_existing_csv(args.retrieval_csv)
    metadata_csv = ensure_existing_csv(args.metadata_csv)
    human_gold_csv = ensure_existing_csv(args.human_gold_csv)
    for name, value in [
        ("--min-hit-rate", args.min_hit_rate),
        ("--min-mrr", args.min_mrr),
        ("--min-human-ndcg", args.min_human_ndcg),
    ]:
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be between 0 and 1")

    proxy = write_benchmark(retrieval_csv, metadata_csv, args.output_dir, k=args.k)
    human = write_human_benchmark(
        retrieval_csv,
        human_gold_csv,
        args.output_dir,
        k=args.k,
        min_judgments_per_query=args.min_human_judgments_per_query,
        min_relevant_per_query=args.min_human_relevant_per_query,
    )
    hit_key = f"hit_rate_at_{args.k}"
    mrr_key = f"mrr_at_{args.k}"
    ndcg_key = f"ndcg_at_{args.k}"
    failures = []
    if proxy["query_count"] < args.min_queries:
        failures.append(f"query_count {proxy['query_count']} < {args.min_queries}")
    if proxy[hit_key] < args.min_hit_rate:
        failures.append(f"{hit_key} {proxy[hit_key]:.4f} < {args.min_hit_rate:.4f}")
    if proxy[mrr_key] < args.min_mrr:
        failures.append(f"{mrr_key} {proxy[mrr_key]:.4f} < {args.min_mrr:.4f}")

    human_gate_active = human["query_count"] >= args.min_human_queries
    if human_gate_active and human[ndcg_key] < args.min_human_ndcg:
        failures.append(
            f"human {ndcg_key} {human[ndcg_key]:.4f} < {args.min_human_ndcg:.4f}"
        )

    result = {
        "proxy": proxy,
        "human": human,
        "human_gate_active": human_gate_active,
        "human_gate_min_queries": args.min_human_queries,
        "human_gate_eligibility_policy": human["eligibility_policy"],
    }
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit("RAG benchmark gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
