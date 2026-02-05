#!/usr/bin/env python3
"""
CLI entry point for running RAG evaluation on the legacy FAISS system.

Usage:
    python -m app.evaluation.run_evaluation
    python -m app.evaluation.run_evaluation --k 5 --search-type similarity
    python -m app.evaluation.run_evaluation --category hr-policy
"""

from __future__ import annotations

import argparse

from .evaluate import run_evaluation, save_results
from .metrics import format_metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation on the legacy FAISS system",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--search-type", choices=["mmr", "similarity"], default="mmr")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--faiss-dir", type=str, default=None)
    args = parser.parse_args()

    report = run_evaluation(
        k=args.k,
        search_type=args.search_type,
        category=args.category,
        faiss_dir=args.faiss_dir,
    )

    print()
    print(format_metrics_summary(report.metrics))
    print()

    if report.query_results:
        print("=== Per-Query Results ===")
        for r in report.query_results:
            status = "HIT" if r.reciprocal_rank > 0 else "MISS"
            print(
                f"  [{status}] {r.query_id}: RR={r.reciprocal_rank:.2f}  "
                f"Recall@3={r.recall.get(3, 0):.2f}  "
                f"relevant={r.relevant_retrieved}/{r.total_relevant}  "
                f"\"{r.query}\""
            )
        print()

    save_results(report, args.output)


if __name__ == "__main__":
    main()
