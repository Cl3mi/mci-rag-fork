"""
Retrieval Evaluation Metrics

Implements standard IR metrics for evaluating retrieval quality:
- MRR (Mean Reciprocal Rank)
- Recall@K
- Precision@K
- NDCG@K (Normalized Discounted Cumulative Gain)

Port of src/lib/server/rag/evaluation/metrics.ts
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Type definitions (mirrors types.ts)
# ---------------------------------------------------------------------------

@dataclass
class TestQuery:
    """Test query with ground truth relevance data."""
    id: str
    query: str
    relevant_chunk_ids: list[str]
    relevance_scores: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class QueryEvaluationResult:
    """Result from evaluating a single query."""
    query_id: str
    query: str
    reciprocal_rank: float
    recall: dict[int, float]
    precision: dict[int, float]
    ndcg: dict[int, float]
    retrieved_ids: list[str]
    relevant_retrieved: int
    total_relevant: int


@dataclass
class MetricsResult:
    """Aggregated metrics across all test queries."""
    mrr: float
    recall: dict[int, float]
    precision: dict[int, float]
    ndcg: dict[int, float]
    queries_evaluated: int
    k_values: list[int]


@dataclass
class EvaluationReport:
    """Full evaluation report."""
    metrics: MetricsResult
    query_results: list[QueryEvaluationResult]
    meta: dict[str, Any]


@dataclass
class EvaluationConfig:
    """Configuration for running evaluation."""
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    max_results: int = 20
    include_details: bool = True


DEFAULT_EVALUATION_CONFIG = EvaluationConfig()


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def calculate_reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """RR = 1 / (position of first relevant result). 0 if none found."""
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def calculate_recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@K = |relevant ∩ top-K| / |relevant|"""
    if len(relevant_ids) == 0:
        return 0.0
    top_k = ranked_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def calculate_precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@K = |relevant ∩ top-K| / K"""
    top_k = ranked_ids[:k]
    if len(top_k) == 0:
        return 0.0
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(top_k)


def _calculate_dcg(ranked_ids: list[str], relevance_scores: dict[str, int], k: int) -> float:
    """DCG@K = Σ (2^rel_i - 1) / log2(i + 2)"""
    top_k = ranked_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        relevance = relevance_scores.get(doc_id, 0)
        dcg += (2 ** relevance - 1) / math.log2(i + 2)
    return dcg


def _calculate_ideal_dcg(relevance_scores: dict[str, int], k: int) -> float:
    """Ideal DCG (for normalization)."""
    sorted_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, score in enumerate(sorted_scores):
        idcg += (2 ** score - 1) / math.log2(i + 2)
    return idcg


def calculate_ndcg(ranked_ids: list[str], relevance_scores: dict[str, int], k: int) -> float:
    """NDCG@K = DCG@K / IDCG@K"""
    dcg = _calculate_dcg(ranked_ids, relevance_scores, k)
    idcg = _calculate_ideal_dcg(relevance_scores, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_query(
    query: TestQuery,
    retrieved_ids: list[str],
    config: EvaluationConfig | None = None,
) -> QueryEvaluationResult:
    """Evaluate a single query against ground truth."""
    if config is None:
        config = DEFAULT_EVALUATION_CONFIG

    relevant_ids = set(query.relevant_chunk_ids)
    relevance_scores = (
        query.relevance_scores
        if query.relevance_scores
        else {cid: 1 for cid in query.relevant_chunk_ids}
    )

    recall: dict[int, float] = {}
    precision: dict[int, float] = {}
    ndcg: dict[int, float] = {}

    for k in config.k_values:
        recall[k] = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
        precision[k] = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
        ndcg[k] = calculate_ndcg(retrieved_ids, relevance_scores, k)

    relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)

    return QueryEvaluationResult(
        query_id=query.id,
        query=query.query,
        reciprocal_rank=calculate_reciprocal_rank(retrieved_ids, relevant_ids),
        recall=recall,
        precision=precision,
        ndcg=ndcg,
        retrieved_ids=retrieved_ids,
        relevant_retrieved=relevant_retrieved,
        total_relevant=len(relevant_ids),
    )


def aggregate_metrics(
    query_results: list[QueryEvaluationResult],
    k_values: list[int] | None = None,
) -> MetricsResult:
    """Aggregate metrics from multiple query evaluations."""
    if k_values is None:
        k_values = DEFAULT_EVALUATION_CONFIG.k_values

    if len(query_results) == 0:
        return MetricsResult(
            mrr=0.0,
            recall={k: 0.0 for k in k_values},
            precision={k: 0.0 for k in k_values},
            ndcg={k: 0.0 for k in k_values},
            queries_evaluated=0,
            k_values=k_values,
        )

    n = len(query_results)
    mrr = sum(r.reciprocal_rank for r in query_results) / n

    recall: dict[int, float] = {}
    precision: dict[int, float] = {}
    ndcg: dict[int, float] = {}

    for k in k_values:
        recall[k] = sum(r.recall.get(k, 0.0) for r in query_results) / n
        precision[k] = sum(r.precision.get(k, 0.0) for r in query_results) / n
        ndcg[k] = sum(r.ndcg.get(k, 0.0) for r in query_results) / n

    return MetricsResult(
        mrr=mrr,
        recall=recall,
        precision=precision,
        ndcg=ndcg,
        queries_evaluated=n,
        k_values=k_values,
    )


def format_metrics_summary(metrics: MetricsResult) -> str:
    """Format metrics as a readable summary string."""
    lines = [
        "=== Retrieval Evaluation Metrics ===",
        f"Queries evaluated: {metrics.queries_evaluated}",
        "",
        f"MRR (Mean Reciprocal Rank): {metrics.mrr:.4f}",
        "",
        "Recall@K:",
        *(f"  @{k}: {metrics.recall[k] * 100:.2f}%" for k in metrics.k_values),
        "",
        "Precision@K:",
        *(f"  @{k}: {metrics.precision[k] * 100:.2f}%" for k in metrics.k_values),
        "",
        "NDCG@K:",
        *(f"  @{k}: {metrics.ndcg[k]:.4f}" for k in metrics.k_values),
    ]
    return "\n".join(lines)
