"""
Evaluation Runner

Loads the FAISS index, runs test queries, determines relevance via
content-based matching, and computes IR metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .metrics import (
    EvaluationConfig,
    EvaluationReport,
    QueryEvaluationResult,
    aggregate_metrics,
    evaluate_query,
    format_metrics_summary,
    TestQuery,
)
from .test_queries import ContentTestQuery, get_test_queries


def _is_relevant(doc: Document, query: ContentTestQuery) -> bool:
    """A document is relevant when its source matches AND content has a keyword."""
    source = str(doc.metadata.get("source", ""))
    source_match = any(src in source for src in query.relevant_sources)
    if not source_match:
        return False
    content_lower = doc.page_content.lower()
    return any(kw.lower() in content_lower for kw in query.relevant_keywords)


def _build_ground_truth(docs: list[Document], query: ContentTestQuery) -> TestQuery:
    """Convert ContentTestQuery to metrics-compatible TestQuery using index positions as IDs."""
    relevant_ids: list[str] = []
    relevance_scores: dict[str, int] = {}
    for i, doc in enumerate(docs):
        doc_id = str(i)
        if _is_relevant(doc, query):
            relevant_ids.append(doc_id)
            relevance_scores[doc_id] = query.relevance_weight
    return TestQuery(
        id=query.id,
        query=query.query,
        relevant_chunk_ids=relevant_ids,
        relevance_scores=relevance_scores if relevance_scores else None,
        metadata=query.metadata,
    )


def load_faiss_index(
    persist_dir: str | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> FAISS:
    """Load the FAISS index from disk."""
    if persist_dir is None:
        persist_dir = str(
            Path(__file__).resolve().parent.parent / "agent" / "faiss_ei"
        )
    print(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"Loading FAISS index from: {persist_dir}")
    db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return db


def run_evaluation(
    k: int = 3,
    search_type: str = "mmr",
    category: str | None = None,
    faiss_dir: str | None = None,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Run the full evaluation pipeline."""
    if config is None:
        config = EvaluationConfig()

    db = load_faiss_index(persist_dir=faiss_dir)
    retriever = db.as_retriever(k=k, search_type=search_type)

    queries = get_test_queries(category=category)
    print(f"Starting evaluation of {len(queries)} queries (k={k}, search_type={search_type})...")

    start_time = time.time()
    query_results: list[QueryEvaluationResult] = []

    for i, test_query in enumerate(queries):
        try:
            docs = retriever.invoke(test_query.query)
            retrieved_ids = [str(j) for j in range(len(docs))]
            ground_truth = _build_ground_truth(docs, test_query)
            result = evaluate_query(ground_truth, retrieved_ids, config)
            query_results.append(result)

            if (i + 1) % 10 == 0 or i == len(queries) - 1:
                print(f"Evaluated {i + 1}/{len(queries)} queries")
        except Exception as e:
            print(f"Error evaluating query \"{test_query.id}\": {e}")

    duration_ms = int((time.time() - start_time) * 1000)
    metrics = aggregate_metrics(query_results, config.k_values)

    print(f"Evaluation complete in {duration_ms}ms")
    print(f"MRR: {metrics.mrr:.4f}")
    print(f"Recall@5: {metrics.recall.get(5, 0):.4f}")
    print(f"NDCG@5: {metrics.ndcg.get(5, 0):.4f}")

    return EvaluationReport(
        metrics=metrics,
        query_results=query_results if config.include_details else [],
        meta={
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "total_queries": len(queries),
            "duration_ms": duration_ms,
            "k": k,
            "search_type": search_type,
            "system": "legacy-faiss",
        },
    )


def save_results(report: EvaluationReport, output_path: str) -> None:
    """Serialize an evaluation report to JSON."""
    def _serialize(obj):
        if hasattr(obj, "__dict__"):
            return {k: _serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_serialize(report), f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")
