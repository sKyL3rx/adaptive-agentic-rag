from __future__ import annotations

from prometheus_client import (
    Counter,
    Histogram,
)


# ============================================================
# Histogram buckets
# ============================================================

RETRIEVAL_STAGE_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
)


RETRIEVAL_E2E_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    12.0,
    20.0,
    30.0,
)


RETRIEVAL_BATCH_SIZE_BUCKETS = (
    1,
    2,
    3,
    4,
    6,
    8,
)

RERANKER_PAIR_BUCKETS = (
    1,
    2,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
)


# ============================================================
# Embedding
# ============================================================

RAG_EMBEDDING_BATCH_SIZE = Histogram(
    "rag_embedding_batch_size",
    (
        "Logical queries submitted to one "
        "hybrid embedding stage"
    ),
    buckets=RETRIEVAL_BATCH_SIZE_BUCKETS,
)


RAG_EMBEDDING_DURATION_SECONDS = Histogram(
    "rag_embedding_duration_seconds",
    (
        "Dense plus sparse query embedding "
        "stage duration"
    ),
    buckets=RETRIEVAL_STAGE_BUCKETS,
)


# ============================================================
# Qdrant
# ============================================================

RAG_QDRANT_BATCH_SIZE = Histogram(
    "rag_qdrant_batch_size",
    (
        "Logical queries submitted in one "
        "Qdrant batch network request"
    ),
    buckets=RETRIEVAL_BATCH_SIZE_BUCKETS,
)


RAG_QDRANT_DURATION_SECONDS = Histogram(
    "rag_qdrant_duration_seconds",
    (
        "Qdrant batch network request "
        "duration"
    ),
    buckets=RETRIEVAL_STAGE_BUCKETS,
)


# ============================================================
# Reranker
# ============================================================

RAG_RERANKER_PAIRS = Histogram(
    "rag_reranker_pairs",
    (
        "Query-document pairs submitted "
        "to one reranker invocation"
    ),
    buckets=RERANKER_PAIR_BUCKETS,
)


RAG_RERANKER_DURATION_SECONDS = Histogram(
    "rag_reranker_duration_seconds",
    (
        "ONNX reranker invocation "
        "duration"
    ),
    buckets=RETRIEVAL_STAGE_BUCKETS,
)


# ============================================================
# Retrieval end-to-end
# ============================================================

RAG_RETRIEVAL_DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    (
        "Online retrieval pipeline duration "
        "excluding lazy pipeline initialization"
    ),
    buckets=RETRIEVAL_E2E_BUCKETS,
)