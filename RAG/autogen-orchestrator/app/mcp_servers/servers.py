from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from prometheus_client import (
    Histogram,
    start_http_server,
)

from retrieval_contracts import (
    BatchSemanticSearchInput,
    BatchSemanticSearchOutput,
)
from retrieval_pipeline import (
    get_retrieval_pipeline,
)
from retrieval_runtime import (
    get_dense_model,
    get_pair_reranker,
    get_sparse_model,
)
from retrieval_metrics import (
    RAG_RETRIEVAL_DURATION_SECONDS,
)


ENV_FILE = Path(__file__).resolve().with_name(
    ".env"
)

load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s [%(levelname)s] "
        "%(name)s - %(message)s"
    ),
)

logger = logging.getLogger(
    "vector_mcp"
)


# ---------------------------------------------------------
# Runtime config
# ---------------------------------------------------------

PORT = int(
    os.getenv(
        "PORT",
        "8002",
    )
)

PROMETHEUS_PORT = int(
    os.getenv(
        "PROMETHEUS_PORT",
        "9102",
    )
)


# ---------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------

mcp = FastMCP(
    name="vector",
    include_fastmcp_meta=False,
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

RAG_BATCH_RETRIEVAL_SECONDS = Histogram(
    "rag_batch_retrieval_seconds",
    (
        "End-to-end execution time for "
        "batch semantic retrieval"
    ),
    [
        "status",
    ],
    buckets=(
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        3,
        5,
        8,
        12,
        20,
        30,
    ),
)


# ---------------------------------------------------------
# Only retrieval tool
# ---------------------------------------------------------

@mcp.tool()
async def batch_semantic_search(
    payload: BatchSemanticSearchInput,
) -> BatchSemanticSearchOutput:
    started_at = time.perf_counter()

    try:
        pipeline = (
            await get_retrieval_pipeline()
        )   
        retrieval_started_at = (
            time.perf_counter()
        )

        try:
            output = await pipeline.search(
                payload
            )
        finally:
            RAG_RETRIEVAL_DURATION_SECONDS.observe(
                time.perf_counter()
                - retrieval_started_at
            )

    except Exception:
        RAG_BATCH_RETRIEVAL_SECONDS.labels(
            status="error",
        ).observe(
            time.perf_counter()
            - started_at
        )

        logger.exception(
            "batch_semantic_search failed | "
            "request_id=%s requests=%d",
            payload.request_id,
            len(payload.requests),
        )

        raise

    RAG_BATCH_RETRIEVAL_SECONDS.labels(
        status="success",
    ).observe(
        time.perf_counter()
        - started_at
    )

    logger.info(
        "batch_semantic_search completed | "
        "request_id=%s requests=%d "
        "qdrant_calls=%d reranker_pairs=%d",
        payload.request_id,
        len(payload.requests),
        (
            output.shared_meta
            .qdrant_network_call_count
        ),
        (
            output.shared_meta
            .reranker_pair_count
        ),
    )

    return output


def warmup() -> None:
    
    logger.info(
        "Loading direct retrieval models"
    )

    get_dense_model()
    get_sparse_model()
    get_pair_reranker()

    logger.info(
        "Direct retrieval models loaded"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    start_http_server(
        PROMETHEUS_PORT,
        addr="0.0.0.0",
    )

    logger.info(
        "Prometheus metrics listening "
        "on 0.0.0.0:%d",
        PROMETHEUS_PORT,
    )

    warmup()

    logger.info(
        "Starting MCP vector server "
        "on 0.0.0.0:%d/mcp",
        PORT,
    )

    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=PORT,
        path="/mcp",
    )