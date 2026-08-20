from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Any

import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from sentence_transformers import (
    SentenceTransformer,
)

from onnx_reranker import (
    OnnxCrossEncoderRerank,
)

from dotenv import load_dotenv
from pathlib import Path

ENV_FILE = Path(__file__).resolve().with_name(
    ".env"
)

load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


QDRANT_HOST = os.getenv(
    "QDRANT_HOST",
    "localhost",
).strip()

QDRANT_PORT = int(
    os.getenv(
        "QDRANT_PORT",
        "6333",
    )
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    (
        f"http://{QDRANT_HOST}:"
        f"{QDRANT_PORT}"
    ),
).strip()

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
)

QDRANT_TIMEOUT_SEC = int(
    os.getenv(
        "QDRANT_TIMEOUT_SEC",
        "30",
    )
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "dmv_handbook_qwen3_06b_c450_o80_v1",
)

DENSE_VECTOR_NAME_OVERRIDE = os.getenv(
    "DENSE_VECTOR_NAME",
) or None

SPARSE_VECTOR_NAME_OVERRIDE = os.getenv(
    "SPARSE_VECTOR_NAME",
) or None

EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    (
        "/app/models/"
        "gte-modernbert-base"
    ),
).strip()

EMBEDDING_ONNX_FILE = os.getenv(
    "EMBEDDING_ONNX_FILE",
    "onnx/model_int8.onnx",
).strip()

EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "4",
    )
)

EMBEDDING_MAX_LENGTH = int(
    os.getenv(
        "EMBEDDING_MAX_LENGTH",
        "512",
    )
)

SPARSE_MODEL = "Qdrant/bm25"

RERANK_MODEL_PATH = os.getenv(
    "RERANK_MODEL_PATH",
    (
        "/app/models/"
        "gte-reranker-modernbert-base"
    ),
).strip()

RERANK_ONNX_FILE = os.getenv(
    "RERANK_ONNX_FILE",
    "onnx/model_quantized.onnx",
).strip()

RERANK_BATCH_SIZE = int(
    os.getenv(
        "RERANK_BATCH_SIZE",
        "2",
    )
)

RERANK_MAX_LENGTH = int(
    os.getenv(
        "RERANK_MAX_LENGTH",
        "512",
    )
)

MAX_RERANK_PAIRS = int(
    os.getenv(
        "MAX_RERANK_PAIRS",
        "12",
    )
)


@dataclass(frozen=True)
class ResolvedVectorNames:
    dense: str
    sparse: str
    dense_size: int

@lru_cache(maxsize=1)
def get_dense_model() -> SentenceTransformer:
    model = SentenceTransformer(
        EMBEDDING_MODEL_PATH,
        backend="onnx",
        model_kwargs={
            "file_name": (
                EMBEDDING_ONNX_FILE
            ),
            "provider": (
                "CPUExecutionProvider"
            ),
            "export": False,
        },
    )

    model.max_seq_length = (
        EMBEDDING_MAX_LENGTH
    )

    return model


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(
        model_name=SPARSE_MODEL,
    )

@lru_cache(maxsize=1)
def get_pair_reranker(
) -> OnnxCrossEncoderRerank:
    return OnnxCrossEncoderRerank(
        model_name_or_path=(
            RERANK_MODEL_PATH
        ),
        model_file=(
            RERANK_ONNX_FILE
        ),
        batch_size=(
            RERANK_BATCH_SIZE
        ),
        max_length=(
            RERANK_MAX_LENGTH
        ),
    )

@lru_cache(maxsize=1)
def get_async_qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT_SEC,
        check_compatibility=True,
        pool_size=20,
    )

def _encode_dense_sync(
    queries: list[str],
) -> np.ndarray:
    if not queries:
        return np.empty(
            shape=(0, 0),
            dtype=np.float32,
        )

    model = get_dense_model()

    prepared_queries = [
        query.strip()
        for query in queries
    ]

    embeddings = model.encode(
        prepared_queries,
        batch_size=min(
            len(prepared_queries),
            EMBEDDING_BATCH_SIZE,
        ),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


async def encode_dense_batch(
    queries: list[str],
) -> np.ndarray:
    return await asyncio.to_thread(
        _encode_dense_sync,
        queries,
    )

def _encode_sparse_sync(
    queries: list[str],
) -> list[Any]:
    if not queries:
        return []

    model = get_sparse_model()

    return list(
        model.query_embed(
            queries,
        )
    )


async def encode_sparse_batch(
    queries: list[str],
) -> list[Any]:
    return await asyncio.to_thread(
        _encode_sparse_sync,
        queries,
    )

async def resolve_vector_names(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    dense_override: str | None = None,
    sparse_override: str | None = None,
) -> ResolvedVectorNames:
    collection = await client.get_collection(
        collection_name=collection_name,
    )

    dense_config = (
        collection.config.params.vectors
    )

    sparse_config = (
        collection.config.params.sparse_vectors
        or {}
    )

    if not isinstance(
        dense_config,
        Mapping,
    ):
        raise RuntimeError(
            "Collection uses an unnamed dense vector. "
            "Migrate it to named vectors before enabling "
            "the direct hybrid production path."
        )

    if not isinstance(
        sparse_config,
        Mapping,
    ):
        raise RuntimeError(
            "Invalid sparse vector configuration."
        )

    dense_names = tuple(
        dense_config.keys()
    )

    sparse_names = tuple(
        sparse_config.keys()
    )

    if dense_override is not None:
        if dense_override not in dense_config:
            raise RuntimeError(
                "Configured dense vector does not exist: "
                f"{dense_override!r}; "
                f"available={dense_names!r}"
            )

        dense_name = dense_override

    elif len(dense_names) == 1:
        dense_name = dense_names[0]

    else:
        raise RuntimeError(
            "Unable to infer dense vector name. "
            f"Available dense vectors: {dense_names!r}"
        )

    if sparse_override is not None:
        if sparse_override not in sparse_config:
            raise RuntimeError(
                "Configured sparse vector does not exist: "
                f"{sparse_override!r}; "
                f"available={sparse_names!r}"
            )

        sparse_name = sparse_override

    elif len(sparse_names) == 1:
        sparse_name = sparse_names[0]

    else:
        raise RuntimeError(
            "Unable to infer sparse vector name. "
            f"Available sparse vectors: {sparse_names!r}"
        )

    dense_size = int(
        dense_config[dense_name].size
    )

    return ResolvedVectorNames(
        dense=dense_name,
        sparse=sparse_name,
        dense_size=dense_size,
    )

async def validate_retrieval_runtime() -> (
    ResolvedVectorNames
):
    client = get_async_qdrant()

    names = await resolve_vector_names(
        client,
        QDRANT_COLLECTION,
        dense_override=(
            DENSE_VECTOR_NAME_OVERRIDE
        ),
        sparse_override=(
            SPARSE_VECTOR_NAME_OVERRIDE
        ),
    )

    probe = await encode_dense_batch(
        ["dimension probe"]
    )

    if probe.ndim != 2:
        raise RuntimeError(
            "Dense model returned invalid shape: "
            f"{probe.shape}"
        )

    model_dimension = int(
        probe.shape[1]
    )

    if model_dimension != names.dense_size:
        raise RuntimeError(
            "Embedding dimension does not match "
            "the Qdrant collection: "
            f"model={model_dimension}, "
            f"collection={names.dense_size}"
        )

    return names

