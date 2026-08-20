from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import models

from retrieval_contracts import (
    BatchSearchRequest,
    BatchSearchResult,
    BatchSemanticSearchInput,
    BatchSemanticSearchOutput,
    CoverageFeatures,
    EvidenceItem,
    RetrievalMeta,
    SearchFilters,
)

from retrieval_metrics import (
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_EMBEDDING_DURATION_SECONDS,
    RAG_QDRANT_BATCH_SIZE,
    RAG_QDRANT_DURATION_SECONDS,
    RAG_RERANKER_DURATION_SECONDS,
    RAG_RERANKER_PAIRS,
)

from retrieval_runtime import (
    MAX_RERANK_PAIRS,
    QDRANT_COLLECTION,
    ResolvedVectorNames,
    encode_dense_batch,
    encode_sparse_batch,
    get_async_qdrant,
    get_pair_reranker,
    validate_retrieval_runtime,
)

logger = logging.getLogger(
    "vector_mcp.retrieval_pipeline"
)

@dataclass(
    frozen=True,
    slots=True,
)
class RetrievalCandidate:
    query_index: int

    point_id: str
    chunk_id: str
    text: str

    retrieval_score: float

    heading_path: tuple[str, ...]
    section_id: str | None
    source_file: str | None

    metadata: dict[str, Any]

def parse_heading_path(
    value: Any,
) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        return tuple(
            part.strip()
            for part in value.split(">")
            if part.strip()
        )

    if isinstance(
        value,
        (list, tuple),
    ):
        return tuple(
            str(part).strip()
            for part in value
            if str(part).strip()
        )

    cleaned = str(value).strip()

    return (
        (cleaned,)
        if cleaned
        else ()
    )

def parse_qdrant_point(
    point: models.ScoredPoint,
    *,
    query_index: int,
) -> RetrievalCandidate | None:
    raw_payload = dict(
        point.payload or {}
    )

    node_data: dict[str, Any] = {}

    serialized_node = raw_payload.get(
        "_node_content"
    )

    if isinstance(
        serialized_node,
        str,
    ):
        try:
            decoded = json.loads(
                serialized_node
            )

            if isinstance(
                decoded,
                dict,
            ):
                node_data = decoded

        except json.JSONDecodeError:
            logger.warning(
                "invalid_node_content_json | "
                "point_id=%s",
                point.id,
            )

    raw_node_metadata = node_data.get(
        "metadata",
        {},
    )

    node_metadata = (
        dict(raw_node_metadata)
        if isinstance(
            raw_node_metadata,
            dict,
        )
        else {}
    )

    metadata = {
        **node_metadata,
        **raw_payload,
    }

    text = (
        raw_payload.get("text")
        or node_data.get("text")
        or metadata.get("text")
        or ""
    )

    text = str(text).strip()

    if not text:
        logger.warning(
            "empty_qdrant_text_skipped | "
            "point_id=%s query_index=%d",
            point.id,
            query_index,
        )

        return None

    chunk_id = str(
        metadata.get(
            "stable_chunk_id"
        )
        or metadata.get(
            "chunk_id"
        )
        or point.id
    )

    source_file_value = (
        metadata.get("file_name")
        or metadata.get("source_file")
    )

    section_id_value = metadata.get(
        "section_id"
    )

    public_metadata_keys = (
        "document_id",
        "source_document_id",
        "title",
        "url",
        "page",
        "section_name",
        "corpus_version",
    )

    public_metadata = {
        key: metadata[key]
        for key in public_metadata_keys
        if key in metadata
    }

    return RetrievalCandidate(
        query_index=query_index,
        point_id=str(point.id),
        chunk_id=chunk_id,
        text=text,
        retrieval_score=float(
            point.score or 0.0
        ),
        heading_path=(
            parse_heading_path(
                metadata.get(
                    "heading_path"
                )
            )
        ),
        section_id=(
            str(section_id_value).strip()
            if section_id_value
            else None
        ),
        source_file=(
            str(source_file_value).strip()
            if source_file_value
            else None
        ),
     
        metadata=public_metadata,
    )

def allocate_rerank_candidates(
    candidates_by_query: Sequence[
        Sequence[
            RetrievalCandidate
        ]
    ],
    *,
    max_pairs: int,
) -> tuple[
    tuple[
        int,
        RetrievalCandidate,
    ],
    ...,
]:
    if max_pairs <= 0:
        return ()

    allocated: list[
        tuple[
            int,
            RetrievalCandidate,
        ]
    ] = []

    positions = [
        0 
        for _ in candidates_by_query
    ]

    while len(allocated) < max_pairs:
        added_in_round = False

        for query_index, candidates in enumerate(
            candidates_by_query
        ):
            position = positions[
                query_index
            ]

            if position >= len(candidates):
                continue

            allocated.append(
                (
                    query_index,
                    candidates[position],
                )
            )

            positions[
                query_index
            ] += 1

            added_in_round = True

            if len(allocated) >= max_pairs:
                break

        if not added_in_round:
            break

    return tuple(allocated)

async def rerank_candidates(
    *,
    requests: tuple[
        BatchSearchRequest,
        ...,
    ],
    candidates_by_query: tuple[
        tuple[
            RetrievalCandidate,
            ...,
        ],
        ...,
    ],
    max_pairs: int,
) -> tuple[
    tuple[
        tuple[
            EvidenceItem,
            ...,
        ],
        ...,
    ],
    int,
]:
    allocated = allocate_rerank_candidates(
        candidates_by_query,
        max_pairs=max_pairs,
    )

    if not allocated:
        return (
            tuple(
                ()
                for _ in requests
            ),
            0,
        )

    pairs = [
        (
            requests[
                query_index
            ].query,
            candidate.text,
        )
        for (
            query_index,
            candidate,
        ) in allocated
    ]

    reranker = get_pair_reranker()

    RAG_RERANKER_PAIRS.observe(
        len(pairs)
    )
    reranker_started_at = (
        time.perf_counter()
    )
    
    try:
        scores = await asyncio.to_thread(
            reranker.predict_pairs,
            pairs,
        )

    finally:
        RAG_RERANKER_DURATION_SECONDS.observe(
            time.perf_counter()
            - reranker_started_at
        )

    if len(scores) != len(pairs):
        raise RuntimeError(
            "Reranker score count mismatch: "
            f"scores={len(scores)}, "
            f"pairs={len(pairs)}"
        )

    scored_by_query: list[
        list[
            tuple[
                RetrievalCandidate,
                float,
            ]
        ]
    ] = [
        []
        for _ in requests
    ]

    for (
        query_index,
        candidate,
    ), score in zip(
        allocated,
        scores,
        strict=True,
    ):
        scored_by_query[
            query_index
        ].append(
            (
                candidate,
                float(score),
            )
        )

    evidence_by_query: list[
        tuple[
            EvidenceItem,
            ...,
        ]
    ] = []

    for query_index, scored in enumerate(
        scored_by_query
    ):
        scored.sort(
            key=lambda item: ( item[1], item[0].retrieval_score ),
            reverse=True,
        )

        top_k = requests[
            query_index
        ].top_k

        selected = scored[:top_k]

        evidence_by_query.append(
            tuple(
                EvidenceItem(
                    chunk_id=(
                        candidate.chunk_id
                    ),
                    text=candidate.text,
                    retrieval_score=(
                        candidate
                        .retrieval_score
                    ),
                    rerank_score=score,
                    heading_path=(
                        candidate
                        .heading_path
                    ),
                    section_id=(
                        candidate
                        .section_id
                    ),
                    source_file=(
                        candidate
                        .source_file
                    ),
                    metadata=(
                        candidate.metadata
                    ),
                )
                for candidate, score
                in selected
            )
        )

    return (
        tuple(evidence_by_query),
        len(pairs),
    )

def build_qdrant_filter(
    filters: SearchFilters | None,
) -> models.Filter | None:
    if filters is None:
        return None

    must: list[
        models.FieldCondition
    ] = []

    if filters.section_ids:
        must.append(
            models.FieldCondition(
                key="section_id",
                match=models.MatchAny(
                    any=list(
                        filters.section_ids
                    )
                ),
            )
        )

    if filters.source_files:
        must.append(
            models.FieldCondition(
                key="file_name",
                match=models.MatchAny(
                    any=list(
                        filters.source_files
                    )
                ),
            )
        )

    if not must:
        return None

    return models.Filter(
        must=must
    )

def sparse_embedding_to_lists(
    sparse_embedding: Any,
) -> tuple[
    list[int],
    list[float],
]:
    raw_indices = getattr(
        sparse_embedding,
        "indices",
        None,
    )

    raw_values = getattr(
        sparse_embedding,
        "values",
        None,
    )

    if raw_indices is None:
        raise RuntimeError(
            "Sparse embedding has no indices"
        )

    if raw_values is None:
        raise RuntimeError(
            "Sparse embedding has no values"
        )

    indices = (
        raw_indices.tolist()
        if hasattr(
            raw_indices,
            "tolist",
        )
        else list(raw_indices)
    )

    values = (
        raw_values.tolist()
        if hasattr(
            raw_values,
            "tolist",
        )
        else list(raw_values)
    )

    if len(indices) != len(values):
        raise RuntimeError(
            "Sparse indices/value length mismatch"
        )

    return (
        [
            int(index)
            for index in indices
        ],
        [
            float(value)
            for value in values
        ],
    )

def build_hybrid_request(
    *,
    dense_vector: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    dense_vector_name: str,
    sparse_vector_name: str,
    candidate_k: int,
    query_filter: models.Filter | None,
) -> models.QueryRequest:
    if not dense_vector:
        raise ValueError(
            "dense_vector must not be empty"
        )

    if len(sparse_indices) != len(
        sparse_values
    ):
        raise ValueError(
            "Sparse indices and values "
            "must have equal lengths"
        )

    return models.QueryRequest(
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=dense_vector_name,
                limit=candidate_k,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using=sparse_vector_name,
                limit=candidate_k,
                filter=query_filter,
            ),
        ],
        query=models.RrfQuery(
            rrf=models.Rrf(),
        ),
        filter=query_filter,
        limit=candidate_k,
        with_payload=True,
        with_vector=False,
    )

def build_coverage_features(
    *,
    candidates: tuple[
        RetrievalCandidate,
        ...,
    ],
    evidence: tuple[
        EvidenceItem,
        ...,
    ],
) -> CoverageFeatures:
    sections = {
        item.section_id
        for item in evidence
        if item.section_id
    }

    headings = {
        item.heading_path
        for item in evidence
        if item.heading_path
    }

    return CoverageFeatures(
        candidate_count=len(candidates),
        selected_count=len(evidence),
        distinct_sections=len(
            sections
        ),
        distinct_headings=len(
            headings
        ),
    )

class RetrievalPipeline:
    def __init__(
        self,
        *,
        vector_names: ResolvedVectorNames,
        max_rerank_pairs: int,
    ) -> None:
        self._client = (
            get_async_qdrant()
        )

        self._vector_names = (
            vector_names
        )

        self._max_rerank_pairs = (
            max_rerank_pairs
        )

    async def search(
        self,
        payload: BatchSemanticSearchInput,
    ) -> BatchSemanticSearchOutput:
        requests = payload.requests

        queries = [
            request.query.strip()
            for request in requests
        ]

        if not queries:
            raise ValueError(
                "Batch contains no queries"
            )
        RAG_EMBEDDING_BATCH_SIZE.observe(
            len(queries)
        )
        
        embedding_started_at = (
            time.perf_counter()
        )

        try:

            dense_vectors = (
                await encode_dense_batch(
                    queries
                )
            )

            sparse_vectors = (
                await encode_sparse_batch(
                    queries
                )
            )

        finally:
            RAG_EMBEDDING_DURATION_SECONDS.observe(
                time.perf_counter()
                - embedding_started_at
            )

        if len(dense_vectors) != len(
            requests
        ):
            raise RuntimeError(
                "Dense embedding count mismatch: "
                f"embeddings={len(dense_vectors)}, "
                f"requests={len(requests)}"
            )

        if len(sparse_vectors) != len(
            requests
        ):
            raise RuntimeError(
                "Sparse embedding count mismatch: "
                f"embeddings={len(sparse_vectors)}, "
                f"requests={len(requests)}"
            )

        qdrant_requests: list[
            models.QueryRequest
        ] = []

        for request_index, request in enumerate(
            requests
        ):
            (
                sparse_indices,
                sparse_values,
            ) = sparse_embedding_to_lists(
                sparse_vectors[
                    request_index
                ]
            )

            dense_vector = [
                float(value)
                for value
                in dense_vectors[
                    request_index
                ].tolist()
            ]

            qdrant_requests.append(
                build_hybrid_request(
                    dense_vector=(
                        dense_vector
                    ),
                    sparse_indices=(
                        sparse_indices
                    ),
                    sparse_values=(
                        sparse_values
                    ),
                    dense_vector_name=(
                        self
                        ._vector_names
                        .dense
                    ),
                    sparse_vector_name=(
                        self
                        ._vector_names
                        .sparse
                    ),
                    candidate_k=(
                        request.candidate_k
                    ),
                    query_filter=(
                        build_qdrant_filter(
                            request.filters
                        )
                    ),
                )
            )


        RAG_QDRANT_BATCH_SIZE.observe(
            len(qdrant_requests)
        )

        qdrant_started_at = (
            time.perf_counter()
        )

        try:
            responses = await (
                self._client
                .query_batch_points(
                    collection_name=(
                        QDRANT_COLLECTION
                    ),
                    requests=(
                        qdrant_requests
                    ),
                )
            )

        finally:
            RAG_QDRANT_DURATION_SECONDS.observe(
                time.perf_counter()
                - qdrant_started_at
            )

        if len(responses) != len(
            requests
        ):
            raise RuntimeError(
                "Qdrant response count mismatch: "
                f"responses={len(responses)}, "
                f"requests={len(requests)}"
            )

        candidates_by_query: list[
            tuple[
                RetrievalCandidate,
                ...,
            ]
        ] = []

        for query_index, response in enumerate(
            responses
        ):
            candidates: list[
                RetrievalCandidate
            ] = []

            seen_chunk_ids: set[str] = set()

            for point in response.points:
                candidate = (
                    parse_qdrant_point(
                        point,
                        query_index=(
                            query_index
                        ),
                    )
                )

                if candidate is None:
                    continue

                if (
                    candidate.chunk_id
                    in seen_chunk_ids
                ):
                    continue

                seen_chunk_ids.add(
                    candidate.chunk_id
                )

                candidates.append(
                    candidate
                )

            candidates_by_query.append(
                tuple(candidates)
            )

        (
            evidence_by_query,
            reranker_pair_count,
        ) = await rerank_candidates(
            requests=requests,
            candidates_by_query=tuple(
                candidates_by_query
            ),
            max_pairs=(
                self._max_rerank_pairs
            ),
        )

        results = tuple(
            BatchSearchResult(
                sub_question_id=(
                    request
                    .sub_question_id
                ),
                query=request.query,
                evidence=(
                    evidence_by_query[
                        request_index
                    ]
                ),
                coverage_features=(
                    build_coverage_features(
                        candidates=(
                            candidates_by_query[
                                request_index
                            ]
                        ),
                        evidence=(
                            evidence_by_query[
                                request_index
                            ]
                        ),
                    )
                ),
            )
            for request_index, request
            in enumerate(requests)
        )

        return BatchSemanticSearchOutput(
            results=results,
            shared_meta=RetrievalMeta(
                request_id=(
                    payload.request_id
                ),
                corpus_version=(
                    payload.corpus_version
                ),
                input_request_count=(
                    len(requests)
                ),
                embedding_batch_count=1,
                sparse_embedding_batch_count=1,
                qdrant_network_call_count=1,
                qdrant_logical_query_count=(
                    len(qdrant_requests)
                ),
                reranker_batch_count=(
                    1
                    if reranker_pair_count > 0
                    else 0
                ),
                reranker_pair_count=(
                    reranker_pair_count
                ),
            ),
        )

_pipeline: RetrievalPipeline | None = None
_pipeline_lock = asyncio.Lock()


async def get_retrieval_pipeline(
) -> RetrievalPipeline:
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    async with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        vector_names = (
            await validate_retrieval_runtime()
        )

        _pipeline = RetrievalPipeline(
            vector_names=vector_names,
            max_rerank_pairs=(
                MAX_RERANK_PAIRS
            ),
        )

        return _pipeline

     