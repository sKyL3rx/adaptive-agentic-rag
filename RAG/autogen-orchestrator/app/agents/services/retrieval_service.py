from __future__ import annotations

import asyncio

from domain.contracts import (
    EvidenceItem,
    RetrievalMetadata,
)
from providers.retrieval_mcp import (
    BatchSearchCall,
    BatchSearchResultResponse,
    BatchSemanticSearchResponse,
)
from workflow.budget import (
    reserve_retrieval_round,
)
from observability import (
    RAG_RETRIEVAL_ROUNDS_TOTAL,
)
from workflow.messages import (
    RetrievalBatchResult,
    RetrievalFailure,
    RetrievalPrepared,
    RetrievalResult,
    RetrievalTask,
)
from workflow.resources import (
    WorkflowResources,
)
from workflow.timeouts import (
    RequestDeadlineExceededError,
    StageTimeoutError,
    run_stage,
)



def _metadata_text(
    metadata: dict[str, object],
    key: str,
) -> str | None:
    value = metadata.get(key)

    if value is None:
        return None

    normalized = str(value).strip()
    return normalized or None


def normalize_batch_result(
    *,
    task: RetrievalTask,
    raw: BatchSearchResultResponse,
) -> RetrievalResult:
    """
    Convert one MCP batch-search result into the
    RetrievalResult.
    """
    evidence = tuple(
        EvidenceItem(
            node_id=(
                f"{task.sub_question_id}:"
                f"r{task.round_number}:"
                f"e{index}"
            ),

            chunk_id=hit.chunk_id,

            sub_question_id=(
                task.sub_question_id
            ),

            text=hit.text,

            source=hit.source_file,

            title=_metadata_text(
                hit.metadata,
                "title",
            ),

            url=_metadata_text(
                hit.metadata,
                "url",
            ),

            heading_path=(
                " > ".join(
                    hit.heading_path
                )
                if hit.heading_path
                else None
            ),

            section_id=hit.section_id,

            retrieval_score=(
                hit.retrieval_score
            ),

            rerank_score=(
                hit.rerank_score
            ),
        )
        for index, hit in enumerate(
            raw.evidence,
            start=1,
        )
    )

    return RetrievalResult(
        request_id=task.request_id,
        round_number=(
            task.round_number
        ),
        sub_question_id=(
            task.sub_question_id
        ),
        query=task.query,
        evidence=evidence,
        metadata=RetrievalMetadata(
            cache_hit=False,
            degraded=False,
            reason=None,
            query_used=raw.query,
        ),
    )


def is_retryable(
    exc: Exception,
) -> bool:
    """
    Decide whether an entire retrieval batch may be
    retried by the surrounding workflow.
    """
    if isinstance(
        exc,
        RequestDeadlineExceededError,
    ):
        return False

    return isinstance(
        exc,
        (
            StageTimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    )


def build_failure_result(
    *,
    task: RetrievalTask,
    round_number: int,
    error_type: str,
    message: str,
    retryable: bool,
) -> RetrievalResult:
    """
    Build a failure result for one 
    sub-question while preserving its ID.
    """
    return RetrievalResult(
        request_id=task.request_id,
        round_number=round_number,
        sub_question_id=(
            task.sub_question_id
        ),
        query=task.query,
        failure=RetrievalFailure(
            error_type=error_type,
            message=message,
            retryable=retryable,
        ),
    )


async def retrieve_batch(
    message: RetrievalPrepared,
    resources: WorkflowResources,
) -> RetrievalBatchResult:
    """
    Execute all logical retrieval tasks through one
    batch_semantic_search MCP call.

    this function should produce:

    - one MCP tool call
    - one dense embedding batch
    - one sparse embedding batch
    - one Qdrant query_batch_points call
    - at most one reranker batch
    """
    if not message.tasks:
        raise ValueError(
            "Retrieval batch requires at "
            "least one task"
        )

    state = reserve_retrieval_round(
        message.state
    )
    
    RAG_RETRIEVAL_ROUNDS_TOTAL.inc()

    current_round = (
        state.retrieval_rounds_used
    )

    valid_tasks: list[
        RetrievalTask
    ] = []

    invalid_round_results: dict[
        str,
        RetrievalResult,
    ] = {}

    for task in message.tasks:
        if (
            task.round_number
            != current_round
        ):
            invalid_round_results[
                task.sub_question_id
            ] = build_failure_result(
                task=task,
                round_number=current_round,
                error_type=(
                    "INVALID_ROUND_NUMBER"
                ),
                message=(
                    "Retrieval task round does "
                    "not match the current "
                    "workflow retrieval round"
                ),
                retryable=False,
            )

            continue

        valid_tasks.append(
            task
        )

    raw_batch: (
        BatchSemanticSearchResponse
        | None
    ) = None

    batch_error: Exception | None = None

    if valid_tasks:
        batch_requests = tuple(
            BatchSearchCall(
                sub_question_id=(
                    task.sub_question_id
                ),
                query=task.query,
                top_k=task.top_k,
                candidate_k=(
                    task.candidate_k
                ),
            )
            for task in valid_tasks
        )

        try:
            raw_batch = await run_stage(
                state,
                stage="batch_retrieval",
                stage_timeout_seconds=(
                    resources.settings
                    .retrieval_timeout_seconds
                ),
                operation=lambda: (
                    resources.retrieval
                    .search_batch(
                        request_id=(
                            state.request_id
                        ),
                        corpus_version=(
                            resources.settings
                            .retrieval_corpus_version
                        ),
                        requests=(
                            batch_requests
                        ),
                    )
                ),
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            batch_error = exc

    raw_by_sub_question_id: dict[
        str,
        BatchSearchResultResponse,
    ] = {}

    duplicate_result_ids: set[
        str
    ] = set()

    if raw_batch is not None:
        for raw_result in raw_batch.results:
            sub_question_id = (
                raw_result
                .sub_question_id
            )

            if (
                sub_question_id
                in raw_by_sub_question_id
            ):
                duplicate_result_ids.add(
                    sub_question_id
                )
                continue

            raw_by_sub_question_id[
                sub_question_id
            ] = raw_result

    final_results: list[
        RetrievalResult
    ] = []


    for task in message.tasks:
        invalid_result = (
            invalid_round_results.get(
                task.sub_question_id
            )
        )

        if invalid_result is not None:
            final_results.append(
                invalid_result
            )
            continue

        if batch_error is not None:
            final_results.append(
                build_failure_result(
                    task=task,
                    round_number=(
                        current_round
                    ),
                    error_type=(
                        type(
                            batch_error
                        ).__name__
                    ),
                    message=str(
                        batch_error
                    ),
                    retryable=(
                        is_retryable(
                            batch_error
                        )
                    ),
                )
            )

            continue

        if (
            task.sub_question_id
            in duplicate_result_ids
        ):
            final_results.append(
                build_failure_result(
                    task=task,
                    round_number=(
                        current_round
                    ),
                    error_type=(
                        "DUPLICATE_BATCH_RESULT"
                    ),
                    message=(
                        "Batch retrieval returned "
                        "more than one result for "
                        "the same sub-question"
                    ),
                    retryable=False,
                )
            )

            continue

        raw_result = (
            raw_by_sub_question_id.get(
                task.sub_question_id
            )
        )

        if raw_result is None:
            final_results.append(
                build_failure_result(
                    task=task,
                    round_number=(
                        current_round
                    ),
                    error_type=(
                        "MISSING_BATCH_RESULT"
                    ),
                    message=(
                        "Batch retrieval did not "
                        "return a result for this "
                        "sub-question"
                    ),
                    retryable=False,
                )
            )

            continue

        try:
            normalized_result = (
                normalize_batch_result(
                    task=task,
                    raw=raw_result,
                )
            )

        except Exception as exc:
            final_results.append(
                build_failure_result(
                    task=task,
                    round_number=(
                        current_round
                    ),
                    error_type=(
                        "INVALID_BATCH_RESULT"
                    ),
                    message=str(exc),
                    retryable=False,
                )
            )

            continue

        final_results.append(
            normalized_result
        )

    return RetrievalBatchResult(
        state=state,
        results=tuple(
            final_results
        ),
    )