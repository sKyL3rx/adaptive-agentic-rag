from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agent_framework import (
    MCPStreamableHTTPTool,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

class BatchSearchCall(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    sub_question_id: str
    query: str

    top_k: int
    candidate_k: int


class BatchEvidenceHit(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    chunk_id: str
    text: str

    retrieval_score: float
    rerank_score: float | None = None

    heading_path: tuple[
        str,
        ...,
    ] = ()

    section_id: str | None = None
    source_file: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class BatchCoverageFeatures(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    candidate_count: int = 0
    selected_count: int = 0
    distinct_sections: int = 0
    distinct_headings: int = 0


class BatchSearchResultResponse(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    sub_question_id: str
    query: str

    evidence: tuple[
        BatchEvidenceHit,
        ...,
    ] = ()

    coverage_features: (
        BatchCoverageFeatures
    ) = Field(
        default_factory=(
            BatchCoverageFeatures
        )
    )


class BatchRetrievalMeta(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    request_id: str = ""
    corpus_version: str = ""

    input_request_count: int = 0
    embedding_batch_count: int = 0
    sparse_embedding_batch_count: int = 0

    qdrant_network_call_count: int = 0
    qdrant_logical_query_count: int = 0

    reranker_batch_count: int = 0
    reranker_pair_count: int = 0


class BatchSemanticSearchResponse(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    results: tuple[
        BatchSearchResultResponse,
        ...,
    ] = ()

    shared_meta: (
        BatchRetrievalMeta
    ) = Field(
        default_factory=(
            BatchRetrievalMeta
        )
    )


def _coerce_mapping(
    value: Any,
) -> dict[str, Any] | None:
    """
    Convert common MCP content values into a dictionary.
    """
    if value is None:
        return None

    if isinstance(value, BaseModel):
        return value.model_dump(
            mode="json"
        )

    if isinstance(value, Mapping):
        return dict(value)

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None

    if isinstance(value, str):
        normalized = value.strip()

        if not normalized:
            return None

        try:
            decoded = json.loads(normalized)
        except json.JSONDecodeError:
            return None

        if isinstance(decoded, Mapping):
            return dict(decoded)

    return None


def _unwrap_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Support common wrappers produced by MCP clients and servers.
    """
    current = payload

    for key in (
        "structuredContent",
        "structured_content",
        "result",
        "data",
    ):
        nested = current.get(key)

        nested_mapping = _coerce_mapping(
            nested
        )

        if nested_mapping is not None:
            current = nested_mapping
            break

    return current


def _content_error_message(
    content: object,
) -> str | None:
    content_type = getattr(
        content,
        "type",
        None,
    )

    if content_type != "error":
        return None

    message = getattr(
        content,
        "message",
        None,
    )

    if isinstance(message, str):
        return message

    details = getattr(
        content,
        "details",
        None,
    )

    if isinstance(details, str):
        return details

    return "Unknown MCP content error"


def _content_payload_candidates(
    content: object,
) -> tuple[Any, ...]:
    """
    Agent Framework versions may expose MCP results through different
    attributes. 
    """
    return (
        getattr(content, "result", None),
        getattr(content, "text", None),
        getattr(content, "data", None),
        getattr(
            content,
            "additional_properties",
            None,
        ),
        getattr(
            content,
            "raw_representation",
            None,
        ),
    )

def decode_batch_semantic_search_response(
    contents: Sequence[object],
) -> BatchSemanticSearchResponse:
    errors: list[str] = []

    for content in contents:
        error_message = (
            _content_error_message(
                content
            )
        )

        if error_message is not None:
            errors.append(
                error_message
            )
            continue

        for candidate in (
            _content_payload_candidates(
                content
            )
        ):
            payload = _coerce_mapping(
                candidate
            )

            if payload is None:
                continue

            payload = _unwrap_payload(
                payload
            )

            try:
                return (
                    BatchSemanticSearchResponse
                    .model_validate(
                        payload
                    )
                )
            except Exception:
                continue

    if errors:
        raise RuntimeError(
            "batch_semantic_search failed: "
            + "; ".join(errors)
        )

    raise ValueError(
        "batch_semantic_search returned "
        "no valid payload"
    )

class RetrievalMCPClient:
    def __init__(
        self,
        *,
        url: str,
        tool_name: str,
        timeout_seconds: int,
    ) -> None:
        self._tool_name = tool_name

        self._client = (
            MCPStreamableHTTPTool(
                name="dmv-retrieval",
                url=url,
                load_tools=True,
                load_prompts=False,
                allowed_tools=[
                    tool_name,
                ],
                request_timeout=(
                    timeout_seconds
                ),
            )
        )

    async def connect(self) -> None:
        await self._client.connect()

    async def close(self) -> None:
        await self._client.close()

    async def search_batch(
        self,
        *,
        request_id: str,
        corpus_version: str,
        requests: Sequence[
            BatchSearchCall
        ],
    ) -> BatchSemanticSearchResponse:
        if not requests:
            raise ValueError(
                "search_batch requires "
                "at least one request"
            )

        contents = await (
            self._client.call_tool(
                self._tool_name,
                payload={
                    "request_id": request_id,
                    "corpus_version": (
                        corpus_version
                    ),
                    "requests": [
                        request.model_dump(
                            mode="json"
                        )
                        for request
                        in requests
                    ],
                },
            )
        )

        return (
            decode_batch_semantic_search_response(
                contents
            )
        )