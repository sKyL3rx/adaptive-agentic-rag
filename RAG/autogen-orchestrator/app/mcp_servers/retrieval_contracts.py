from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class SearchFilters(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    section_ids: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()


class BatchSearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    sub_question_id: str = Field(
        min_length=1,
        max_length=100,
    )

    query: str = Field(
        min_length=1,
        max_length=2_000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    candidate_k: int = Field(
        default=20,
        ge=5,
        le=100,
    )

    filters: SearchFilters | None = None

    @model_validator(mode="after")
    def validate_budgets(
        self,
    ) -> "BatchSearchRequest":
        if self.candidate_k < self.top_k:
            raise ValueError(
                "candidate_k must be >= top_k"
            )

        return self

class BatchSemanticSearchInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    requests: tuple[
        BatchSearchRequest,
        ...,
    ] = Field(
        min_length=1,
        max_length=8,
    )

    corpus_version: str = Field(
        min_length=1,
    )

    request_id: str = Field(
        min_length=1,
    )

class EvidenceItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: str
    text: str

    retrieval_score: float
    rerank_score: float | None = None

    heading_path: tuple[str, ...] = ()
    section_id: str | None = None
    source_file: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

class CoverageFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    candidate_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    distinct_sections: int = Field(ge=0)
    distinct_headings: int = Field(ge=0)


class BatchSearchResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    sub_question_id: str
    query: str

    evidence: tuple[
        EvidenceItem,
        ...,
    ]

    coverage_features: CoverageFeatures

class RetrievalMeta(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    request_id: str
    corpus_version: str

    input_request_count: int = Field(
        ge=1,
    )

    embedding_batch_count: int = Field(
        ge=0,
    )

    sparse_embedding_batch_count: int = Field(
        ge=0,
    )

    qdrant_network_call_count: int = Field(
        ge=0,
    )

    qdrant_logical_query_count: int = Field(
        ge=0,
    )

    reranker_batch_count: int = Field(
        ge=0,
    )

    reranker_pair_count: int = Field(
        ge=0,
    )
class BatchSemanticSearchOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    results: tuple[
        BatchSearchResult,
        ...,
    ]

    shared_meta: RetrievalMeta

