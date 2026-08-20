from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

class QuestionShape(str, Enum):
    CASUAL_CONVERSATION = "casual_conversation"
    CONVERSATION_RECALL = "conversation_recall"

    SINGLE_FOCUSED = "single_focused"
    BROAD_COVERAGE = "broad_coverage"
    MULTI_PART = "multi_part"
    COMPARISON = "comparison"
    CONTEXT_DEPENDENT = "context_dependent"

ConversationRecallTarget = Literal[
    "none",
    "last_user_message",
    "last_assistant_message",
]

class QuestionShapeDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    version: Literal["v1"] = "v1"
    shape: QuestionShape
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    reasoning: str = ""

    direct_response: str | None = None

    recall_target: ConversationRecallTarget = (
        "none"
    )

    @model_validator(mode="after")
    def validate_response_mode(
        self,
    ) -> "QuestionShapeDecision":
        direct_response = (
            self.direct_response or ""
        ).strip()

        if (
            self.shape
            == QuestionShape.CASUAL_CONVERSATION
        ):
            if not direct_response:
                raise ValueError(
                    "casual_conversation requires "
                    "direct_response"
                )

            if self.recall_target != "none":
                raise ValueError(
                    "casual_conversation must not set "
                    "recall_target"
                )

            return self

        if (
            self.shape
            == QuestionShape.CONVERSATION_RECALL
        ):
            if self.recall_target == "none":
                raise ValueError(
                    "conversation_recall requires "
                    "recall_target"
                )

            if direct_response:
                raise ValueError(
                    "conversation_recall must not return "
                    "direct_response"
                )

            return self

        if direct_response:
            raise ValueError(
                "Retrieval question shapes must not "
                "return direct_response"
            )

        if self.recall_target != "none":
            raise ValueError(
                "Retrieval question shapes must not "
                "set recall_target"
            )

        return self


class SubQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    id: str
    question: str
    query: str

class PlannedSubQuestion(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    query: str = Field(min_length=1)

class RetrievalPlanProposal(BaseModel):
    """
    Raw structured output of planner model.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    version: Literal["v1"] = "v1"
    sub_questions: tuple[
        PlannedSubQuestion,
        ...
    ]
    parallelizable: bool = True

    @model_validator(mode="after")
    def validate_questions(
        self,
    ) -> "RetrievalPlanProposal":
        if not self.sub_questions:
            raise ValueError(
                "Planner must return at least "
                "one sub-question"
            )

        return self
    
class RetrievalPlan(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    version: Literal["v1"] = "v1"
    mode: Literal["single", "decomposed"]
    shape: QuestionShape
    sub_questions: tuple[SubQuestion, ...]
    parallelizable: bool = False

    @model_validator(mode="after")
    def validate_plan(
        self,
    ) -> "RetrievalPlan":
        if not self.sub_questions:
            raise ValueError(
                "RetrievalPlan requires at least "
                "one sub-question"
            )

        ids = [
            item.id
            for item in self.sub_questions
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Sub-question IDs must be unique"
            )

        return self

class EvidenceItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    node_id: str
    chunk_id: str
    sub_question_id: str
    text: str

    source: str | None = None
    title: str | None = None
    url: str | None = None

    heading_path: str | None = None
    section_id: str | None = None

    retrieval_score: float | None = None
    rerank_score: float | None = None

class CitationSource(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source_id: str
    title: str

    source_file: str | None = None
    section_id: str | None = None
    heading_path: str | None = None
    url: str | None = None
    excerpt: str | None = None


class RetrievalMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    cache_hit: bool = False
    degraded: bool = False
    reason: str | None = None
    query_used: str | None = None

class CoverageStatus(str, Enum):
    SUFFICIENT = "sufficient"
    RECOVERABLE_GAP = "recoverable_gap"
    INSUFFICIENT = "insufficient"

class CoverageDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: CoverageStatus
    covered_sub_questions: tuple[str, ...] = ()
    missing_sub_questions: tuple[str, ...] = ()
    next_queries: tuple[SubQuestion, ...] = ()
    reason: str = ""


class DraftSupport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    sub_question_id: str
    evidence_node_ids: tuple[str, ...]

class DraftPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    version: Literal["v1"] = "v1"
    support: tuple[DraftSupport, ...]
    merged_answer: str

class ReviewVerdict(str, Enum):
    APPROVE = "approve"
    NEEDS_MORE = "needs_more"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

class ReviewQueryProposal(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    question: str = Field(min_length=1)
    query: str = Field(min_length=1)

class ReviewDecisionProposal(BaseModel):
    """
    Raw output of reviewer.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    version: Literal["v1"] = "v1"
    verdict: ReviewVerdict

    missing_sub_questions: tuple[str, ...] = ()
    next_queries: tuple[
        ReviewQueryProposal,
        ...
    ] = ()

    reason: str | None = None

class ReviewDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    version: Literal["v1"] = "v1"
    verdict: ReviewVerdict
    missing_sub_questions: tuple[str, ...] = ()
    next_queries: tuple[SubQuestion, ...] = ()
    reason: str | None = None

class ContextRewritePayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    version: Literal["v1"] = "v1"
    can_resolve: bool
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    dependency_type: str
    anchor_topic: str | None = None
    standalone_question: str | None = None
    query: str | None = None

    rewritten_shape: QuestionShape
    needs_full_planner: bool = False

    @model_validator(mode="after")
    def validate_resolution(
        self,
    ) -> "ContextRewritePayload":
        standalone = (
            self.standalone_question or ""
        ).strip()

        query = (
            self.query or ""
        ).strip()

        if self.can_resolve:
            if not standalone and not query:
                raise ValueError(
                    "Resolved context requires a "
                    "standalone question or query"
                )

            if self.rewritten_shape in {
                QuestionShape.CASUAL_CONVERSATION,
                QuestionShape.CONVERSATION_RECALL,
                QuestionShape.CONTEXT_DEPENDENT,
            }:
                raise ValueError(
                    "Resolved DMV context must become a "
                    "retrieval question shape"
                )

        return self