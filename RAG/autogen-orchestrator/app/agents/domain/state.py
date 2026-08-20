from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from domain.contracts import (
    ContextRewritePayload,
    ConversationRecallTarget,
    CoverageDecision,
    DraftPayload,
    EvidenceItem,
    QuestionShape,
    RetrievalPlan,
    ReviewDecision,
    SubQuestion,
)


class Route(str, Enum):

    CASUAL_RESPONSE = "casual_response"
    RECALL_RESPONSE = "recall_response"

    SINGLE_FAST = "single_fast"
    BROAD_ADAPTIVE = "broad_adaptive"
    CONTEXT_FAST = "context_fast"
    MULTI_BATCH = "multi_batch"
    COMPARISON_BATCH = "comparison_batch"


class TerminationReason(str, Enum):
    CASUAL_RESPONSE = "CASUAL_RESPONSE"
    CONVERSATION_RECALL = "CONVERSATION_RECALL"

    FINAL_ANSWER = "FINAL_ANSWER"
    AUTO_APPROVED = "AUTO_APPROVED"
    APPROVED_REVIEW = "APPROVED_REVIEW"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class HistoryTurn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    role: str
    content: str


class ExecutionBudget(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    max_model_calls: int = Field(default=3, ge=0)
    max_retrieval_rounds: int = Field(default=2, ge=0)
    max_sub_questions: int = Field(default=4, ge=1)
    deadline_ms: int = Field(default=30_000, ge=1)


class FailureDetail(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    stage: str
    error_type: str
    message: str
    retryable: bool = False


class RequestState(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    request_id: str
    session_id: str
    user_query: str

    received_at: datetime
    deadline_at: datetime

    recent_history: tuple[HistoryTurn, ...] = ()

    shape: QuestionShape | None = None
    shape_confidence: float = 0.0

    resolved_query: str = ""
    context_was_rewritten: bool = False
    context_rewrite: ContextRewritePayload | None = None

    route: Route | None = None

    retrieval_plan: RetrievalPlan | None = None
    evidence: tuple[EvidenceItem, ...] = ()
    coverage: CoverageDecision | None = None

    draft: DraftPayload | None = None
    review: ReviewDecision | None = None

    pending_gap_queries: tuple[SubQuestion, ...] = ()

    budget: ExecutionBudget = Field(
        default_factory=ExecutionBudget
    )

    model_calls_used: int = 0
    retrieval_rounds_used: int = 0

    termination_reason: TerminationReason | None = None
    failure: FailureDetail | None = None

    direct_response: str | None = None
    recall_target: ConversationRecallTarget = ( "none" )    

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        session_id: str,
        user_query: str,
        recent_history: tuple[HistoryTurn, ...],
        budget: ExecutionBudget,
    ) -> "RequestState":
        now = datetime.now(timezone.utc)

        return cls(
            request_id=request_id,
            session_id=session_id,
            user_query=user_query.strip(),
            resolved_query=user_query.strip(),
            received_at=now,
            deadline_at=now
            + timedelta(milliseconds=budget.deadline_ms),
            recent_history=recent_history,
            budget=budget,
        )

    def remaining_seconds(self) -> float:
        remaining = (
            self.deadline_at
            - datetime.now(timezone.utc)
        ).total_seconds()

        return max(0.0, remaining)