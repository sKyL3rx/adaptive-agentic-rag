from typing import Literal

from pydantic import BaseModel, ConfigDict

from domain.contracts import (
    CitationSource,
    CoverageStatus,
    DraftPayload,
    EvidenceItem,
    QuestionShape,
    RetrievalMetadata,
    ReviewDecision,
)
from domain.state import RequestState, Route, TerminationReason

class WorkflowMessage(BaseModel):
    model_config = ConfigDict(
        extra = "forbid",
        frozen = True,
    )

class WorkflowInput(WorkflowMessage):
    state: RequestState


class CompiledRequest(WorkflowMessage):
    state: RequestState


class ContextResolved(WorkflowMessage):
    state: RequestState

class RetrievalTask(WorkflowMessage):
    request_id: str
    round_number: int
    sub_question_id: str
    query: str
    top_k: int
    candidate_k: int


class RetrievalPrepared(WorkflowMessage):
    state: RequestState
    tasks: tuple[RetrievalTask, ...]

class RetrievalFailure(WorkflowMessage):
    error_type: str
    message: str
    retryable: bool = False


class RetrievalResult(WorkflowMessage):
    request_id: str
    round_number: int
    sub_question_id: str
    query: str

    evidence: tuple[EvidenceItem, ...] = ()
    metadata: RetrievalMetadata | None = None
    failure: RetrievalFailure | None = None

class RetrievalBatchResult(WorkflowMessage):
    state: RequestState
    results: tuple[RetrievalResult, ...]

class EvidenceReady(WorkflowMessage):
    state: RequestState


class AnswerCandidate(WorkflowMessage):
    state: RequestState
    draft: DraftPayload


class ReviewCompleted(WorkflowMessage):
    state: RequestState
    decision: ReviewDecision


class BoundedAnswer(WorkflowMessage):
    state: RequestState
    answer: str
    reason: TerminationReason


class RetrievalEvaluationGroup(WorkflowMessage):
    sub_question_id: str
    query: str
    chunk_ids: tuple[str, ...] = ()


class EvaluationTelemetry(WorkflowMessage):
    question_shape: QuestionShape | None = None
    route: Route | None = None
    retrieval_mode: Literal["single", "decomposed"] | None = None
    coverage_status: CoverageStatus | None = None
    retrieval_groups: tuple[
        RetrievalEvaluationGroup,
        ...,
    ] = ()


class FinalAnswer(WorkflowMessage):
    request_id: str
    answer: str
    citations: tuple[CitationSource, ...]
    termination_reason: TerminationReason
    model_calls_used: int
    retrieval_rounds_used: int
    evaluation: EvaluationTelemetry | None = None

