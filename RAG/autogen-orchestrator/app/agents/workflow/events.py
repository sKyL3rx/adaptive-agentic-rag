from __future__ import annotations

from typing import Literal

from agent_framework import WorkflowEvent
from pydantic import BaseModel, ConfigDict

from workflow.messages import FinalAnswer


class RetrievalProgressSignal(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    sub_question_id: str
    chunks_found: int
    failed: bool = False


class ChatEvent(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    type: str
    request_id: str


class StageEvent(ChatEvent):
    type: Literal["stage"]
    stage: str
    status: Literal[
        "started",
        "completed",
        "failed",
    ]


class RetrievalProgressEvent(ChatEvent):
    type: Literal["retrieval_progress"]
    sub_question_id: str
    chunks_found: int
    failed: bool = False


class FinalEvent(ChatEvent):
    type: Literal["final"]
    result: FinalAnswer


class ErrorEvent(ChatEvent):
    type: Literal["error"]
    error_type: str
    message: str


def map_workflow_event(
    event: WorkflowEvent,
    *,
    request_id: str,
) -> ChatEvent | None:
    if event.type == "executor_invoked":
        if event.executor_id is None:
            return None

        return StageEvent(
            type="stage",
            request_id=request_id,
            stage=event.executor_id,
            status="started",
        )

    if event.type == "executor_completed":
        if event.executor_id is None:
            return None

        return StageEvent(
            type="stage",
            request_id=request_id,
            stage=event.executor_id,
            status="completed",
        )

    if event.type == "executor_failed":
        if event.executor_id is None:
            return None

        return StageEvent(
            type="stage",
            request_id=request_id,
            stage=event.executor_id,
            status="failed",
        )

    if event.type == "retrieval_progress":
        data = event.data

        if isinstance(
            data,
            RetrievalProgressSignal,
        ):
            return RetrievalProgressEvent(
                type="retrieval_progress",
                request_id=request_id,
                sub_question_id=(
                    data.sub_question_id
                ),
                chunks_found=(
                    data.chunks_found
                ),
                failed=data.failed,
            )

        return None

    if event.type == "output":
        if not isinstance(
            event.data,
            FinalAnswer,
        ):
            return ErrorEvent(
                type="error",
                request_id=request_id,
                error_type=(
                    "INVALID_WORKFLOW_OUTPUT"
                ),
                message=(
                    "Workflow produced an "
                    "unexpected output type."
                ),
            )

        return FinalEvent(
            type="final",
            request_id=request_id,
            result=event.data,
        )

    if event.type == "failed":
        details = event.details

        return ErrorEvent(
            type="error",
            request_id=request_id,
            error_type=(
                details.error_type
                if details is not None
                else "WORKFLOW_FAILED"
            ),
            message=(
                details.message
                if details is not None
                else "Workflow execution failed."
            ),
        )

    if event.type == "error":
        exception = event.data

        return ErrorEvent(
            type="error",
            request_id=request_id,
            error_type=(
                type(exception).__name__
            ),
            message=str(exception),
        )

    return None