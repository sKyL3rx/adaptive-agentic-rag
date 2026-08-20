from __future__ import annotations

import os

from typing import Any, Literal

from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict

from domain.state import TerminationReason
from workflow.events import (
    ChatEvent,
    ErrorEvent,
    FinalEvent,
    RetrievalProgressEvent,
    StageEvent,
)


PublicEventType = Literal[
    "request_accepted",
    "retrieval_started",
    "evaluation",
    "answer_delta",
    "citation",
    "completed",
    "error",
]


ENABLE_EVAL_TELEMETRY = os.getenv(
    "ENABLE_EVAL_TELEMETRY",
    "false",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class PublicChatEvent(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    type: PublicEventType
    request_id: str
    sequence: int
    payload: dict[str, Any]


class ChatEventStream:
    def __init__(
        self,
        *,
        websocket: WebSocket,
        request_id: str,
    ) -> None:
        self._websocket = websocket
        self._request_id = request_id
        self._sequence = 0

        self._retrieval_started_sent = False

    async def send(
        self,
        event_type: PublicEventType,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._sequence += 1

        event = PublicChatEvent(
            type=event_type,
            request_id=self._request_id,
            sequence=self._sequence,
            payload=payload or {},
        )

        await self._websocket.send_json(
            event.model_dump(
                mode="json"
            )
        )

    async def send_retrieval_started_once(
        self,
    ) -> None:
        if self._retrieval_started_sent:
            return

        self._retrieval_started_sent = True

        await self.send(
            "retrieval_started",
            {
                "message": (
                    "Searching relevant DMV "
                    "handbook sections."
                ),
            },
        )


async def send_chat_event(
    stream: ChatEventStream,
    event: ChatEvent,
) -> None:

    # Internal executor lifecycle stays internal.
    if isinstance(
        event,
        StageEvent,
    ):
        return

    # Avoid exposing sub-question IDs or other
    # internal retrieval planning information.
    if isinstance(
        event,
        RetrievalProgressEvent,
    ):
        await stream.send_retrieval_started_once()
        return

    # Workflow errors are handled in main.py where
    # they can be mapped to a safe public message.
    if isinstance(
        event,
        ErrorEvent,
    ):
        return

    if isinstance(
        event,
        FinalEvent,
    ):
        result = event.result

        reason = result.termination_reason

        if (
            ENABLE_EVAL_TELEMETRY
            and result.evaluation is not None
        ):
            await stream.send(
                "evaluation",
                result.evaluation.model_dump(
                    mode="json"
                ),
            )

        for index, citation in enumerate(
            result.citations,
            start=1,
        ):
            await stream.send(
                "citation",
                {
                    "index": index,
                    "source": (
                        citation.model_dump(
                            mode="json"
                        )
                    ),
                },
            )

        await stream.send(
            "completed",
            {
                "answer": result.answer,

                "termination_reason":
                    reason.value,

                "model_calls_used":
                    result.model_calls_used,

                "retrieval_rounds_used":
                    result.retrieval_rounds_used,

                "insufficient_evidence":
                    reason
                    == TerminationReason
                    .INSUFFICIENT_EVIDENCE,

                "budget_exhausted":
                    reason
                    == TerminationReason
                    .BUDGET_EXHAUSTED,
            },
        )