from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from workflow.builder import (
    build_request_workflow,
)
from workflow.events import (
    ChatEvent,
    ErrorEvent,
    FinalEvent,
    map_workflow_event,
)
from workflow.messages import WorkflowInput
from workflow.requests import ChatRequest
from workflow.resources import WorkflowResources
from domain.state import RequestState

logger = logging.getLogger(__name__)

class AgentFrameworkEngine:
    def __init__(
        self,
        resources: WorkflowResources,
    ) -> None:
        self._resources = resources

    async def stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[ChatEvent]:
        state = RequestState.create(
            request_id=request.request_id,
            session_id=request.session_id,
            user_query=request.user_query,
            recent_history=request.recent_history,
            budget=request.budget,
        )

        workflow = build_request_workflow(
            self._resources
        )

        terminal_event_seen = False

        try:
            async with asyncio.timeout(
                state.remaining_seconds()
            ):
                event_stream = workflow.run(
                    WorkflowInput(state=state),
                    stream=True,
                )

                async for event in event_stream:
                    mapped = map_workflow_event(
                        event,
                        request_id=state.request_id,
                    )

                    if mapped is None:
                        continue

                    yield mapped


                    if isinstance(
                        mapped,
                        FinalEvent,
                    ):
                        terminal_event_seen = True

                    if (
                        isinstance(
                            mapped,
                            ErrorEvent,
                        )
                        and event.type == "failed"
                    ):
                        terminal_event_seen = True

        except asyncio.CancelledError:
            raise

        except TimeoutError:
            terminal_event_seen = True

            yield ErrorEvent(
                type="error",
                request_id=state.request_id,
                error_type="TIMEOUT",
                message=(
                    "Request deadline exceeded."
                ),
            )

        except Exception as exc:
            terminal_event_seen = True

            logger.exception(
                "workflow_execution_failed "
                "request_id=%s",
                state.request_id,
            )

            yield ErrorEvent(
                type="error",
                request_id=state.request_id,
                error_type=(
                    type(exc).__name__
                ),
                message=(
                    "Workflow execution failed."
                ),
            )

        if not terminal_event_seen:
            yield ErrorEvent(
                type="error",
                request_id=state.request_id,
                error_type=(
                    "NO_TERMINAL_OUTPUT"
                ),
                message=(
                    "Workflow finished without "
                    "a final output."
                ),
            )

