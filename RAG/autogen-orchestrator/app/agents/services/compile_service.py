from __future__ import annotations

import logging

from domain.contracts import (
    QuestionShape,
    QuestionShapeDecision,
)
from domain.state import (
    RequestState,
    Route,
)
from observability import (
    RAG_ROUTE_TOTAL,
)
from prompts.compile import build_compile_prompt
from services.history_utils import (
    build_conversation_recall_response,
    build_recent_history_preview,
)
from workflow.errors import (
    InvalidModelOutputError,
)
from workflow.policies import (
    determine_initial_route,
)
from workflow.resources import WorkflowResources


logger = logging.getLogger(__name__)


async def compile_request(
    state: RequestState,
    resources: WorkflowResources,
) -> RequestState:
    """
    Classify every request with one compiler-model call.
    """
    query = state.user_query.strip()

    if not query:
        raise ValueError(
            "Cannot compile an empty user query"
        )

    has_history = any(
        turn.content.strip()
        for turn in state.recent_history
    )

    history_preview = (
        build_recent_history_preview(
            state.recent_history,
            max_turns=(
                resources.settings
                .compile_history_max_turns
            ),
            max_total_chars=(
                resources.settings
                .compile_history_max_chars
            ),
        )
    )

    prompt = build_compile_prompt(
        user_query=query,
        has_recent_history=has_history,
        recent_history_preview=history_preview,
    )

    charged_state, decision = (
        await resources
        .model_gateway
        .run_structured(
            state=state,
            stage="compile",
            timeout_seconds=(
                resources.settings
                .compile_timeout_seconds
            ),
            agent=resources.models.compiler,
            prompt=prompt,
            response_model=(
                QuestionShapeDecision
            ),
        )
    )

    if (
        decision.shape
        == QuestionShape.CONTEXT_DEPENDENT
        and not has_history
    ):
        raise InvalidModelOutputError(
            state=charged_state,
            stage="compile",
            expected=(
                "context_dependent only when recent "
                "conversation history exists"
            ),
        )

    direct_response = (
        decision.direct_response or ""
    ).strip() or None
    
    if (
        decision.shape
        == QuestionShape.CONVERSATION_RECALL
    ):
        if decision.recall_target == "none":
            raise InvalidModelOutputError(
                state=charged_state,
                stage="compile",
                expected=(
                    "conversation_recall with a valid "
                    "recall_target"
                ),
            )

        direct_response = (
            build_conversation_recall_response(
                charged_state.recent_history,
                target=decision.recall_target,
            )
        )

    compiled_state = (
        charged_state.model_copy(
            update={
                "shape": decision.shape,
                "shape_confidence": (
                    decision.confidence
                ),
                "direct_response": (
                    direct_response
                ),
                "recall_target": (
                    decision.recall_target
                ),
            }
        )
    )

    route = determine_initial_route(
        compiled_state
    )
    
    if route != Route.CONTEXT_FAST:
        RAG_ROUTE_TOTAL.labels(
            route=route.value
        ).inc()

    compiled_state = (
        compiled_state.model_copy(
            update={
                "route": route,
            }
        )
    )

    logger.info(
        "request_compiled "
        "request_id=%s "
        "classification_source=model "
        "shape=%s "
        "confidence=%.3f "
        "route=%s "
        "recall_target=%s "
        "model_calls_used=%d",
        compiled_state.request_id,
        decision.shape.value,
        decision.confidence,
        route.value,
        decision.recall_target,
        compiled_state.model_calls_used,
    )

    return compiled_state