from __future__ import annotations

import asyncio
import logging

from domain.contracts import (
    ContextRewritePayload,
    QuestionShape,
)
from domain.state import RequestState
from prompts.context import (
    build_context_prompt,
)
from workflow.errors import ModelStageError
from workflow.resources import WorkflowResources

logger = logging.getLogger(__name__)

def _unresolved_context(
    *,
    dependency_type: str,
    reason: str,
) -> ContextRewritePayload:
    return ContextRewritePayload(
        version="v1",
        can_resolve=False,
        confidence=0.0,
        dependency_type=dependency_type,
        anchor_topic=None,
        standalone_question=None,
        query=None,
        rewritten_shape=(
            QuestionShape.CONTEXT_DEPENDENT
        ),
        needs_full_planner=False,
    )

async def resolve_context(
    state: RequestState,
    resources: WorkflowResources,
) -> RequestState:
    """
    Rewrite a context-dependent request.
    """
    if (
        state.shape
        != QuestionShape.CONTEXT_DEPENDENT
    ):
        raise ValueError(
            "resolve_context requires a "
            "context_dependent request"
        )

    if not state.recent_history:
        rewrite = _unresolved_context(
            dependency_type="missing_history",
            reason="No recent history available",
        )

        return state.model_copy(
            update={
                "context_rewrite": rewrite,
                "context_was_rewritten": False,
            }
        )

    prompt = build_context_prompt(
        state,
        max_history_turns=(
            resources.settings
            .context_history_max_turns
        ),
        max_history_chars=(
            resources.settings
            .context_history_max_chars
        ),
    )

    try:
        charged_state, rewrite = (
            await resources
            .model_gateway
            .run_structured(
                state=state,
                stage="resolve_context",
                timeout_seconds=(
                    resources.settings
                    .context_timeout_seconds
                ),
                agent=(
                    resources.models
                    .context_rewriter
                ),
                prompt=prompt,
                response_model=(
                    ContextRewritePayload
                ),
            )
        )

        state = charged_state

    except asyncio.CancelledError:
        raise

    except ModelStageError as exc:
        state = exc.state

        logger.warning(
            "context_resolution_fallback "
            "request_id=%s error_type=%s",
            state.request_id,
            exc.error_type,
        )

        rewrite = _unresolved_context(
            dependency_type="model_failure",
            reason=exc.error_type,
        )

    if not rewrite.can_resolve:
        return state.model_copy(
            update={
                "context_rewrite": rewrite,
                "context_was_rewritten": False,
            }
        )

    resolved_query = (
        rewrite.query
        or rewrite.standalone_question
        or ""
    ).strip()

    if not resolved_query:
        raise ValueError(
            "Resolved context produced an empty query"
        )

    return state.model_copy(
        update={
            "resolved_query": resolved_query,
            "context_was_rewritten": True,
            "context_rewrite": rewrite,
            "shape": rewrite.rewritten_shape,
        }
    )