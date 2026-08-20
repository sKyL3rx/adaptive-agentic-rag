from __future__ import annotations

from domain.contracts import (
    CoverageStatus,
    QuestionShape,
    ReviewVerdict,
)
from domain.state import RequestState, Route
from workflow.budget import remaining_model_calls
from workflow.messages import (
    AnswerCandidate,
    CompiledRequest,
    ContextResolved,
    EvidenceReady,
    ReviewCompleted,
)


def determine_initial_route(
    state: RequestState,
) -> Route:
    if state.shape is None:
        raise ValueError(
            "Cannot route before shape classification"
        )

    route_by_shape = {
        QuestionShape.CASUAL_CONVERSATION:
            Route.CASUAL_RESPONSE,

        QuestionShape.CONVERSATION_RECALL:
            Route.RECALL_RESPONSE,

        QuestionShape.SINGLE_FOCUSED:
            Route.SINGLE_FAST,

        QuestionShape.BROAD_COVERAGE:
            Route.BROAD_ADAPTIVE,

        QuestionShape.CONTEXT_DEPENDENT:
            Route.CONTEXT_FAST,

        QuestionShape.MULTI_PART:
            Route.MULTI_BATCH,

        QuestionShape.COMPARISON:
            Route.COMPARISON_BATCH,
    }

    return route_by_shape[state.shape]


def determine_route_after_context(
    state: RequestState,
) -> Route:
    rewrite = state.context_rewrite

    if (
        rewrite is None
        or not rewrite.can_resolve
    ):
        return Route.CONTEXT_FAST

    if state.shape is None:
        raise ValueError(
            "Resolved context has no question shape"
        )

    return determine_initial_route(state)

def returns_directly(
    message: CompiledRequest,
) -> bool:
    return message.state.route in {
        Route.CASUAL_RESPONSE,
        Route.RECALL_RESPONSE,
    }


def needs_context(
    message: CompiledRequest,
) -> bool:
    return (
        message.state.route
        == Route.CONTEXT_FAST
    )


def uses_fast_plan(
    message: CompiledRequest,
) -> bool:
    return message.state.route in {
        Route.SINGLE_FAST,
        Route.BROAD_ADAPTIVE,
    }


def context_uses_fast_plan(
    message: ContextResolved,
) -> bool:
    state = message.state
    rewrite = state.context_rewrite

    if (
        rewrite is None
        or not rewrite.can_resolve
        or rewrite.needs_full_planner
    ):
        return False

    return state.route in {
        Route.SINGLE_FAST,
        Route.BROAD_ADAPTIVE,
    }


def context_requires_planner(
    message: ContextResolved,
) -> bool:
    state = message.state
    rewrite = state.context_rewrite

    if (
        rewrite is None
        or not rewrite.can_resolve
    ):
        return False

    return state.route in {
        Route.MULTI_BATCH,
        Route.COMPARISON_BATCH,
    }


def coverage_is_sufficient(
    message: EvidenceReady,
) -> bool:
    coverage = message.state.coverage

    return bool(
        coverage is not None
        and coverage.status
        == CoverageStatus.SUFFICIENT
    )


def coverage_is_recoverable(
    message: EvidenceReady,
) -> bool:
    state = message.state
    coverage = state.coverage

    return bool(
        coverage is not None
        and coverage.status
        == CoverageStatus.RECOVERABLE_GAP
        and state.pending_gap_queries
        and state.retrieval_rounds_used
        < state.budget.max_retrieval_rounds
    )


def should_verify(
    message: AnswerCandidate,
) -> bool:
    state = message.state

    if remaining_model_calls(state) <= 0:
        return False

    return state.shape in {
        QuestionShape.MULTI_PART,
        QuestionShape.COMPARISON,
    }


def review_approved(
    message: ReviewCompleted,
) -> bool:
    return (
        message.decision.verdict
        == ReviewVerdict.APPROVE
    )


def review_recoverable(
    message: ReviewCompleted,
) -> bool:
    state = message.state

    return bool(
        message.decision.verdict
        == ReviewVerdict.NEEDS_MORE
        and state.pending_gap_queries
        and state.retrieval_rounds_used
        < state.budget.max_retrieval_rounds
    )