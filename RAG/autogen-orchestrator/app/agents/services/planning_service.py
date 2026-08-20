from __future__ import annotations

import re

from domain.contracts import (
    QuestionShape,
    RetrievalPlan,
    RetrievalPlanProposal,
    SubQuestion,
)

from domain.state import RequestState, Route
from prompts.planning import (
    build_planner_prompt,
    build_planner_repair_prompt,
)

from workflow.errors import (
    InvalidModelOutputError,
)
from workflow.messages import (
    RetrievalPrepared,
    RetrievalTask,
)
from workflow.resources import WorkflowResources

def _normalize_query_key(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        value.strip().lower(),
    )


def canonicalize_plan(
    proposal: RetrievalPlanProposal,
    *,
    state: RequestState,
    max_sub_questions: int,
) -> RetrievalPlan:
    """
    Canonicalize model-provided retrieval queries.
    """
    usable_queries: list[str] = []
    seen_queries: set[str] = set()

    for item in proposal.sub_questions:
        query = item.query.strip()

        if not query:
            continue

        key = _normalize_query_key(query)

        if key in seen_queries:
            continue

        seen_queries.add(key)
        usable_queries.append(query)

        if len(usable_queries) >= max_sub_questions:
            break

    if not usable_queries:
        raise ValueError(
            "Planner returned no usable "
            "sub-questions"
        )

    if (
        state.shape
        in {
            QuestionShape.MULTI_PART,
            QuestionShape.COMPARISON,
        }
        and len(usable_queries) < 2
    ):
        raise ValueError(
            "Complex request requires at least "
            "two usable sub-questions"
        )

    sub_questions = tuple(
        SubQuestion(
            id=f"sq{index}",
            question=query,
            query=query,
        )
        for index, query in enumerate(
            usable_queries,
            start=1,
        )
    )

    return RetrievalPlan(
        version="v1",
        mode=(
            "decomposed"
            if len(sub_questions) > 1
            else "single"
        ),
        shape=state.shape,
        sub_questions=sub_questions,
        parallelizable=(
            proposal.parallelizable
            and len(sub_questions) > 1
        ),
    )
    
    
def _retrieval_sizes(
    state: RequestState,
    resources: WorkflowResources,
) -> tuple[int, int]:
    settings = resources.settings

    if state.route == Route.SINGLE_FAST:
        top_k = settings.single_top_k

    elif state.route == Route.BROAD_ADAPTIVE:
        top_k = settings.broad_top_k

    else:
        top_k = settings.complex_top_k

    top_k = max(
        1,
        min(top_k, 20),
    )

    candidate_k = max(
        top_k,
        top_k
        * settings.retrieval_candidate_multiplier,
    )

    candidate_k = min(candidate_k, 50)

    return top_k, candidate_k

def build_retrieval_tasks(
    state: RequestState,
    plan: RetrievalPlan,
    resources: WorkflowResources,
) -> tuple[RetrievalTask, ...]:
    top_k, candidate_k = _retrieval_sizes(
        state,
        resources,
    )

    round_number = (
        state.retrieval_rounds_used + 1
    )

    return tuple(
        RetrievalTask(
            request_id=state.request_id,
            round_number=round_number,
            sub_question_id=item.id,
            query=item.query,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        for item in plan.sub_questions
    )

async def plan_retrieval(
    state: RequestState,
    resources: WorkflowResources,
) -> RetrievalPrepared:
    if state.route not in {
        Route.MULTI_BATCH,
        Route.COMPARISON_BATCH,
    }:
        raise ValueError(
            "Planner may only run for multi_batch "
            "or comparison_batch routes"
        )

    state, proposal = (
        await resources
        .model_gateway
        .run_structured(
            state=state,
            stage="plan",
            timeout_seconds=(
                resources.settings
                .plan_timeout_seconds
            ),
            agent=resources.models.planner,
            prompt=build_planner_prompt(state),
            response_model=RetrievalPlanProposal,
        )
    )

    try:
        # Normal path:
        # planner output is valid -> continue normally.
        plan = canonicalize_plan(
            proposal,
            state=state,
            max_sub_questions=(
                state.budget.max_sub_questions
            ),
        )

    except ValueError as first_error:
        # Repair path:
        # only runs if canonicalize_plan()
        # rejects the first planner output.
        repair_prompt = (
            build_planner_repair_prompt(
                state,
                failed_proposal=proposal,
                validation_error=str(
                    first_error
                ),
            )
        )

        state, repaired_proposal = (
            await resources
            .model_gateway
            .run_structured(
                state=state,
                stage="plan_repair",
                timeout_seconds=(
                    resources.settings
                    .plan_timeout_seconds
                ),
                agent=resources.models.planner,
                prompt=repair_prompt,
                response_model=(
                    RetrievalPlanProposal
                ),
            )
        )

        try:
            plan = canonicalize_plan(
                repaired_proposal,
                state=state,
                max_sub_questions=(
                    state.budget
                    .max_sub_questions
                ),
            )

        except ValueError as second_error:
            raise InvalidModelOutputError(
                state=state,
                stage="plan",
                expected=(
                    "a usable retrieval plan "
                    "with valid distinct "
                    "sub-questions after repair"
                ),
            ) from second_error
        
    state = state.model_copy(
        update={
            "retrieval_plan": plan,
        }
    )

    return RetrievalPrepared(
        state=state,
        tasks=build_retrieval_tasks(
            state,
            plan,
            resources,
        ),
    )

def prepare_fast_retrieval(
    state: RequestState,
    resources: WorkflowResources,
) -> RetrievalPrepared:
    if state.route not in {
        Route.SINGLE_FAST,
        Route.BROAD_ADAPTIVE,
    }:
        raise ValueError(
            "Fast retrieval requires a single "
            "or broad route"
        )

    if state.shape not in {
        QuestionShape.SINGLE_FOCUSED,
        QuestionShape.BROAD_COVERAGE,
    }:
        raise ValueError(
            "Fast retrieval received an "
            "incompatible question shape"
        )

    resolved_query = (
        state.resolved_query
        or state.user_query
    ).strip()

    if not resolved_query:
        raise ValueError(
            "Fast retrieval query is empty"
        )

    plan = RetrievalPlan(
        version="v1",
        mode="single",
        shape=state.shape,
        sub_questions=(
            SubQuestion(
                id="sq1",
                question=resolved_query,
                query=resolved_query,
            ),
        ),
        parallelizable=False,
    )

    state = state.model_copy(
        update={
            "retrieval_plan": plan,
        }
    )

    return RetrievalPrepared(
        state=state,
        tasks=build_retrieval_tasks(
            state,
            plan,
            resources,
        ),
    )

def prepare_gap_retrieval(
    state: RequestState,
    resources: WorkflowResources,
) -> RetrievalPrepared:
    if not state.pending_gap_queries:
        raise ValueError(
            "No pending gap queries to retrieve"
        )

    top_k = max(
        1,
        min(
            resources.settings.recovery_top_k,
            20,
        ),
    )

    candidate_k = min(
        50,
        max(
            top_k,
            top_k
            * resources.settings
            .retrieval_candidate_multiplier,
        ),
    )

    round_number = (
        state.retrieval_rounds_used + 1
    )

    tasks = tuple(
        RetrievalTask(
            request_id=state.request_id,
            round_number=round_number,
            sub_question_id=item.id,
            query=item.query,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        for item in state.pending_gap_queries[
            :state.budget.max_sub_questions
        ]
    )

    return RetrievalPrepared(
        state=state,
        tasks=tasks,
    )