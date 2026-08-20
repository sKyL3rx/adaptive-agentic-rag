from __future__ import annotations

from domain.contracts import (
    ReviewDecision,
    ReviewDecisionProposal,
    ReviewVerdict,
    SubQuestion,
)
from prompts.review import (
    build_review_prompt,
)
from workflow.messages import (
    AnswerCandidate,
    ReviewCompleted,
)
from workflow.resources import WorkflowResources


def canonicalize_gap_queries(
    proposal: ReviewDecisionProposal,
    *,
    candidate: AnswerCandidate,
) -> tuple[SubQuestion, ...]:
    state = candidate.state
    plan = state.retrieval_plan

    if plan is None:
        return ()

    plan_by_id = {
        item.id: item
        for item in plan.sub_questions
    }

    allowed_missing_ids: list[str] = []

    for sub_question_id in (
        proposal.missing_sub_questions
    ):
        if (
            sub_question_id in plan_by_id
            and sub_question_id
            not in allowed_missing_ids
        ):
            allowed_missing_ids.append(
                sub_question_id
            )

        if (
            len(allowed_missing_ids)
            >= state.budget.max_sub_questions
        ):
            break

    if (
        not allowed_missing_ids
        and state.coverage is not None
    ):
        for sub_question_id in (
            state.coverage
            .missing_sub_questions
        ):
            if sub_question_id in plan_by_id:
                allowed_missing_ids.append(
                    sub_question_id
                )

    suggestions = list(
        proposal.next_queries
    )

    canonical: list[SubQuestion] = []

    for index, sub_question_id in enumerate(
        allowed_missing_ids
    ):
        original = plan_by_id[
            sub_question_id
        ]

        if index < len(suggestions):
            suggestion = suggestions[index]

            question = (
                suggestion.question.strip()
                or original.question
            )

            query = (
                suggestion.query.strip()
                or original.query
            )

        else:
            question = original.question
            query = original.query

        canonical.append(
            SubQuestion(
                id=sub_question_id,
                question=question,
                query=query,
            )
        )

    return tuple(canonical)


def canonicalize_review_decision(
    proposal: ReviewDecisionProposal,
    *,
    candidate: AnswerCandidate,
) -> ReviewDecision:
    if (
        proposal.verdict
        == ReviewVerdict.APPROVE
    ):
        return ReviewDecision(
            version="v1",
            verdict=ReviewVerdict.APPROVE,
            missing_sub_questions=(),
            next_queries=(),
            reason=proposal.reason,
        )

    if (
        proposal.verdict
        == ReviewVerdict.INSUFFICIENT_EVIDENCE
    ):
        return ReviewDecision(
            version="v1",
            verdict=(
                ReviewVerdict
                .INSUFFICIENT_EVIDENCE
            ),
            missing_sub_questions=tuple(
                proposal
                .missing_sub_questions
            ),
            next_queries=(),
            reason=proposal.reason,
        )

    gap_queries = canonicalize_gap_queries(
        proposal,
        candidate=candidate,
    )

    if not gap_queries:
        return ReviewDecision(
            version="v1",
            verdict=(
                ReviewVerdict
                .INSUFFICIENT_EVIDENCE
            ),
            missing_sub_questions=(),
            next_queries=(),
            reason=(
                proposal.reason
                or (
                    "Reviewer requested recovery "
                    "without valid gap queries"
                )
            ),
        )

    return ReviewDecision(
        version="v1",
        verdict=ReviewVerdict.NEEDS_MORE,
        missing_sub_questions=tuple(
            item.id
            for item in gap_queries
        ),
        next_queries=gap_queries,
        reason=proposal.reason,
    )


async def verify_answer(
    candidate: AnswerCandidate,
    resources: WorkflowResources,
) -> ReviewCompleted:
    state = candidate.state

    if state.draft is None:
        raise ValueError(
            "Cannot review without a draft"
        )

    charged_state, proposal = (
        await resources
        .model_gateway
        .run_structured(
            state=state,
            stage="verify",
            timeout_seconds=(
                resources.settings
                .verify_timeout_seconds
            ),
            agent=resources.models.reviewer,
            prompt=build_review_prompt(
                state,
                candidate.draft,
                max_chars_per_evidence=(
                    resources.settings
                    .review_evidence_chars_per_item
                ),
            ),
            response_model=(
                ReviewDecisionProposal
            ),
        )
    )

    candidate = candidate.model_copy(
        update={
            "state": charged_state,
        }
    )

    decision = canonicalize_review_decision(
        proposal,
        candidate=candidate,
    )

    state = charged_state.model_copy(
        update={
            "review": decision,
            "pending_gap_queries":
                decision.next_queries,
        }
    )

    return ReviewCompleted(
        state=state,
        decision=decision,
    )   