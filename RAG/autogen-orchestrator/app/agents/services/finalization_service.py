from __future__ import annotations

import re

from domain.contracts import (
    CitationSource,
    ReviewVerdict,
)
from domain.state import (
    RequestState,
    Route,
    TerminationReason,
)
from workflow.messages import (
    AnswerCandidate,
    BoundedAnswer,
    CompiledRequest,
    ContextResolved,
    EvaluationTelemetry,
    EvidenceReady,
    FinalAnswer,
    RetrievalEvaluationGroup,
    ReviewCompleted,
)


_DEFAULT_SOURCE_TITLE = "California Driver's Handbook"
_MAX_EXCERPT_CHARS = 360
_WHITESPACE_RE = re.compile(r"\s+")


def _evaluation_telemetry(
    state: RequestState,
) -> EvaluationTelemetry:
    plan = state.retrieval_plan

    evidence_by_sub_question: dict[
        str,
        list[str],
    ] = {}

    for item in state.evidence:
        chunk_id = item.chunk_id.strip()

        if not chunk_id:
            continue

        bucket = evidence_by_sub_question.setdefault(
            item.sub_question_id,
            [],
        )

        if chunk_id not in bucket:
            bucket.append(chunk_id)

    groups: list[RetrievalEvaluationGroup] = []

    if plan is not None:
        for sub_question in plan.sub_questions:
            groups.append(
                RetrievalEvaluationGroup(
                    sub_question_id=sub_question.id,
                    query=sub_question.query,
                    chunk_ids=tuple(
                        evidence_by_sub_question.get(
                            sub_question.id,
                            [],
                        )
                    ),
                )
            )

    else:
        for sub_question_id in sorted(
            evidence_by_sub_question
        ):
            groups.append(
                RetrievalEvaluationGroup(
                    sub_question_id=sub_question_id,
                    query="",
                    chunk_ids=tuple(
                        evidence_by_sub_question[
                            sub_question_id
                        ]
                    ),
                )
            )

    return EvaluationTelemetry(
        question_shape=state.shape,
        route=state.route,
        retrieval_mode=(
            plan.mode
            if plan is not None
            else None
        ),
        coverage_status=(
            state.coverage.status
            if state.coverage is not None
            else None
        ),
        retrieval_groups=tuple(groups),
    )


def _clean_excerpt(text: str) -> str | None:
    normalized = _WHITESPACE_RE.sub(
        " ",
        (text or "").strip(),
    )

    if not normalized:
        return None

    if len(normalized) <= _MAX_EXCERPT_CHARS:
        return normalized

    return (
        normalized[: _MAX_EXCERPT_CHARS - 3]
        .rstrip()
        + "..."
    )


def _citation_sources(
    state: RequestState,
) -> tuple[CitationSource, ...]:
    draft = state.draft

    if draft is None:
        return ()

    evidence_by_node = {
        item.node_id: item
        for item in state.evidence
    }

    ordered: list[CitationSource] = []
    seen_source_ids: set[str] = set()

    for support in draft.support:
        for node_id in support.evidence_node_ids:
            item = evidence_by_node.get(node_id)

            if item is None:
                continue

            source_id = item.chunk_id.strip()

            if not source_id:
                continue

            if source_id in seen_source_ids:
                continue

            seen_source_ids.add(source_id)

            ordered.append(
                CitationSource(
                    source_id=source_id,
                    title=(
                        (item.title or "").strip()
                        or _DEFAULT_SOURCE_TITLE
                    ),
                    source_file=(
                        (item.source or "").strip()
                        or None
                    ),
                    section_id=(
                        (item.section_id or "").strip()
                        or None
                    ),
                    heading_path=(
                        (item.heading_path or "").strip()
                        or None
                    ),
                    url=(
                        (item.url or "").strip()
                        or None
                    ),
                    excerpt=_clean_excerpt(
                        item.text
                    ),
                )
            )

    return tuple(ordered)


def finalize_compiled(
    message: CompiledRequest,
) -> FinalAnswer:
    state = message.state

    answer = (
        state.direct_response or ""
    ).strip()

    if state.route not in {
        Route.CASUAL_RESPONSE,
        Route.RECALL_RESPONSE,
    }:
        raise ValueError(
            "Compiled request is not a "
            "direct-response route"
        )

    if not answer:
        raise ValueError(
            "Direct-response route has no "
            "response text"
        )

    if state.route == Route.RECALL_RESPONSE:
        reason = (
            TerminationReason
            .CONVERSATION_RECALL
        )
    else:
        reason = (
            TerminationReason
            .CASUAL_RESPONSE
        )

    return FinalAnswer(
        request_id=state.request_id,
        answer=answer,
        citations=(),
        termination_reason=reason,
        model_calls_used=(
            state.model_calls_used
        ),
        retrieval_rounds_used=(
            state.retrieval_rounds_used
        ),
        evaluation=_evaluation_telemetry(
            state
        ),
    )


def build_context_bounded_answer(
    message: ContextResolved,
) -> BoundedAnswer:
    state = message.state.model_copy(
        update={
            "termination_reason":
                TerminationReason
                .INSUFFICIENT_EVIDENCE,
        }
    )

    return BoundedAnswer(
        state=state,
        answer=(
            "I could not determine what the latest "
            "question refers to from the available "
            "conversation context. Please restate "
            "the question as a standalone request."
        ),
        reason=(
            TerminationReason
            .INSUFFICIENT_EVIDENCE
        ),
    )


def build_evidence_bounded_answer(
    message: EvidenceReady,
) -> BoundedAnswer:
    state = message.state

    if (
        state.retrieval_rounds_used
        >= state.budget.max_retrieval_rounds
    ):
        reason = (
            TerminationReason
            .BUDGET_EXHAUSTED
        )
    else:
        reason = (
            TerminationReason
            .INSUFFICIENT_EVIDENCE
        )

    state = state.model_copy(
        update={
            "termination_reason": reason,
        }
    )

    return BoundedAnswer(
        state=state,
        answer=(
            "I could not find enough supporting "
            "evidence to provide a reliable answer."
        ),
        reason=reason,
    )


def build_review_bounded_answer(
    message: ReviewCompleted,
) -> BoundedAnswer:
    state = message.state

    if (
        message.decision.verdict
        == ReviewVerdict.NEEDS_MORE
        and state.retrieval_rounds_used
        >= state.budget.max_retrieval_rounds
    ):
        reason = (
            TerminationReason
            .BUDGET_EXHAUSTED
        )
    else:
        reason = (
            TerminationReason
            .INSUFFICIENT_EVIDENCE
        )

    state = state.model_copy(
        update={
            "termination_reason": reason,
        }
    )

    return BoundedAnswer(
        state=state,
        answer=(
            "The available evidence was not "
            "sufficient to verify a complete answer."
        ),
        reason=reason,
    )


def finalize_candidate(
    message: AnswerCandidate,
) -> FinalAnswer:
    state = message.state

    return FinalAnswer(
        request_id=state.request_id,
        answer=message.draft.merged_answer,
        citations=_citation_sources(state),
        termination_reason=(
            TerminationReason.AUTO_APPROVED
        ),
        model_calls_used=(
            state.model_calls_used
        ),
        retrieval_rounds_used=(
            state.retrieval_rounds_used
        ),
        evaluation=_evaluation_telemetry(
            state
        ),
    )


def finalize_review(
    message: ReviewCompleted,
) -> FinalAnswer:
    state = message.state

    if (
        message.decision.verdict
        != ReviewVerdict.APPROVE
    ):
        raise ValueError(
            "Cannot finalize an unapproved review"
        )

    if state.draft is None:
        raise ValueError(
            "Cannot finalize without a draft"
        )

    return FinalAnswer(
        request_id=state.request_id,
        answer=state.draft.merged_answer,
        citations=_citation_sources(state),
        termination_reason=(
            TerminationReason.APPROVED_REVIEW
        ),
        model_calls_used=(
            state.model_calls_used
        ),
        retrieval_rounds_used=(
            state.retrieval_rounds_used
        ),
        evaluation=_evaluation_telemetry(
            state
        ),
    )


def finalize_bounded(
    message: BoundedAnswer,
) -> FinalAnswer:
    state = message.state

    return FinalAnswer(
        request_id=state.request_id,
        answer=message.answer,
        citations=(),
        termination_reason=message.reason,
        model_calls_used=(
            state.model_calls_used
        ),
        retrieval_rounds_used=(
            state.retrieval_rounds_used
        ),
        evaluation=_evaluation_telemetry(
            state
        ),
    )
