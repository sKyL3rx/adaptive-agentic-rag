from __future__ import annotations

from domain.contracts import (
    DraftPayload,
    DraftSupport,
    EvidenceItem,
    RetrievalPlan,
)

from domain.state import RequestState
from prompts.answer import (
    build_answer_prompt,
    build_answer_repair_prompt,
)
from workflow.errors import (
    InvalidModelOutputError,
)
from workflow.messages import (
    AnswerCandidate,
)
from workflow.resources import WorkflowResources

def canonicalize_draft_support(
    draft: DraftPayload,
) -> DraftPayload:
    grouped: dict[
        str,
        list[str],
    ] = {}

    support_order: list[str] = []

    for support in draft.support:
        sub_question_id = (
            support.sub_question_id.strip()
        )

        if sub_question_id not in grouped:
            grouped[sub_question_id] = []
            support_order.append(
                sub_question_id
            )

        target = grouped[
            sub_question_id
        ]

        for node_id in (
            support.evidence_node_ids
        ):
            normalized = node_id.strip()

            if (
                normalized
                and normalized not in target
            ):
                target.append(normalized)

    rebuilt_support = tuple(
        DraftSupport(
            sub_question_id=(
                sub_question_id
            ),
            evidence_node_ids=tuple(
                grouped[sub_question_id]
            ),
        )
        for sub_question_id
        in support_order
    )

    return draft.model_copy(
        update={
            "support": rebuilt_support,
        }
    )
    

def validate_draft_support(
    *,
    draft: DraftPayload,
    evidence: tuple[EvidenceItem, ...],
    plan: RetrievalPlan,
) -> None:
    if not draft.merged_answer.strip():
        raise ValueError(
            "Draft answer is empty"
        )

    planned_ids = {
        item.id
        for item in plan.sub_questions
    }

    evidence_by_question: dict[
        str,
        set[str],
    ] = {
        sub_question_id: set()
        for sub_question_id
        in planned_ids
    }

    for item in evidence:
        evidence_by_question.setdefault(
            item.sub_question_id,
            set(),
        ).add(item.node_id)

    supported_ids: set[str] = set()

    for support in draft.support:
        if (
            support.sub_question_id
            not in planned_ids
        ):
            raise ValueError(
                "Draft references an unknown "
                "sub-question ID"
            )

        if not support.evidence_node_ids:
            raise ValueError(
                "Draft support requires at least "
                "one evidence node ID"
            )

        allowed_nodes = (
            evidence_by_question.get(
                support.sub_question_id,
                set(),
            )
        )

        invalid_nodes = (
            set(support.evidence_node_ids)
            - allowed_nodes
        )

        if invalid_nodes:
            raise ValueError(
                "Draft references evidence nodes "
                "that do not support its "
                "sub-question"
            )

        supported_ids.add(
            support.sub_question_id
        )

    if not supported_ids:
        raise ValueError(
            "Draft does not contain "
            "any grounded support"
        )

    missing_ids = (
        planned_ids
        - supported_ids
    )

    if missing_ids:
        raise ValueError(
            "Draft is missing grounded "
            "support for sub-questions: "
            f"{sorted(missing_ids)}"
        )

    if supported_ids != planned_ids:
        raise ValueError(
            "Draft support does not exactly "
            "cover the retrieval plan"
        )


async def generate_answer(
    state: RequestState,
    resources: WorkflowResources,
) -> AnswerCandidate:
    plan = state.retrieval_plan

    if plan is None:
        raise ValueError(
            "Cannot answer without a "
            "retrieval plan"
        )

    if not state.evidence:
        raise ValueError(
            "Cannot answer without evidence"
        )

    state, draft = (
        await resources
        .model_gateway
        .run_structured(
            state=state,
            stage="answer",
            timeout_seconds=(
                resources.settings
                .answer_timeout_seconds
            ),
            agent=(
                resources.models
                .answer_generator
            ),
            prompt=build_answer_prompt(
                state,
                max_chars_per_evidence=(
                    resources.settings
                    .answer_evidence_chars_per_item
                ),
            ),
            response_model=DraftPayload,
        )
    )

    draft = canonicalize_draft_support(
        draft
    )

    try:
        validate_draft_support(
            draft=draft,
            evidence=state.evidence,
            plan=plan,
        )

    except ValueError as first_error:
        repair_prompt = (
            build_answer_repair_prompt(
                state,
                previous_draft=draft,
                validation_error=str(
                    first_error
                ),
                max_chars_per_evidence=(
                    resources.settings
                    .answer_evidence_chars_per_item
                ),
            )
        )

        state, repaired_draft = (
            await resources
            .model_gateway
            .run_structured(
                state=state,
                stage="answer_repair",
                timeout_seconds=(
                    resources.settings
                    .answer_timeout_seconds
                ),
                agent=(
                    resources.models
                    .answer_generator
                ),
                prompt=repair_prompt,
                response_model=DraftPayload,
            )
        )

        draft = canonicalize_draft_support(
            repaired_draft
        )

        try:
            validate_draft_support(
                draft=draft,
                evidence=state.evidence,
                plan=plan,
            )

        except ValueError as second_error:
            raise InvalidModelOutputError(
                state=state,
                stage="answer",
                expected=(
                    "a grounded DraftPayload "
                    "with valid support after "
                    "one semantic repair"
                ),
            ) from second_error

    state = state.model_copy(
        update={
            "draft": draft,
        }
    )

    return AnswerCandidate(
        state=state,
        draft=draft,
    )