from __future__ import annotations

from collections import defaultdict

from domain.contracts import (
    CoverageDecision,
    CoverageStatus,
    EvidenceItem,
    QuestionShape,
    RetrievalPlan,
)
from workflow.messages import (
    EvidenceReady,
    RetrievalBatchResult,
    RetrievalFailure,
)


def _evidence_score(
    item: EvidenceItem,
) -> float:
    if item.rerank_score is not None:
        return item.rerank_score

    if item.retrieval_score is not None:
        return item.retrieval_score

    return float("-inf")


def deduplicate_evidence(
    items: tuple[EvidenceItem, ...],
) -> tuple[EvidenceItem, ...]:
    best_by_key: dict[
        tuple[str, str],
        EvidenceItem,
    ] = {}

    for item in items:
        key = (
            item.sub_question_id,
            item.chunk_id,
        )

        current = best_by_key.get(key)

        if (
            current is None
            or _evidence_score(item)
            > _evidence_score(current)
        ):
            best_by_key[key] = item

    return tuple(
        sorted(
            best_by_key.values(),
            key=lambda item: (
                item.sub_question_id,
                -_evidence_score(item),
                item.chunk_id,
            ),
        )
    )


def evaluate_coverage(
    *,
    plan: RetrievalPlan,
    evidence: tuple[EvidenceItem, ...],
    failures: tuple[
        RetrievalFailure,
        ...
    ],
    broad_min_items: int = 2,
) -> CoverageDecision:
    evidence_by_question: dict[
        str,
        list[EvidenceItem],
    ] = defaultdict(list)

    for item in evidence:
        evidence_by_question[
            item.sub_question_id
        ].append(item)

    covered: list[str] = []
    missing: list[str] = []

    for sub_question in plan.sub_questions:
        items = evidence_by_question.get(
            sub_question.id,
            [],
        )

        required_items = 1

        if (
            plan.shape
            == QuestionShape.BROAD_COVERAGE
        ):
            required_items = max(
                1,
                broad_min_items,
            )

        if len(items) >= required_items:
            covered.append(
                sub_question.id
            )
        else:
            missing.append(
                sub_question.id
            )

    if not missing:
        status = CoverageStatus.SUFFICIENT
        next_queries = ()

    elif len(missing) == 1:
        status = (
            CoverageStatus.RECOVERABLE_GAP
        )

        missing_id = missing[0]

        next_queries = tuple(
            item
            for item in plan.sub_questions
            if item.id == missing_id
        )

    else:
        status = CoverageStatus.INSUFFICIENT
        next_queries = ()

    return CoverageDecision(
        status=status,
        covered_sub_questions=tuple(
            covered
        ),
        missing_sub_questions=tuple(
            missing
        ),
        next_queries=next_queries,
        reason=(
            f"covered={len(covered)};"
            f"missing={len(missing)};"
            f"failures={len(failures)}"
        ),
    )


def aggregate_evidence(
    batch: RetrievalBatchResult,
    *,
    broad_min_items: int = 2,
) -> EvidenceReady:
    plan = batch.state.retrieval_plan

    if plan is None:
        raise ValueError(
            "Cannot aggregate evidence "
            "without a retrieval plan"
        )

    new_items = tuple(
        item
        for result in batch.results
        for item in result.evidence
    )

    evidence = deduplicate_evidence(
        (
            *batch.state.evidence,
            *new_items,
        )
    )

    failures = tuple(
        result.failure
        for result in batch.results
        if result.failure is not None
    )

    coverage = evaluate_coverage(
        plan=plan,
        evidence=evidence,
        failures=failures,
        broad_min_items=broad_min_items,
    )

    state = batch.state.model_copy(
        update={
            "evidence": evidence,
            "coverage": coverage,
            "pending_gap_queries":
                coverage.next_queries,
        }
    )

    return EvidenceReady(
        state=state
    )