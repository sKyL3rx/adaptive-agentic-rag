from __future__ import annotations

from typing_extensions import Never

from agent_framework import (
    Executor,
    WorkflowContext,
    WorkflowEvent,
    handler,
)

from services.answer_service import (
    generate_answer,
)
from services.compile_service import (
    compile_request,
)
from services.context_service import (
    resolve_context,
)
from services.evidence_service import (
    aggregate_evidence,
)
from services.finalization_service import (
    build_context_bounded_answer,
    build_evidence_bounded_answer,
    build_review_bounded_answer,
    finalize_bounded,
    finalize_compiled,
    finalize_candidate,
    finalize_review,
)
from services.planning_service import (
    plan_retrieval,
    prepare_fast_retrieval,
    prepare_gap_retrieval,
)
from services.retrieval_service import (
    retrieve_batch,
)
from services.review_service import (
    verify_answer,
)
from workflow.events import (
    RetrievalProgressSignal,
)
from workflow.messages import (
    AnswerCandidate,
    BoundedAnswer,
    CompiledRequest,
    ContextResolved,
    EvidenceReady,
    FinalAnswer,
    RetrievalBatchResult,
    RetrievalPrepared,
    ReviewCompleted,
    WorkflowInput,
)
from workflow.policies import (
    determine_route_after_context,
)
from workflow.resources import WorkflowResources

from domain.contracts import (
    QuestionShape,
)

from observability import (
    RAG_REVIEWER_ACTIVATION_TOTAL,
    RAG_ROUTE_TOTAL,
)


class CompileExecutor(Executor):
    def __init__(
        self,
        resources: WorkflowResources,
        *,
        id: str = "compile"
    ) -> None:
        super().__init__(id=id)
        self._resources = resources

    @handler
    async def handle(
        self,
        message: WorkflowInput,
        ctx: WorkflowContext[
            CompiledRequest
        ],
    ) -> None:
        state = await compile_request(
            message.state,
            self._resources,
        )

        await ctx.send_message(
            CompiledRequest(state=state)
        )


class ResolveContextExecutor(Executor):
    def __init__(
        self,
        resources: WorkflowResources,
        *,
        id: str = "resolve_context",
    ) -> None:
        super().__init__(id=id)
        self._resources = resources

    @handler
    async def handle(
        self,
        message: CompiledRequest,
        ctx: WorkflowContext[
            ContextResolved
        ],
    ) -> None:
        state = await resolve_context(
            message.state,
            self._resources,
        )

        # Context service chỉ rewrite.
        # Pure policy chịu trách nhiệm route.
        if (
            state.context_rewrite is not None
            and state.context_rewrite.can_resolve
        ):
            state = state.model_copy(
                update={
                    "route":
                        determine_route_after_context(
                            state
                        ),
                }
            )

        if state.route is not None:
            RAG_ROUTE_TOTAL.labels(
                route=state.route.value
            ).inc()

        await ctx.send_message(
            ContextResolved(state=state)
        )

class PrepareFastPlanExecutor(Executor):
    def __init__(
        self,
        resources: WorkflowResources,
        *,
        id: str = "prepare_fast",
    ) -> None:
        super().__init__(id=id)
        self._resources = resources

    async def _prepare(
        self,
        state,
        ctx: WorkflowContext[
            RetrievalPrepared
        ],
    ) -> None:
        prepared = prepare_fast_retrieval(
            state,
            self._resources,
        )

        await ctx.send_message(prepared)

    @handler
    async def from_compiled(
        self,
        message: CompiledRequest,
        ctx: WorkflowContext[
            RetrievalPrepared
        ],
    ) -> None:
        await self._prepare(
            message.state,
            ctx,
        )

    @handler
    async def from_context(
        self,
        message: ContextResolved,
        ctx: WorkflowContext[
            RetrievalPrepared
        ],
    ) -> None:
        await self._prepare(
            message.state,
            ctx,
        )

class PlanExecutor(Executor):
    def __init__(
        self,
        resources: WorkflowResources,
        *,
        id: str = "plan",
    ) -> None:
        super().__init__(id=id)
        self._resources = resources

    async def _plan(
        self,
        state,
        ctx: WorkflowContext[
            RetrievalPrepared
        ],
    ) -> None:
        prepared = await plan_retrieval(
            state,
            self._resources,
        )

        await ctx.send_message(prepared)

    @handler
    async def from_compiled(
        self,
        message: CompiledRequest,
        ctx: WorkflowContext[
            RetrievalPrepared
        ],
    ) -> None:
        await self._plan(
            message.state,
            ctx,
        )

    @handler
    async def from_context(
        self,
        message: ContextResolved,
        ctx: WorkflowContext[
            RetrievalPrepared
        ],
    ) -> None:
        await self._plan(
            message.state,
            ctx,
        )


class RetrieveExecutor(Executor):
    def __init__(
        self,
        resources: WorkflowResources,
        *,
        id: str = "retrieve",
    ) -> None:
        super().__init__(id=id)
        self._resources = resources

    @handler
    async def handle(
        self,
        message: RetrievalPrepared,
        ctx: WorkflowContext[
            RetrievalBatchResult
        ],
    ) -> None:
        result = await retrieve_batch(
            message,
            self._resources,
        )

        for retrieval_result in result.results:
            await ctx.add_event(
                WorkflowEvent(
                    "retrieval_progress",
                    data=RetrievalProgressSignal(
                        sub_question_id=(
                            retrieval_result
                            .sub_question_id
                        ),
                        chunks_found=len(
                            retrieval_result.evidence
                        ),
                        failed=(
                            retrieval_result.failure
                            is not None
                        ),
                    ),
                    executor_id=self.id,
                )
            )

        await ctx.send_message(result)

class AggregateExecutor(Executor):
    def __init__(
        self,
        resources: WorkflowResources,
        *,
        id: str = "aggregate",
    ) -> None:
        super().__init__(id=id)
        self._resources = resources

    @handler
    async def handle(
        self,
        message: RetrievalBatchResult,
        ctx: WorkflowContext[
            EvidenceReady
        ],
    ) -> None:
        result = aggregate_evidence(
            message,
            broad_min_items=(
                self._resources.settings
                .broad_min_evidence_items
            ),
        )

        await ctx.send_message(result)

class ExpandPlanExecutor(Executor):
    def __init__(
        self,
        resources: WorkflowResources,
        *,
        id: str = "expand",
    ) -> None:
        super().__init__(id=id)
        self._resources = resources

    async def _expand(
        self,
        state,
        ctx: WorkflowContext[
            RetrievalPrepared
        ],
    ) -> None:
        prepared = prepare_gap_retrieval(
            state,
            self._resources,
        )

        await ctx.send_message(prepared)

    @handler
    async def from_evidence(
        self,
        message: EvidenceReady,
        ctx: WorkflowContext[
            RetrievalPrepared
        ],
    ) -> None:
        await self._expand(
            message.state,
            ctx,
        )

    @handler
    async def from_review(
        self,
        message: ReviewCompleted,
        ctx: WorkflowContext[
            RetrievalPrepared
        ],
    ) -> None:
        await self._expand(
            message.state,
            ctx,
        )

class AnswerExecutor(Executor):
    def __init__(
        self,
        resources: WorkflowResources,
        *,
        id: str = "answer",
    ) -> None:
        super().__init__(id=id)
        self._resources = resources

    @handler
    async def handle(
        self,
        message: EvidenceReady,
        ctx: WorkflowContext[
            AnswerCandidate
        ],
    ) -> None:
        candidate = await generate_answer(
            message.state,
            self._resources,
        )

        await ctx.send_message(candidate)

class VerifyExecutor(Executor):
    def __init__(
        self,
        resources: WorkflowResources,
        *,
        id: str = "verify",
    ) -> None:
        super().__init__(id=id)
        self._resources = resources

    @handler
    async def handle(
        self,
        message: AnswerCandidate,
        ctx: WorkflowContext[
            ReviewCompleted
        ],
    ) -> None:
        state = message.state

        if (
            state.shape
            == QuestionShape.MULTI_PART
        ):
            activation_reason = (
                "multi_part_validation"
            )

        elif (
            state.shape
            == QuestionShape.COMPARISON
        ):
            activation_reason = (
                "comparison_validation"
            )

        else:
            activation_reason = "policy"
        
        RAG_REVIEWER_ACTIVATION_TOTAL.labels(
            reason=activation_reason
        ).inc()


        result = await verify_answer(
            message,
            self._resources,
        )

        await ctx.send_message(result)


class BoundedAnswerExecutor(Executor):
    def __init__(
        self,
        *,
        id: str = "bounded",
    ) -> None:
        super().__init__(id=id)

    @handler
    async def from_context(
        self,
        message: ContextResolved,
        ctx: WorkflowContext[
            BoundedAnswer
        ],
    ) -> None:
        await ctx.send_message(
            build_context_bounded_answer(
                message
            )
        )

    @handler
    async def from_evidence(
        self,
        message: EvidenceReady,
        ctx: WorkflowContext[
            BoundedAnswer
        ],
    ) -> None:
        await ctx.send_message(
            build_evidence_bounded_answer(
                message
            )
        )

    @handler
    async def from_review(
        self,
        message: ReviewCompleted,
        ctx: WorkflowContext[
            BoundedAnswer
        ],
    ) -> None:
        await ctx.send_message(
            build_review_bounded_answer(
                message
            )
        )

class FinalizeExecutor(Executor):
    def __init__(
        self,
        *,
        id: str = "finalize",
    ) -> None:
        super().__init__(id=id)

    @handler
    async def from_compiled(
        self,
        message: CompiledRequest,
        ctx: WorkflowContext[
            Never,
            FinalAnswer,
        ],
    ) -> None:
        await ctx.yield_output(
            finalize_compiled(message)
        )
    @handler
    async def from_candidate(
        self,
        message: AnswerCandidate,
        ctx: WorkflowContext[
            Never,
            FinalAnswer,
        ],
    ) -> None:
        await ctx.yield_output(
            finalize_candidate(message)
        )

    @handler
    async def from_review(
        self,
        message: ReviewCompleted,
        ctx: WorkflowContext[
            Never,
            FinalAnswer,
        ],
    ) -> None:
        await ctx.yield_output(
            finalize_review(message)
        )

    @handler
    async def from_bounded(
        self,
        message: BoundedAnswer,
        ctx: WorkflowContext[
            Never,
            FinalAnswer,
        ],
    ) -> None:
        await ctx.yield_output(
            finalize_bounded(message)
        )


