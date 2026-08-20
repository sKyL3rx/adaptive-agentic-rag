from __future__ import annotations

from agent_framework import (
    Case,
    Default,
    WorkflowBuilder,
)

from workflow.executors import (
    AggregateExecutor,
    AnswerExecutor,
    BoundedAnswerExecutor,
    CompileExecutor,
    ExpandPlanExecutor,
    FinalizeExecutor,
    PlanExecutor,
    PrepareFastPlanExecutor,
    ResolveContextExecutor,
    RetrieveExecutor,
    VerifyExecutor,
)
from workflow.policies import (
    context_requires_planner,
    context_uses_fast_plan,
    coverage_is_recoverable,
    coverage_is_sufficient,
    needs_context,
    review_approved,
    review_recoverable,
    returns_directly,
    should_verify,
    uses_fast_plan,
)

from workflow.resources import WorkflowResources


def build_request_workflow(
    resources: WorkflowResources,
):
    compile_executor = CompileExecutor(
        resources
    )

    resolve_context = ResolveContextExecutor(
        resources
    )

    prepare_fast = PrepareFastPlanExecutor(
        resources
    )

    planner = PlanExecutor(
        resources
    )

    retrieve = RetrieveExecutor(
        resources
    )

    aggregate = AggregateExecutor(
        resources
    )

    expand = ExpandPlanExecutor(
        resources
    )

    answer = AnswerExecutor(
        resources
    )

    verify = VerifyExecutor(
        resources
    )

    bounded = BoundedAnswerExecutor()

    finalize = FinalizeExecutor()

    builder = WorkflowBuilder(
        max_iterations=(
            resources.settings
            .workflow_max_iterations
        ),
        name="dmv-rag-request",
        start_executor=compile_executor,
        output_from=[finalize],
    )

    builder.add_switch_case_edge_group(
        compile_executor,
        [   
            Case(
                condition=returns_directly,
                target=finalize,
            ),
            Case(
                condition=needs_context,
                target=resolve_context,
            ),
            Case(
                condition=uses_fast_plan,
                target=prepare_fast,
            ),
            Default(
                target=planner,
            ),
        ],
    )

    builder.add_switch_case_edge_group(
        resolve_context,
        [
            Case(
                condition=context_uses_fast_plan,
                target=prepare_fast,
            ),
            Case(
                condition=context_requires_planner,
                target=planner,
            ),
            Default(
                target=bounded,
            ),
        ],
    )

    builder.add_edge(
        prepare_fast,
        retrieve,
    )

    builder.add_edge(
        planner,
        retrieve,
    )

    builder.add_edge(
        expand,
        retrieve,
    )

    builder.add_edge(
        retrieve,
        aggregate,
    )


    builder.add_switch_case_edge_group(
        aggregate,
        [
            Case(
                condition=coverage_is_sufficient,
                target=answer,
            ),
            Case(
                condition=coverage_is_recoverable,
                target=expand,
            ),
            Default(
                target=bounded,
            ),
        ],
    )

    builder.add_switch_case_edge_group(
        answer,
        [
            Case(
                condition=should_verify,
                target=verify,
            ),
            Default(
                target=finalize,
            ),
        ],
    )

    builder.add_switch_case_edge_group(
        verify,
        [
            Case(
                condition=review_approved,
                target=finalize,
            ),
            Case(
                condition=review_recoverable,
                target=expand,
            ),
            Default(
                target=bounded,
            ),
        ],
    )

    builder.add_edge(   
        bounded,
        finalize,
    )


    return builder.build()