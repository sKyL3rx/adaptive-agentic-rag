from __future__ import annotations

import json

from domain.contracts import (
    QuestionShape,
    RetrievalPlanProposal,
)
from domain.state import RequestState


PLANNING_INSTRUCTIONS = """
Decompose a complex DMV request into independently retrievable queries.

CRITICAL RULES:
- Create one query for each explicit user intent.
- Preserve every explicit intent from the user's request.
- Never merge distinct explicit intents into one query.
- If the user asks exactly two explicit things, return exactly two queries.
- If the user asks exactly three explicit things, return exactly three queries.
- Each query must be independently usable for semantic retrieval.
- Queries must be semantically distinct.
- Do not add related topics that were not explicitly requested.
- Do not answer the user.
- Do not retrieve documents.
- Do not assign sub-question IDs.
- Do not decide workflow routing.

Return a RetrievalPlanProposal structured response.
""".strip()


def build_planner_prompt(
    state: RequestState,
) -> str:
    payload = {
        "resolved_query":
            state.resolved_query
            or state.user_query,

        "question_shape":
            (
                state.shape.value
                if state.shape is not None
                else None
            ),

        "max_sub_questions":
            state.budget.max_sub_questions,
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

def build_planner_repair_prompt(
    state: RequestState,
    *,
    failed_proposal: RetrievalPlanProposal,
    validation_error: str,
) -> str:
    minimum_sub_questions = (
        2
        if state.shape in {
            QuestionShape.MULTI_PART,
            QuestionShape.COMPARISON,
        }
        else 1
    )

    payload = {
        "task": "repair_retrieval_plan",

        "resolved_query":
            state.resolved_query
            or state.user_query,

        "question_shape":
            (
                state.shape.value
                if state.shape is not None
                else None
            ),

        "previous_output":
            failed_proposal.model_dump(
                mode="json"
            ),

        "validation_error":
            validation_error,

        "requirements": {
            "minimum_sub_questions":
                minimum_sub_questions,

            "maximum_sub_questions":
                state.budget.max_sub_questions,

            "preserve_every_explicit_intent":
                True,

            "never_merge_distinct_intents":
                True,

            "return_only_retrieval_queries":
                True,
        },
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )