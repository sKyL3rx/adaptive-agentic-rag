from __future__ import annotations

import json

from domain.contracts import DraftPayload
from domain.state import RequestState


REVIEW_INSTRUCTIONS = """
Review whether the draft answer is fully supported by the supplied evidence.

Return a ReviewDecisionProposal structured response.

Allowed verdicts:
- approve
- needs_more
- insufficient_evidence

Decision rules:
- Use approve when every requested sub-question is answered and the answer is supported by its evidence.
- For approve:
  - reason MUST be null.
  - missing_sub_questions MUST be empty.
  - next_queries MUST be empty.
- For needs_more:
  - reason must be one short sentence, maximum 20 words.
  - include only genuinely missing sub-question IDs.
  - provide only the minimum targeted retrieval queries needed to close the gap.
- For insufficient_evidence:
  - reason must be one short sentence, maximum 20 words.
  - next_queries MUST be empty.

Efficiency rules:
- Do not explain evidence item by item.
- Do not quote evidence.
- Do not restate the draft.
- Do not restate the user's question.
- Do not describe your reasoning process.
- Do not rewrite the answer.
- Do not retrieve documents.
- Do not assign new sub-question IDs.
- missing_sub_questions may contain only existing IDs supplied in the prompt.
- next_queries must not contain IDs.
- Do not decide workflow routing.
""".strip()


def build_review_prompt(
    state: RequestState,
    draft: DraftPayload,
    *,
    max_chars_per_evidence: int,
) -> str:
    if state.retrieval_plan is None:
        raise ValueError(
            "Review prompt requires a "
            "retrieval plan"
        )

    payload = {
        "user_query":
            state.resolved_query
            or state.user_query,

        "allowed_sub_question_ids": [
            item.id
            for item
            in state.retrieval_plan
            .sub_questions
        ],

        "sub_questions": [
            item.model_dump(
                mode="json"
            )
            for item
            in state.retrieval_plan
            .sub_questions
        ],

        "draft":
            draft.model_dump(
                mode="json"
            ),

        "evidence": [
            {
                "node_id":
                    item.node_id,

                "sub_question_id":
                    item.sub_question_id,

                "text":
                    item.text[
                        :max_chars_per_evidence
                    ],
            }
            for item in state.evidence
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )