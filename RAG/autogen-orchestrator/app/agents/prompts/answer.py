from __future__ import annotations

import json

from domain.state import RequestState
from domain.contracts import DraftPayload

ANSWER_INSTRUCTIONS = """
Generate a concise, grounded DMV answer using only the supplied evidence.

Return a DraftPayload structured response.

Grounding rules:
- Each sub-question contains its own evidence array.
- Evidence nested under one sub-question belongs ONLY to that sub-question.
- Return exactly one DraftSupport entry for each supplied sub-question.
- If sub_question_id is "sqN", every evidence_node_id in that support MUST start with "sqN:".
- The prefix before the first ":" of every evidence_node_id MUST exactly equal sub_question_id.
- Before returning, verify this prefix rule for every support entry.
- For every DraftSupport entry, use only node_id values nested under the same sub-question.
- Never reference evidence belonging to another sub-question.
- Copy node_id values exactly as supplied.
- Never invent or modify a node_id.
- Every supplied sub-question must have grounded support.
- If evidence is insufficient, do not use outside knowledge and do not borrow evidence from another sub-question.

Answer rules:
- Answer every requested intent directly.
- Prefer 1-2 short sentences per sub-question.
- Keep merged_answer proportional to the number of sub-questions, usually about 30-50 words per sub-question.
- Treat this as a soft target; use more only when essential conditions or exceptions would otherwise be lost.
- Include only requirements, actions, conditions, and exceptions needed to answer the question.
- Do not restate the question.
- Do not summarize every retrieved evidence item.
- Do not include examples, background information, or related rules unless necessary.
""".strip()

def _build_answer_payload(
    state: RequestState,
    *,
    max_chars_per_evidence: int,
) -> dict:
    if state.retrieval_plan is None:
        raise ValueError(
            "Answer prompt requires a "
            "retrieval plan"
        )

    evidence_by_question: dict[
        str,
        list[dict],
    ] = {
        item.id: []
        for item
        in state.retrieval_plan.sub_questions
    }

    for item in state.evidence:
        evidence_by_question.setdefault(
            item.sub_question_id,
            [],
        ).append(
            {
                "node_id":
                    item.node_id,

                "text":
                    item.text[
                        :max_chars_per_evidence
                    ],
            }
        )

    return {
        "user_query":
            state.resolved_query
            or state.user_query,

        "sub_questions": [
            {
                "id":
                    item.id,

                "question":
                    item.question,

                "evidence":
                    evidence_by_question.get(
                        item.id,
                        [],
                    ),
            }
            for item
            in state.retrieval_plan
            .sub_questions
        ],
    }
    
def build_answer_prompt(
    state: RequestState,
    *,
    max_chars_per_evidence: int,
) -> str:
    return json.dumps(
        _build_answer_payload(
            state,
            max_chars_per_evidence=(
                max_chars_per_evidence
            ),
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    
def build_answer_repair_prompt(
    state: RequestState,
    *,
    previous_draft: DraftPayload,
    validation_error: str,
    max_chars_per_evidence: int,
) -> str:
    payload = {
        "task": "repair_grounded_answer",

        "validation_error":
            validation_error,

        "previous_draft":
            previous_draft.model_dump(
                mode="json"
            ),

        "input":
            _build_answer_payload(
                state,
                max_chars_per_evidence=(
                    max_chars_per_evidence
                ),
            ),

        "requirements": [
            (
                "Every sub-question must "
                "have grounded support."
            ),
            (
                "Every evidence node ID must "
                "come from the evidence array "
                "of the same sub-question."
            ),
            (
                "Never move evidence from one "
                "sub-question to another."
            ),
            (
                "Never invent or modify "
                "evidence node IDs."
            ),
            (
                "Rewrite merged_answer if "
                "necessary to remain grounded."
            ),
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )