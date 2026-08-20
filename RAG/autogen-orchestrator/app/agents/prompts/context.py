from __future__ import annotations

import json

from domain.state import RequestState
from services.history_utils import (
    build_recent_history_preview,
)

CONTEXT_INSTRUCTIONS = """
Rewrite a context-dependent DMV question into a standalone
question.

Return a ContextRewritePayload structured response.

Requirements:

- Use only the provided recent history.
- Do not answer the user's question.
- Do not retrieve documents.
- If the reference cannot be resolved reliably, set
  can_resolve=false.
- If can_resolve=true, provide a standalone_question
  or query.
- rewritten_shape must describe the standalone DMV request.
- When can_resolve=true, rewritten_shape must be exactly
  one of:
  - single_focused
  - broad_coverage
  - multi_part
  - comparison
- Never return casual_conversation, conversation_recall,
  or context_dependent as the rewritten shape of a
  resolved DMV follow-up.
- needs_full_planner=true only for multi-part or
  comparison requests.
""".strip()


def build_context_prompt(
    state: RequestState,
    *,
    max_history_turns: int,
    max_history_chars: int,
) -> str:
    payload = {
        "latest_user_query":
            state.user_query,

        "recent_history_preview":
            build_recent_history_preview(
                state.recent_history,
                max_turns=max_history_turns,
                max_total_chars=max_history_chars,
            ),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )