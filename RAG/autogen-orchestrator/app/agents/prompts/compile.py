from __future__ import annotations

import json


COMPILER_INSTRUCTIONS = """
Classify the latest user message into exactly one request shape.

Allowed shapes:

casual_conversation:
A greeting, thanks, goodbye, conversational acknowledgment,
or lightweight social interaction that does not require
California DMV handbook information.

conversation_recall:
The user asks what they previously asked, what the assistant
previously said, or requests the latest user or assistant
message from this conversation.

single_focused:
One standalone California DMV information request with one
main intent.

broad_coverage:
A broad overview, guide, summary, or explanation of one
coherent California DMV topic.

multi_part:
Two or more distinct California DMV questions or tasks that
should become separate retrieval queries.

comparison:
A request to compare two or more California DMV rules,
documents, procedures, or situations.

context_dependent:
A California DMV information request that cannot be
understood without recent conversation history. This does
not include asking what was previously said.

Output requirements:

- Return only a QuestionShapeDecision structured response.
- Classify every request with the model.
- Do not rely on keyword or regex rules.
- Keep reasoning to one short sentence.

For casual_conversation:

- Write the complete concise reply in direct_response.
- Set recall_target to "none".

For conversation_recall:

- Set direct_response to null.
- Set recall_target to exactly one of:
  - "last_user_message"
  - "last_assistant_message"

For all California DMV retrieval shapes:

- Set direct_response to null.
- Set recall_target to "none".

Additional rules:

- Do not answer California DMV information questions.
- Do not retrieve documents.
- Do not create sub-questions.
- Do not decide workflow routing.
- Do not choose context_dependent when
  has_recent_history is false.

Important distinctions:

"Hello"
-> casual_conversation

"What did I just ask?"
-> conversation_recall
-> last_user_message

"What did you just tell me?"
-> conversation_recall
-> last_assistant_message

"What about drivers under 18?"
when prior DMV context is required
-> context_dependent
""".strip()


def build_compile_prompt(
    *,
    user_query: str,
    has_recent_history: bool,
    recent_history_preview: str,
) -> str:
    payload = {
        "latest_user_query": user_query,
        "has_recent_history": (
            has_recent_history
        ),
        "recent_history_preview": (
            recent_history_preview
            if has_recent_history
            else ""
        ),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )