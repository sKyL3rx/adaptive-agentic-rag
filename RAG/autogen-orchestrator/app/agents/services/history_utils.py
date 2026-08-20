from __future__ import annotations

from collections.abc import Sequence

from domain.contracts import (
    ConversationRecallTarget,
)
from domain.state import HistoryTurn


def build_recent_history_preview(
    history: Sequence[HistoryTurn],
    *,
    max_turns: int = 4,
    max_total_chars: int = 1_600,
    max_chars_per_turn: int = 500,
) -> str:
    if (
        not history
        or max_turns <= 0
        or max_total_chars <= 0
        or max_chars_per_turn <= 0
    ):
        return ""

    selected_turns = history[-max_turns:]
    lines: list[str] = []

    for turn in selected_turns:
        role = turn.role.strip().lower()

        content = " ".join(
            turn.content.strip().split()
        )

        if not content:
            continue

        lines.append(
            f"{role}: "
            f"{content[:max_chars_per_turn]}"
        )

    preview = "\n".join(lines)

    if len(preview) > max_total_chars:
        preview = preview[-max_total_chars:]

    return preview


def build_conversation_recall_response(
    history: Sequence[HistoryTurn],
    *,
    target: ConversationRecallTarget,
) -> str:
    if target == "last_user_message":
        wanted_role = "user"

        prefix = (
            "Your previous question was:"
        )

        empty_message = (
            "There is no previous user question "
            "available in this conversation."
        )

    elif target == "last_assistant_message":
        wanted_role = "assistant"

        prefix = (
            "My previous response was:"
        )

        empty_message = (
            "There is no previous assistant response "
            "available in this conversation."
        )

    else:
        raise ValueError(
            "Conversation recall target must identify "
            "a user or assistant message"
        )

    for turn in reversed(history):
        role = turn.role.strip().lower()
        content = turn.content.strip()

        if (
            role == wanted_role
            and content
        ):
            return (
                f"{prefix}\n\n"
                f"{content}"
            )

    return empty_message