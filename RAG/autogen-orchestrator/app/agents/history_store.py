from __future__ import annotations

import json
import os
import time

import redis.asyncio as redis

from observability import hash_user_id


HISTORY_TTL_SECONDS = int(
    os.getenv(
        "HISTORY_TTL_SECONDS",
        "1800",
    )
)
HISTORY_MAX_MESSAGES = int(
    os.getenv(
        "HISTORY_MAX_MESSAGES",
        "30",
    )
)


def _truncate(
    text: str,
    max_len_chars: int = 900,
) -> str:
    text = (text or "").strip()

    if len(text) <= max_len_chars:
        return text

    return (
        text[: max_len_chars - 3]
        .rstrip()
        + "..."
    )


def _redis_key_qa(
    session_id: str,
) -> str:
    return (
        "chat:"
        f"{hash_user_id(session_id)}"
        ":qa"
    )


def _pack_qa(
    question: str,
    short_answer: str,
) -> str:
    return json.dumps(
        {
            "q": question,
            "a": short_answer,
            "ts": time.time(),
        }
    )


async def append_history(
    r: redis.Redis,
    session_id: str,
    question: str,
    short_answer: str,
) -> None:
    key = _redis_key_qa(
        session_id
    )

    async with r.pipeline(
        transaction=False
    ) as pipe:
        pipe.rpush(
            key,
            _pack_qa(
                question,
                short_answer,
            ),
        )

        pipe.ltrim(
            key,
            -HISTORY_MAX_MESSAGES,
            -1,
        )

        pipe.expire(
            key,
            HISTORY_TTL_SECONDS,
        )

        await pipe.execute()


async def get_recent_history(
    r: redis.Redis,
    session_id: str,
    *,
    last_n: int = 4,
) -> list[dict]:
    """Load latest QA turns for conversation-aware follow-up."""
    key = _redis_key_qa(
        session_id
    )

    raw_items = await r.lrange(
        key,
        -last_n,
        -1,
    )

    items: list[dict] = []

    for raw in raw_items:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            obj = json.loads(raw)
        except Exception:
            continue

        q = _truncate(
            str(obj.get("q", "")),
            700,
        )
        a = _truncate(
            str(obj.get("a", "")),
            900,
        )

        if q or a:
            items.append(
                {
                    "q": q,
                    "a": a,
                    "ts": obj.get("ts"),
                }
            )

    return items


def format_recent_history(
    items: list[dict],
) -> str:
    """Format recent QA turns for the agent task prompt."""
    if not items:
        return (
            "No previous conversation "
            "context is available."
        )

    chunks: list[str] = []

    for idx, item in enumerate(
        items,
        start=1,
    ):
        chunks.append(
            f"Turn {idx}:\n"
            f"User: {item.get('q', '')}\n"
            f"Assistant: {item.get('a', '')}"
        )

    return "\n\n".join(chunks)
