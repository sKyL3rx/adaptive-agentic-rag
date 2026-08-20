from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import redis.asyncio as redis
import websockets


EVAL_RESULT_VERSION = "e2e_v3_eval_telemetry"

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

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


def hash_session_id(session_id: str) -> str:
    return hashlib.sha256(
        session_id.encode("utf-8")
    ).hexdigest()[:16]


def redis_history_key_for_session(
    session_id: str,
) -> str:
    return (
        f"chat:{hash_session_id(session_id)}:qa"
    )


def pack_qa(
    question: str,
    short_answer: str,
) -> str:
    return json.dumps(
        {
            "q": question,
            "a": short_answer,
            "ts": time.time(),
        },
        ensure_ascii=False,
    )


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def append_jsonl_row(
    path: Path,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                row,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        file.flush()
        os.fsync(file.fileno())


def read_existing_results_by_id(
    path: Path,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    results: dict[str, dict[str, Any]] = {}

    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(
                    json.dumps(
                        {
                            "warning": "skip_bad_jsonl_line",
                            "path": str(path),
                            "line_no": line_no,
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            item_id = str(
                row.get("id", "")
            ).strip()

            if item_id:
                # Latest line wins for duplicate IDs.
                results[item_id] = row

    return results


def is_successful_result(
    row: dict[str, Any],
) -> bool:
    if (
        row.get("eval_version")
        != EVAL_RESULT_VERSION
    ):
        return False

    rag_meta = row.get("rag_meta", {}) or {}

    if rag_meta.get("error"):
        return False

    if not rag_meta.get(
        "eval_telemetry_present",
        False,
    ):
        return False

    return bool(
        str(
            row.get("rag_answer", "")
        ).strip()
    )


def get_conversation_turns(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    inputs = row.get("inputs", {}) or {}
    turns = inputs.get(
        "conversation_turns",
        [],
    )

    if not isinstance(turns, list):
        return []

    return [
        turn
        for turn in turns
        if isinstance(turn, dict)
    ]


def get_turn_role(
    turn: dict[str, Any],
) -> str:
    return str(
        turn.get(
            "role",
            turn.get("speaker", ""),
        )
    ).strip().lower()


def get_turn_content(
    turn: dict[str, Any],
) -> str:
    for key in (
        "content",
        "message",
        "text",
        "utterance",
    ):
        value = turn.get(key)

        if value:
            return str(value).strip()

    return ""


def pair_conversation_turns(
    turns: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Convert dataset conversation turns into Redis QA pairs."""
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None

    for turn in turns:
        role = get_turn_role(turn)
        content = get_turn_content(turn)

        if not content:
            continue

        if role in {
            "assistant",
            "bot",
            "system",
        }:
            if pending_user is not None:
                pairs.append(
                    (
                        pending_user,
                        content,
                    )
                )
                pending_user = None

            continue

        if pending_user is not None:
            pairs.append(
                (
                    pending_user,
                    "",
                )
            )

        pending_user = content

    if pending_user is not None:
        pairs.append(
            (
                pending_user,
                "",
            )
        )

    return pairs


async def seed_redis_history_from_dataset(
    *,
    redis_client: redis.Redis,
    session_id: str,
    conversation_turns: list[dict[str, Any]],
) -> int:
    key = redis_history_key_for_session(
        session_id
    )

    await redis_client.delete(key)

    pairs = pair_conversation_turns(
        conversation_turns
    )

    for question, answer in pairs:
        await redis_client.rpush(
            key,
            pack_qa(
                question,
                answer,
            ),
        )

    await redis_client.ltrim(
        key,
        -HISTORY_MAX_MESSAGES,
        -1,
    )
    await redis_client.expire(
        key,
        HISTORY_TTL_SECONDS,
    )

    return len(pairs)


async def clear_redis_history_for_session(
    *,
    redis_client: redis.Redis,
    session_id: str,
) -> None:
    await redis_client.delete(
        redis_history_key_for_session(
            session_id
        )
    )


def normalize_retrieval_groups(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    groups: list[dict[str, Any]] = []

    for raw_group in value:
        if not isinstance(raw_group, dict):
            continue

        sub_question_id = str(
            raw_group.get(
                "sub_question_id",
                "",
            )
        ).strip()

        query = str(
            raw_group.get("query", "")
        ).strip()

        raw_chunk_ids = raw_group.get(
            "chunk_ids",
            [],
        )

        chunk_ids: list[str] = []
        seen: set[str] = set()

        if isinstance(
            raw_chunk_ids,
            (list, tuple),
        ):
            for raw_chunk_id in raw_chunk_ids:
                chunk_id = str(
                    raw_chunk_id
                ).strip()

                if (
                    chunk_id
                    and chunk_id not in seen
                ):
                    seen.add(chunk_id)
                    chunk_ids.append(chunk_id)

        groups.append(
            {
                "sub_question_id": (
                    sub_question_id
                ),
                "query": query,
                "chunk_ids": chunk_ids,
            }
        )

    return groups


def flatten_retrieval_groups(
    groups: list[dict[str, Any]],
    *,
    per_group_k: int | None = None,
) -> list[str]:
    flattened: list[str] = []
    seen: set[str] = set()

    for group in groups:
        chunk_ids = list(
            group.get("chunk_ids", [])
            or []
        )

        if per_group_k is not None:
            chunk_ids = chunk_ids[
                :per_group_k
            ]

        for chunk_id in chunk_ids:
            chunk_id = str(chunk_id).strip()

            if (
                chunk_id
                and chunk_id not in seen
            ):
                seen.add(chunk_id)
                flattened.append(chunk_id)

    return flattened


def hit_at_k_by_group(
    gold_ids: list[str],
    groups: list[dict[str, Any]],
    k: int,
) -> float:
    gold = set(gold_ids)

    if not gold:
        return 0.0

    retrieved = set(
        flatten_retrieval_groups(
            groups,
            per_group_k=k,
        )
    )

    return 1.0 if gold & retrieved else 0.0


def recall_at_k_by_group(
    gold_ids: list[str],
    groups: list[dict[str, Any]],
    k: int,
) -> float:
    gold = set(gold_ids)

    if not gold:
        return 0.0

    retrieved = set(
        flatten_retrieval_groups(
            groups,
            per_group_k=k,
        )
    )

    return len(gold & retrieved) / len(gold)


def exact_match_at_k_by_group(
    gold_ids: list[str],
    groups: list[dict[str, Any]],
    k: int,
) -> float:
    gold = set(gold_ids)

    if not gold:
        return 0.0

    retrieved = set(
        flatten_retrieval_groups(
            groups,
            per_group_k=k,
        )
    )

    return 1.0 if gold.issubset(retrieved) else 0.0


def group_hit_rate_at_k(
    gold_ids: list[str],
    groups: list[dict[str, Any]],
    k: int,
) -> float:
    if not groups:
        return 0.0

    gold = set(gold_ids)

    if not gold:
        return 0.0

    hits = 0

    for group in groups:
        group_ids = {
            str(chunk_id).strip()
            for chunk_id in (
                group.get("chunk_ids", [])
                or []
            )[:k]
            if str(chunk_id).strip()
        }

        if gold & group_ids:
            hits += 1

    return hits / len(groups)


def reciprocal_rank_for_group(
    gold_ids: set[str],
    chunk_ids: list[str],
) -> float:
    for rank, chunk_id in enumerate(
        chunk_ids,
        start=1,
    ):
        if chunk_id in gold_ids:
            return 1.0 / rank

    return 0.0


def mean_group_mrr(
    gold_ids: list[str],
    groups: list[dict[str, Any]],
) -> float:
    """Mean reciprocal rank of the first gold hit within each retrieval group."""
    if not groups:
        return 0.0

    gold = set(gold_ids)

    if not gold:
        return 0.0

    values = [
        reciprocal_rank_for_group(
            gold,
            [
                str(chunk_id)
                for chunk_id in (
                    group.get(
                        "chunk_ids",
                        [],
                    )
                    or []
                )
            ],
        )
        for group in groups
    ]

    return sum(values) / len(values)


def gold_mrr(
    gold_ids: list[str],
    groups: list[dict[str, Any]],
) -> float:
    """Mean best reciprocal rank for every gold chunk across all groups."""
    if not gold_ids:
        return 0.0

    values: list[float] = []

    for gold_id in dict.fromkeys(gold_ids):
        best_rr = 0.0

        for group in groups:
            chunk_ids = [
                str(chunk_id)
                for chunk_id in (
                    group.get(
                        "chunk_ids",
                        [],
                    )
                    or []
                )
            ]

            for rank, chunk_id in enumerate(
                chunk_ids,
                start=1,
            ):
                if chunk_id == gold_id:
                    best_rr = max(
                        best_rr,
                        1.0 / rank,
                    )
                    break

        values.append(best_rr)

    return sum(values) / len(values)


def binary_match(
    expected: Any,
    actual: Any,
) -> float:
    expected_text = str(
        expected or ""
    ).strip().lower()
    actual_text = str(
        actual or ""
    ).strip().lower()

    if not expected_text:
        return 0.0

    return (
        1.0
        if expected_text == actual_text
        else 0.0
    )


def citation_gold_recall(
    gold_ids: list[str],
    citation_ids: list[str],
) -> float:
    gold = set(gold_ids)

    if not gold:
        return 0.0

    cited = set(citation_ids)
    return len(gold & cited) / len(gold)


def citation_gold_precision(
    gold_ids: list[str],
    citation_ids: list[str],
) -> float:
    cited = set(citation_ids)

    if not cited:
        return 0.0

    gold = set(gold_ids)
    return len(gold & cited) / len(cited)


def build_metrics(
    *,
    gold_chunk_ids: list[str],
    retrieval_groups: list[dict[str, Any]],
    citation_source_ids: list[str],
    expected_question_shape: Any,
    actual_question_shape: Any,
    expected_retrieval_mode: Any,
    actual_retrieval_mode: Any,
    top_k: int,
) -> dict[str, float]:
    return {
        f"hit@{top_k}": hit_at_k_by_group(
            gold_chunk_ids,
            retrieval_groups,
            top_k,
        ),
        f"recall@{top_k}": recall_at_k_by_group(
            gold_chunk_ids,
            retrieval_groups,
            top_k,
        ),
        f"exact_match@{top_k}": (
            exact_match_at_k_by_group(
                gold_chunk_ids,
                retrieval_groups,
                top_k,
            )
        ),
        f"group_hit_rate@{top_k}": (
            group_hit_rate_at_k(
                gold_chunk_ids,
                retrieval_groups,
                top_k,
            )
        ),
        "group_mrr": mean_group_mrr(
            gold_chunk_ids,
            retrieval_groups,
        ),
        "gold_mrr": gold_mrr(
            gold_chunk_ids,
            retrieval_groups,
        ),
        "citation_gold_recall": (
            citation_gold_recall(
                gold_chunk_ids,
                citation_source_ids,
            )
        ),
        "citation_gold_precision": (
            citation_gold_precision(
                gold_chunk_ids,
                citation_source_ids,
            )
        ),
        "shape_match": binary_match(
            expected_question_shape,
            actual_question_shape,
        ),
        "retrieval_mode_match": (
            binary_match(
                expected_retrieval_mode,
                actual_retrieval_mode,
            )
        ),
    }


def zero_metrics(
    *,
    top_k: int,
) -> dict[str, float]:
    return {
        f"hit@{top_k}": 0.0,
        f"recall@{top_k}": 0.0,
        f"exact_match@{top_k}": 0.0,
        f"group_hit_rate@{top_k}": 0.0,
        "group_mrr": 0.0,
        "gold_mrr": 0.0,
        "citation_gold_recall": 0.0,
        "citation_gold_precision": 0.0,
        "shape_match": 0.0,
        "retrieval_mode_match": 0.0,
    }


def build_error_result(
    *,
    item_id: str,
    query: str,
    expectations: dict[str, Any],
    gold_chunk_ids: list[str],
    top_k: int,
    exc: BaseException,
    eval_setup_latency_ms: float,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "eval_version": EVAL_RESULT_VERSION,
        "id": item_id,
        "query": query,
        "question_shape": expectations.get(
            "question_shape"
        ),
        "expected_retrieval_mode": (
            expectations.get(
                "expected_retrieval_mode"
            )
        ),
        "gold_chunk_ids": gold_chunk_ids,
        "retrieved_chunk_ids": [],
        "retrieval_groups": [],
        "citations": [],
        "reference_answer": expectations.get(
            "reference_answer",
            "",
        ),
        "rag_answer": "",
        "rag_meta": {
            "error": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "eval_telemetry_present": False,
            "eval_setup_latency_ms": round(
                eval_setup_latency_ms,
                2,
            ),
        },
        "latency_ms": round(
            latency_ms,
            2,
        ),
        "metrics": zero_metrics(
            top_k=top_k
        ),
    }


async def ask_rag_ws(
    *,
    ws_url: str,
    query: str,
    session_id: str,
    timeout_sec: float,
    require_eval_telemetry: bool,
) -> dict[str, Any]:
    call_started_at = time.perf_counter()

    async with websockets.connect(
        ws_url,
        ping_interval=30,
        ping_timeout=None,
        close_timeout=5,
        max_size=None,
    ) as websocket:
        connected_at = time.perf_counter()

        await websocket.send(
            json.dumps(
                {
                    "session_id": session_id,
                    "content": query,
                },
                ensure_ascii=False,
            )
        )

        request_sent_at = time.perf_counter()

        events: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        evaluation_payload: dict[str, Any] | None = None
        request_id: str | None = None

        request_accepted_at: float | None = None
        retrieval_started_at: float | None = None
        completed_at: float | None = None

        while True:
            raw = await asyncio.wait_for(
                websocket.recv(),
                timeout=timeout_sec,
            )

            event = json.loads(raw)

            if not isinstance(event, dict):
                continue

            events.append(event)

            event_type = str(
                event.get("type", "")
            ).strip()

            event_request_id = str(
                event.get(
                    "request_id",
                    "",
                )
            ).strip()

            if event_request_id:
                request_id = event_request_id

            payload = event.get(
                "payload",
                {},
            ) or {}

            if not isinstance(payload, dict):
                payload = {}

            if event_type == "request_accepted":
                request_accepted_at = (
                    request_accepted_at
                    or time.perf_counter()
                )
                continue

            if event_type == "retrieval_started":
                retrieval_started_at = (
                    retrieval_started_at
                    or time.perf_counter()
                )
                continue

            if event_type == "evaluation":
                evaluation_payload = dict(payload)
                continue

            if event_type == "citation":
                source = payload.get("source")

                if isinstance(source, dict):
                    citations.append(
                        dict(source)
                    )

                continue

            if event_type == "error":
                message = str(
                    payload.get(
                        "message",
                        "Application returned an error event.",
                    )
                ).strip()

                raise RuntimeError(
                    message
                    or "Application returned an error event."
                )

            if event_type != "completed":
                continue

            completed_at = time.perf_counter()

            if not request_id:
                raise RuntimeError(
                    "completed event did not include request_id"
                )

            if (
                require_eval_telemetry
                and evaluation_payload is None
            ):
                raise RuntimeError(
                    "Evaluation telemetry was not received. "
                    "Start the application with "
                    "ENABLE_EVAL_TELEMETRY=true."
                )

            await websocket.send(
                json.dumps(
                    {
                        "type": "final_ack",
                        "request_id": request_id,
                    },
                    ensure_ascii=False,
                )
            )

            ack_sent_at = time.perf_counter()

            def elapsed_ms(
                end: float | None,
                start: float,
            ) -> float | None:
                if end is None:
                    return None

                return round(
                    (end - start) * 1000,
                    2,
                )

            return {
                "request_id": request_id,
                "answer": str(
                    payload.get(
                        "answer",
                        "",
                    )
                ),
                "completed_payload": dict(payload),
                "evaluation": (
                    dict(evaluation_payload)
                    if evaluation_payload is not None
                    else {}
                ),
                "citations": citations,
                "events": events,
                "timing": {
                    "ws_connect_ms": round(
                        (
                            connected_at
                            - call_started_at
                        )
                        * 1000,
                        2,
                    ),
                    "time_to_request_accepted_ms": (
                        elapsed_ms(
                            request_accepted_at,
                            request_sent_at,
                        )
                    ),
                    "time_to_retrieval_started_ms": (
                        elapsed_ms(
                            retrieval_started_at,
                            request_sent_at,
                        )
                    ),
                    "time_to_completed_ms": (
                        elapsed_ms(
                            completed_at,
                            request_sent_at,
                        )
                    ),
                    "client_e2e_ms": round(
                        (
                            ack_sent_at
                            - call_started_at
                        )
                        * 1000,
                        2,
                    ),
                },
            }


async def evaluate_one(
    *,
    row: dict[str, Any],
    ws_url: str,
    top_k: int,
    timeout_sec: float,
    replay_history: bool,
    redis_client: redis.Redis | None,
    require_eval_telemetry: bool,
) -> dict[str, Any]:
    item_id = str(row["id"])
    query = str(
        row["inputs"]["user_query"]
    )

    expectations = row.get(
        "expectations",
        {},
    ) or {}

    gold_chunk_ids = [
        str(chunk_id)
        for chunk_id in expectations.get(
            "gold_chunk_ids",
            [],
        )
    ]

    session_id = f"eval:{item_id}"

    replayed_turns = 0
    seeded_history_pairs = 0
    history_mode = "none"

    setup_started_at = time.perf_counter()

    try:
        conversation_turns = (
            get_conversation_turns(row)
        )

        if replay_history and conversation_turns:
            if redis_client is None:
                raise RuntimeError(
                    "redis_client is required when replay_history=True "
                    "and dataset conversation_turns are present"
                )

            seeded_history_pairs = (
                await seed_redis_history_from_dataset(
                    redis_client=redis_client,
                    session_id=session_id,
                    conversation_turns=(
                        conversation_turns
                    ),
                )
            )
            history_mode = (
                "redis_seeded_from_dataset"
            )

        elif replay_history:
            history_mode = (
                "empty_dataset_history"
            )

            if redis_client is not None:
                await clear_redis_history_for_session(
                    redis_client=redis_client,
                    session_id=session_id,
                )

        elif redis_client is not None:
            history_mode = "disabled_and_cleared"

            await clear_redis_history_for_session(
                redis_client=redis_client,
                session_id=session_id,
            )

        eval_setup_latency_ms = (
            time.perf_counter()
            - setup_started_at
        ) * 1000

        request_started_at = time.perf_counter()

        rag_result = await ask_rag_ws(
            ws_url=ws_url,
            query=query,
            session_id=session_id,
            timeout_sec=timeout_sec,
            require_eval_telemetry=(
                require_eval_telemetry
            ),
        )

        request_latency_ms = (
            time.perf_counter()
            - request_started_at
        ) * 1000

    except Exception as exc:
        eval_setup_latency_ms = (
            time.perf_counter()
            - setup_started_at
        ) * 1000

        request_latency_ms = 0.0

        if "request_started_at" in locals():
            request_latency_ms = (
                time.perf_counter()
                - request_started_at
            ) * 1000

        return build_error_result(
            item_id=item_id,
            query=query,
            expectations=expectations,
            gold_chunk_ids=gold_chunk_ids,
            top_k=top_k,
            exc=exc,
            eval_setup_latency_ms=(
                eval_setup_latency_ms
            ),
            latency_ms=request_latency_ms,
        )

    completed_payload = (
        rag_result.get(
            "completed_payload",
            {},
        )
        or {}
    )

    evaluation = (
        rag_result.get("evaluation", {})
        or {}
    )

    retrieval_groups = (
        normalize_retrieval_groups(
            evaluation.get(
                "retrieval_groups",
                [],
            )
        )
    )

    retrieved_chunk_ids = (
        flatten_retrieval_groups(
            retrieval_groups
        )
    )

    citations = [
        citation
        for citation in (
            rag_result.get(
                "citations",
                [],
            )
            or []
        )
        if isinstance(citation, dict)
    ]

    citation_source_ids = [
        str(
            citation.get(
                "source_id",
                "",
            )
        ).strip()
        for citation in citations
        if str(
            citation.get(
                "source_id",
                "",
            )
        ).strip()
    ]

    actual_question_shape = evaluation.get(
        "question_shape"
    )
    actual_retrieval_mode = evaluation.get(
        "retrieval_mode"
    )

    metrics = build_metrics(
        gold_chunk_ids=gold_chunk_ids,
        retrieval_groups=retrieval_groups,
        citation_source_ids=citation_source_ids,
        expected_question_shape=(
            expectations.get(
                "question_shape"
            )
        ),
        actual_question_shape=(
            actual_question_shape
        ),
        expected_retrieval_mode=(
            expectations.get(
                "expected_retrieval_mode"
            )
        ),
        actual_retrieval_mode=(
            actual_retrieval_mode
        ),
        top_k=top_k,
    )

    timing = rag_result.get(
        "timing",
        {},
    ) or {}

    client_e2e_ms = timing.get(
        "client_e2e_ms"
    )

    latency_ms = (
        float(client_e2e_ms)
        if client_e2e_ms is not None
        else request_latency_ms
    )

    rag_meta = {
        "error": False,
        "request_id": rag_result.get(
            "request_id"
        ),
        "eval_telemetry_present": bool(
            evaluation
        ),
        "actual_question_shape": (
            actual_question_shape
        ),
        "actual_route": evaluation.get(
            "route"
        ),
        "actual_retrieval_mode": (
            actual_retrieval_mode
        ),
        "coverage_status": evaluation.get(
            "coverage_status"
        ),
        "termination_reason": (
            completed_payload.get(
                "termination_reason"
            )
        ),
        "blocked": bool(
            completed_payload.get(
                "blocked",
                False,
            )
        ),
        "insufficient_evidence": bool(
            completed_payload.get(
                "insufficient_evidence",
                False,
            )
        ),
        "budget_exhausted": bool(
            completed_payload.get(
                "budget_exhausted",
                False,
            )
        ),
        "model_calls_used": int(
            completed_payload.get(
                "model_calls_used",
                0,
            )
            or 0
        ),
        "retrieval_rounds_used": int(
            completed_payload.get(
                "retrieval_rounds_used",
                0,
            )
            or 0
        ),
        "retrieval_group_count": len(
            retrieval_groups
        ),
        "citation_count": len(citations),
        "citation_source_ids": (
            citation_source_ids
        ),
        "eval_replayed_turns": replayed_turns,
        "eval_seeded_history_pairs": (
            seeded_history_pairs
        ),
        "eval_history_mode": history_mode,
        "eval_setup_latency_ms": round(
            eval_setup_latency_ms,
            2,
        ),
        "timing": timing,
    }

    return {
        "eval_version": EVAL_RESULT_VERSION,
        "id": item_id,
        "query": query,
        "question_shape": expectations.get(
            "question_shape"
        ),
        "expected_retrieval_mode": (
            expectations.get(
                "expected_retrieval_mode"
            )
        ),
        "gold_chunk_ids": gold_chunk_ids,
        "retrieved_chunk_ids": (
            retrieved_chunk_ids
        ),
        "retrieval_groups": retrieval_groups,
        "citations": citations,
        "reference_answer": expectations.get(
            "reference_answer",
            "",
        ),
        "rag_answer": rag_result.get(
            "answer",
            "",
        ),
        "rag_meta": rag_meta,
        "latency_ms": round(
            latency_ms,
            2,
        ),
        "metrics": metrics,
    }


def percentile_nearest_rank(
    values: list[float],
    p: float,
) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return round(
            sorted_values[0],
            2,
        )

    index = int(
        round(
            (len(sorted_values) - 1)
            * p
        )
    )

    return round(
        sorted_values[index],
        2,
    )


def numeric_distribution(
    values: list[float],
) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "avg": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "count": len(values),
        "avg": round(
            sum(values) / len(values),
            2,
        ),
        "p50": percentile_nearest_rank(
            values,
            0.50,
        ),
        "p90": percentile_nearest_rank(
            values,
            0.90,
        ),
        "p95": percentile_nearest_rank(
            values,
            0.95,
        ),
        "p99": percentile_nearest_rank(
            values,
            0.99,
        ),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def build_summary(
    *,
    rows: list[dict[str, Any]],
    dataset_path: Path,
    output_path: Path,
    top_k: int,
    evaluated_now: int,
    skipped_successful: int,
) -> dict[str, Any]:
    metric_names = [
        f"hit@{top_k}",
        f"recall@{top_k}",
        f"exact_match@{top_k}",
        f"group_hit_rate@{top_k}",
        "group_mrr",
        "gold_mrr",
        "citation_gold_recall",
        "citation_gold_precision",
        "shape_match",
        "retrieval_mode_match",
    ]

    quality: dict[str, float] = {}

    for metric_name in metric_names:
        values = [
            float(row["metrics"][metric_name])
            for row in rows
            if isinstance(
                row.get("metrics"),
                dict,
            )
            and metric_name in row["metrics"]
        ]

        quality[metric_name] = round(
            (
                sum(values) / len(values)
                if values
                else 0.0
            ),
            6,
        )

    success_rows = [
        row
        for row in rows
        if not (
            row.get("rag_meta", {}) or {}
        ).get("error")
    ]

    error_count = len(rows) - len(
        success_rows
    )

    latency_values = [
        float(row["latency_ms"])
        for row in success_rows
        if row.get("latency_ms") is not None
    ]

    time_to_completed_values = [
        float(value)
        for row in success_rows
        for value in [
            (
                (row.get("rag_meta", {}) or {})
                .get("timing", {})
                .get("time_to_completed_ms")
            )
        ]
        if value is not None
    ]

    model_calls = [
        float(
            (row.get("rag_meta", {}) or {})
            .get("model_calls_used", 0)
        )
        for row in success_rows
    ]

    retrieval_rounds = [
        float(
            (row.get("rag_meta", {}) or {})
            .get("retrieval_rounds_used", 0)
        )
        for row in success_rows
    ]

    citation_counts = [
        float(
            (row.get("rag_meta", {}) or {})
            .get("citation_count", 0)
        )
        for row in success_rows
    ]

    termination_counts = Counter(
        str(
            (row.get("rag_meta", {}) or {})
            .get("termination_reason", "UNKNOWN")
        )
        for row in success_rows
    )

    actual_shape_counts = Counter(
        str(
            (row.get("rag_meta", {}) or {})
            .get("actual_question_shape", "UNKNOWN")
        )
        for row in success_rows
    )

    route_counts = Counter(
        str(
            (row.get("rag_meta", {}) or {})
            .get("actual_route", "UNKNOWN")
        )
        for row in success_rows
    )

    total = len(rows)

    return {
        "eval_version": EVAL_RESULT_VERSION,
        "dataset": str(dataset_path),
        "out": str(output_path),
        "evaluated_total_latest": total,
        "evaluated_now": evaluated_now,
        "skipped_successful": skipped_successful,
        "errors_latest": error_count,
        "top_k_per_retrieval_group": top_k,
        "reliability": {
            "successes": len(success_rows),
            "errors": error_count,
            "success_rate": round(
                (
                    len(success_rows) / total
                    if total
                    else 0.0
                ),
                6,
            ),
        },
        "quality": quality,
        "runtime": {
            "client_e2e_latency_ms": (
                numeric_distribution(
                    latency_values
                )
            ),
            "time_to_completed_ms": (
                numeric_distribution(
                    time_to_completed_values
                )
            ),
            "model_calls_used": (
                numeric_distribution(
                    model_calls
                )
            ),
            "retrieval_rounds_used": (
                numeric_distribution(
                    retrieval_rounds
                )
            ),
            "citation_count": (
                numeric_distribution(
                    citation_counts
                )
            ),
        },
        "termination_counts": dict(
            sorted(
                termination_counts.items()
            )
        ),
        "actual_shape_counts": dict(
            sorted(
                actual_shape_counts.items()
            )
        ),
        "route_counts": dict(
            sorted(route_counts.items())
        ),
    }


async def run_eval(
    *,
    dataset_path: Path,
    output_path: Path,
    ws_url: str,
    top_k: int,
    limit: int,
    timeout_sec: float,
    resume: bool,
    replay_history: bool,
    require_eval_telemetry: bool,
) -> None:
    rows = read_jsonl(dataset_path)

    if limit > 0:
        rows = rows[:limit]

    if not resume and output_path.exists():
        output_path.unlink()

    existing_by_id = (
        read_existing_results_by_id(
            output_path
        )
        if resume
        else {}
    )

    completed_ids = {
        item_id
        for item_id, result in (
            existing_by_id.items()
        )
        if is_successful_result(result)
    }

    results_by_id: dict[
        str,
        dict[str, Any],
    ] = dict(existing_by_id)

    total = len(rows)
    skipped_successful = 0
    evaluated_now = 0

    redis_client: redis.Redis | None = None

    try:
        redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=False,
        )

        for index, row in enumerate(
            rows,
            start=1,
        ):
            item_id = str(row["id"])

            if item_id in completed_ids:
                skipped_successful += 1
                existing = results_by_id[
                    item_id
                ]

                print(
                    json.dumps(
                        {
                            "skip": index,
                            "total": total,
                            "id": item_id,
                            "reason": (
                                "already_successful"
                            ),
                            "latency_ms": (
                                existing.get(
                                    "latency_ms"
                                )
                            ),
                            "metrics": (
                                existing.get(
                                    "metrics",
                                    {},
                                )
                            ),
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            result = await evaluate_one(
                row=row,
                ws_url=ws_url,
                top_k=top_k,
                timeout_sec=timeout_sec,
                replay_history=replay_history,
                redis_client=redis_client,
                require_eval_telemetry=(
                    require_eval_telemetry
                ),
            )

            evaluated_now += 1

            append_jsonl_row(
                output_path,
                result,
            )

            results_by_id[item_id] = result

            rag_meta = (
                result.get("rag_meta", {})
                or {}
            )

            print(
                json.dumps(
                    {
                        "done": index,
                        "total": total,
                        "id": result["id"],
                        "expected_shape": (
                            result.get(
                                "question_shape"
                            )
                        ),
                        "actual_shape": (
                            rag_meta.get(
                                "actual_question_shape"
                            )
                        ),
                        "expected_mode": (
                            result.get(
                                "expected_retrieval_mode"
                            )
                        ),
                        "actual_mode": (
                            rag_meta.get(
                                "actual_retrieval_mode"
                            )
                        ),
                        "gold": result[
                            "gold_chunk_ids"
                        ],
                        "retrieved": result[
                            "retrieved_chunk_ids"
                        ],
                        "retrieval_groups": result.get(
                            "retrieval_groups",
                            [],
                        ),
                        "citations": rag_meta.get(
                            "citation_source_ids",
                            [],
                        ),
                        "latency_ms": result.get(
                            "latency_ms"
                        ),
                        "model_calls": rag_meta.get(
                            "model_calls_used"
                        ),
                        "retrieval_rounds": (
                            rag_meta.get(
                                "retrieval_rounds_used"
                            )
                        ),
                        "termination": rag_meta.get(
                            "termination_reason"
                        ),
                        "error": rag_meta.get(
                            "error",
                            False,
                        ),
                        "history_mode": rag_meta.get(
                            "eval_history_mode"
                        ),
                        "seeded_history_pairs": (
                            rag_meta.get(
                                "eval_seeded_history_pairs"
                            )
                        ),
                        "metrics": result["metrics"],
                    },
                    ensure_ascii=False,
                )
            )

    finally:
        if redis_client is not None:
            await redis_client.aclose()
            await (
                redis_client
                .connection_pool
                .disconnect()
            )

    ordered_results: list[
        dict[str, Any]
    ] = []

    for row in rows:
        item_id = str(row["id"])

        if item_id in results_by_id:
            ordered_results.append(
                results_by_id[item_id]
            )

    summary = build_summary(
        rows=ordered_results,
        dataset_path=dataset_path,
        output_path=output_path,
        top_k=top_k,
        evaluated_now=evaluated_now,
        skipped_successful=(
            skipped_successful
        ),
    )

    summary_path = output_path.with_name(
        output_path.stem
        + "_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "eval_version": (
                    EVAL_RESULT_VERSION
                ),
                "evaluated_total_latest": (
                    summary[
                        "evaluated_total_latest"
                    ]
                ),
                "evaluated_now": summary[
                    "evaluated_now"
                ],
                "skipped_successful": (
                    summary[
                        "skipped_successful"
                    ]
                ),
                "errors_latest": summary[
                    "errors_latest"
                ],
                "out": str(output_path),
                "summary": str(summary_path),
                "quality": summary["quality"],
                "runtime": summary["runtime"],
                "reliability": summary[
                    "reliability"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--ws-url",
        default="ws://localhost:8000/ws/chat",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Evaluate the first K results inside each "
            "retrieval group/sub-question."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable resume mode. If output exists, delete it and "
            "restart"
        ),
    )

    parser.add_argument(
        "--no-replay-history",
        action="store_true",
        help=(
            "Disable seeding inputs.conversation_turns into Redis before "
            "the latest query."
        ),
    )

    parser.add_argument(
        "--allow-missing-eval-telemetry",
        action="store_true",
        help=(
            "Allow runs against an application that was not started with "
            "ENABLE_EVAL_TELEMETRY=true. Retrieval metrics will then be "
            "unavailable, so this is not recommended for benchmark runs."
        ),
    )

    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit(
            "--top-k must be >= 1"
        )

    asyncio.run(
        run_eval(
            dataset_path=Path(args.dataset),
            output_path=Path(args.out),
            ws_url=args.ws_url,
            top_k=args.top_k,
            limit=args.limit,
            timeout_sec=args.timeout_sec,
            resume=not args.no_resume,
            replay_history=(
                not args.no_replay_history
            ),
            require_eval_telemetry=(
                not args.allow_missing_eval_telemetry
            ),
        )
    )


if __name__ == "__main__":
    main()
