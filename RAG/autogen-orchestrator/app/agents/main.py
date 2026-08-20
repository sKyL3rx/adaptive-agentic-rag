from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import redis.asyncio as redis
from cachetools import TTLCache
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from opentelemetry.trace import Status, StatusCode
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from config import load_model_role_settings, load_workflow_settings
from domain.state import (
    ExecutionBudget,
    HistoryTurn,
    TerminationReason,
)
from history_store import append_history, get_recent_history
from observability import (
    RAG_CLIENT_ACK_WAIT_SECONDS,
    RAG_E2E_ACKS_TOTAL,
    RAG_E2E_LATENCY_SECONDS,
    RAG_REQUEST_CANCELLATIONS_TOTAL,
    RAG_REQUEST_DURATION_SECONDS,
    RAG_REQUESTS_TOTAL,
    RAG_TERMINATION_TOTAL,
    RAG_TIME_TO_FINAL_READY_SECONDS,
    RAG_TIME_TO_FINAL_SECONDS,
    RAG_TTFT_SECONDS,
    RAG_WEBSOCKET_DISCONNECTS_TOTAL,
    _trace_id_hex,
    hash_user_id,
    shutdown_otel,
    tracer,
)

from providers.model_clients import build_model_agents
from providers.retrieval_mcp import RetrievalMCPClient
from safety import run_input_guardrails
from services.model_gateway import ModelGateway
from transport.websocket import (
    ChatEventStream,
    send_chat_event,
)
from workflow.engine import AgentFrameworkEngine
from workflow.events import ErrorEvent, FinalEvent
from workflow.requests import ChatRequest
from workflow.resources import WorkflowResources
from workflow.lifecycle import (
    RequestExecutionLifecycle,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
os.makedirs(LOG_DIR, exist_ok=True)

_LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _configure_logging() -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    if not any(getattr(h, "_rag_main_handler", False) for h in root_logger.handlers):
        formatter = logging.Formatter(_LOG_FORMAT)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(formatter)
        console_handler._rag_main_handler = True  

        file_handler = RotatingFileHandler(
            _LOG_PATH,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        file_handler._rag_main_handler = True  

        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    configured = logging.getLogger(__name__)
    configured.info("Logging initialized. Writing to %s", _LOG_PATH)
    return configured


logger = _configure_logging()


# ---------------------------------------------------------------------------
# API-layer settings
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ACK_TIMEOUT_SECONDS = float(os.getenv("ACK_TIMEOUT_SECONDS", "8"))
UI_FILE = os.getenv("UI_FILE", str(Path(__file__).with_name("ui.html")))
ENABLE_INPUT_GUARDRAILS = os.getenv("ENABLE_INPUT_GUARDRAILS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

SESSION_LOCK_CACHE_MAXSIZE = int(os.getenv("SESSION_LOCK_CACHE_MAXSIZE", "1000"))
SESSION_LOCK_CACHE_TTL_SECONDS = int(os.getenv("SESSION_LOCK_CACHE_TTL_SECONDS", "600"))

SYSTEM_BLOCK_MESSAGE = os.getenv(
    "SYSTEM_BLOCK_MESSAGE",
    "I cannot help with that request. Please ask a safe, DMV-related question.",
)
PUBLIC_WORKFLOW_ERROR_MESSAGE = os.getenv(
    "PUBLIC_WORKFLOW_ERROR_MESSAGE",
    "I could not complete the request reliably. Please try again.",
)

MAX_CONCURRENT_REQUESTS = max(
    1,
    int(
        os.getenv(
            "MAX_CONCURRENT_REQUESTS",
            "4",
        )
    ),
)


def _split_csv_env(name: str, default: str) -> list[str]:
    values = [part.strip() for part in os.getenv(name, default).split(",") if part.strip()]
    return values or ["*"]


CORS_ALLOW_ORIGINS = _split_csv_env("CORS_ALLOW_ORIGINS", "*")
CORS_ALLOW_METHODS = _split_csv_env("CORS_ALLOW_METHODS", "GET,POST,OPTIONS")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_request_id() -> str:
    return uuid.uuid4().hex

def _failure_termination_reason(
    error_type: str,
) -> TerminationReason:
    normalized = (
        error_type
        or ""
    ).strip().upper()

    if normalized in {
        "TIMEOUT",
        "TIMEOUTERROR",
        "STAGETIMEOUTERROR",
        "REQUESTDEADLINEEXCEEDEDERROR",
    }:
        return TerminationReason.TIMEOUT

    if normalized in {
        "BUDGETEXHAUSTEDERROR",
        "BUDGET_EXHAUSTED",
    }:
        return (
            TerminationReason
            .BUDGET_EXHAUSTED
        )

    if normalized in {
        "INVALIDMODELOUTPUTERROR",
        "INVALID_MODEL_OUTPUT",
    }:
        return (
            TerminationReason
            .INVALID_MODEL_OUTPUT
        )

    if normalized in {
        "MODELINVOCATIONERROR",
        "MODEL_ERROR",
    }:
        return TerminationReason.MODEL_ERROR

    if normalized in {
        "RETRIEVALINVOCATIONERROR",
        "RETRIEVAL_ERROR",
    }:
        return (
            TerminationReason
            .RETRIEVAL_ERROR
        )

    return TerminationReason.INTERNAL_ERROR

def _to_history_turns(items: list[dict[str, Any]]) -> tuple[HistoryTurn, ...]:
    turns: list[HistoryTurn] = []

    for item in items:
        question = str(item.get("q", "")).strip()
        answer = str(item.get("a", "")).strip()

        if question:
            turns.append(HistoryTurn(role="user", content=question))
        if answer:
            turns.append(HistoryTurn(role="assistant", content=answer))

    return tuple(turns)


async def _get_session_lock(app: FastAPI, session_key: str) -> asyncio.Lock:
    async with app.state.session_locks_guard:
        lock = app.state.session_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            app.state.session_locks[session_key] = lock
        return lock


async def _wait_for_final_ack(websocket: WebSocket, req_id: str) -> tuple[str, float]:
    try:
        message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=ACK_TIMEOUT_SECONDS,
        )
        valid = (
            isinstance(
                message,
                dict,
            )
            and message.get("type")
            == "final_ack"
            and message.get("request_id")
            == req_id
        )
        status = "success" if valid else "error"
    except WebSocketDisconnect:
        raise
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        status = "timeout"
    except Exception:
        status = "error"

    return status, time.perf_counter()


def _record_completed_request_metrics(
    *,
    request_started_at: float,
    final_ready_at: float,
    ack_finished_at: float,
    ack_status: str,
) -> None:
    RAG_E2E_ACKS_TOTAL.labels(
        status=ack_status
    ).inc()

    RAG_CLIENT_ACK_WAIT_SECONDS.labels(
        status=ack_status
    ).observe(
        ack_finished_at - final_ready_at
    )

    RAG_E2E_LATENCY_SECONDS.labels(
        status=ack_status
    ).observe(
        ack_finished_at - request_started_at
    )


def _record_failed_request_metrics(
    request_started_at: float,
    status: str = "error",
) -> None:
    RAG_E2E_LATENCY_SECONDS.labels(
        status=status
    ).observe(
        time.perf_counter()
        - request_started_at
    )


async def _deliver_guardrail_block(
    *,
    websocket: WebSocket,
    event_stream: ChatEventStream,
    redis_client: redis.Redis,
    session_key: str,
    user_query: str,
    req_id: str,
    trace_id: str,
    request_started_at: float,
) -> None:
    
    try:
        await append_history(
            redis_client,
            session_key,
            user_query,
            SYSTEM_BLOCK_MESSAGE,
        )
    except Exception:
        logger.exception(
            "Failed to persist guardrail response trace_id=%s req_id=%s",
            trace_id,
            req_id,
        )

    await event_stream.send(
        "completed",
        {
            "answer": SYSTEM_BLOCK_MESSAGE,
            "termination_reason":
                "GUARDRAIL_BLOCK",
            "blocked": True,
            "model_calls_used": 0,
            "retrieval_rounds_used": 0,
            "insufficient_evidence": False,
            "budget_exhausted": False,
        },
    )

    final_ready_at = (
        time.perf_counter()
    )

    server_duration = (
        final_ready_at
        - request_started_at
    )

    RAG_TIME_TO_FINAL_READY_SECONDS.labels(
        status="success"
    ).observe(
        server_duration
    )

    RAG_REQUEST_DURATION_SECONDS.observe(
        server_duration
    )

    RAG_TIME_TO_FINAL_SECONDS.observe(
        server_duration
    )

    RAG_TTFT_SECONDS.observe(
        server_duration
    )

    RAG_TERMINATION_TOTAL.labels(
        reason="GUARDRAIL_BLOCK"
    ).inc()

    ack_status, ack_finished_at = (
        await _wait_for_final_ack(
            websocket,
            req_id,
        )
    )

    _record_completed_request_metrics(
        request_started_at=request_started_at,
        final_ready_at=final_ready_at,
        ack_finished_at=ack_finished_at,
        ack_status=ack_status,
    )

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = redis.from_url(REDIS_URL)
    retrieval_client: RetrievalMCPClient | None = None

    app.state.redis = redis_client
    app.state.session_locks = TTLCache(
        maxsize=SESSION_LOCK_CACHE_MAXSIZE,
        ttl=SESSION_LOCK_CACHE_TTL_SECONDS,
    )
    app.state.session_locks_guard = asyncio.Lock()
    app.state.request_semaphore = (
        asyncio.Semaphore(
            MAX_CONCURRENT_REQUESTS
        )
    )

    try:

        workflow_settings = load_workflow_settings()
        model_role_settings = load_model_role_settings()
        models = build_model_agents(model_role_settings)

        retrieval_client = RetrievalMCPClient(
            url=workflow_settings.mcp_url,
            tool_name=workflow_settings.mcp_tool_name,
            timeout_seconds=max(1, int(workflow_settings.retrieval_timeout_seconds)),
        )
        await retrieval_client.connect()

        resources = WorkflowResources(
            models=models,
            model_gateway=ModelGateway(),
            retrieval=retrieval_client,
            settings=workflow_settings,
        )

        app.state.workflow_settings = workflow_settings
        app.state.orchestration_engine = AgentFrameworkEngine(resources)

        logger.info("Application resources initialized")
        yield
    finally:
        if retrieval_client is not None:
            try:
                await retrieval_client.close()
            except Exception:
                logger.exception("Failed to close MCP client")

        try:
            await redis_client.aclose()
        except Exception:
            logger.exception("Failed to close Redis client")

        try:
            shutdown_otel()
        except Exception:
            logger.exception("Failed to shut down OpenTelemetry")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=["*"],
    allow_credentials=False,
)


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(UI_FILE)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get(
    "/livez",
    include_in_schema=False,
)
async def livez() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.get(
    "/readyz",
    include_in_schema=False,
)
async def readyz() -> dict[str, str]:
    if not hasattr(
        app.state,
        "orchestration_engine",
    ):
        raise HTTPException(
            status_code=503,
            detail="not ready",
        )

    return {
        "status": "ready",
    }
    

async def _process_chat_request(websocket: WebSocket, data: dict[str, Any]) -> None:
    client_session_id = str(
        data.get("session_id") or ""
    ).strip()
    legacy_user_id = str(
        data.get("user_id") or ""
    ).strip()
    user_query = str(
        data.get("content") or ""
    ).strip()

    if not user_query:
        return

    RAG_REQUESTS_TOTAL.inc()

    request_started_at = time.perf_counter()
    req_id = _new_request_id()
    session_key = (
        client_session_id
        or legacy_user_id
        or f"anon:{req_id}"
    )

    with tracer.start_as_current_span("ws.ask") as span:
        span.set_attribute("request.kind", "websocket")
        span.set_attribute("request.id", req_id)
        span.set_attribute(
            "session.id_hash",
            hash_user_id(session_key),
        )
        span.set_attribute("question.length", len(user_query))

        otel_trace_id = _trace_id_hex()
        trace_id = otel_trace_id or f"local-{req_id}"
        span.set_attribute("trace.id", trace_id)
        span.set_attribute("otel.trace_available", bool(otel_trace_id))
        
        event_stream = ChatEventStream(
            websocket=websocket,
            request_id=req_id,
        )

        logger.info(
            "ws.ask start trace_id=%s req_id=%s session_key=%s question=%s",
            trace_id,
            req_id,
            session_key,
            user_query[:500],
        )

        await event_stream.send(
            "request_accepted",
            {
                "trace_id": trace_id,
                "session_id": session_key,
            },
        )

        try:
            if ENABLE_INPUT_GUARDRAILS:
                guard_decision = await run_input_guardrails(user_query)
                span.set_attribute("guardrail.enabled", True)
                span.set_attribute("guardrail.blocked", guard_decision.blocked)
                span.set_attribute("guardrail.flow", guard_decision.flow)

                if guard_decision.blocked:
                    await _deliver_guardrail_block(
                        websocket=websocket,
                        event_stream=event_stream,
                        redis_client=(
                            websocket.app.state.redis
                        ),
                        session_key=session_key,
                        user_query=user_query,
                        req_id=req_id,
                        trace_id=trace_id,
                        request_started_at=request_started_at,
                    )
                    span.set_attribute("termination.reason", "GUARDRAIL_BLOCK")
                    span.set_status(Status(StatusCode.OK))
                    return
            else:
                span.set_attribute("guardrail.enabled", False)

            workflow_settings = websocket.app.state.workflow_settings
            history_pairs_to_load = max(
                1,
                workflow_settings.compile_history_max_turns,
                workflow_settings.context_history_max_turns,
            )

            try:
                recent_history = await get_recent_history(
                    websocket.app.state.redis,
                    session_key,
                    last_n=history_pairs_to_load,
                )
            except Exception:
                logger.exception(
                    "Failed to load recent history trace_id=%s req_id=%s",
                    trace_id,
                    req_id,
                )
                recent_history = []

            history_turns = _to_history_turns(recent_history)
            span.set_attribute("history.turns", len(history_turns))

            request = ChatRequest(
                request_id=req_id,
                session_id=session_key,
                user_query=user_query,
                recent_history=history_turns,
                budget=ExecutionBudget(
                    max_model_calls=workflow_settings.max_model_calls,
                    max_retrieval_rounds=workflow_settings.max_retrieval_rounds,
                    max_sub_questions=workflow_settings.max_sub_questions,
                    deadline_ms=workflow_settings.workflow_deadline_ms,
                ),
            )

            engine: AgentFrameworkEngine = websocket.app.state.orchestration_engine
            session_lock = await _get_session_lock(websocket.app, session_key)

            final_event: FinalEvent | None = None
            workflow_error: ErrorEvent | None = None
            final_ready_at: float | None = None
            

            async with RequestExecutionLifecycle(
                session_lock=session_lock,
                request_semaphore=(
                    websocket.app.state
                    .request_semaphore
                ),
            ):
                async for event in engine.stream(
                    request
                ):
                    if isinstance(
                        event,
                        ErrorEvent,
                    ):
                        workflow_error = event
                        continue

                    await send_chat_event(
                        event_stream,
                        event,
                    )

                    if isinstance(
                        event,
                        FinalEvent,
                    ):
                        final_event = event
                        final_ready_at = (
                            time.perf_counter()
                        )
                        continue

            if final_event is None:
                error_type = (
                    workflow_error.error_type
                    if workflow_error
                    else "NO_FINAL_OUTPUT"
                )
                
                termination_reason = (
                    _failure_termination_reason(
                        error_type
                    )
                )

                internal_message = (
                    workflow_error.message
                    if workflow_error
                    else (
                        "Workflow completed without "
                        "a typed final output."
                    )
                )

                logger.error(
                    "ws.ask workflow_error "
                    "trace_id=%s req_id=%s "
                    "error_type=%s message=%s",
                    trace_id,
                    req_id,
                    error_type,
                    internal_message,
                )

                RAG_REQUEST_DURATION_SECONDS.observe(
                    time.perf_counter()
                    - request_started_at
                )

                _record_failed_request_metrics(
                    request_started_at
                )

                await event_stream.send(
                    "error",
                    {
                        "message":
                            PUBLIC_WORKFLOW_ERROR_MESSAGE,
                    },
                )
                span.set_attribute(
                    "termination.reason",
                    termination_reason.value,
                )

                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        internal_message,
                    )
                )
                
                RAG_TERMINATION_TOTAL.labels(
                    reason=termination_reason.value
                ).inc()

                return

            result = final_event.result

            final_ready_at = (
                final_ready_at
                or time.perf_counter()
            )

            server_duration = (
                final_ready_at
                - request_started_at
            )

            RAG_TIME_TO_FINAL_READY_SECONDS.labels(
                status="success"
            ).observe(
                server_duration
            )

            RAG_REQUEST_DURATION_SECONDS.observe(
                server_duration
            )

            RAG_TIME_TO_FINAL_SECONDS.observe(
                server_duration
            )

            RAG_TTFT_SECONDS.observe(
                server_duration
            )
            
            RAG_TERMINATION_TOTAL.labels(
                reason=(
                    result
                    .termination_reason
                    .value
                )
            ).inc()

            if (
                result.termination_reason 
                == TerminationReason.CONVERSATION_RECALL
            ):
                logger.info(
                    "conversation_memory_persist_skipped "
                    "trace_id=%s "
                    "req_id=%s "
                    "reason=%s",
                    trace_id,
                    req_id,
                    result.termination_reason.value,
                )
            else:
                try:
                    await append_history(
                        websocket.app.state.redis,
                        session_key,
                        user_query,
                        result.answer,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist conversation history "
                        "trace_id=%s req_id=%s",
                        trace_id,
                        req_id,
                    )


            span.set_attribute("termination.reason", result.termination_reason.value)
            span.set_attribute("model_calls.used", result.model_calls_used)
            span.set_attribute("retrieval_rounds.used", result.retrieval_rounds_used)
            span.set_attribute("citations.count", len(result.citations))

            logger.info(
                "ws.ask final trace_id=%s req_id=%s termination=%s "
                "model_calls=%d retrieval_rounds=%d",
                trace_id,
                req_id,
                result.termination_reason.value,
                result.model_calls_used,
                result.retrieval_rounds_used,
            )

            ack_status, ack_finished_at = await _wait_for_final_ack(websocket, req_id)
            _record_completed_request_metrics(
                request_started_at=request_started_at,
                final_ready_at=final_ready_at,
                ack_finished_at=ack_finished_at,
                ack_status=ack_status,
            )

            span.set_attribute(
                "latency.ms",
                int((time.perf_counter() - request_started_at) * 1000),
            )
            span.set_status(Status(StatusCode.OK))

        except WebSocketDisconnect:
            _record_failed_request_metrics(request_started_at, status="cancelled")
            logger.info("client_disconnected trace_id=%s req_id=%s", trace_id, req_id)
            raise
        except asyncio.CancelledError:
            RAG_REQUEST_CANCELLATIONS_TOTAL.inc()

            RAG_TERMINATION_TOTAL.labels(
                reason=(
                    TerminationReason
                    .CANCELLED
                    .value
                )
            ).inc()

            RAG_REQUEST_DURATION_SECONDS.observe(
                time.perf_counter()
                - request_started_at
            )

            _record_failed_request_metrics(
                request_started_at,
                status="cancelled",
            )

            logger.info(
                "request_cancelled "
                "trace_id=%s req_id=%s",
                trace_id,
                req_id,
            )

            raise
        except Exception as exc:
            RAG_REQUEST_DURATION_SECONDS.observe(
                time.perf_counter()
                - request_started_at
            )

            RAG_TERMINATION_TOTAL.labels(
                reason=(
                    TerminationReason
                    .INTERNAL_ERROR
                    .value
                )
            ).inc()

            _record_failed_request_metrics(
                request_started_at
            )

            span.record_exception(exc)

            span.set_attribute(
                "termination.reason",
                type(exc).__name__,
            )

            span.set_status(
                Status(
                    StatusCode.ERROR,
                    str(exc),
                )
            )

            logger.exception(
                "ws.ask error "
                "trace_id=%s req_id=%s",
                trace_id,
                req_id,
            )

            try:
                await event_stream.send(
                    "error",
                    {
                        "message":
                            PUBLIC_WORKFLOW_ERROR_MESSAGE,
                    },
                )
            except WebSocketDisconnect:
                raise

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()

            if not isinstance(payload, dict):
                continue

            if payload.get("type") == "final_ack":
                continue

            await _process_chat_request(websocket, payload)
    except WebSocketDisconnect:
        RAG_WEBSOCKET_DISCONNECTS_TOTAL.inc()

        logger.info(
            "WebSocket disconnected"
        )
    except asyncio.CancelledError:
        logger.info("WebSocket handler cancelled")
        raise
    except Exception:
        logger.exception(
            "ws_chat error"
        )

        try:
            await websocket.close(
                code=1011,
            )
        except Exception:
            pass