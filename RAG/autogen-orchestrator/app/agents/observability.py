from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sized
from typing import Any

from agent_framework.observability import enable_instrumentation
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


def _env_bool(
    name: str,
    default: bool = False,
) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized = raw_value.strip().casefold()

    if normalized in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    return default


def _env_text(
    name: str,
    default: str = "",
) -> str:
    return os.getenv(name, default).strip()


# =========================
# ENV / CONSTANTS
# =========================

OTEL_SERVICE_NAME = _env_text(
    "OTEL_SERVICE_NAME",
    "rag-orchestrator",
)

OTEL_SERVICE_VERSION = _env_text(
    "OTEL_SERVICE_VERSION",
    "2.0.0",
)

OTEL_DEPLOYMENT_ENVIRONMENT = _env_text(
    "OTEL_DEPLOYMENT_ENVIRONMENT",
    "",
)

OTEL_EXPORTER_OTLP_ENDPOINT = _env_text(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "",
)

OTEL_ENABLED = _env_bool(
    "OTEL_ENABLED",
    False,
)

OTEL_SDK_DISABLED = _env_bool(
    "OTEL_SDK_DISABLED",
    False,
)

OTEL_EXPORTER_OTLP_INSECURE = _env_bool(
    "OTEL_EXPORTER_OTLP_INSECURE",
    True,
)

ENABLE_INSTRUMENTATION = _env_bool(
    "ENABLE_INSTRUMENTATION",
    False,
)

ENABLE_SENSITIVE_DATA = _env_bool(
    "ENABLE_SENSITIVE_DATA",
    False,
)

ENABLE_MCP_DEBUG_LOGS = _env_bool(
    "ENABLE_MCP_DEBUG_LOGS",
    False,
)

_TRACER_PROVIDER: TracerProvider | None = None
_OWNS_TRACER_PROVIDER = False
_AGENT_FRAMEWORK_INSTRUMENTATION_ENABLED = False


# =========================
# OTEL PROVIDER
# =========================


def setup_otel() -> bool:
    global _TRACER_PROVIDER
    global _OWNS_TRACER_PROVIDER

    if _TRACER_PROVIDER is not None:
        logger.debug("otel_already_configured")
        return True

    if OTEL_SDK_DISABLED:
        logger.info(
            "otel_export_disabled "
            "reason=otel_sdk_disabled"
        )
        return False

    if not OTEL_ENABLED:
        logger.info(
            "otel_export_disabled "
            "reason=otel_enabled_false"
        )
        return False

    if not OTEL_EXPORTER_OTLP_ENDPOINT:
        logger.warning(
            "otel_export_disabled "
            "reason=missing_otlp_endpoint"
        )
        return False

    current_provider = trace.get_tracer_provider()

    if isinstance(current_provider, TracerProvider):
        _TRACER_PROVIDER = current_provider
        _OWNS_TRACER_PROVIDER = False
        logger.info(
            "otel_provider_reused "
            "provider_type=%s",
            type(current_provider).__name__,
        )
        return True

    resource_attributes: dict[str, str] = {
        "service.name": OTEL_SERVICE_NAME,
        "service.version": OTEL_SERVICE_VERSION,
    }

    if OTEL_DEPLOYMENT_ENVIRONMENT:
        resource_attributes[
            "deployment.environment.name"
        ] = OTEL_DEPLOYMENT_ENVIRONMENT

    resource = Resource.create(
        resource_attributes
    )

    provider = TracerProvider(
        resource=resource
    )

    exporter = OTLPSpanExporter(
        endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
        insecure=OTEL_EXPORTER_OTLP_INSECURE,
    )

    provider.add_span_processor(
        BatchSpanProcessor(exporter)
    )

    trace.set_tracer_provider(provider)
    _TRACER_PROVIDER = provider
    _OWNS_TRACER_PROVIDER = True

    logger.info(
        "otel_export_enabled "
        "service_name=%s "
        "service_version=%s "
        "endpoint=%s",
        OTEL_SERVICE_NAME,
        OTEL_SERVICE_VERSION,
        OTEL_EXPORTER_OTLP_ENDPOINT,
    )

    return True


def enable_agent_framework_observability() -> bool:
    global _AGENT_FRAMEWORK_INSTRUMENTATION_ENABLED

    if _AGENT_FRAMEWORK_INSTRUMENTATION_ENABLED:
        logger.debug(
            "agent_framework_instrumentation_"
            "already_enabled"
        )
        return True

    if not ENABLE_INSTRUMENTATION:
        logger.info(
            "agent_framework_instrumentation_"
            "disabled"
        )
        return False

    enable_instrumentation(
        enable_sensitive_data=(
            ENABLE_SENSITIVE_DATA
        )
    )

    _AGENT_FRAMEWORK_INSTRUMENTATION_ENABLED = True

    logger.info(
        "agent_framework_instrumentation_"
        "enabled sensitive_data=%s",
        ENABLE_SENSITIVE_DATA,
    )

    return True


def setup_observability() -> None:
    setup_otel()
    enable_agent_framework_observability()


def shutdown_otel() -> None:
    global _TRACER_PROVIDER
    global _OWNS_TRACER_PROVIDER

    if _TRACER_PROVIDER is None:
        return

    if not _OWNS_TRACER_PROVIDER:
        logger.info(
            "otel_shutdown_skipped "
            "reason=provider_owned_elsewhere"
        )
        _TRACER_PROVIDER = None
        return

    try:
        _TRACER_PROVIDER.force_flush(
            timeout_millis=5_000
        )
        _TRACER_PROVIDER.shutdown()
        logger.info("otel_shutdown_complete")

    except Exception:
        logger.exception("otel_shutdown_failed")

    finally:
        _TRACER_PROVIDER = None
        _OWNS_TRACER_PROVIDER = False

setup_observability()

tracer = trace.get_tracer(
    "rag-orchestrator",
    OTEL_SERVICE_VERSION,
)


# =========================
# TRACE HELPERS
# =========================


def _trace_id_hex() -> str:
    context = (
        trace.get_current_span()
        .get_span_context()
    )

    if context and context.is_valid:
        return f"{context.trace_id:032x}"

    return ""


def hash_user_id(
    user_id: str,
) -> str:
    return hashlib.sha256(
        user_id.encode("utf-8")
    ).hexdigest()[:16]


def _enum_or_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    enum_value = getattr(
        value,
        "value",
        None,
    )

    if enum_value is not None:
        return str(enum_value)

    return str(value)


def _safe_len(
    value: Any,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, Sized):
        return len(value)

    return None


def set_final_result_span_attributes(
    span: Span,
    result: Any,
) -> None:
    request_id = getattr(
        result,
        "request_id",
        None,
    )

    if request_id:
        span.set_attribute(
            "request.id",
            str(request_id),
        )

    model_calls_used = getattr(
        result,
        "model_calls_used",
        None,
    )

    if model_calls_used is not None:
        span.set_attribute(
            "model_calls.used",
            int(model_calls_used),
        )

    retrieval_rounds_used = getattr(
        result,
        "retrieval_rounds_used",
        None,
    )

    if retrieval_rounds_used is not None:
        span.set_attribute(
            "retrieval_rounds.used",
            int(retrieval_rounds_used),
        )

    termination_reason = _enum_or_text(
        getattr(
            result,
            "termination_reason",
            None,
        )
    )

    if termination_reason:
        span.set_attribute(
            "termination.reason",
            termination_reason,
        )

    citations_count = _safe_len(
        getattr(
            result,
            "citations",
            None,
        )
    )

    if citations_count is not None:
        span.set_attribute(
            "citations.count",
            citations_count,
        )

    route = _enum_or_text(
        getattr(result, "route", None)
    )

    if route:
        span.set_attribute(
            "workflow.route",
            route,
        )

    shape = _enum_or_text(
        getattr(result, "shape", None)
    )

    if shape:
        span.set_attribute(
            "question.shape",
            shape,
        )

    sub_questions_count = getattr(
        result,
        "sub_questions_count",
        None,
    )

    if sub_questions_count is not None:
        span.set_attribute(
            "sub_questions.count",
            int(sub_questions_count),
        )

    evidence_count = getattr(
        result,
        "evidence_count",
        None,
    )

    if evidence_count is not None:
        span.set_attribute(
            "evidence.count",
            int(evidence_count),
        )


# =========================
# OPTIONAL DEBUG LOGGING
# =========================


def enable_mcp_debug_logs() -> None:
    formatter = logging.Formatter(
        (
            "%(asctime)s %(levelname)s "
            "[%(name)s] %(message)s"
        )
    )


    targets = [
        "agent_framework",
        "mcp",
        "mcp.client",
        "mcp.client.sse",
        "httpx",
        "httpcore",
    ]

    for name in targets:
        target_logger = logging.getLogger(
            name
        )
        target_logger.setLevel(logging.DEBUG)

        already_installed = any(
            getattr(
                handler,
                "_rag_mcp_debug_handler",
                False,
            )
            for handler in target_logger.handlers
        )

        if not already_installed:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)

            setattr(
                handler,
                "_rag_mcp_debug_handler",
                True,
            )

            target_logger.addHandler(handler)

        target_logger.propagate = False


if ENABLE_MCP_DEBUG_LOGS:
    enable_mcp_debug_logs()


# =========================
# PROMETHEUS METRICS
# =========================

RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total number of accepted RAG requests",
)

RAG_INFLIGHT_REQUESTS = Gauge(
    "rag_inflight_requests",
    (
        "Requests currently executing "
        "an orchestration run"
    ),
)

RAG_PENDING_REQUESTS = Gauge(
    "rag_pending_requests",
    (
        "Requests waiting for either the "
        "session lock or global request permit"
    ),
)

RAG_E2E_LATENCY_SECONDS = Histogram(
    "rag_e2e_latency_seconds",
    (
        "E2E time from server receiving the question "
        "to client ACK after rendering final answer"
    ),
    ["status"],
    buckets=(
        0.5,
        1,
        2,
        3,
        5,
        8,
        12,
        20,
        30,
        60,
    ),
)

RAG_E2E_ACKS_TOTAL = Counter(
    "rag_e2e_acks_total",
    "Number of client ACKs for E2E measurement",
    ["status"],
)

RAG_TIME_TO_FINAL_READY_SECONDS = Histogram(
    "rag_time_to_final_ready_seconds",
    (
        "Time from receiving the user request "
        "until the final answer is sent"
    ),
    ["status"],
    buckets=(
        0.5,
        1,
        2,
        3,
        5,
        8,
        12,
        20,
        30,
        45,
        60,
        90,
        120,
    ),
)

RAG_CLIENT_ACK_WAIT_SECONDS = Histogram(
    "rag_client_ack_wait_seconds",
    (
        "Time spent waiting for the client "
        "to acknowledge the final answer"
    ),
    ["status"],
    buckets=(
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        4,
        8,
    ),
)

RAG_REQUEST_DURATION_SECONDS = Histogram(
    "rag_request_duration_seconds",
    (
        "Server-owned request duration from request "
        "acceptance until final/error/cancellation"
    ),
    buckets=(
        0.1,
        0.25,
        0.5,
        1,
        2,
        3,
        5,
        8,
        12,
        20,
        30,
        45,
        60,
        90,
        120,
    ),
)


RAG_TTFT_SECONDS = Histogram(
    "rag_ttft_seconds",
    (
        "Time from request acceptance until the first "
        "user-visible answer content is sent"
    ),
    buckets=(
        0.1,
        0.25,
        0.5,
        1,
        2,
        3,
        5,
        8,
        12,
        20,
        30,
        45,
        60,
    ),
)


RAG_TIME_TO_FINAL_SECONDS = Histogram(
    "rag_time_to_final_seconds",
    (
        "Time from request acceptance until the final "
        "answer is sent"
    ),
    buckets=(
        0.1,
        0.25,
        0.5,
        1,
        2,
        3,
        5,
        8,
        12,
        20,
        30,
        45,
        60,
        90,
        120,
    ),
)


RAG_WEBSOCKET_DISCONNECTS_TOTAL = Counter(
    "rag_websocket_disconnects_total",
    "Total WebSocket connection disconnects",
)


RAG_REQUEST_CANCELLATIONS_TOTAL = Counter(
    "rag_request_cancellations_total",
    "Total orchestration requests cancelled before completion",
)

RAG_ROUTE_TOTAL = Counter(
    "rag_route_total",
    "Requests by resolved execution route",
    ["route"],
)

RAG_MODEL_CALLS_TOTAL = Counter(
    "rag_model_calls_total",
    "Attempted model invocations by logical model role",
    ["role"],
)


RAG_RETRIEVAL_ROUNDS_TOTAL = Counter(
    "rag_retrieval_rounds_total",
    "Total retrieval rounds executed",
)


RAG_REVIEWER_ACTIVATION_TOTAL = Counter(
    "rag_reviewer_activation_total",
    "Reviewer activations by bounded activation reason",
    ["reason"],
)


RAG_TERMINATION_TOTAL = Counter(
    "rag_termination_total",
    "Requests by bounded workflow termination reason",
    ["reason"],
)

