from __future__ import annotations

import asyncio
from typing import TypeVar

from agent_framework import Agent
from pydantic import BaseModel

from domain.state import RequestState
from observability import (
    RAG_MODEL_CALLS_TOTAL,
)
from workflow.budget import reserve_model_call
from workflow.errors import (
    InvalidModelOutputError,
    ModelInvocationError,
)
from workflow.timeouts import (
    RequestDeadlineExceededError,
    StageTimeoutError,
    run_stage,
)

T = TypeVar("T", bound=BaseModel)

_MODEL_ROLE_BY_STAGE = {
    "compile": "compiler",
    "resolve_context": "context_rewriter",
    "plan": "planner",
    "plan_repair": "planner",
    "answer": "answer_generator",
    "answer_repair": "answer_generator",
    "verify": "reviewer",
}


class ModelGateway:
    async def run_structured(
        self,
        *,
        state: RequestState,
        stage: str,
        timeout_seconds: float,
        agent: Agent,
        prompt: str,
        response_model: type[T],
    ) -> tuple[RequestState, T]:
        charged_state = reserve_model_call(
            state,
            stage=stage,
        )
        
        role = _MODEL_ROLE_BY_STAGE.get(
            stage,
            "other",
        )

        RAG_MODEL_CALLS_TOTAL.labels(
            role=role
        ).inc()

        try:
            response = await run_stage(
                charged_state,
                stage=stage,
                stage_timeout_seconds=(
                    timeout_seconds
                ),
                operation=lambda: agent.run(
                    prompt,
                    options={
                        "response_format":
                            response_model,
                    },
                ),
            )

        except asyncio.CancelledError:
            raise

        except (
            StageTimeoutError,
            RequestDeadlineExceededError,
        ) as exc:
            raise ModelInvocationError(
                state=charged_state,
                stage=stage,
                error_type=type(exc).__name__,
                message=str(exc),
            ) from exc

        except Exception as exc:
            raise ModelInvocationError(
                state=charged_state,
                stage=stage,
                error_type=type(exc).__name__,
                message=(
                    f"Model invocation failed "
                    f"at stage '{stage}'"
                ),
            ) from exc

        value = response.value

        if not isinstance(value, response_model):
            raise InvalidModelOutputError(
                state=charged_state,
                stage=stage,
                expected=response_model.__name__,
            )

        return charged_state, value