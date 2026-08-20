from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient

from config import BaseAgentSettings, ModelRoleSettings
from prompts.answer import ANSWER_INSTRUCTIONS
from prompts.compile import COMPILER_INSTRUCTIONS
from prompts.context import CONTEXT_INSTRUCTIONS
from prompts.planning import PLANNING_INSTRUCTIONS
from prompts.review import REVIEW_INSTRUCTIONS


SUPPORTED_MODEL_PROVIDERS = frozenset({"openai", "runpod"})


def build_runpod_base_url(endpoint_id: str) -> str:
    """Convert a RunPod endpoint ID to an OpenAI-compatible base URL."""

    normalized_endpoint_id = endpoint_id.strip()

    if not normalized_endpoint_id:
        raise ValueError(
            "RunPod endpoint ID must not be empty"
        )

    return (
        "https://api.runpod.ai/v2/"
        f"{normalized_endpoint_id}/openai/v1"
    )


def validate_agent_settings(
    settings: BaseAgentSettings,
    *,
    role_name: str,
) -> None:
    """Fail when required provider configuration is missing."""

    provider = settings.model_provider.strip().lower()
    missing: list[str] = []

    if provider not in SUPPORTED_MODEL_PROVIDERS:
        raise ValueError(
            f"Unsupported model provider '{provider}' "
            f"for role '{role_name}'"
        )

    if not settings.openai_model.strip():
        missing.append(
            f"{settings.prefix}_MODEL"
        )

    if settings.openai_max_tokens <= 0:
        raise ValueError(
            f"{settings.prefix}_OPENAI_MAX_TOKENS "
            "must be greater than zero"
        )

    if not 0.0 <= settings.openai_temperature <= 2.0:
        raise ValueError(
            f"{settings.prefix}_OPENAI_TEMPERATURE "
            "must be between 0 and 2"
        )

    if not 0.0 < settings.openai_top_p <= 1.0:
        raise ValueError(
            f"{settings.prefix}_OPENAI_TOP_P "
            "must be greater than 0 and at most 1"
        )

    if provider == "openai":
        if not settings.openai_api_key.strip():
            missing.append("OPENAI_API_KEY")

    elif provider == "runpod":
        if not settings.runpod_api_key.strip():
            missing.append("RUNPOD_API_KEY")

        if not settings.openai_base_id.strip():
            missing.append(
                f"{settings.prefix}_MODEL_ID"
            )

    if missing:
        raise ValueError(
            f"Missing model configuration for role "
            f"'{role_name}': {', '.join(missing)}"
        )


def build_client(
    settings: BaseAgentSettings,
    *,
    role_name: str,
) -> OpenAIChatCompletionClient:
    """Create an OpenAI Chat Completions client for OpenAI or RunPod."""

    validate_agent_settings(
        settings,
        role_name=role_name,
    )

    provider = settings.model_provider.strip().lower()

    if provider == "openai":
        client_kwargs: dict[str, Any] = {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
        }

        if settings.openai_base_url.strip():
            client_kwargs["base_url"] = (
                settings.openai_base_url.strip()
            )

        return OpenAIChatCompletionClient(
            **client_kwargs
        )

    return OpenAIChatCompletionClient(
        model=settings.openai_model,
        api_key=settings.runpod_api_key,
        base_url=build_runpod_base_url(
            settings.openai_base_id
        ),
    )


def build_default_options(
    settings: BaseAgentSettings,
) -> dict[str, Any]:
    """Return default generation options for one workflow role."""

    options: dict[str, Any] = {
        "temperature": settings.openai_temperature,
        "max_tokens": settings.openai_max_tokens,
        "top_p": settings.openai_top_p,
    }
    
    provider = settings.model_provider.strip().lower()
    model = settings.openai_model.strip()

    if provider == "openai":
        options["presence_penalty"] = (
            settings.openai_presence_penalty
        )
        options["frequency_penalty"] = (
            settings.openai_frequency_penalty
        )

    if (
            provider == "runpod"
            and model.startswith("Qwen/Qwen3.6-")
        ):
            options["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            }

    return options


def build_agent(
    *,
    name: str,
    client: OpenAIChatCompletionClient,
    instructions: str,
    settings: BaseAgentSettings,
) -> Agent:

    return client.as_agent(
        name=name,
        instructions=instructions,
        default_options=build_default_options(
            settings
        ),
    )


@dataclass(frozen=True)
class ModelAgents:
    compiler: Agent
    context_rewriter: Agent
    planner: Agent
    answer_generator: Agent
    reviewer: Agent


def build_model_agents(
    settings: ModelRoleSettings,
) -> ModelAgents:
    
    compiler_client = build_client(
        settings.compiler,
        role_name="compiler",
    )
    context_client = build_client(
        settings.context,
        role_name="context_rewriter",
    )
    planner_client = build_client(
        settings.planner,
        role_name="planner",
    )
    answer_client = build_client(
        settings.answer,
        role_name="answer_generator",
    )
    reviewer_client = build_client(
        settings.reviewer,
        role_name="reviewer",
    )

    return ModelAgents(
        compiler=build_agent(
            name="RequestCompiler",
            client=compiler_client,
            instructions=COMPILER_INSTRUCTIONS,
            settings=settings.compiler,
        ),
        context_rewriter=build_agent(
            name="ContextResolver",
            client=context_client,
            instructions=CONTEXT_INSTRUCTIONS,
            settings=settings.context,
        ),
        planner=build_agent(
            name="RetrievalPlanner",
            client=planner_client,
            instructions=PLANNING_INSTRUCTIONS,
            settings=settings.planner,
        ),
        answer_generator=build_agent(
            name="GroundedAnswerGenerator",
            client=answer_client,
            instructions=ANSWER_INSTRUCTIONS,
            settings=settings.answer,
        ),
        reviewer=build_agent(
            name="GroundedAnswerReviewer",
            client=reviewer_client,
            instructions=REVIEW_INSTRUCTIONS,
            settings=settings.reviewer,
        ),
    )