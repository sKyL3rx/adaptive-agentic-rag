from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_framework import Agent

from providers.retrieval_mcp import RetrievalMCPClient
from services.model_gateway import ModelGateway


class WorkflowModels(Protocol):
    compiler: Agent
    context_rewriter: Agent
    planner: Agent
    answer_generator: Agent
    reviewer: Agent



class WorkflowSettingsProtocol(Protocol):
    workflow_max_iterations: int
    
    # Compile
    compile_history_max_turns: int
    compile_history_max_chars: int
    compile_timeout_seconds: float

    # Context
    context_history_max_turns: int
    context_history_max_chars: int
    context_timeout_seconds: float

    # Planning
    plan_timeout_seconds: float

    # Retrieval
    retrieval_timeout_seconds: float
    retrieval_corpus_version: str

    single_top_k: int
    broad_top_k: int
    complex_top_k: int
    recovery_top_k: int

    retrieval_candidate_multiplier: int

    # Evidence
    broad_min_evidence_items: int

    # Answer/review
    answer_timeout_seconds: float
    verify_timeout_seconds: float

    answer_evidence_chars_per_item: int
    review_evidence_chars_per_item: int


@dataclass(frozen=True)
class WorkflowResources:
    models: WorkflowModels
    model_gateway: ModelGateway
    retrieval: RetrievalMCPClient
    settings: WorkflowSettingsProtocol