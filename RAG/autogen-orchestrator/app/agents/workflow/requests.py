from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from domain.state import (
    ExecutionBudget,
    HistoryTurn,
)

class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    request_id: str
    session_id: str
    user_query: str

    recent_history: tuple[
        HistoryTurn,
        ...
    ] = ()

    budget: ExecutionBudget
