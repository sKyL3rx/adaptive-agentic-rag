import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from domain.state import RequestState

T = TypeVar("T")

class RequestDeadlineExceededError(TimeoutError):
    def __init__(self, *, stage: str) -> None:
        self.stage = stage

        super().__init__(
            f"Request deadline reached before stage '{stage}'"
        )

class StageTimeoutError(TimeoutError):
    def __init__(
        self,
        *,
        stage: str,
        timeout_seconds: float,
    ) -> None:
        self.stage = stage
        self.timeout_seconds = timeout_seconds

        super().__init__(
            f"Stage '{stage}' exceeded "
            f"{timeout_seconds:.3f} seconds"
        )

    
async def run_stage(
    state: RequestState,
    *,
    stage: str,
    stage_timeout_seconds: float,
    operation: Callable[[], Awaitable[T]],
) -> T:
    remaining_seconds = state.remaining_seconds()

    if remaining_seconds <= 0:
        raise RequestDeadlineExceededError(
            stage=stage,
        )

    effective_timeout = min(
        remaining_seconds,
        stage_timeout_seconds,
    )

    try:
        async with asyncio.timeout(
            effective_timeout
        ):
            return await operation()

    except asyncio.CancelledError:
        raise

    except TimeoutError as exc:
        raise StageTimeoutError(
            stage=stage,
            timeout_seconds=effective_timeout,
        ) from exc