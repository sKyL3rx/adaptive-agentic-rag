from __future__ import annotations

from domain.state import RequestState


class WorkflowApplicationError(RuntimeError):
    pass


class ModelStageError(WorkflowApplicationError):
    def __init__(
        self,
        *,
        state: RequestState,
        stage: str,
        error_type: str,
        message: str,
    ) -> None:
        self.state = state
        self.stage = stage
        self.error_type = error_type

        super().__init__(message)


class ModelInvocationError(ModelStageError):
    pass


class InvalidModelOutputError(ModelStageError):
    def __init__(
        self,
        *,
        state: RequestState,
        stage: str,
        expected: str,
    ) -> None:
        self.expected = expected

        super().__init__(
            state=state,
            stage=stage,
            error_type="INVALID_MODEL_OUTPUT",
            message=(
                f"Stage '{stage}' did not return "
                f"the expected '{expected}' value"
            ),
        )


class RetrievalInvocationError(
    WorkflowApplicationError
):
    pass
