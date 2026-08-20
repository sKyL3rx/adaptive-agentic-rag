from domain.state import RequestState

class BudgetExhaustedError(RuntimeError):
    pass

def reserve_model_call(
    state: RequestState,
    *,
    stage: str,
) -> RequestState:
    if state.model_calls_used >= state.budget.max_model_calls:
        raise BudgetExhaustedError(
            f"Model-call budget exhausted before stage={stage}"
        )

    return state.model_copy(
        update = {
            "model_calls_used":
                state.model_calls_used + 1,
        }
    )

def reserve_retrieval_round(
    state: RequestState,
) -> RequestState:
    if (
        state.retrieval_rounds_used
        >= state.budget.max_retrieval_rounds
    ):
        raise BudgetExhaustedError(
            "Retrieval-round budget exhausted"
        )

    return state.model_copy(
        update={
            "retrieval_rounds_used":
                state.retrieval_rounds_used + 1,
        }
    )

def remaining_model_calls(
    state: RequestState,
) -> int:
    return max(
        0,
        state.budget.max_model_calls
        - state.model_calls_used,
    )