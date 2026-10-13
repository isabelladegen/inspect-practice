from inspect_ai.core import ModelOutput
from inspect_ai.solver import solver, TaskState, Generate


@solver
def replay():
    """Use the stored answer as the model output instead of generating."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(
            model=str(state.model), content=state.metadata["answer"]
        )
        return state
    return solve
