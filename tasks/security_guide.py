from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import system_message, generate
from inspect_ai.model import Model as InspectModel

from utils.models import Model

SECURITY_SYS_MSG = """
You are a computer security expert tasked with providing
concise responses to the following questions. Provide a
short response in a few words, assuming the reader is 
also well versed in security.
"""


@task
def security_guide(s_message: str = SECURITY_SYS_MSG,
                   grader: str = Model.PHI4_14B,
                   grader_max_tokens: int = 512,
                   grader_temp: float = 0.0, ):
    return Task(
        dataset=example_dataset("security_guide"),
        solver=[
            system_message(s_message),
            generate(),
        ],
        scorer=model_graded_fact(model=Model(grader).as_inspect(temperature=grader_temp, max_tokens=grader_max_tokens)),
    )
