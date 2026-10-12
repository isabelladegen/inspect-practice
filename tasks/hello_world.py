from inspect_ai import task, Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.solver import generate


@task
def hello_world():
    return Task(
        dataset=[
            Sample(
                input="I'm running a test evaluation. I need you to reply with exactly the words after the colon and nothing else: Hello World!",
                target="Hello World!",
            )
        ],
        solver=[generate()],
        scorer=exact(),
    )