from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog

from solvers.replay import replay


def replay_task(log: EvalLog, scorers: list) -> Task:
    """A task that replays the answers in `log` and scores them with `scorers`."""
    dataset = MemoryDataset([
        Sample(id=s.id, input=s.input, target=s.target,
               metadata={"answer": s.output.completion})
        for s in log.samples
    ])
    return Task(
        name=log.eval.task,
        dataset=dataset,
        solver=replay(),
        scorer=scorers,
        metadata={
            "kind": "rescore",
            "rescores": {
                "log": log.location,
                "run_id": log.eval.run_id,
                "eval_id": log.eval.eval_id,
                "task_args": log.eval.task_args,
            },
        },
    )
