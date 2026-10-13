from inspect_ai import Task, Epochs
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog

from src.scorers.score_reducers import single
from src.solvers.replay import replay


def replay_sample(id, input, target, answer: str) -> Sample:
    """A sample whose stored `answer` the replay solver will use as the model output."""
    return Sample(id=id, input=input, target=target, metadata={"answer": answer})


def rescore_task(samples: list[Sample], scorers: list, name: str,
                 metadata: dict | None = None) -> Task:
    """The rescoring task: replays the stored answers in `samples` and scores them with `scorers`."""
    return Task(
        name=name,
        dataset=MemoryDataset(samples),
        solver=replay(),
        scorer=scorers,
        epochs=Epochs(1, single()),
        metadata=metadata or {},
    )


def replay_task(log: EvalLog, scorers: list) -> Task:
    """Rescoring task for the answers stored in `log`."""
    samples = [replay_sample(s.id, s.input, s.target, s.output.completion) for s in log.samples]
    return rescore_task(
        samples, scorers, name=log.eval.task,
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