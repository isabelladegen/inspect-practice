import random
from string import ascii_uppercase

from inspect_ai import Task, Epochs
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog, EvalSample

from src.scorers.score_reducers import single
from src.solvers.replay import replay


def stored_answer(sample: EvalSample) -> str:
    """The answer the model gave in a logged sample."""
    return sample.output.completion


def source_metadata(log: EvalLog) -> dict:
    """What identifies the run a log came from."""
    return {
        "log": log.location,
        "run_id": log.eval.run_id,
        "eval_id": log.eval.eval_id,
        "task_args": log.eval.task_args,
    }


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
    samples = [replay_sample(s.id, s.input, s.target, stored_answer(s)) for s in log.samples]
    return rescore_task(
        samples, scorers, name=log.eval.task,
        metadata={
            "kind": "rescore",
            "rescores": source_metadata(log),
        },
    )


def answers_by_item(logs: list[EvalLog], seed: int) -> list[Sample]:
    """One sample per item holding every model's answer, under letters in a seeded
    random order per item. The letter-to-model mapping is kept in the metadata."""
    by_model = {}
    for log in logs:
        model = log.eval.model
        if model in by_model:
            raise ValueError(f"More than one log for {model}")
        by_model[model] = {s.id: s for s in log.samples}

    models = sorted(by_model)
    ids = sorted(set().union(*(s.keys() for s in by_model.values())))
    return [_item_sample(id, by_model, models, seed) for id in ids]


def _item_sample(id, by_model: dict, models: list[str], seed: int) -> Sample:
    missing = [m for m in models if id not in by_model[m]]
    if missing:
        raise ValueError(f"Item {id} has no answer from {missing}")
    samples = [by_model[m][id] for m in models]
    first = samples[0]
    if any(s.input != first.input or s.target != first.target for s in samples):
        raise ValueError(f"Item {id} has differing questions or keys across models")

    order = list(models)
    random.Random(f"{seed}:{id}").shuffle(order)  # per item, independent of item order
    letters = ascii_uppercase[:len(order)]
    return Sample(
        id=id, input=first.input, target=first.target,
        metadata={
            "answers": {l: stored_answer(by_model[m][id]) for l, m in zip(letters, order)},
            "models": dict(zip(letters, order)),
            "order_seed": seed,
        },
    )
