import random

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
    item_ids = sorted(set().union(*(s.keys() for s in by_model.values())))
    return [_item_sample(item_id, by_model, models, seed) for item_id in item_ids]


def _item_sample(item_id, by_model: dict, models: list[str], seed: int) -> Sample:
    models_without_answer = [model for model in models if item_id not in by_model[model]]
    if models_without_answer:
        raise ValueError(f"Item {item_id} has no answer from models {models_without_answer}")

    samples = {model: by_model[model][item_id] for model in models}
    first = samples[models[0]]
    if any(s.input != first.input or s.target != first.target for s in samples.values()):
        raise ValueError(f"Item {item_id} has differing questions or keys across models")

    answers = {model: stored_answer(sample) for model, sample in samples.items()}
    models_with_empty_answer = [model for model, answer in answers.items() if not answer.strip()]
    if models_with_empty_answer:
        raise ValueError(f"Item {item_id} has an empty answer from models {models_with_empty_answer}")

    order = list(models)
    random.Random(f"{seed}:{item_id}").shuffle(order)  # per item, independent of item order
    labels = [f"A{i}" for i in range(1, len(order) + 1)]
    return Sample(
        id=item_id, input=first.input, target=first.target,
        metadata={
            "answers": {label: answers[model] for label, model in zip(labels, order)},
            "models": dict(zip(labels, order)),
            "order_seed": seed,
        },
    )
