import random
from dataclasses import dataclass

from inspect_ai import Epochs, Task
from inspect_ai.core import ChatMessage
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.log import EvalLog

from src.scorers.score_reducers import single
from src.solvers.replay import replay
from src.utils.inspect_logs import logged_samples


@dataclass(frozen=True)
class StoredAnswer:
    """What a model answered to one item."""
    item_id: int | str
    question: str | list[ChatMessage]
    key: str | list[str]
    answer: str


def stored_answers(log: EvalLog) -> list[StoredAnswer]:
    """The answers stored in `log`; raises if it has no samples."""
    return [StoredAnswer(s.id, s.input, s.target, s.output.completion) for s in logged_samples(log)]


def source_metadata(log: EvalLog) -> dict:
    """What identifies the run a log came from."""
    return {
        "log": log.location,
        "run_id": log.eval.run_id,
        "eval_id": log.eval.eval_id,
        "task_args": log.eval.task_args,
    }


def answers_by_model(logs: list[EvalLog]) -> dict[str, list[StoredAnswer]]:
    """The stored answers of each log under its model; raises if a model has more than one log."""
    by_model = {}
    for log in logs:
        model = log.eval.model
        if model in by_model:
            raise ValueError(f"More than one log for model {model}")
        by_model[model] = stored_answers(log)
    return by_model


def answers_by_item(answers: dict[str, list[StoredAnswer]], seed: int) -> list[Sample]:
    """One sample per item whose replayed answer holds every model's answer, each under a label,
    in a seeded random order per item. The label-to-model mapping is kept in the metadata."""
    by_model = {model: {a.item_id: a for a in model_answers} for model, model_answers in answers.items()}
    models = sorted(by_model)
    item_ids = sorted(set().union(*(items.keys() for items in by_model.values())))
    return [_item_sample(item_id, by_model, models, seed) for item_id in item_ids]


def replay_sample(id, input, target, answer: str, metadata: dict | None = None) -> Sample:
    """A sample whose stored `answer` the replay solver will use as the model output."""
    return Sample(id=id, input=input, target=target, metadata={"answer": answer, **(metadata or {})})


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
    samples = [replay_sample(a.item_id, a.question, a.key, a.answer) for a in stored_answers(log)]
    return rescore_task(
        samples, scorers, name=log.eval.task,
        metadata={
            "kind": "rescore",
            "rescores": source_metadata(log),
        },
    )


def _item_sample(item_id, by_model: dict[str, dict], models: list[str], seed: int) -> Sample:
    models_without_answer = [model for model in models if item_id not in by_model[model]]
    if models_without_answer:
        raise ValueError(f"Item {item_id} has no answer from models {models_without_answer}")

    stored = {model: by_model[model][item_id] for model in models}
    first = stored[models[0]]
    if any(a.question != first.question or a.key != first.key for a in stored.values()):
        raise ValueError(f"Item {item_id} has differing questions or keys across models")

    models_with_empty_answer = [model for model, a in stored.items() if not a.answer.strip()]
    if models_with_empty_answer:
        raise ValueError(f"Item {item_id} has an empty answer from models {models_with_empty_answer}")

    order = list(models)
    random.Random(f"{seed}:{item_id}").shuffle(order)  # per item, independent of item order
    labels = [f"A{i}" for i in range(1, len(order) + 1)]
    submissions = "\n".join(f"[Submission {label}] {stored[model].answer}" for label, model in zip(labels, order))
    return replay_sample(item_id,
                         first.question,
                         first.key,
                         submissions,
                         metadata={"models": dict(zip(labels, order)),
                                   "order_seed": seed
                                   }
                         )


def several_answers_task(logs: list[EvalLog], scorers: list, order_seed: int) -> Task:
    """Rescoring task for every item's answers from all `logs` together."""
    if not logs:
        raise ValueError("No source logs to rescore")
    return rescore_task(
        answers_by_item(answers_by_model(logs), seed=order_seed), scorers, name=logs[0].eval.task,
        metadata={
            "kind": "rescore_several_answers",
            "rescores": [source_metadata(log) for log in logs],
        },
    )
