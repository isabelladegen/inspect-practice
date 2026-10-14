import math
from dataclasses import dataclass, field

from inspect_ai.log import EvalLog, EvalMetric, EvalSampleReductions, EvalSampleScore
from inspect_ai.scorer import SampleScore

from src.utils.inspect_registery import registered_name, scorer_metrics


@dataclass(frozen=True)
class MetricChange:
    name: str
    old: object
    new: object


@dataclass(frozen=True)
class SampleChange:
    sample_id: object
    old: object
    new: object


@dataclass(frozen=True)
class Changes:
    scorer: str
    metrics: list[MetricChange] = field(default_factory=list)
    reducer: tuple | None = None  # (old name, new name)
    samples: list[SampleChange] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.metrics or self.reducer or self.samples)


def _missing_as_none(value):
    """NaN and None both mean 'no value' (NaN is stored as null), so report both as None."""
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def compute_changes(log: EvalLog, scorer, reducer) -> Changes:
    """What `scorer`'s metrics and reductions should be, given its stored per-sample scores
    and `reducer`, compared with what the log holds. Does not modify the log."""
    name = registered_name(scorer)
    stored = sorted(((s.id, s.scores[name]) for s in log.samples), key=lambda x: x[0])
    sample_scores = [SampleScore(score=score, sample_id=sid) for sid, score in stored]

    result = next(r for r in log.results.scores if r.name == name)
    metric_changes = []
    for metric in scorer_metrics(scorer):
        metric_name = registered_name(metric)
        old = _missing_as_none(result.metrics[metric_name].value
                               if metric_name in result.metrics else None)
        new = _missing_as_none(metric(sample_scores))
        if old != new:
            metric_changes.append(MetricChange(metric_name, old, new))

    reducer_name = registered_name(reducer)
    current = next((r for r in (log.reductions or []) if r.scorer == name), None)
    old_reducer = current.reducer if current else None
    reducer_change = None if old_reducer == reducer_name else (old_reducer, reducer_name)

    current_values = {s.sample_id: s.value for s in current.samples} if current else {}
    sample_changes = [
        SampleChange(sid, current_values.get(sid), score.value)
        for sid, score in stored
        if current_values.get(sid) != score.value
    ]

    return Changes(scorer=name, metrics=metric_changes, reducer=reducer_change, samples=sample_changes)


def apply_changes(log: EvalLog, changes: Changes, tag: str) -> None:
    """Write `changes` into the log and tag it."""
    result = next(r for r in log.results.scores if r.name == changes.scorer)
    for change in changes.metrics:
        value = math.nan if change.new is None else change.new
        if change.name in result.metrics:
            result.metrics[change.name].value = value
        else:
            result.metrics[change.name] = EvalMetric(name=change.name, value=value)

    if changes.reducer or changes.samples:
        current = next((r for r in (log.reductions or []) if r.scorer == changes.scorer), None)
        reducer_name = changes.reducer[1] if changes.reducer else current.reducer
        # A single reducer's reduced score is the stored score itself, so the stored
        # score is used whole, keeping its answer, explanation and metadata.
        reduction = EvalSampleReductions(
            scorer=changes.scorer,
            reducer=reducer_name,
            samples=[EvalSampleScore(**s.scores[changes.scorer].model_dump(), sample_id=s.id)
                     for s in sorted(log.samples, key=lambda s: s.id)],
        )
        log.reductions = [r for r in (log.reductions or [])
                          if r.scorer != changes.scorer] + [reduction]

    log.eval.tags = list(dict.fromkeys((log.eval.tags or []) + [tag]))
