import math

from inspect_ai.scorer import Metric, SampleScore, metric

from src.scorers.grades import Grade


def na_metrics() -> list:
    """The metrics that include na as a value, e.g. reported for the additions ."""
    return [m() for m in na_metrics_names()]

def na_metrics_names() -> list:
    return [mean_excl_na, stderr_excl_na, share_na]


def _graded(scores: list[SampleScore]) -> list[float]:
    """Numeric values of the applicable grades; N and unknown values are left out."""
    grades = (Grade.parse(s.score.value) for s in scores)
    return [g.numeric for g in grades if g is not None and g.applicable]


@metric
def mean_excl_na() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        v = _graded(scores)
        return sum(v) / len(v) if v else math.nan

    return compute


@metric
def stderr_excl_na() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        v = _graded(scores)
        if len(v) < 2:
            return math.nan
        m = sum(v) / len(v)
        var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
        return math.sqrt(var / len(v))

    return compute


@metric
def share_na() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return math.nan
        return sum(Grade.parse(s.score.value) is Grade.N for s in scores) / len(scores)
    return compute
