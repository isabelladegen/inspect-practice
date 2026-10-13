import math

from inspect_ai.scorer import Metric, SampleScore, metric

GRADE_VALUE = {"C": 1.0, "P": 0.5, "I": 0.0}


def _graded(scores: list[SampleScore]) -> list[float]:
    """Numeric values of the C/P/I grades; N and anything else is left out."""
    return [GRADE_VALUE[s.score.value] for s in scores if s.score.value in GRADE_VALUE]


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
        return sum(s.score.value == "N" for s in scores) / len(scores) if scores else math.nan
    return compute