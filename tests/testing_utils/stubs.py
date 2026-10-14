from inspect_ai.scorer import Score, scorer, mean, stderr

from src.metrics.metric_ignoring_na import na_metrics
from src.scorers.completeness_and_additions import completeness, additions


def stub_grader(grades: list[str], metrics: list, name="stub_grader"):
    """Stand-in for a model grader: sample i gets grades[i], no model is called.

    Takes the metrics to report, so a test can attach the same metrics as the
    production scorer it replaces.
    """

    @scorer(metrics=metrics, name=name)
    def _stub():
        async def score(state, target):
            return Score(value=grades[state.sample_id])

        return score

    return _stub()


def stub_addition_completeness_scorers(additions_grades, completeness_grades):
    return [
        stub_grader(completeness_grades, [mean(), stderr()], name=completeness.__name__),
        stub_grader(additions_grades, na_metrics(), name=additions.__name__),
    ]
