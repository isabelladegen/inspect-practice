from inspect_ai.scorer import Score, scorer


def stub_grader(grades: list[str], metrics: list):
    """Stand-in for a model grader: sample i gets grades[i], no model is called.

    Takes the metrics to report, so a test can attach the same metrics as the
    production scorer it replaces.
    """
    @scorer(metrics=metrics, name="stub_grader")
    def _stub():
        async def score(state, target):
            return Score(value=grades[state.sample_id])
        return score

    return _stub()