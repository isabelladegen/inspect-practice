from inspect_ai.scorer import score_reducer, ScoreReducer, Score


@score_reducer(name="single")
def single() -> ScoreReducer:
    """Pass a sample's score through unchanged; there must be exactly one."""
    def reduce(scores: list[Score]) -> Score:
        if len(scores) != 1:
            raise ValueError(
                f"single reducer expects exactly one score per sample, got {len(scores)}"
            )
        return scores[0]
    return reduce