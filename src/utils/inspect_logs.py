from inspect_ai.log import EvalLog, EvalResults, EvalSample
from inspect_ai.scorer import Score


def logged_samples(log: EvalLog) -> list[EvalSample]:
    """The samples of `log`; raises if it has none (read header-only, or a failed run)."""
    if log.samples is None:
        raise ValueError(f"Log {log.location} has no samples: read header-only, or the run failed")
    return log.samples


def logged_results(log: EvalLog) -> EvalResults:
    """The results of `log`; raises if it has none."""
    if log.results is None:
        raise ValueError(f"Log {log.location} has no results: the run failed or nothing was scored")
    return log.results


def sample_score(sample: EvalSample, scorer: str) -> Score:
    """`sample`'s score from `scorer`; raises if it has none."""
    if not sample.scores or scorer not in sample.scores:
        raise ValueError(f"Sample {sample.id} has no score from scorer {scorer}")
    return sample.scores[scorer]