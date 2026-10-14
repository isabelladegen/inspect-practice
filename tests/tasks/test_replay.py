from hamcrest import *

from src.metrics.metric_ignoring_na import na_metrics
from src.scorers.grades import Grade
from src.tasks.replay import replay_sample, rescore_task
from tests.testing_utils.stubs import stub_grader
from inspect_ai import eval as inspect_eval


def additions_metrics_after_rescoring(grades: list[str], log_dir: str) -> dict:
    """Rescore one sample per grade through the real rescore task; return the reported metrics."""
    samples = [replay_sample(id=i, input="q", target="t", answer="a") for i in range(len(grades))]
    task = rescore_task(samples, scorers=[stub_grader(grades, na_metrics())],
                        name="rescore_test")
    log = inspect_eval(task, model="mockllm/model", log_dir=log_dir, display="none")[0]
    return {name: m.value for name, m in log.results.scores[0].metrics.items()}


def test_rescoring_excludes_na_from_additions_mean(tmp_path):
    metrics = additions_metrics_after_rescoring([Grade.C, Grade.I, Grade.N, Grade.N], str(tmp_path))
    assert_that(metrics["mean_excl_na"], close_to(0.5, 1e-9))
    assert_that(metrics["share_na"], close_to(0.5, 1e-9))