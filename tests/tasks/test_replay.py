from hamcrest import *
from inspect_ai.log import read_eval_log, EvalLog, list_eval_logs

from src.metrics.metric_ignoring_na import na_metrics
from src.scorers.grades import Grade
from src.tasks.replay import replay_sample, rescore_task, stored_answer
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


def replayed_log(model: str, answers: list[str], log_dir: str) -> EvalLog:
    """Replay one sample per answer through the real rescore task as `model`; return the log read from disk."""
    samples = [replay_sample(id=i, input="q", target="t", answer=a) for i, a in enumerate(answers)]
    task = rescore_task(samples, scorers=[], name="replay_test")
    log = inspect_eval(task, model=model, log_dir=log_dir, display="none")[0]
    return read_eval_log(log.location)


def test_stored_answer_is_read_back_per_model_and_id_from_disk(tmp_path):
    replayed_log("mockllm/gemma", ["gemma to 0", "gemma to 1"], str(tmp_path))
    replayed_log("mockllm/phi", ["phi to 0", "phi to 1"], str(tmp_path))

    logs = {log.eval.model: log
            for log in (read_eval_log(info) for info in list_eval_logs(str(tmp_path)))}

    assert_that(set(logs), equal_to({"mockllm/gemma", "mockllm/phi"}))
    assert_that({s.id: stored_answer(s) for s in logs["mockllm/gemma"].samples},
                equal_to({0: "gemma to 0", 1: "gemma to 1"}))
    assert_that({s.id: stored_answer(s) for s in logs["mockllm/phi"].samples},
                equal_to({0: "phi to 0", 1: "phi to 1"}))