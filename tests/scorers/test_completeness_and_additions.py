from hamcrest import *
from inspect_ai.core import ModelOutput
from inspect_ai.model import Model as InspectModel, get_model
from inspect_ai import eval as inspect_eval

from src.scorers.completeness_and_additions import completeness_of_several_answers, additions_of_several_answers
from src.tasks.replay import several_answers_task
from src.utils.inspect_registery import registered_name
from tests.testing_utils.stub_logs import replayed_log


def replying(name: str, reply: str) -> InspectModel:
    """A grader that always replies `reply`."""
    return get_model(f"mockllm/{name}",
                     custom_outputs=[ModelOutput.from_content(model=f"mockllm/{name}", content=reply)])


def several_answers_log(tmp_path, completeness_reply: str, additions_reply: str):
    logs = [replayed_log("mockllm/gemma", ["gemma to 0"], str(tmp_path)),
            replayed_log("mockllm/phi", ["phi to 0"], str(tmp_path))]
    scorers = [completeness_of_several_answers(grader=replying("completeness", completeness_reply)),
               additions_of_several_answers(grader=replying("additions", additions_reply))]
    return inspect_eval(several_answers_task(logs, scorers, order_seed=1), model="mockllm/model",
                        log_dir=str(tmp_path / "graded"), display="none")[0]


def metric(log, scorer, model: str, name: str) -> float:
    return next(s for s in log.results.scores if s.scorer == registered_name(scorer) and s.name == model).metrics[
        name].value


def test_metrics_are_reported_per_answering_model(tmp_path):
    log = several_answers_log(tmp_path, "GRADES: A1=C, A2=P", "GRADES: A1=C, A2=N")

    models = log.samples[0].metadata["models"]
    assert_that(metric(log, completeness_of_several_answers, models["A1"], "mean"), equal_to(1.0))
    assert_that(metric(log, completeness_of_several_answers, models["A2"], "mean"), equal_to(0.5))
    assert_that(metric(log, additions_of_several_answers, models["A1"], "mean_excl_na"), equal_to(1.0))
    assert_that(metric(log, additions_of_several_answers, models["A2"], "share_na"), equal_to(1.0))


def test_a_reply_missing_a_label_fails_the_run(tmp_path):
    log = several_answers_log(tmp_path, "GRADES: A1=C", "GRADES: A1=C, A2=C")

    assert_that(log.status, equal_to("error"))
    assert_that(log.error.message, contains_string("no grade for A2"))
