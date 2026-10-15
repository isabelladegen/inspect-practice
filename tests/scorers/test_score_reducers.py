from hamcrest import *
from inspect_ai.log import read_eval_log

from src.scorers.grades import Grade
from src.tasks.replay import rescore_task, replay_sample
from inspect_ai import eval as inspect_eval

from src.utils.inspect_registery import registered_name
from tests.testing_utils.stubs import stub_dict_scorer


def test_dict_of_letter_grades_is_stored_and_read_back_unchanged(tmp_path):
    value = {"mockllm/gemma": Grade.C, "mockllm/phi": Grade.N}
    task = rescore_task([replay_sample(id=0, input="q", target="t", answer="a")],
                        scorers=[stub_dict_scorer(value)], name="dict_score_test")

    log = read_eval_log(inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path),
                                     display="none")[0].location)

    assert_that(log.status, equal_to("success"))
    assert_that(log.samples[0].scores[registered_name(stub_dict_scorer)].value, equal_to(value))
