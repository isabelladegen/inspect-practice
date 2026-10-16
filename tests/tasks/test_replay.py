from hamcrest import *
from inspect_ai.log import read_eval_log, list_eval_logs

from src.metrics.metric_ignoring_na import na_metrics
from src.scorers.grades import Grade
from src.tasks.replay import replay_sample, rescore_task, answers_by_item, replay_task, answers_by_model, StoredAnswer, \
    several_answers_task
from tests.testing_utils.stub_logs import replayed_log
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


def test_answers_are_read_back_per_model_and_item_from_disk(tmp_path):
    replayed_log("mockllm/gemma", ["gemma to 0", "gemma to 1"], str(tmp_path))
    replayed_log("mockllm/phi", ["phi to 0", "phi to 1"], str(tmp_path))
    logs = [read_eval_log(info) for info in list_eval_logs(str(tmp_path))]

    answers = answers_by_model(logs)

    assert_that(set(answers), equal_to({"mockllm/gemma", "mockllm/phi"}))
    assert_that({a.item_id: a.answer for a in answers["mockllm/gemma"]},
                equal_to({0: "gemma to 0", 1: "gemma to 1"}))
    assert_that({a.item_id: a.answer for a in answers["mockllm/phi"]},
                equal_to({0: "phi to 0", 1: "phi to 1"}))


def test_two_logs_for_one_model_raise(tmp_path):
    logs = [replayed_log("mockllm/gemma", ["g"], str(tmp_path)),
            replayed_log("mockllm/gemma", ["g"], str(tmp_path))]

    assert_that(calling(answers_by_model).with_args(logs), raises(ValueError))


def test_replaying_a_header_only_log_raises(tmp_path):
    location = replayed_log("mockllm/model", ["a"], str(tmp_path)).location
    header = read_eval_log(location, header_only=True)

    assert_that(calling(replay_task).with_args(header, []), raises(ValueError, "no samples"))


def test_answers_of_a_header_only_log_raise(tmp_path):
    location = replayed_log("mockllm/gemma", ["g"], str(tmp_path)).location
    header = read_eval_log(location, header_only=True)

    assert_that(calling(answers_by_model).with_args([header]), raises(ValueError, "no samples"))


#########

def test_each_label_holds_the_answer_of_the_model_mapped_to_it():
    answers = {"gemma": [StoredAnswer(0, "q", "t", "gemma to 0"), StoredAnswer(1, "q", "t", "gemma to 1")],
               "phi": [StoredAnswer(0, "q", "t", "phi to 0"), StoredAnswer(1, "q", "t", "phi to 1")]}
    expected = {("gemma", 0): "gemma to 0", ("gemma", 1): "gemma to 1",
                ("phi", 0): "phi to 0", ("phi", 1): "phi to 1"}

    samples = answers_by_item(answers, seed=1)

    assert_that([s.id for s in samples], equal_to([0, 1]))
    for s in samples:
        for label, model in s.metadata["models"].items():
            assert_that(s.metadata["answer"], contains_string(f"[Submission {label}] {expected[(model, s.id)]}"))


def test_same_seed_gives_same_order():
    answers = {"gemma": [StoredAnswer(0, "q", "t", "g")],
               "phi": [StoredAnswer(0, "q", "t", "p")]}
    reversed_answers = {"phi": answers["phi"], "gemma": answers["gemma"]}

    first = answers_by_item(answers, seed=1)
    second = answers_by_item(reversed_answers, seed=1)

    assert_that(second[0].metadata["models"], equal_to(first[0].metadata["models"]))


def test_missing_answer_raises():
    answers = {"gemma": [StoredAnswer(0, "q", "t", "g0"), StoredAnswer(1, "q", "t", "g1")],
               "phi": [StoredAnswer(0, "q", "t", "p0")]}

    assert_that(calling(answers_by_item).with_args(answers, seed=1), raises(ValueError))


def test_differing_question_raises():
    answers = {"gemma": [StoredAnswer(0, "q1", "t", "g")],
               "phi": [StoredAnswer(0, "q2", "t", "p")]}

    assert_that(calling(answers_by_item).with_args(answers, seed=1), raises(ValueError))


def test_empty_answer_raises():
    answers = {"gemma": [StoredAnswer(0, "q", "t", "g")],
               "phi": [StoredAnswer(0, "q", "t", "")]}

    assert_that(calling(answers_by_item).with_args(answers, seed=1), raises(ValueError))


def test_several_answers_task_replays_each_models_answer_under_its_label(tmp_path):
    logs = [replayed_log("mockllm/gemma", ["gemma to 0"], str(tmp_path)),
            replayed_log("mockllm/phi", ["phi to 0"], str(tmp_path))]
    expected = {"mockllm/gemma": "gemma to 0", "mockllm/phi": "phi to 0"}

    task = several_answers_task(logs, scorers=[], order_seed=1)
    log = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path / "several"), display="none")[0]

    sample = log.samples[0]
    for label, model in sample.metadata["models"].items():
        assert_that(sample.output.completion, contains_string(f"[Submission {label}] {expected[model]}"))

