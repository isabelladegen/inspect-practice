from hamcrest import *
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import Score

from src.maintainance.recompute import SampleChange, MetricChange, Changes, apply_changes, compute_changes
from src.scorers.completeness_and_additions import additions, completeness
from src.scorers.grades import Grade
from src.scorers.score_reducers import single
from src.metrics.metric_ignoring_na import mean_excl_na, stderr_excl_na, share_na
from src.utils.inspect_registery import registered_name
from tests.testing_utils.stub_logs import stub_broken_inspect_log, stub_correct_inspect_log, stub_no_changes

test_addition_grades = [Grade.C, Grade.I, Grade.N, Grade.N]
test_completeness_grades = [Grade.C, Grade.P, Grade.I, Grade.C]


# --- compute_changes: decides what differs and calculates the new values ---
def changes_of(log):
    # method under test here
    return compute_changes(log, scorer=additions, reducer=single)


def test_computes_changes_for_broken_log(tmp_path):
    stubbed_log = stub_broken_inspect_log(tmp_path, test_addition_grades, test_completeness_grades)
    changes = changes_of(stubbed_log)
    expected_changes = Changes(
        scorer=registered_name(additions),
        metrics=[
            MetricChange(registered_name(mean_excl_na), None, 0.5),
            MetricChange(registered_name(stderr_excl_na), None, 0.5),
            MetricChange(registered_name(share_na), 0.0, 0.5),
        ],
        reducer=(None, registered_name(single)),  # "mean": the name Inspect writes for its default
        samples=[
            SampleChange(0, 1.0, Grade.C),
            SampleChange(1, 0.0, Grade.I),
            SampleChange(2, 0.0, Grade.N),
            SampleChange(3, 0.0, Grade.N),
        ],
    )
    assert_that(changes, equal_to(expected_changes))


def test_computes_no_changes_for_correct_log(tmp_path):
    stubbed_log = stub_correct_inspect_log(tmp_path, test_addition_grades, test_completeness_grades)
    changes = changes_of(stubbed_log)
    assert_that(changes, equal_to(stub_no_changes()))
    assert_that(bool(changes), is_(False))


def test_computes_no_changes_for_correct_all_na_log(tmp_path):
    log = stub_correct_inspect_log(tmp_path, additions_grades=[Grade.N, Grade.N],
                                   completeness_grades=[Grade.C, Grade.I])
    changes = changes_of(log)
    assert_that(changes, equal_to(stub_no_changes()))


def test_computing_changes_does_not_modify_log(tmp_path):
    log = stub_broken_inspect_log(tmp_path, test_addition_grades, test_completeness_grades)
    before = log.model_dump()
    changes_of(log)  # calling recompute log does not change the log itself!
    assert_that(log.model_dump(), equal_to(before))


# --- apply_changes: writes given changes into the log (trusts compute_changes) ---

def test_applied_changes_leave_nothing_to_change(tmp_path):
    log = stub_broken_inspect_log(tmp_path, test_addition_grades, test_completeness_grades)
    apply_changes(log, changes_of(log), "some test")
    assert_that(changes_of(log), equal_to(stub_no_changes()))


def test_applying_changes_tags_log(tmp_path):
    log = stub_broken_inspect_log(tmp_path, test_addition_grades, test_completeness_grades)
    tag = "testing"
    apply_changes(log, changes_of(log), tag)
    assert_that(log.eval.tags or [], has_item(tag))


def test_applying_changes_leaves_other_scorer_and_samples_untouched(tmp_path):
    log = stub_broken_inspect_log(tmp_path, test_addition_grades, test_completeness_grades)
    name = completeness.__name__
    before_result = next(r for r in log.results.scores if r.name == name).model_dump()
    before_reduction = next(r for r in log.reductions if r.scorer == name).model_dump()
    before_samples = [s.scores for s in log.samples]

    apply_changes(log, changes_of(log), "some test")

    assert_that(next(r for r in log.results.scores if r.name == name).model_dump(),
                equal_to(before_result))
    assert_that(next(r for r in log.reductions if r.scorer == name).model_dump(),
                equal_to(before_reduction))
    assert_that([s.scores for s in log.samples], equal_to(before_samples))


def test_applying_changes_keeps_rich_score_content_in_reductions(tmp_path):
    rich_score = Score(
        value=Grade.I,
        answer="I don't store any personal data.",
        explanation="Both additions are not relevant to the question.\n\nGRADE: I",
        metadata={"grading": [
            {"id": "u1", "content": "You are evaluating only the content of a submission...",
             "role": "user"},
            {"id": "a1", "content": "Both additions are not relevant...\n\nGRADE: I",
             "source": "generate", "role": "assistant", "model": "phi4:14b"},
        ]},
    )
    log = stub_broken_inspect_log(tmp_path, [rich_score, Grade.C], [Grade.C, Grade.I])

    apply_changes(log, compute_changes(log, scorer=additions, reducer=single), "metrics_recomputed")

    reduction = next(r for r in log.reductions if r.scorer == registered_name(additions))
    reduced = next(s for s in reduction.samples if s.sample_id == 0)
    assert_that(reduced.value, equal_to(rich_score.value))
    assert_that(reduced.answer, equal_to(rich_score.answer))
    assert_that(reduced.explanation, equal_to(rich_score.explanation))
    assert_that(reduced.metadata, equal_to(rich_score.metadata))


def test_computing_changes_on_a_header_only_log_raises(tmp_path):
    location = stub_correct_inspect_log(tmp_path, [Grade.C], [Grade.C]).location
    header = read_eval_log(location, header_only=True)

    assert_that(calling(compute_changes).with_args(header, additions, single()), raises(ValueError, "no samples"))
