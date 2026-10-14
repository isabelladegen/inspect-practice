from hamcrest import *
from inspect_ai.log import write_eval_log, read_eval_log

from src.maintainance.recompute import compute_changes
from src.maintainance.recompute_log_file import recompute_log_file
from src.scorers.completeness_and_additions import additions
from src.scorers.grades import Grade
from src.scorers.score_reducers import single
from tests.testing_utils.stub_logs import stub_broken_inspect_log, stub_no_changes, stub_correct_inspect_log, file_bytes

test_addition_grades = [Grade.C, Grade.I, Grade.N, Grade.N]
test_completeness_grades = [Grade.C, Grade.P, Grade.I, Grade.C]

def recompute_file(location):
    # method under test!
    return recompute_log_file(location, scorer=additions, reducer=single, tag="some test")


def test_broken_log_file_is_fixed_on_disk(tmp_path):
    location = stub_broken_inspect_log(tmp_path, test_addition_grades, test_completeness_grades).location
    assert_that(bool(recompute_file(location)), is_(True))
    changes = compute_changes(read_eval_log(location), scorer=additions, reducer=single)
    assert_that(changes, equal_to(stub_no_changes()))


def test_correct_log_file_is_left_alone(tmp_path):
    location = stub_correct_inspect_log(tmp_path, test_addition_grades, test_completeness_grades).location
    original = file_bytes(location)
    assert_that(recompute_file(location), equal_to(stub_no_changes()))
    assert_that(file_bytes(location), equal_to(original))


def test_fixed_log_file_is_left_alone_on_second_run(tmp_path):
    location = stub_broken_inspect_log(tmp_path, test_addition_grades, test_completeness_grades).location
    recompute_file(location)
    after_first = file_bytes(location)
    assert_that(recompute_file(location), equal_to(stub_no_changes()))
    assert_that(file_bytes(location), equal_to(after_first))


def test_unsuccessful_log_file_is_left_alone(tmp_path):
    log = stub_broken_inspect_log(tmp_path, test_addition_grades, test_completeness_grades)
    log.status = "error"
    write_eval_log(log, log.location)
    original = file_bytes(log.location)
    assert_that(recompute_file(log.location), equal_to(stub_no_changes()))
    assert_that(file_bytes(log.location), equal_to(original))
