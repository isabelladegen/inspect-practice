import re

from hamcrest import *

from src.scorers.grades import Grade

all_grades = list(Grade)
standard_grades = [Grade.C, Grade.P, Grade.I]


def test_position_values_of_the_scale():
    assert_that([g.position for g in Grade], equal_to([1.0, 0.5, 0.0, None]))


def test_numeric_values_of_the_grades_on_the_scale():
    assert_that([Grade.C.numeric, Grade.P.numeric, Grade.I.numeric], equal_to([1.0, 0.5, 0.0]))


def test_numeric_of_a_grade_not_on_the_scale_raises():
    assert_that(calling(lambda: Grade.N.numeric), raises(ValueError))


def test_only_n_is_not_applicable():
    assert_that([g.applicable for g in Grade], equal_to([True, True, True, False]))


def test_parse_known_and_unknown_values():
    assert_that(Grade.parse("P"), is_(Grade.P))
    assert_that(Grade.parse(1.0), is_(none()))
    assert_that(Grade.parse("X"), is_(none()))


def test_pattern_with_and_without_na():
    reply = "Some reasoning.\nGRADE: N"
    assert_that(re.search(Grade.pattern(all_grades), reply).group(1), equal_to("N"))
    assert_that(re.search(Grade.pattern(standard_grades), reply), is_(none()))


def test_grades_are_parsed_per_label_from_a_grades_line():
    pairs = Grade.parse_grades_line("A1=C, A2=I, A3=P, A4=N", ["A1", "A2", "A3", "A4"], Grade)
    assert_that(pairs, equal_to([("A1", Grade.C), ("A2", Grade.I), ("A3", Grade.P), ("A4", Grade.N)]))


def test_a_grade_not_in_use_is_absent():
    pairs = Grade.parse_grades_line("A1=N, A2=C", ["A1", "A2"], [Grade.C, Grade.P, Grade.I])
    assert_that(pairs, equal_to([("A2", Grade.C)]))


def test_the_last_grades_line_is_captured():
    reply = "GRADES: A1=I, A2=I\nOn review, A1 is complete.\nGRADES: A1=C, A2=I"
    assert_that(re.search(Grade.several_answers_pattern(), reply).group(1), equal_to("A1=C, A2=I"))


def test_no_grades_line_is_not_captured():
    assert_that(re.search(Grade.several_answers_pattern(), "GRADE: C"), none())