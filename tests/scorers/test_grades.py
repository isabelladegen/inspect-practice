import re

from hamcrest import *

from src.scorers.grades import Grade

all_grades = list(Grade)
standard_grades = [Grade.C, Grade.P, Grade.I]


def test_numeric_values_of_the_scale():
    assert_that([g.numeric for g in Grade], equal_to([1.0, 0.5, 0.0, None]))


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


def test_grades_are_parsed_per_label_from_the_grades_line():
    reply = "A1 covers all points...\nGRADES: A1=C, A2=I, A3=P, A4=N"

    grades = Grade.parse_several(reply, ["A1", "A2", "A3", "A4"], all_grades)

    assert_that(grades, equal_to([("A1", Grade.C), ("A2", Grade.I), ("A3", Grade.P), ("A4", Grade.N)]))


def test_n_is_absent_when_not_allowed():
    grades = Grade.parse_several("GRADES: A1=N, A2=C", ["A1", "A2"], standard_grades)

    assert_that(grades, equal_to([("A2", Grade.C)]))


def test_last_grades_line_wins():
    reply = "GRADES: A1=I, A2=I\nOn review, A1 is complete.\nGRADES: A1=C, A2=I"

    grades = Grade.parse_several(reply, ["A1", "A2"], standard_grades)

    assert_that(grades, equal_to([("A1", Grade.C), ("A2", Grade.I)]))


def test_no_grades_line_gives_no_grades():
    assert_that(Grade.parse_several("GRADE: C", ["A1"], standard_grades), equal_to([]))
