import re

from hamcrest import *

from src.scorers.grades import Grade


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
    assert_that(re.search(Grade.pattern(include_na=True), reply).group(1), equal_to("N"))
    assert_that(re.search(Grade.pattern(include_na=False), reply), is_(none()))
