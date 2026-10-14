from hamcrest import *
from inspect_ai.scorer import mean

from src.metrics.metric_ignoring_na import mean_excl_na
from src.scorers.completeness_and_additions import additions
from src.scorers.score_reducers import single
from src.utils.inspect_registery import registered_name, scorer_metrics


def test_registered_name_of_metric_instance():
    assert_that(registered_name(mean_excl_na()), equal_to("mean_excl_na"))


def test_registered_name_of_metric_factory():
    assert_that(registered_name(mean_excl_na), equal_to("mean_excl_na"))


def test_registered_name_strips_inspect_package_prefix():
    assert_that(registered_name(mean()), equal_to("mean"))


def test_registered_name_of_reducer():
    assert_that(registered_name(single), equal_to("single"))


def test_registered_name_of_scorer():
    assert_that(registered_name(additions), equal_to("additions"))


def test_scorer_metrics_are_those_declared_by_the_scorer():
    names = [registered_name(m) for m in scorer_metrics(additions)]
    assert_that(names, equal_to(["mean_excl_na", "stderr_excl_na", "share_na"]))