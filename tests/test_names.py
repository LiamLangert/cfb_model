"""betting.names: Kalshi's team names against CFBD's. No network."""

import pytest

from betting import names


@pytest.mark.parametrize("a, b", [("Iowa St.", "Iowa State"), ("San Jose St.", "San José State"),
                                  ("North Carolina St.", "NC State"), ("Appalachian St.", "App State"),
                                  ("Louisiana-Monroe", "UL Monroe"), ("Southern Miss", "Southern Miss"),
                                  ("Hawai'i", "Hawai'i"),
                                  ("Texas A&M", "Texas A&M"), ("Miami (OH)", "Miami (OH)")])
def test_names_that_must_match(a, b):
    assert names.similarity(a, b) == 1.0


def test_names_that_must_not_match_better_than_the_right_one():
    assert names.similarity("North Carolina St.", "NC State") > names.similarity("North Carolina", "NC State")
    assert names.similarity("Miami (FL)", "Miami") > names.similarity("Miami (FL)", "Miami (OH)")
