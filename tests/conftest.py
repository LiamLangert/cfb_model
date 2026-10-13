"""The fake accounts in these tests hold only trading money: no reserve (kalshi.maker.RESERVE) unless a test
passes --reserve."""

import pytest

from kalshi import maker


@pytest.fixture(autouse=True)
def no_reserve(monkeypatch):
    monkeypatch.setattr(maker, "RESERVE", 0.0)
