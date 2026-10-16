from research import pull_kalshi as pk


def test_a_candle_parses_from_either_endpoint():
    """/historical spells the fields close, volume, open_interest; the live endpoint close_dollars, volume_fp,
    open_interest_fp. Both come out the same."""
    side = {"open": "0.17", "low": "0.16", "high": "0.18", "close": "0.17"}
    hist = {"end_period_ts": 60, "yes_bid": side, "yes_ask": {k: "0.37" for k in side},
            "price": {"open": "0.20", "low": "0.19", "high": "0.21", "close": "0.20", "mean": "0.20"},
            "volume": "2.00", "open_interest": "5.00"}
    live = {"end_period_ts": 60, "yes_bid": {k + "_dollars": v for k, v in side.items()},
            "yes_ask": {k + "_dollars": "0.3700" for k in side},
            "price": {k + "_dollars": v for k, v in hist["price"].items()},
            "volume_fp": "2.00", "open_interest_fp": "5.00"}
    assert pk.parse_candle(live) == pk.parse_candle(hist)
    assert pk.parse_candle(hist)[4] == 0.17 and pk.parse_candle(hist)[8] == 0.37
    # a minute with no trades has no price
    assert pk.parse_candle(live | {"price": {}})[9:14] == [None] * 5


def test_seasons_are_named_by_the_year_they_start():
    assert pk.in_season("KXNCAAFGAME-25AUG30OSUTEX", 2025)
    assert pk.in_season("KXNCAAFGAME-26JAN19MIAIND", 2025)
    assert not pk.in_season("KXNCAAFGAME-26AUG29OSUTEX", 2025)
