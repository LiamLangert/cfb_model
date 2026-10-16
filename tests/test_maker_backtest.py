from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from research import maker_backtest as mb

ET = ZoneInfo("America/New_York")


def ts(*args):
    return datetime(*args, tzinfo=ET).timestamp()


def test_best_price_improves_when_there_is_room_and_edge_after_the_fee():
    fee = mb.fee(0.51)
    assert fee > 0   # football charges makers
    assert mb.best_price(0.60, 0.50, 0.53) == 0.51
    # no room to improve: joins the bid
    assert mb.best_price(0.60, 0.50, 0.51) == 0.50
    # improving would leave under MIN_EDGE after the fee, joining does not
    assert mb.best_price(0.51 + mb.MIN_EDGE + fee - 0.001, 0.50, 0.55) == 0.50
    assert mb.best_price(0.51, 0.50, 0.55) is None


def test_fair_value_moves_weight_of_the_way_in_log_odds():
    assert mb.fair_value(0.5, 0.7, 0.0) == pytest.approx(0.5)
    assert mb.fair_value(0.5, 0.7, 1.0) == pytest.approx(0.7)
    assert mb.fair_value(0.5, 0.7, 0.25) == pytest.approx(1 / (1 + (3 / 7) ** 0.25))


def candle(t, bid, ask):
    r = [None] * 16
    r[0], r[4], r[8] = t, bid, ask
    return r


def test_market_quote_and_what_is_sold_into_a_bid():
    m = mb.Market([candle(100, 0.40, 0.43), candle(200, 0.0, 1.0), candle(300, 0.45, 0.47)],
                  [[150_000, 0.40, 10, 0], [160_000, 0.39, 5, 0], [170_000, 0.42, 7, 1], [400_000, 0.41, 3, 0]])
    assert m.quote(150) == (0.40, 0.43) and m.quote(250) == (None, None) and m.quote(50) == (None, None)
    assert m.sold_into(100, 200, 0.40, strict=False) == (15.0, 150.0)
    assert m.sold_into(100, 200, 0.40, strict=True) == (5.0, 160.0)
    assert m.sold_into(165, 200, 0.45, strict=False) == (0.0, None)   # a buyer never fills a bid


def test_rappa_is_known_from_the_tuesday_of_the_games_week():
    sat = datetime(2025, 10, 11, 15, 30, tzinfo=ET)
    assert mb.known_from(sat) == datetime(2025, 10, 7, 10, 0, tzinfo=ET)
    tue = datetime(2025, 10, 7, 19, 0, tzinfo=ET)
    assert mb.known_from(tue) == datetime(2025, 10, 7, 10, 0, tzinfo=ET)


def test_daily_runs_are_4pm_between_30_and_2_hours_out_and_hourly_ones_every_hour():
    noon = SimpleNamespace(kick_ts=ts(2025, 10, 11, 12, 0))
    assert [datetime.fromtimestamp(r, ET).day for r in mb.runs(noon)] == [10]        # Friday 4pm, 20h out
    late = SimpleNamespace(kick_ts=ts(2025, 10, 11, 22, 30))
    assert [datetime.fromtimestamp(r, ET).day for r in mb.runs(late)] == [11]        # Friday's run is 30.5h out
    hourly = mb.place_times(SimpleNamespace(kick_ts=ts(2025, 10, 11, 12, 0)), "hourly:6")
    assert [datetime.fromtimestamp(r, ET).hour for r in hourly] == [6, 7, 8, 9, 10]
    assert mb.variant_lead("daily") == mb.MAX_LEAD and mb.variant_lead("hourly:12").total_seconds() == 12 * 3600


def game(**kw):
    base = dict(season=2025, week=7, game_id=1, kick_ts=10 * 3600, known_ts=0, open_ts=0, settle_ts=14 * 3600,
                home_ticker="H", away_ticker="A", p_home=0.70, home_win=1)
    return SimpleNamespace(**(base | kw))


def test_decide_waits_for_rappa_and_keeps_the_guards():
    markets = {"H": mb.Market([candle(0, 0.50, 0.52)], []), "A": mb.Market([candle(0, 0.48, 0.50)], [])}
    o, why = mb.decide(game(), 3600, markets, 1.0)
    assert why == "" and o["side"] == "home" and o["price"] == 0.51
    assert o["edge"] == pytest.approx(o["fair"] - 0.51 - mb.fee(0.51))
    assert mb.decide(game(known_ts=7200), 3600, markets, 1.0) == (None, "RAPPA not handed over yet")
    assert mb.decide(game(), 9 * 3600, markets, 1.0) == (None, "kickoff too soon")
    assert mb.decide(game(p_home=0.95), 3600, markets, 1.0) == (None, "RAPPA too far from the market")


def test_a_reviewed_order_is_replaced_when_outbid_and_fills_net_of_the_fee(monkeypatch):
    # The home book moves up an hour later: the daily review replaces the bid at the new best bid plus a cent.
    home = mb.Market([candle(0, 0.50, 0.53), candle(3600, 0.52, 0.55)],
                     [[1800_000, 0.50, 4, 0], [5400_000, 0.53, 6, 0]])
    markets = {"H": home, "A": mb.Market([candle(0, 0.47, 0.50), candle(3600, 0.45, 0.48)], [])}
    monkeypatch.setattr(mb, "place_times", lambda g, v: [0.0, 3600.0])   # two runs, an hour apart
    row = mb.simulate(game(kick_ts=6 * 3600), "daily", markets, 1.0)
    assert row["decision"] == "order" and row["price"] == 0.51 and row["review"] == "replaced at 0.53"
    assert row["fills_at"] == [(4.0, 0.51), (6.0, 0.53)]
    assert row["pnl_at"] == pytest.approx(1 - 0.51 - mb.fee(0.51))


def test_the_plan_caps_what_is_active_at_once(monkeypatch):
    monkeypatch.setattr(mb, "KELLY_FRACTION", 1.0)    # the sizes below are full Kelly's
    monkeypatch.setattr(mb, "FEE_TYPE", "quadratic")  # and fee-free, to keep the arithmetic plain

    def order(gid, placed, rest_end, settle, filled):
        fills = [(1000.0, 0.50)] if filled else [(0.0, 0.50)]
        return {"game_id": gid, "ticker": gid, "variant": "daily", "weight": 1.0, "decision": "order", "placed": placed,
                "rest_end": rest_end, "settle": settle, "fair": 0.60, "price": 0.50, "edge": 0.10, "y": 1,
                "fills_at": fills, "fills_through": fills}

    # A rests unfilled until 10; B fills and settles at 20; C finds both active; D comes once A is gone, E once B is
    df = pd.DataFrame([order("A", 0, 10, 30, False), order("B", 0, 10, 20, True), order("C", 5, 10, 30, True),
                       order("D", 15, 16, 30, False), order("E", 25, 26, 40, False)])
    plan, ledger = mb.plan_table(df, 1000.0, max_orders=2, max_share=0.5)
    at = plan[plan.fills == "at"].iloc[0]
    assert ledger[ledger.fills == "at"].game_id.tolist() == ["A", "B", "D", "E"]
    assert (at.capped_games, at.capped_money, at.most_active) == (1, 0, 2)
    e = ledger[(ledger.fills == "at") & (ledger.game_id == "E")].iloc[0]
    assert e.bankroll_before == pytest.approx(1200.0) and at.pnl == pytest.approx(200.0)
    # each order is $200 (20% of $1,000): with 30% of it to hold at once, B has no room beside A
    plan, ledger = mb.plan_table(df, 1000.0, max_share=0.30)
    assert ledger[ledger.fills == "at"].game_id.tolist() == ["A", "D", "E"]
    assert plan.iloc[0].capped_money == 2
