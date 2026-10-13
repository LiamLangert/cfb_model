"""betting.orders: pricing, matching, guards, caps, and that nothing is sent unless asked. No network."""

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from betting import orders as ko
from betting.kalshi import Book, Client, maker_fee

NOW = datetime(2026, 10, 6, 14, 0, tzinfo=timezone.utc)  # Tuesday 10am ET
BANKROLL = 50.0


def game(gid, home, away, wp, day="2026-10-10", time="15:30", score=None):
    return {"game_id": gid, "home_team": home, "away_team": away, "rappa_wp": wp, "gameday": day,
            "gametime": time, "home_score": score}


def markets_for(event, home_name, away_name, home_code="H", away_code="A"):
    return [{"event_ticker": event, "ticker": f"{event}-{home_code}", "yes_sub_title": home_name},
            {"event_ticker": event, "ticker": f"{event}-{away_code}", "yes_sub_title": away_name}]


def book_pair(home_bid, width=0.01):
    """Two consistent books: the home team bid at home_bid, the away team the mirror image."""
    return Book(home_bid, round(home_bid + width, 2), 100, 100), Book(round(1 - home_bid - width, 2), round(1 - home_bid, 2), 100, 100)


def setup(wp, home_bid, width=0.01, **kw):
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", wp)])
    mk = markets_for("KXNCAAFGAME-26OCT10PSUOSU", "Ohio St.", "Penn St.")
    matches = ko.match_events(games, mk)
    hb, ab = book_pair(home_bid, width)
    books = {mk[0]["ticker"]: hb, mk[1]["ticker"]: ab}
    return ko.plan(games, matches, books, set(), NOW, 2026, 6, **{"bankroll": BANKROLL, **kw}), mk


# ---------- pricing ----------

def test_fair_value_is_a_log_odds_step_toward_rappa():
    assert ko.fair_value(0.5, 0.5) == pytest.approx(0.5)
    assert ko.fair_value(0.6, 0.6) == pytest.approx(0.6)
    assert ko.fair_value(0.5, 0.8, weight=0) == pytest.approx(0.5)
    assert ko.fair_value(0.5, 0.8, weight=1) == pytest.approx(0.8)
    f = ko.fair_value(0.5, 0.8, weight=0.25)
    assert f == pytest.approx(1 / (1 + math.exp(-0.25 * math.log(4))))
    assert 0.5 < f < 0.8
    assert 0 < ko.fair_value(0.5, 0.0) < 0.5 and 0.5 < ko.fair_value(0.5, 1.0) < 1  # no log(0)


@pytest.mark.parametrize("bid, ask", [(0.40, 0.41), (0.40, 0.42), (0.40, 0.45), (0.12, 0.13), (0.88, 0.89)])
@pytest.mark.parametrize("fair", [0.30, 0.45, 0.50, 0.60, 0.95])
def test_best_price_always_rests_and_always_clears_the_edge(bid, ask, fair):
    book = Book(bid, ask)
    price = ko.best_price(fair, book, min_edge=0.02, max_below_bid=0.02)
    if price is None:
        # then no price from bid - 2c up to the top clears it
        for c in range(round((bid - 0.02) * 100), round(min(bid + 0.01, ask - 0.01) * 100) + 1):
            p = c / 100
            assert fair - p - maker_fee(p) < 0.02
        return
    assert price < ask - 1e-9, "would cross the ask"
    assert price <= bid + 0.01 + 1e-9
    assert price >= bid - 0.02 - 1e-9
    assert fair - price - maker_fee(price) >= 0.02 - 1e-9
    assert abs(price * 100 - round(price * 100)) < 1e-9, "whole cents only"
    # and it is the highest such price
    higher = round(price + 0.01, 2)
    if higher < ask and higher <= bid + 0.01 + 1e-9:
        assert fair - higher - maker_fee(higher) < 0.02


def test_best_price_by_default_never_rests_below_the_bid():
    assert ko.best_price(0.455, Book(0.44, 0.45)) is None         # edge only exists below the bid
    assert ko.best_price(0.455, Book(0.44, 0.45), max_below_bid=0.02) == 0.43  # 0.455 - 0.43 - fee = 2.07c
    assert ko.best_price(0.60, Book(0.44, 0.47)) == 0.45          # improves the bid by a cent, below the ask


def test_best_price_needs_both_sides():
    assert ko.best_price(0.9, Book(None, 0.5)) is None
    assert ko.best_price(0.9, Book(0.5, None)) is None


# ---------- matching ----------

def test_match_events_orients_home_and_away_by_name():
    games = pd.DataFrame([game(1, "NC State", "North Carolina", 0.5)])
    mk = markets_for("KXNCAAFGAME-26OCT10UNCNCST", "North Carolina", "North Carolina St.", "UNC", "NCST")
    hit = ko.match_events(games, mk)[1]
    assert hit["home"]["yes_sub_title"] == "North Carolina St."
    assert hit["away"]["yes_sub_title"] == "North Carolina"


def test_match_events_needs_the_date_and_both_names():
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", 0.5, day="2026-10-10"),
                          game(2, "Georgia State", "Northern Illinois", 0.5, day="2026-10-10")])
    far = markets_for("KXNCAAFGAME-26OCT24PSUOSU", "Ohio St.", "Penn St.")            # two weeks off
    fcs = markets_for("KXNCAAFGAME-26OCT10SIUMUR", "Southern Illinois", "Murray St.")  # not these teams
    one_side = markets_for("KXNCAAFGAME-26OCT10RUTOSU", "Ohio St.", "Rutgers")         # only one team right
    assert ko.match_events(games, far + fcs + one_side) == {}


def test_match_events_one_event_per_game_and_game_per_event():
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", 0.5)])
    good = markets_for("KXNCAAFGAME-26OCT10PSUOSU", "Ohio St.", "Penn St.")
    worse = markets_for("KXNCAAFGAME-26OCT10PSUOSUB", "Ohio St.", "Penn")
    m = ko.match_events(games, good + worse)
    assert list(m) == [1] and m[1]["event"] == "KXNCAAFGAME-26OCT10PSUOSU"


def test_match_events_skips_events_without_exactly_two_markets():
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", 0.5)])
    mk = markets_for("KXNCAAFGAME-26OCT10PSUOSU", "Ohio St.", "Penn St.")
    assert ko.match_events(games, mk[:1]) == {}
    tie = {"event_ticker": mk[0]["event_ticker"], "ticker": mk[0]["event_ticker"] + "-TIE", "yes_sub_title": "Tie"}
    assert ko.match_events(games, mk + [tie]) == {}


def test_load_week_leaves_out_games_already_kicked_off(tmp_path):
    out = tmp_path / "cfb"
    (out / "2026").mkdir(parents=True)
    pd.DataFrame([game(1, "Troy", "Southern Miss", 0.6, day="2026-10-06", time="08:00"),      # this morning
                  game(2, "Liberty", "Sam Houston", 0.6, day="2026-10-05", time=None),        # yesterday, time TBD
                  game(3, "Ohio State", "Penn State", 0.6, day="2026-10-06", time="20:00"),   # tonight
                  game(4, "Georgia", "Alabama", 0.6, day="2026-10-06", time=None),            # today, time TBD
                  game(5, "Texas", "Oklahoma", 0.6, score=31)]                                # scored
                 ).to_csv(out / "2026" / "parts_wk06.csv", index=False)
    assert list(ko.load_week(2026, 6, out).game_id) == [1, 2, 3, 4]
    assert list(ko.load_week(2026, 6, out, NOW).game_id) == [3, 4]  # NOW is Tuesday 10am ET


# ---------- the plan ----------

def test_plan_buys_the_side_rappa_likes_at_a_maker_price():
    p, mk = setup(wp=0.72, home_bid=0.55)
    row = p.iloc[0]
    assert row.decision == "order" and row.side == "home" and row.ticker == mk[0]["ticker"]
    assert row.price <= 0.55 + 1e-9  # 1c book: the bid, never the ask
    assert row.edge >= ko.MIN_EDGE - 1e-9
    per = row.price + maker_fee(row.price)
    assert row["count"] == ko.contracts(row.fair, per, BANKROLL)[0] >= 1
    assert row.cost == pytest.approx(row["count"] * (row.price + maker_fee(row.price)), abs=1e-4)
    assert datetime.fromisoformat(row.expires) == datetime(2026, 10, 10, 19, 0, tzinfo=timezone.utc)  # 3:30 ET - 30m


def test_plan_buys_the_away_side_when_rappa_likes_it():
    p, mk = setup(wp=0.38, home_bid=0.55)
    assert p.iloc[0].decision == "order" and p.iloc[0].side == "away" and p.iloc[0].ticker == mk[1]["ticker"]


def test_plan_skips_small_disagreements():
    p, _ = setup(wp=0.57, home_bid=0.55)
    assert p.iloc[0].decision == "skip" and p.iloc[0].reason == "no edge at a maker price"


@pytest.mark.parametrize("kw, reason", [
    (dict(wp=0.90, home_bid=0.55), "RAPPA too far"),
    (dict(wp=0.72, home_bid=0.55, width=0.05), "book too wide"),
    (dict(wp=0.70, home_bid=0.93), "RAPPA too far"),
])
def test_plan_guards(kw, reason):
    p, _ = setup(**kw)
    assert p.iloc[0].decision == "skip" and p.iloc[0].reason.startswith(reason)


def test_plan_price_bounds():
    # a favourite RAPPA likes even more, priced above MAX_PRICE: no order
    p, _ = setup(wp=0.99, home_bid=0.91, weight=0.25)
    assert p.iloc[0].decision == "skip"


def test_plan_skips_kickoff_soon_and_tbd():
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", 0.8, day="2026-10-06", time="11:00"),
                          game(2, "Georgia", "Alabama", 0.8, time=None)])
    mk = markets_for("KXNCAAFGAME-26OCT06PSUOSU", "Ohio St.", "Penn St.") + \
        markets_for("KXNCAAFGAME-26OCT10ALAUGA", "Georgia", "Alabama")
    books = {m["ticker"]: b for m, b in zip(mk, book_pair(0.55) + book_pair(0.55))}
    p = ko.plan(games, ko.match_events(games, mk), books, set(), NOW, 2026, 6, bankroll=BANKROLL).set_index("game_id")
    assert p.loc[1].reason == "kickoff too soon" and p.loc[2].reason == "kickoff time TBD"


def test_plan_skips_games_already_held():
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", 0.8)])
    mk = markets_for("KXNCAAFGAME-26OCT10PSUOSU", "Ohio St.", "Penn St.")
    hb, ab = book_pair(0.55)
    p = ko.plan(games, ko.match_events(games, mk), {mk[0]["ticker"]: hb, mk[1]["ticker"]: ab},
                {"KXNCAAFGAME-26OCT10PSUOSU"}, NOW, 2026, 6, bankroll=BANKROLL)
    assert p.iloc[0].reason.startswith("already have")


def test_plan_skips_inconsistent_books():
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", 0.8)])
    mk = markets_for("KXNCAAFGAME-26OCT10PSUOSU", "Ohio St.", "Penn St.")
    books = {mk[0]["ticker"]: Book(0.55, 0.56), mk[1]["ticker"]: Book(0.55, 0.56)}  # sums to 1.11
    p = ko.plan(games, ko.match_events(games, mk), books, set(), NOW, 2026, 6, bankroll=BANKROLL)
    assert p.iloc[0].reason == "the two books disagree"


def many(n, wp=0.72, bid=0.55):
    teams = [(f"Home{i}", f"Away{i}") for i in range(n)]
    games = pd.DataFrame([game(i, h, a, wp) for i, (h, a) in enumerate(teams)])
    mk, books = [], {}
    for i, (h, a) in enumerate(teams):
        m = markets_for(f"KXNCAAFGAME-26OCT10G{i:02d}", h, a)
        mk += m
        books[m[0]["ticker"]], books[m[1]["ticker"]] = book_pair(bid)
    return games, ko.match_events(games, mk), books


def test_caps_on_orders_total_cost_and_balance():
    games, matches, books = many(12)
    p = ko.plan(games, matches, books, set(), NOW, 2026, 6, bankroll=BANKROLL, max_orders=3)
    assert (p.decision == "order").sum() == 3 and (p.reason == "order count cap reached").sum() == 9

    p = ko.plan(games, matches, books, set(), NOW, 2026, 6, bankroll=BANKROLL, max_total_cost=6.0)
    assert p[p.decision == "order"].cost.sum() <= 6.0 + 1e-9

    p = ko.plan(games, matches, books, set(), NOW, 2026, 6, bankroll=BANKROLL, balance=0.0)
    assert (p.decision == "order").sum() == 0 and (p.reason == "not enough balance").all()


def test_caps_go_to_the_biggest_edges_first():
    # five games, RAPPA's disagreement growing down the slate; room for two orders
    teams = [(f"Home{i}", f"Away{i}") for i in range(5)]
    games = pd.DataFrame([game(i, h, a, 0.62 + 0.02 * i) for i, (h, a) in enumerate(teams)])
    mk, books = [], {}
    for i, (h, a) in enumerate(teams):
        m = markets_for(f"KXNCAAFGAME-26OCT10G{i:02d}", h, a)
        mk += m
        books[m[0]["ticker"]], books[m[1]["ticker"]] = book_pair(0.55)
    p = ko.plan(games, ko.match_events(games, mk), books, set(), NOW, 2026, 6, bankroll=BANKROLL, max_orders=2)
    assert sorted(p[p.decision == "order"].game_id) == [3, 4]
    assert p.set_index("game_id").edge.dropna().is_monotonic_increasing  # game 0 is too close to trade


def test_per_order_cap_shrinks_the_count():
    p, _ = setup(wp=0.72, home_bid=0.55, bankroll=10_000, max_order_cost=2.0)
    row = p.iloc[0]
    assert row["count"] == math.floor(2.0 / (row.price + maker_fee(row.price))) and row.cost <= 2.0


EVENT = "KXNCAAFGAME-26OCT10PSUOSU"


def test_a_held_game_is_topped_up_to_its_kelly_size():
    fresh, mk = setup(wp=0.72, home_bid=0.55, bankroll=100)
    target = int(fresh.iloc[0]["count"])
    held = {EVENT: {"ticker": mk[0]["ticker"], "contracts": 2.0, "cost": 1.10}}
    p, _ = setup(wp=0.72, home_bid=0.55, bankroll=100, held=held)
    row = p.iloc[0]
    assert target > 2 and row.decision == "order" and row.ticker == mk[0]["ticker"]
    assert row.topup and row.held == 2 and row["count"] == target - 2

    full = {EVENT: {"ticker": mk[0]["ticker"], "contracts": float(target), "cost": 5.0}}
    p, _ = setup(wp=0.72, home_bid=0.55, bankroll=100, held=full)
    assert p.iloc[0].decision == "skip" and p.iloc[0].reason.startswith(f"holds {target}")


def test_a_top_up_never_hedges_and_respects_the_game_cap():
    _, mk = setup(wp=0.72, home_bid=0.55)
    other = {EVENT: {"ticker": mk[1]["ticker"], "contracts": 2.0, "cost": 0.9}}  # holds Penn State; edge is on Ohio St.
    p, _ = setup(wp=0.72, home_bid=0.55, bankroll=100, held=other)
    assert p.iloc[0].decision == "skip" and p.iloc[0].reason.startswith("the edge is now on the other team")
    capped = {EVENT: {"ticker": mk[0]["ticker"], "contracts": 2.0, "cost": ko.MAX_GAME_SHARE * 100}}
    p, _ = setup(wp=0.72, home_bid=0.55, bankroll=100, held=capped)
    assert p.iloc[0].decision == "skip" and p.iloc[0].reason == "holds the most one game may have"


def test_top_ups_take_no_game_slot_but_compete_for_the_money():
    games, matches, books = many(3)
    held = {matches[0]["event"]: {"ticker": matches[0]["home"]["ticker"], "contracts": 1.0, "cost": 0.55}}
    p = ko.plan(games, matches, books, set(), NOW, 2026, 6, bankroll=BANKROLL, max_orders=0, held=held)
    assert list(p[p.decision == "order"].game_id) == [0]  # no games left, but the held game still grows
    # the budget goes to the biggest edge, top-up or not
    games = pd.DataFrame([game(0, "Home0", "Away0", 0.66), game(1, "Home1", "Away1", 0.74)])
    matches = ko.match_events(games, sum((markets_for(f"KXNCAAFGAME-26OCT10G{i:02d}", f"Home{i}", f"Away{i}")
                                          for i in range(2)), []))
    for small, big in ((0, 1), (1, 0)):
        held = {matches[small]["event"]: {"ticker": matches[small]["home"]["ticker"], "contracts": 1.0, "cost": 0.55}}
        if big == 0:  # the held game has the bigger edge
            games.loc[0, "rappa_wp"], games.loc[1, "rappa_wp"] = 0.74, 0.66
        p = ko.plan(games, matches, books, set(), NOW, 2026, 6, bankroll=BANKROLL, held=held, max_total_cost=6.0,
                    kelly=1.0)  # room for one
        assert list(p[p.decision == "order"].game_id) == [big]


def test_client_order_ids_are_new_for_every_run():
    a = ko.client_order_id("T-1", 2026, 6, NOW)
    assert a == ko.client_order_id("T-1", 2026, 6, NOW)                       # the same run
    assert a != ko.client_order_id("T-1", 2026, 6, NOW + timedelta(seconds=1))  # a rerun
    assert a != ko.client_order_id("T-1", 2026, 7, NOW)


def test_order_body_is_a_post_only_expiring_yes_bid():
    p, _ = setup(wp=0.72, home_bid=0.55)
    body = ko.order_body(p.iloc[0])
    assert body["side"] == "bid" and body["post_only"] is True
    assert body["time_in_force"] == "good_till_canceled"
    assert body["expiration_time"] == int(datetime(2026, 10, 10, 19, 0, tzinfo=timezone.utc).timestamp())
    assert body["price"] == f"{p.iloc[0].price:.4f}" and body["count"] == f"{int(p.iloc[0]['count'])}.00"
    assert body["self_trade_prevention_type"] == "maker"


# ---------- execution ----------

class FakeKalshi:
    """Stands in for betting.kalshi.Client: canned markets and books, and a small exchange that keeps the
    orders it is sent, cancels them, and can be told an order filled or a market settled."""

    def __init__(self, markets, books, balance="100.00", resting=(), positions=(), post_status=None, read_only=True):
        self.markets, self.books, self.balance = markets, books, balance
        self.resting, self.positions = list(resting), list(positions)
        self.post_status, self.read_only, self.signed = post_status or {}, read_only, True
        self.posts, self.deletes, self.orders, self.fills, self.settlements = [], [], {}, [], []
        self.results = {}
        self.sleep = lambda seconds: None
        self.lag = 0  # how many order look-ups 404 before the exchange's reads catch up

    def get(self, path, **params):
        if path == "/portfolio/balance":
            return {"balance_dollars": self.balance}
        if path.endswith("/orderbook"):
            b = self.books[path.split("/")[2]]
            return {"orderbook_fp": {"yes_dollars": [[f"{b.bid:.4f}", "100"]], "no_dollars": [[f"{1 - b.ask:.4f}", "100"]]}}
        if path == "/portfolio/orders/queue_positions":
            return {"queue_positions": [{"order_id": oid, "market_ticker": o["ticker"], "queue_position_fp": "7.00"}
                                        for oid, o in self.orders.items() if o["status"] == "resting"]}
        if path.startswith("/portfolio/orders/"):
            if self.lag:
                self.lag -= 1
                from betting.kalshi import KalshiError
                raise KalshiError(404, {"error": {"code": "not_found"}}, "GET", path)
            return {"order": self.orders[path.rsplit("/", 1)[1]]}
        if path.startswith("/markets/"):
            t = path.split("/")[2]
            b = self.books.get(t)
            return {"market": {"ticker": t, "status": "finalized" if t in self.results else "active",
                               "result": self.results.get(t, ""), "yes_bid_dollars": f"{b.bid:.4f}" if b else None,
                               "yes_ask_dollars": f"{b.ask:.4f}" if b else None}}
        raise AssertionError(path)

    def pages(self, path, key, **params):
        if path == "/portfolio/orders":
            mine = list(self.orders.values())
            if params.get("status"):
                mine = [o for o in mine if o["status"] == params["status"]]
            return self.resting + mine
        return {"/markets": self.markets, "/portfolio/positions": self.positions, "/historical/orders": [],
                "/portfolio/fills": self.fills, "/portfolio/settlements": self.settlements}[path]

    def post(self, path, body):
        assert not self.read_only, "post on a read-only client"
        self.posts.append((path, body))
        status = self.post_status.get(body["ticker"])
        if any(o["client_order_id"] == body["client_order_id"] for o in self.orders.values()):
            status = 409  # Kalshi: order_already_exists, for any id it has seen, cancelled orders' too
        if status:
            from betting.kalshi import KalshiError
            raise KalshiError(status, "err", "POST", path)
        oid = f"oid-{len(self.posts)}"
        self.orders[oid] = {"order_id": oid, "client_order_id": body["client_order_id"], "ticker": body["ticker"],
                            "status": "resting", "yes_price_dollars": body["price"], "initial_count_fp": body["count"],
                            "remaining_count_fp": body["count"], "fill_count_fp": "0.00", "book_side": "bid",
                            "maker_fill_cost_dollars": "0", "taker_fill_cost_dollars": "0", "maker_fees_dollars": "0",
                            "taker_fees_dollars": "0", "expiration_time": body["expiration_time"]}
        return {"order_id": oid, "fill_count": "0.00", "remaining_count": body["count"]}

    def delete(self, path, **params):
        assert not self.read_only
        self.deletes.append((path, params))
        oid = path.rsplit("/", 1)[1]
        if oid in self.orders:
            self.orders[oid]["status"] = "canceled"
            self.orders[oid]["remaining_count_fp"] = "0.00"
        return {"order_id": oid}

    def fill(self, oid, n):
        """n contracts of a resting order fill as a maker."""
        o = self.orders[oid]
        price = float(o["yes_price_dollars"])
        fee = round(0.0175 * price * (1 - price) * n, 6)
        o.update(fill_count_fp=f"{n:.2f}", remaining_count_fp=f"{float(o['initial_count_fp']) - n:.2f}",
                 maker_fill_cost_dollars=f"{price * n:.4f}", maker_fees_dollars=f"{fee:.6f}",
                 status="executed" if float(o["initial_count_fp"]) == n else "resting")
        self.fills.append({"fill_id": f"f-{oid}", "order_id": oid, "ticker": o["ticker"], "count_fp": f"{n:.2f}",
                           "yes_price_dollars": o["yes_price_dollars"], "is_taker": False, "fee_cost": f"{fee:.6f}"})
        self.positions.append({"ticker": o["ticker"], "position_fp": f"{n:.2f}"})

    def settle(self, ticker, result):
        self.results[ticker] = result
        self.settlements.append({"ticker": ticker, "market_result": result, "settled_time": "2026-10-11T03:00:00Z"})


@pytest.fixture
def week(tmp_path):
    out = tmp_path / "cfb"
    (out / "2026").mkdir(parents=True)
    rows = [game(10 + i, h, a, wp) | {"week": 6} for i, (h, a, wp) in
            enumerate([("Ohio State", "Penn State", 0.72), ("Georgia", "Alabama", 0.52), ("Texas", "Oklahoma", 0.38)])]
    pd.DataFrame(rows).to_csv(out / "2026" / "parts_wk06.csv", index=False)
    mk = markets_for("KXNCAAFGAME-26OCT10PSUOSU", "Ohio St.", "Penn St.", "OSU", "PSU") + \
        markets_for("KXNCAAFGAME-26OCT10ALAUGA", "Georgia", "Alabama", "UGA", "ALA") + \
        markets_for("KXNCAAFGAME-26OCT10OUTEX", "Texas", "Oklahoma", "TEX", "OU")
    books = {}
    for i in range(0, 6, 2):
        books[mk[i]["ticker"]], books[mk[i + 1]["ticker"]] = book_pair(0.55)
    return out, mk, books


def run(out, client, *args, confirm=lambda prompt: "no"):
    return ko.main(["--season", "2026", "--week", "6", "--out", str(out), *args], client=client, now=NOW,
                   confirm=confirm)


def test_dry_run_sends_nothing_and_logs_the_plan(week):
    out, mk, books = week
    client = FakeKalshi(mk, books)
    assert run(out, client) == 0
    assert client.posts == [] and client.deletes == []
    logged = pd.read_csv(out.parent / "kalshi" / "orders" / "2026_wk06.csv")
    assert set(logged["mode"]) == {"plan"}
    assert (logged.decision == "order").sum() == 2  # Ohio State and Oklahoma; Georgia-Alabama is too close


def test_a_week_played_out_has_nothing_to_plan(week, capsys):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    sunday = datetime(2026, 10, 11, 17, 0, tzinfo=timezone.utc)  # every fixture game kicked off Saturday
    assert ko.main(["--season", "2026", "--week", "6", "--out", str(out), "--execute", "--yes"], client=client,
                   now=sunday) == 0
    assert client.posts == [] and client.deletes == []
    assert "every game of week 6 has kicked off" in capsys.readouterr().out
    assert not (out.parent / "kalshi" / "orders" / "2026_wk06.csv").exists()


def test_execute_needs_the_typed_confirmation(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    assert run(out, client, "--execute", confirm=lambda prompt: "yes") == 1
    assert client.posts == []
    assert run(out, client, "--execute", confirm=lambda prompt: "PLACE") == 0
    assert len(client.posts) == 2
    for path, body in client.posts:
        assert path == "/portfolio/events/orders" and body["post_only"] is True and body["side"] == "bid"
    logged = pd.read_csv(out.parent / "kalshi" / "orders" / "2026_wk06.csv")
    assert set(logged["mode"]) == {"declined", "execute"}
    assert set(logged[logged["mode"] == "execute"].status.dropna()) >= {"placed"}


def test_execute_stops_at_an_unexpected_error_and_skips_duplicates(week):
    out, mk, books = week
    tex = "KXNCAAFGAME-26OCT10OUTEX-OU"
    client = FakeKalshi(mk, books, read_only=False, post_status={mk[0]["ticker"]: 400})
    assert run(out, client, "--execute", "--yes") == 1
    assert len(client.posts) == 1  # the first failed; the second was never sent

    client = FakeKalshi(mk, books, read_only=False, post_status={mk[0]["ticker"]: 409})
    assert run(out, client, "--execute", "--yes") == 0
    assert [b["ticker"] for _, b in client.posts] == [mk[0]["ticker"], tex]


def test_existing_orders_and_positions_block_their_game(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False,
                        resting=[{"ticker": mk[0]["ticker"], "order_id": "x", "client_order_id": "y"}],
                        positions=[{"ticker": "KXNCAAFGAME-26OCT10OUTEX-TEX", "position_fp": "3.00"}])
    run(out, client, "--execute", "--yes")
    assert client.posts == []


def test_zero_balance_places_nothing(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, balance="0.0000", read_only=False)
    run(out, client, "--execute", "--yes")
    assert client.posts == []


def test_cancel_only_touches_orders_this_module_placed(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    run(out, client, "--execute", "--yes")
    theirs = {"ticker": mk[2]["ticker"], "order_id": "manual", "client_order_id": "placed-by-hand", "status": "resting"}
    client.resting = [theirs]
    assert run(out, client, "--cancel", "--yes") == 0
    cancelled = {p.rsplit("/", 1)[1] for p, _ in client.deletes}
    assert cancelled == {"oid-1", "oid-2"}


def test_the_real_client_is_read_only_unless_executing(week, monkeypatch, tmp_path):
    out, mk, books = week
    key = ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    env = tmp_path / ".env"
    env.write_text(f'KALSHI_ID="id"\nKALSHI_PRIVATE_KEY="{key}"\n')
    made = []

    class Spy(Client):
        def __init__(self, *a, **kw):
            made.append(kw.get("read_only"))
            raise SystemExit(0)  # stop before any request

    monkeypatch.setattr(ko, "Client", Spy)
    for args, expect in (([], True), (["--status"], True), (["--execute"], False), (["--cancel"], False)):
        with pytest.raises(SystemExit):
            ko.main(["--season", "2026", "--week", "6", "--out", str(out), "--env", str(env), *args], now=NOW)
        assert made[-1] is expect


# ---------- smoke test and tracking ----------

def test_smoke_places_checks_and_cancels_one_far_bid(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    assert run(out, client, "--smoke", confirm=lambda prompt: "PLACE") == 0
    assert len(client.posts) == 1
    _, body = client.posts[0]
    assert body["count"] == "1.00" and body["price"] == "0.0100" and body["post_only"] is True
    assert body["expiration_time"] - NOW.timestamp() <= 600
    assert len(client.deletes) == 1 and list(client.orders.values())[0]["status"] == "canceled"
    log_ = pd.read_csv(out.parent / "kalshi" / "orders" / "smoke.csv")
    assert list(log_.step) == ["place", "rests", "cancel", "cancelled"] and log_.ok.all()


def test_smoke_needs_confirmation_and_cancels_even_when_the_check_fails(week, monkeypatch):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    assert run(out, client, "--smoke", confirm=lambda prompt: "") == 1 and client.posts == []

    real_get = client.get
    def bad_get(path, **params):  # the order comes back looking wrong
        r = real_get(path, **params)
        if path.startswith("/portfolio/orders/") and r["order"]["status"] == "resting":
            r = {"order": {**r["order"], "yes_price_dollars": "0.5000"}}
        return r
    client.get = bad_get
    assert run(out, client, "--smoke", "--yes") == 1
    assert len(client.deletes) == 1  # still cancelled


def test_track_records_orders_fills_and_profit(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    run(out, client, "--execute", "--yes")
    folder = out.parent / "kalshi" / "orders"
    orders = pd.read_csv(folder / "orders.csv")
    assert len(orders) == 2 and set(orders.status) == {"resting"}
    for col in ("rappa_home", "fair", "edge", "bid", "ask", "bid_size", "plan_price", "kickoff", "game"):
        assert col in orders and orders[col].notna().all(), col

    first, second = sorted(client.orders)
    client.fill(first, 5)
    client.fill(second, 2)
    t1, t2 = client.orders[first]["ticker"], client.orders[second]["ticker"]
    client.settle(t1, "yes")
    client.settle(t2, "no")
    ko.main(["--season", "2026", "--week", "6", "--out", str(out), "--track"], client=client, now=NOW)
    ko.main(["--season", "2026", "--week", "6", "--out", str(out), "--track"], client=client, now=NOW)

    orders = pd.read_csv(folder / "orders.csv").set_index("ticker")
    p1 = float(client.orders[first]["yes_price_dollars"])
    p2 = float(client.orders[second]["yes_price_dollars"])
    fee = lambda p, n: round(0.0175 * p * (1 - p) * n, 6)
    assert orders.loc[t1, "pnl"] == pytest.approx(5 - 5 * p1 - fee(p1, 5), abs=1e-4)
    assert orders.loc[t2, "pnl"] == pytest.approx(-2 * p2 - fee(p2, 2), abs=1e-4)
    assert orders.loc[t1, "avg_fill_price"] == pytest.approx(p1)
    fills = pd.read_csv(folder / "fills.csv")
    assert len(fills) == 2 and not fills.is_taker.any()          # tracked twice, stored once
    assert len(pd.read_csv(folder / "settlements.csv")) == 2
    snaps = pd.read_csv(folder / "order_snapshots.csv")
    assert len(snaps) == 2 * 3                                   # execute, then two tracks
    assert set(snaps.columns) >= {"snapshot_at", "status", "fill_count", "market_bid", "market_ask", "position"}


def test_track_prints_live_and_filled_orders_only_and_fits_the_terminal(week, capsys, monkeypatch):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    run(out, client, "--execute", "--yes")
    by_team = {o["ticker"].rsplit("-", 1)[1]: oid for oid, o in client.orders.items()}
    client.delete(f"/portfolio/events/orders/{by_team['OSU']}")
    client.fill(by_team["OU"], 2)
    capsys.readouterr()
    monkeypatch.setenv("COLUMNS", "70")
    ko.track(client, out, NOW)
    printed = capsys.readouterr().out
    assert "Oklahoma" in printed and "Ohio State" not in printed       # the cancelled order is not listed
    assert "1 cancelled or expired without a fill" in " ".join(printed.split())
    assert max(map(len, printed.splitlines())) <= 70
    assert len(pd.read_csv(out.parent / "kalshi" / "orders" / "orders.csv")) == 2  # but it is still saved


def test_track_ignores_orders_this_module_did_not_place(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False,
                        resting=[{"ticker": mk[0]["ticker"], "order_id": "manual", "client_order_id": "by-hand",
                                  "status": "resting"}])
    client.resting = []  # let the run place
    run(out, client, "--execute", "--yes")
    client.resting = [{"ticker": mk[0]["ticker"], "order_id": "manual", "client_order_id": "by-hand", "status": "resting"}]
    latest = ko.track(client, out, NOW)
    assert "by-hand" not in set(latest.index)


def test_smoke_waits_for_the_exchange_to_see_the_order(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    client.lag = 3  # Kalshi answered 404 for an order placed a quarter second before
    assert run(out, client, "--smoke", "--yes") == 0
    assert list(client.orders.values())[0]["status"] == "canceled"


def test_no_keyboard_means_not_confirmed(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    def eof(prompt):
        raise EOFError
    assert run(out, client, "--execute", confirm=eof) == 1
    assert run(out, client, "--smoke", confirm=eof) == 1
    assert client.posts == []



# ---------- the queue ----------

def queue_setup(bid_size, volume_24h, width=0.02, wp=0.72):
    """One game, kickoff four days out (expiry 3.5 days after NOW... ~93.5 hours), home book with a given queue."""
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", wp)])
    mk = markets_for("KXNCAAFGAME-26OCT10PSUOSU", "Ohio St.", "Penn St.")
    for m in mk:
        m["volume_24h_fp"] = f"{volume_24h:.2f}"
    hb = Book(0.55, round(0.55 + width, 2), bid_size, 100)
    ab = Book(round(1 - 0.55 - width, 2), 0.45, 100, 100)
    p = ko.plan(games, ko.match_events(games, mk), {mk[0]["ticker"]: hb, mk[1]["ticker"]: ab}, set(), NOW, 2026, 6,
                bankroll=BANKROLL)
    hours = (datetime(2026, 10, 10, 19, 0, tzinfo=timezone.utc) - NOW).total_seconds() / 3600
    return p.iloc[0], 24 * hours  # the row, and hours to expiry as a multiple of a day's volume


def test_short_queue_joins_the_bid_even_with_room_to_improve():
    row, _ = queue_setup(bid_size=10, volume_24h=1000)
    assert row.decision == "order" and row.price == 0.55 and not row.improved and row.queue_ahead == 10


def test_crowded_bid_is_improved_by_a_cent_when_the_spread_allows():
    row, _ = queue_setup(bid_size=2000, volume_24h=1000)  # ~3,900 expected; 2,000 ahead > 10%
    assert row.decision == "order" and row.price == 0.56 and row.improved and row.queue_ahead == 0


def test_hopeless_queue_is_skipped_when_it_cannot_be_improved():
    row, _ = queue_setup(bid_size=500_000, volume_24h=60_000, width=0.01)  # like Southern Miss
    assert row.decision == "skip" and row.reason.startswith("queue too deep")
    row, _ = queue_setup(bid_size=6000, volume_24h=200, width=0.01)       # like Ohio
    assert row.decision == "skip" and row.reason.startswith("queue too deep")


def test_a_queue_within_the_expected_volume_is_kept():
    row, _ = queue_setup(bid_size=12_000, volume_24h=16_000, width=0.01)  # like Oklahoma
    assert row.decision == "order" and row.queue_ahead == 12_000


def test_no_volume_listed_means_no_queue_check():
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", 0.72)])
    mk = markets_for("KXNCAAFGAME-26OCT10PSUOSU", "Ohio St.", "Penn St.")
    hb, ab = book_pair(0.55)
    row = ko.plan(games, ko.match_events(games, mk), {mk[0]["ticker"]: hb, mk[1]["ticker"]: ab}, set(), NOW, 2026, 6,
                  bankroll=BANKROLL).iloc[0]
    assert row.decision == "order" and row.expected_volume is None


def test_track_saves_queue_positions(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    run(out, client, "--execute", "--yes")
    snaps = pd.read_csv(out.parent / "kalshi" / "orders" / "order_snapshots.csv")
    assert (snaps.queue_position == 7).all()
    orders = pd.read_csv(out.parent / "kalshi" / "orders" / "orders.csv")
    assert {"queue_ahead", "improved", "volume_24h", "expected_volume"} <= set(orders.columns)



# ---------- sizing ----------

def test_kelly_share():
    assert ko.kelly_share(0.6, 0.5) == pytest.approx(0.2)
    assert ko.kelly_share(0.5, 0.5) == 0 and ko.kelly_share(0.4, 0.5) == 0
    # the Pittsburgh order of 2026-10-06: fair .667 at .62 plus fee
    c = 0.62 + maker_fee(0.62)
    assert ko.kelly_share(0.6672, c) == pytest.approx((0.6672 - c) / (1 - c))


def test_default_is_half_kelly():
    assert ko.KELLY_FRACTION == 0.5 and ko.parse_args([]).kelly == 0.5
    c = 0.5 + maker_fee(0.5)
    assert ko.contracts(0.6, c, 20, max_game_share=1)[1] == pytest.approx(0.5 * ko.kelly_share(0.6, c) * 20)


def test_contracts_kelly_fraction_whole_contracts():
    c = 0.5 + maker_fee(0.5)
    f = ko.kelly_share(0.6, c)
    n, stake = ko.contracts(0.6, c, 100)
    game_cap = ko.MAX_GAME_SHARE * 100
    assert stake == pytest.approx(ko.KELLY_FRACTION * f * 100) and n == math.floor(min(stake, game_cap) / c)
    assert ko.contracts(0.6, c, 100, max_order_cost=5)[0] == math.floor(min(stake, game_cap, 5) / c)  # fixed cap on top
    n, stake = ko.contracts(0.6, c, 100, kelly=1.0, max_game_share=1)
    assert stake == pytest.approx(f * 100) and n == math.floor(stake / c)  # --kelly 1 still sizes at full


def test_contracts_rounds_up_to_one_only_from_half_a_contract():
    c = 0.5 + maker_fee(0.5)
    f = ko.kelly_share(0.6, c)
    just_over = 0.5 * c / (ko.KELLY_FRACTION * f) * 1.01  # a bankroll whose Kelly stake is just over half a contract
    assert ko.contracts(0.6, c, just_over, max_game_share=1)[0] == 1
    assert ko.contracts(0.6, c, just_over * 0.9, max_game_share=1)[0] == 0


def test_contracts_caps():
    c = 0.5 + maker_fee(0.5)
    assert ko.contracts(0.9, c, 10_000, max_contracts=7)[0] == 7
    assert ko.contracts(0.9, c, 10_000, max_order_cost=3.0)[0] == math.floor(3.0 / c)
    assert ko.contracts(0.9, c, 20, max_game_share=0.10)[0] == math.floor(2.0 / c)
    assert ko.contracts(0.4, c, 10_000)[0] == 0  # no edge, no stake
    assert ko.contracts(0.9, c, 10_000)[0] == math.floor(ko.MAX_GAME_SHARE * 10_000 / c)  # no fixed contract cap


def test_sizes_and_the_run_cap_scale_with_the_bankroll():
    games, matches, books = many(12)
    plans = {b: ko.plan(games, matches, books, set(), NOW, 2026, 6, bankroll=b, kelly=1.0) for b in (50, 5000)}
    for bankroll, p in plans.items():
        orders = p[p.decision == "order"]
        assert len(orders) and orders.cost.sum() <= ko.MAX_TOTAL_SHARE * bankroll + 1e-9
        assert (orders.cost <= ko.MAX_GAME_SHARE * bankroll + 1e-9).all()
    small, big = (plans[b][plans[b].decision == "order"] for b in (50, 5000))
    assert big.cost.mean() == pytest.approx(100 * small.cost.mean(), rel=0.02)  # whole contracts round a little
    assert abs(len(big) - len(small)) <= 1                                        # one more or less fits the cap


def test_plan_sizes_on_the_bankroll_and_explains_tiny_stakes():
    big, _ = setup(wp=0.72, home_bid=0.55, bankroll=200)
    small, _ = setup(wp=0.72, home_bid=0.55, bankroll=10)
    tiny, _ = setup(wp=0.72, home_bid=0.55, bankroll=2)
    assert big.iloc[0]["count"] > small.iloc[0]["count"] == 1
    assert tiny.iloc[0].decision == "skip" and tiny.iloc[0].reason.startswith("Kelly stake")
    for col in ("kelly_share", "kelly_stake", "bankroll"):
        assert col in big


def test_money_committed_to_resting_orders_is_not_spent_twice(week):
    out, mk, books = week
    elsewhere = {"ticker": "KXNFLGAME-26OCT11AAABBB-AAA", "order_id": "x", "client_order_id": "z", "status": "resting",
                 "book_side": "bid", "yes_price_dollars": "0.5000", "remaining_count_fp": "19.00"}
    client = FakeKalshi(mk, books, balance="10.00", read_only=False, resting=[elsewhere])  # ~$9.58 committed
    run(out, client, "--execute", "--yes", "--bankroll", "100")
    assert sum(float(b["count"]) * float(b["price"]) for _, b in client.posts) <= 10.0 - 9.58 + 1e-9


def test_bankroll_counts_positions():
    class Acct:
        def pages(self, path, key, **params):
            if path == "/portfolio/orders":
                return [{"ticker": "T-1", "book_side": "ask", "yes_price_dollars": "0.7000", "remaining_count_fp": "10"}]
            return [{"ticker": "T-2", "position_fp": "4.00", "market_exposure_dollars": "2.4000"},
                    {"ticker": "T-3", "position_fp": "0.00", "market_exposure_dollars": "9.0000"}]
    taken, committed, exposure = ko.account(Acct(), ["T-1", "T-2"])
    assert taken == {"T"} and exposure == pytest.approx(2.4)
    assert committed == pytest.approx(10 * (0.3 + maker_fee(0.3)))  # selling YES at .70 is buying NO at .30


# ---------- cancelling one order ----------

def placed(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    run(out, client, "--execute", "--yes")
    teams = {b["ticker"]: b for _, b in client.posts}
    return out, client, teams


def test_cancel_one_order_by_team_name(week):
    out, client, teams = placed(week)
    assert len(teams) == 2
    assert run(out, client, "--cancel-order", "Ohio St.", "--yes") == 0
    cancelled = [o for o in client.orders.values() if o["status"] == "canceled"]
    assert len(cancelled) == 1 and cancelled[0]["ticker"].endswith("-OSU")
    assert sum(o["status"] == "resting" for o in client.orders.values()) == 1
    log_ = pd.read_csv(out.parent / "kalshi" / "orders" / "cancels.csv")
    assert list(log_.ticker) == [cancelled[0]["ticker"]] and log_.ok.all()


def test_cancel_one_order_by_ticker_or_id(week):
    out, client, teams = placed(week)
    oid = next(oid for oid, o in client.orders.items() if o["ticker"].endswith("-OU"))
    assert run(out, client, "--cancel-order", "outex-ou", "--yes") == 0  # part of the ticker, any case
    assert client.orders[oid]["status"] == "canceled"
    other = next(oid for oid, o in client.orders.items() if o["status"] == "resting")
    assert run(out, client, "--cancel-order", other, "--yes") == 0
    assert all(o["status"] == "canceled" for o in client.orders.values())


def test_cancel_order_refuses_ambiguous_or_unknown_names(week):
    out, client, teams = placed(week)
    assert run(out, client, "--cancel-order", "KXNCAAFGAME", "--yes") == 1      # matches both
    assert run(out, client, "--cancel-order", "Georgia", "--yes") == 1          # no order on that game
    assert run(out, client, "--cancel-order", "Ohio St.", "--cancel-order", "nope", "--yes") == 1  # one bad name: none
    assert client.deletes == []


def test_cancel_order_needs_confirmation_and_ignores_orders_we_did_not_place(week):
    out, client, teams = placed(week)
    assert run(out, client, "--cancel-order", "Ohio St.", confirm=lambda p: "yes") == 1
    assert client.deletes == []
    client.resting = [{"ticker": next(iter(teams)), "order_id": "manual", "client_order_id": "by-hand", "status": "resting"}]
    assert run(out, client, "--cancel", confirm=lambda p: "CANCEL") == 0
    assert "manual" not in {p.rsplit("/", 1)[1] for p, _ in client.deletes}


def test_cancel_takes_a_snapshot_after(week):
    out, client, teams = placed(week)
    before = len(pd.read_csv(out.parent / "kalshi" / "orders" / "order_snapshots.csv"))
    run(out, client, "--cancel-order", "Ohio St.", "--yes")
    snaps = pd.read_csv(out.parent / "kalshi" / "orders" / "order_snapshots.csv")
    assert len(snaps) == before + 2 and "canceled" in set(snaps.status)



def test_a_re_placed_order_is_tracked_apart_from_the_one_it_replaced(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    ko.main(["--season", "2026", "--week", "6", "--out", str(out), "--execute", "--yes"], client=client, now=NOW)
    run(out, client, "--cancel", "--yes")
    later = NOW + timedelta(minutes=10)
    ko.main(["--season", "2026", "--week", "6", "--out", str(out), "--execute", "--yes"], client=client, now=later)
    assert all(status != "duplicate" for status in pd.read_csv(out.parent / "kalshi" / "orders" / "2026_wk06.csv")
               .status.dropna())
    orders = pd.read_csv(out.parent / "kalshi" / "orders" / "orders.csv")
    assert len(orders) == 4 and orders.order_id.is_unique and orders.client_order_id.is_unique
    assert sorted(orders.status) == ["canceled", "canceled", "resting", "resting"]
    assert orders.groupby("ticker").size().eq(2).all()  # each market's cancelled order and its replacement


def test_a_rerun_with_orders_resting_places_nothing_new(week):
    out, mk, books = week
    client = FakeKalshi(mk, books, read_only=False)
    ko.main(["--season", "2026", "--week", "6", "--out", str(out), "--execute", "--yes"], client=client, now=NOW)
    n = len(client.posts)
    ko.main(["--season", "2026", "--week", "6", "--out", str(out), "--execute", "--yes"], client=client,
            now=NOW + timedelta(minutes=5))
    assert len(client.posts) == n  # the games are taken by their resting orders, whatever the ids
