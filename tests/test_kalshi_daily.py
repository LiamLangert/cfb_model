"""cfb.kalshi_daily: the window, the week's budget, the review of resting orders, and that nothing is sent
unless asked. No network."""

import base64
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from cfb import kalshi_daily as kd
from cfb import kalshi_orders as ko
from rappa.kalshi import Book, Client, KalshiError, maker_fee
from test_kalshi_orders import NOW as TUESDAY, FakeKalshi, book_pair, game, markets_for

FRIDAY = datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc)  # 4pm ET; the fixture's games are Saturday 3:30pm ET
KICK = datetime(2026, 10, 10, 19, 30, tzinfo=timezone.utc)
EVENT = "KXNCAAFGAME-26OCT10PSUOSU"


class DepthKalshi(FakeKalshi):
    """FakeKalshi whose books show our own resting bids, as Kalshi's do, and whose cancels can fail."""

    def __init__(self, *a, fail_delete=False, **kw):
        super().__init__(*a, **kw)
        self.fail_delete = fail_delete

    def get(self, path, **params):
        if path.endswith("/orderbook"):
            t = path.split("/")[2]
            b = self.books[t]
            yes = {round(b.bid, 2): b.bid_size or 100.0}
            for o in self.orders.values():
                if o["ticker"] == t and o["status"] == "resting":
                    p = round(float(o["yes_price_dollars"]), 2)
                    yes[p] = yes.get(p, 0.0) + float(o["remaining_count_fp"])
            return {"orderbook_fp": {"yes_dollars": [[f"{p:.4f}", f"{n:.2f}"] for p, n in sorted(yes.items())],
                                     "no_dollars": [[f"{1 - b.ask:.4f}", "100"]]}}
        return super().get(path, **params)

    def delete(self, path, **params):
        if self.fail_delete:
            raise KalshiError(500, "err", "DELETE", path)
        return super().delete(path, **params)


@pytest.fixture
def week(tmp_path):
    out = tmp_path / "cfb"
    (out / "2026").mkdir(parents=True)
    rows = [game(10 + i, h, a, wp) | {"week": 6} for i, (h, a, wp) in
            enumerate([("Ohio State", "Penn State", 0.72), ("Georgia", "Alabama", 0.52), ("Texas", "Oklahoma", 0.38)])]
    pd.DataFrame(rows).to_csv(out / "2026" / "parts_wk06.csv", index=False)
    mk = markets_for(EVENT, "Ohio St.", "Penn St.", "OSU", "PSU") + \
        markets_for("KXNCAAFGAME-26OCT10ALAUGA", "Georgia", "Alabama", "UGA", "ALA") + \
        markets_for("KXNCAAFGAME-26OCT10OUTEX", "Texas", "Oklahoma", "TEX", "OU")
    books = {}
    for i in range(0, 6, 2):
        books[mk[i]["ticker"]], books[mk[i + 1]["ticker"]] = book_pair(0.55)
    return out, mk, books


def daily(out, client, *args, now=FRIDAY, confirm=lambda prompt: "no", notify=None):
    notes = [] if notify is None else notify
    code = kd.main(["--season", "2026", "--week", "6", "--out", str(out), *args], client=client, now=now,
                   confirm=confirm, notify=lambda title, body: notes.append((title, body)))
    return code


def week_log(out):
    return pd.read_csv(out.parent / "kalshi" / "orders" / "2026_wk06.csv")


# ---------- the window ----------

def test_split_holds_orders_beyond_the_window_for_later():
    planned = pd.DataFrame([
        {"decision": "order", "kickoff": (FRIDAY + timedelta(hours=29)).isoformat(), "reason": ""},
        {"decision": "order", "kickoff": (FRIDAY + timedelta(hours=31)).isoformat(), "reason": ""},
        {"decision": "skip", "kickoff": (FRIDAY + timedelta(hours=40)).isoformat(), "reason": "run's total cost cap reached"},
        {"decision": "skip", "kickoff": None, "reason": "no Kalshi market matched"},
    ])
    s = kd.split(planned, FRIDAY)
    assert list(s.decision) == ["order", "later", "skip", "skip"]
    assert list(s.lead_hours[:3]) == [29.0, 31.0, 40.0] and pd.isna(s.lead_hours[3])
    assert s.reason[1].startswith("kicks off in 31h")
    assert s.reason[2] == "the week's budget is held for bigger edges"


def test_split_with_no_kickoffs_at_all():
    s = kd.split(pd.DataFrame([{"decision": "skip", "reason": "no Kalshi market matched"}]), FRIDAY)
    assert list(s.decision) == ["skip"]


def test_tuesday_plans_the_week_but_places_nothing(week):
    out, mk, books = week
    client = DepthKalshi(mk, books, read_only=False)
    notes = []
    assert daily(out, client, "--execute", "--yes", now=TUESDAY, notify=notes) == 0
    assert client.posts == [] and notes == []
    logged = week_log(out)
    assert (logged.decision == "later").sum() == 2 and (logged.decision == "order").sum() == 0
    assert set(logged.planner) == {"daily"} and logged.lead_hours.max() == pytest.approx(101.5)


def test_friday_dry_run_sends_nothing_and_notifies(week):
    out, mk, books = week
    client = DepthKalshi(mk, books)
    notes = []
    assert daily(out, client, "--notify", notify=notes) == 0
    assert client.posts == [] and client.deletes == []
    logged = week_log(out)
    assert set(logged["mode"]) == {"plan"} and (logged.decision == "order").sum() == 2
    assert logged[logged.decision == "order"].lead_hours.eq(23.5).all()
    assert len(notes) == 1 and notes[0][0] == "RAPPA Kalshi: 2 to place" and "--execute" in notes[0][1]


def test_no_notification_without_notify_or_with_nothing_to_do(week):
    out, mk, books = week
    notes = []
    daily(out, DepthKalshi(mk, books), notify=notes)
    daily(out, DepthKalshi(mk, books), "--notify", now=TUESDAY, notify=notes)
    assert notes == []


def test_a_week_played_out_has_nothing_to_plan_and_no_notification(week, capsys):
    out, mk, books = week
    client = DepthKalshi(mk, books, read_only=False)
    notes = []
    sunday = KICK + timedelta(hours=22)
    assert daily(out, client, "--notify", now=sunday, notify=notes) == 0
    assert daily(out, client, "--execute", "--yes", now=sunday, notify=notes) == 0
    assert client.posts == [] and client.deletes == [] and notes == []
    assert "every game of week 6 has kicked off" in capsys.readouterr().out
    assert not (out.parent / "kalshi" / "orders" / "2026_wk06.csv").exists()


def test_friday_execute_needs_the_typed_confirmation(week):
    out, mk, books = week
    client = DepthKalshi(mk, books, read_only=False)
    assert daily(out, client, "--execute", confirm=lambda prompt: "yes") == 1
    assert client.posts == []
    assert daily(out, client, "--execute", confirm=lambda prompt: "PLACE") == 0
    assert len(client.posts) == 2
    for _, body in client.posts:
        assert body["post_only"] is True and body["side"] == "bid"
        assert datetime.fromtimestamp(body["expiration_time"], timezone.utc) == KICK - ko.EXPIRE_BEFORE
    # tracked like cfb.kalshi_orders' own, from the same week log
    orders = pd.read_csv(out.parent / "kalshi" / "orders" / "orders.csv")
    assert len(orders) == 2 and set(orders.status) == {"resting"}
    assert set(ko.our_orders(out).index) == set(client.orders)


def test_a_second_run_keeps_what_rests_and_adds_nothing(week):
    out, mk, books = week
    client = DepthKalshi(mk, books, read_only=False)
    daily(out, client, "--execute", "--yes")
    assert daily(out, client, "--execute", "--yes", now=FRIDAY + timedelta(hours=1)) == 0
    assert len(client.posts) == 2 and client.deletes == []
    reviews = pd.read_csv(out.parent / "kalshi" / "orders" / "reviews.csv")
    assert set(reviews.decision) == {"keep"}


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
            raise SystemExit(0)

    monkeypatch.setattr(kd, "Client", Spy)
    for args, expect in (([], True), (["--notify"], True), (["--execute"], False)):
        with pytest.raises(SystemExit):
            kd.main(["--season", "2026", "--week", "6", "--out", str(out), "--env", str(env), *args], now=FRIDAY,
                    notify=lambda *a: None)
        assert made[-1] is expect


def test_a_failed_run_notifies_and_still_fails(week, tmp_path):
    out, mk, books = week
    notes = []
    with pytest.raises(SystemExit):
        kd.main(["--season", "2026", "--week", "6", "--out", str(out), "--env", str(tmp_path / "none"), "--notify"],
                now=FRIDAY, notify=lambda *a: notes.append(a))
    assert len(notes) == 1 and "failed" in notes[0][0]


# ---------- the week's budget ----------

def fill(ticker, n, price, fee=0.01, **kw):
    return {"ticker": ticker, "count_fp": f"{n:.2f}", "yes_price_dollars": f"{price:.4f}", "fee_cost": f"{fee}", **kw}


def test_week_spent_counts_this_weeks_fills_and_resting_orders():
    first, last = datetime(2026, 10, 6).date(), datetime(2026, 10, 10).date()
    fills = [fill(f"{EVENT}-OSU", 4, 0.60),                                 # $2.41
             fill("KXNCAAFGAME-26OCT07USMTROY-USM", 2, 0.30),               # $0.61, the Tuesday before
             fill("KXNCAAFGAME-26OCT03AAABBB-AAA", 9, 0.50),                # last week
             fill("KXNFLGAME-26OCT11AAABBB-AAA", 9, 0.50),                  # not college
             fill(f"{EVENT}-OSU", 1, 0.70, action="sell"),                  # a sale spends nothing
             fill("KXNCAAFGAME-26OCT10ALAUGA-UGA", 3, 0.25, side="no", no_price_dollars="0.75")]  # $2.26
    resting = [{"ticker": "KXNCAAFGAME-26OCT10OUTEX-OU", "book_side": "bid", "yes_price_dollars": "0.4000",
                "remaining_count_fp": "5.00"},
               {"ticker": "KXNCAAFGAME-26OCT17AAABBB-AAA", "book_side": "bid", "yes_price_dollars": "0.4000",
                "remaining_count_fp": "5.00"}]                              # next week
    spent, events = kd.week_spent(fills, resting, first, last)
    assert spent == pytest.approx(2.41 + 0.61 + 2.26 + 5 * (0.4 + maker_fee(0.4)))
    assert events == {EVENT, "KXNCAAFGAME-26OCT07USMTROY", "KXNCAAFGAME-26OCT10ALAUGA", "KXNCAAFGAME-26OCT10OUTEX"}


def test_what_the_week_already_spent_comes_off_its_budget(week):
    out, mk, books = week
    client = DepthKalshi(mk, books, read_only=False, balance="50.00")
    client.fills = [fill("KXNCAAFGAME-26OCT09AAABBB-AAA", 40, 0.60, fee=0.0)]  # $24 on Friday's game
    daily(out, client, "--execute", "--yes")  # of half the $50 bankroll
    assert client.posts == []
    logged = week_log(out)
    assert logged[logged.team.notna()].reason.eq("the week's budget is held for bigger edges").all()

    client = DepthKalshi(mk, books, read_only=False, balance="100.00")
    client.fills = [fill("KXNCAAFGAME-26OCT09AAABBB-AAA", 40, 0.60, fee=0.0)]
    daily(out, client, "--execute", "--yes")  # the same $24, of half a $100 bankroll: room left
    assert len(client.posts) >= 1
    client = DepthKalshi(mk, books, read_only=False, balance="100.00")
    client.fills = [fill("KXNCAAFGAME-26OCT09AAABBB-AAA", 40, 0.60, fee=0.0)]
    daily(out, client, "--execute", "--yes", "--max-week-share", "0.25")  # a quarter of $100: full again
    assert client.posts == []

    client = DepthKalshi(mk, books, read_only=False)
    client.fills = [fill(f"KXNCAAFGAME-26OCT09G{i}-A", 1, 0.5, fee=0.0) for i in range(9)]  # 9 games of 10
    daily(out, client, "--execute", "--yes")
    assert len(client.posts) == 1


def test_the_budget_goes_to_the_biggest_edge_of_the_week_not_the_first(tmp_path):
    out = tmp_path / "cfb"
    (out / "2026").mkdir(parents=True)
    pd.DataFrame([game(1, "Ohio State", "Penn State", 0.66, day="2026-10-09", time="19:00") | {"week": 6},
                  game(2, "Georgia", "Alabama", 0.74, day="2026-10-10", time="23:00") | {"week": 6}]
                 ).to_csv(out / "2026" / "parts_wk06.csv", index=False)
    mk = markets_for("KXNCAAFGAME-26OCT09PSUOSU", "Ohio St.", "Penn St.") + \
        markets_for("KXNCAAFGAME-26OCT10ALAUGA", "Georgia", "Alabama")
    books = dict(zip([m["ticker"] for m in mk], book_pair(0.55) + book_pair(0.55)))
    client = DepthKalshi(mk, books, read_only=False)
    daily(out, client, "--execute", "--yes", "--max-week-cost", "12", "--kelly", "1")
    assert client.posts == []  # Friday's game is in the window, but Saturday night's edge is bigger
    logged = week_log(out).set_index("game_id")
    assert logged.loc[2].decision == "later" and logged.loc[1].reason == "the week's budget is held for bigger edges"

    client = DepthKalshi(mk, books, read_only=False)
    daily(out, client, "--execute", "--yes", "--max-week-cost", "25", "--kelly", "1")
    assert [b["ticker"] for _, b in client.posts] == [mk[0]["ticker"]]


# ---------- the review ----------

def payload(levels, ask):
    """An orderbook response: YES bids {price: size} and the YES ask."""
    return {"orderbook_fp": {"yes_dollars": [[f"{p:.4f}", f"{n:.2f}"] for p, n in sorted(levels.items())],
                             "no_dollars": [[f"{1 - ask:.4f}", "100"]]}}


def mirror(levels, ask):
    """The other team's book for a home book with this best bid and ask."""
    bid = max(levels)
    return payload({round(1 - ask, 2): 100}, round(1 - bid, 2))


def reviewed(wp, levels, ask, price=0.55, left=5, now=FRIDAY, **kw):
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", wp)])
    mk = markets_for(EVENT, "Ohio St.", "Penn St.", "OSU", "PSU")
    matches = ko.match_events(games, mk)
    order = {"order_id": "o1", "client_order_id": "c1", "ticker": mk[0]["ticker"], "status": "resting",
             "book_side": "bid", "yes_price_dollars": f"{price:.4f}", "remaining_count_fp": f"{left:.2f}"}
    payloads = {mk[0]["ticker"]: payload(levels, ask), mk[1]["ticker"]: mirror(levels, ask)}
    reviews, replacements = kd.review([order], games, matches, payloads, now, 2026, 6, 100, **kw)
    return reviews.iloc[0], replacements


def test_a_position_is_topped_up_on_the_same_team(week):
    out, mk, books = week
    osu = mk[0]["ticker"]
    client = DepthKalshi(mk, books, read_only=False,
                         positions=[{"ticker": osu, "position_fp": "2.00", "market_exposure_dollars": "1.1000"}])
    client.fills = [fill(osu, 2, 0.55, fee=0.0)]
    daily(out, client, "--execute", "--yes")
    logged = week_log(out)
    row = logged[(logged["mode"] == "execute") & (logged.ticker == osu)].iloc[0]
    per = row.price + maker_fee(row.price)
    assert row.decision == "order" and row.topup and row.held == 2
    assert row["count"] == ko.contracts(row.fair, per, row.bankroll)[0] - 2
    assert [b["count"] for _, b in client.posts if b["ticker"] == osu] == [f"{int(row['count'])}.00"]


def test_odd_holdings_are_left_alone(week):
    out, mk, books = week
    client = DepthKalshi(mk, books, read_only=False,
                         positions=[{"ticker": mk[0]["ticker"], "position_fp": "-3.00", "market_exposure_dollars": "1.35"}])
    daily(out, client, "--execute", "--yes")
    assert not any(b["ticker"].startswith(EVENT) for _, b in client.posts)
    logged = week_log(out)
    assert logged[logged.event == EVENT].reason.eq("held on both teams or as NO contracts").all()


def test_holdings_add_positions_and_resting_bids_per_game():
    held, odd = kd.holdings(
        [{"ticker": f"{EVENT}-OSU", "book_side": "bid", "yes_price_dollars": "0.5500", "remaining_count_fp": "3.00"},
         {"ticker": "KXNCAAFGAME-26OCT10ALAUGA-UGA", "book_side": "bid", "yes_price_dollars": "0.40",
          "remaining_count_fp": "1.00"}],
        [{"ticker": f"{EVENT}-OSU", "position_fp": "2.00", "market_exposure_dollars": "1.10"},
         {"ticker": "KXNCAAFGAME-26OCT10ALAUGA-ALA", "position_fp": "1.00", "market_exposure_dollars": "0.60"},
         {"ticker": "KXNCAAFGAME-26OCT10OUTEX-TEX", "position_fp": "0.00", "market_exposure_dollars": "0"}])
    assert odd == {"KXNCAAFGAME-26OCT10ALAUGA"}  # both teams
    assert held[EVENT]["contracts"] == 5 and held[EVENT]["cost"] == pytest.approx(1.10 + 3 * (0.55 + maker_fee(0.55)))
    assert set(held) == {EVENT}


def test_book_without_takes_our_bids_out():
    p = payload({0.55: 100, 0.56: 5}, 0.57)
    assert kd.book_without(p, {0.56: 5}) == Book(0.55, 0.57, 100, 100)
    assert kd.book_without(p, {0.55: 30}) == Book(0.56, 0.57, 5, 100)
    assert kd.book_without(p, None) == Book(0.56, 0.57, 5, 100)


def test_review_keeps_an_order_whose_edge_holds():
    row, rep = reviewed(0.72, {0.55: 105}, 0.56)
    assert row.decision == "keep" and rep.empty and row.bid_size == 100  # its own 5 contracts are not the market


def test_review_cancels_when_the_edge_is_gone():
    row, rep = reviewed(0.58, {0.50: 100, 0.52: 5}, 0.53, price=0.52)  # alone at .52; RAPPA no longer clears it
    assert row.decision == "cancel" and row.reason.startswith("edge gone") and rep.empty
    assert row.edge < kd.KEEP_EDGE
    row, _ = reviewed(0.58, {0.50: 100, 0.52: 5}, 0.53, price=0.52, keep_edge=0.005)
    assert row.decision == "keep"  # still above the looser threshold


def test_review_cancels_what_it_cannot_price_any_more():
    row, _ = reviewed(0.72, {0.48: 100, 0.55: 5}, 0.56)  # without us the book is .48-.56
    assert row.decision == "cancel" and row.reason.startswith("can't price it now: book too wide")


def test_review_replaces_an_outbid_order_at_the_new_bid():
    row, rep = reviewed(0.72, {0.55: 5, 0.56: 100}, 0.57, left=3)
    assert row.decision == "replace" and row.new_price == 0.56 and row.new_count == 3  # no more than it had left
    r = rep.iloc[0]
    assert r.replaces == "o1" and r.decision == "order" and r.price == 0.56 and r["count"] == 3
    assert r.edge >= ko.MIN_EDGE and datetime.fromisoformat(r.expires) == KICK - ko.EXPIRE_BEFORE
    assert r.client_order_id != "c1" and ko.order_body(r)["price"] == "0.5600"


def test_review_cancels_an_outbid_order_with_no_price_left():
    row, rep = reviewed(0.60, {0.55: 5, 0.56: 100}, 0.57)
    assert row.decision == "cancel" and "no price above the bid" in row.reason and rep.empty


def test_review_does_not_re_place_close_to_kickoff():
    row, rep = reviewed(0.72, {0.55: 5, 0.56: 100}, 0.57, now=KICK - timedelta(hours=1))
    assert row.decision == "cancel" and "too close to kickoff" in row.reason and rep.empty


def test_review_ignores_orders_on_other_weeks_games():
    games = pd.DataFrame([game(1, "Ohio State", "Penn State", 0.72)])
    mk = markets_for(EVENT, "Ohio St.", "Penn St.", "OSU", "PSU")
    other = {"order_id": "o9", "ticker": "KXNCAAFGAME-26OCT17AAABBB-AAA", "yes_price_dollars": "0.5000",
             "remaining_count_fp": "1.00"}
    reviews, rep = kd.review([other], games, ko.match_events(games, mk), {}, FRIDAY, 2026, 6, 100)
    assert reviews.empty and rep.empty


# ---------- cancelling and re-placing, end to end ----------

def test_an_outbid_order_is_cancelled_then_re_placed_and_tracked(week):
    out, mk, books = week
    client = DepthKalshi(mk, books, read_only=False)
    daily(out, client, "--execute", "--yes")
    osu = next(oid for oid, o in client.orders.items() if o["ticker"].endswith("-OSU"))
    assert client.orders[osu]["yes_price_dollars"] == "0.5500"
    books[mk[0]["ticker"]], books[mk[1]["ticker"]] = book_pair(0.56)  # someone bids .56 above us
    notes = []
    assert daily(out, client, "--notify", notify=notes) == 0  # the plan says so, and sends nothing
    assert client.deletes == [] and notes[0][0] == "RAPPA Kalshi: 1 to replace"

    assert daily(out, client, "--execute", "--yes") == 0
    assert [p.rsplit("/", 1)[1] for p, _ in client.deletes] == [osu]
    new = [b for _, b in client.posts[2:]]
    assert len(new) == 1 and new[0]["ticker"] == mk[0]["ticker"] and new[0]["price"] == "0.5600"
    assert client.orders[osu]["status"] == "canceled"
    cancels = pd.read_csv(out.parent / "kalshi" / "orders" / "cancels.csv")
    assert list(cancels.order_id) == [osu] and cancels.reason[0].startswith("outbid")
    tracked = pd.read_csv(out.parent / "kalshi" / "orders" / "orders.csv")
    assert len(tracked) == 3 and set(tracked.status) == {"resting", "canceled"}


def test_a_cancel_that_fails_stops_everything(week):
    out, mk, books = week
    client = DepthKalshi(mk, books, read_only=False)
    daily(out, client, "--execute", "--yes")
    books[mk[0]["ticker"]], books[mk[1]["ticker"]] = book_pair(0.56)
    client.fail_delete = True
    assert daily(out, client, "--execute", "--yes") == 1
    assert len(client.posts) == 2  # nothing re-placed


def test_a_declined_review_cancels_nothing(week):
    out, mk, books = week
    client = DepthKalshi(mk, books, read_only=False)
    daily(out, client, "--execute", "--yes")
    books[mk[0]["ticker"]], books[mk[1]["ticker"]] = book_pair(0.56)
    assert daily(out, client, "--execute", confirm=lambda prompt: "") == 1
    assert client.deletes == [] and len(client.posts) == 2


# ---------- predictions and the notification ----------

def write_games(out, kickoffs):
    pd.DataFrame([{"season": 2026, "week": w, "season_type": "regular", "start_date": k.isoformat(),
                   "completed": True, "home_classification": "fbs", "away_classification": "fbs"}
                  for w, k in kickoffs]).to_csv(out / "2026" / "games.csv", index=False)


def test_next_weeks_predictions_wait_for_the_pull(tmp_path, monkeypatch):
    out = tmp_path / "cfb"
    (out / "2026").mkdir(parents=True)
    built = []
    monkeypatch.setattr("cfb.week_parts.main", lambda argv: built.append(argv))
    future = datetime.now(timezone.utc) + timedelta(days=1)
    write_games(out, [(6, future - timedelta(days=3)), (6, future)])  # pulled before week 6 was over
    assert kd.ensure_parts(2026, 7, out) == "waiting" and built == []
    assert kd.main(["--season", "2026", "--week", "7", "--out", str(out)], client=object()) == 0

    write_games(out, [(6, datetime.now(timezone.utc) - timedelta(days=2))])  # pulled after
    assert kd.ensure_parts(2026, 7, out) == "built" and built == [["2026", "7", "--out", str(out)]]


def test_stale_predictions_are_rebuilt(tmp_path, monkeypatch):
    out = tmp_path / "cfb"
    (out / "2026").mkdir(parents=True)
    built = []
    monkeypatch.setattr("cfb.week_parts.main", lambda argv: built.append(argv))
    parts = out / "2026" / "parts_wk06.csv"
    parts.write_text("x\n")
    write_games(out, [(5, datetime(2026, 10, 3, tzinfo=timezone.utc))])
    import os
    os.utime(parts, (1, 1))  # older than games.csv
    assert kd.ensure_parts(2026, 6, out) == "built"
    os.utime(parts, None)
    os.utime(out / "2026" / "games.csv", (1, 1))
    assert kd.ensure_parts(2026, 6, out) == "ok" and len(built) == 1


def test_toast_hands_powershell_an_encoded_script():
    calls = []

    class Done:
        returncode, stderr = 0, b""

    kd.toast("RAPPA Kalshi: 1 to place", "Ohio St. 3 @ .55; it's here", run=lambda cmd, **kw: calls.append(cmd) or Done())
    cmd = calls[0]
    assert cmd[0] == "powershell.exe" and cmd[-2] == "-EncodedCommand"
    script = base64.b64decode(cmd[-1]).decode("utf-16-le")
    assert "'Ohio St. 3 @ .55; it''s here'" in script and "ToastText02" in script


def test_toast_never_breaks_the_run():
    def missing(cmd, **kw):
        raise FileNotFoundError("powershell.exe")
    kd.toast("t", "b", run=missing)  # logged, not raised
