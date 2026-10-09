"""Maker orders on Kalshi's college game markets, priced off RAPPA's win probability for the coming week.

For every FBS-vs-FBS game in cfb.week_parts' predictions for the week
(data/cfb/<season>/parts_wk<NN>.csv) not yet kicked off, this finds Kalshi's two winner markets
(series KXNCAAFGAME, one YES contract per team), reads both books, and
decides whether to rest a buy on one side.

Fair value is not RAPPA's probability itself. Next to Kalshi's Tuesday price,
RAPPA's walk-forward win probability got a weight of about 0.26 +/- 0.17 in
predicting the winner (935 FBS games, 2025-26), and the price drifted toward
it by about 5% of the gap by kickoff. So fair value moves the market's mid
WEIGHT of the way toward RAPPA, in log-odds:

    logit(fair) = logit(mid) + WEIGHT * (logit(rappa) - logit(mid))

An order is a post-only bid for YES on the team whose fair value clears the
price by MIN_EDGE after the maker fee:

    fair - price - maker_fee(price) >= MIN_EDGE

priced at the best bid or a cent above it, and always below the ask (a
post-only order that would cross is cancelled, never filled as a taker).
Below the bid it would fill only once the market moved through it, which
is mostly on news against the side bought. A maker's real cost is who fills it: a resting bid gets hit most by
someone who knows more (an injury, a starter out), and MIN_EDGE is the
cushion for that. The edge here has not been shown to be profitable; see
the numbers above.

Guards, each logged as the reason a game was skipped:

  - both books tight (MAX_WIDTH) and agreeing (their mids sum to within
    MAX_BOOK_GAP of 1), the price inside [MIN_PRICE, MAX_PRICE]
  - RAPPA no further than MAX_DISAGREE from the market: a gap that large is
    more likely news RAPPA lacks (CFBD lists no starters ahead of a game) than
    an edge
  - kickoff at least MIN_LEAD away, and a kickoff time listed; every order
    expires EXPIRE_BEFORE kickoff, so nothing rests into the game
  - no order already resting, and no position held, in the game's event
    (cfb.kalshi_daily tops such a game up to its Kelly size instead; see plan)
  - the queue: a cent above the best bid when the bid is already crowded
    (IMPROVE_SHARE of the volume expected before expiry), joining it
    otherwise; and no order with more than MAX_QUEUE_SHARE of that volume
    still ahead of it, which would fill only if the price fell through it
  - one order per game, sized at full Kelly on fair value (KELLY_FRACTION,
    MIN_CONTRACT_SHARE) and at most MAX_GAME_SHARE of the bankroll; the run
    capped at MAX_TOTAL_SHARE of the bankroll (fees included) and MAX_ORDERS
    games, and never more than the balance not already committed to resting
    orders, filled from the biggest edge down. The money caps are shares of
    the bankroll, so every size grows with the account

By default nothing is sent: the plan is printed and logged. --execute asks
for a typed confirmation (or --yes), then places the orders one at a time,
stopping at the first error it does not expect. Every run appends to
data/kalshi/orders/<season>_wk<NN>.csv. --status lists this week's resting
orders and positions; --cancel cancels the resting orders the log says this
module placed, and --cancel-order NAME just the one NAME picks out (an order
id, ticker, or part of a ticker or team name; it must match exactly one).
Both ask for a typed CANCEL, log to data/kalshi/orders/cancels.csv, and
take a --track snapshot after. --smoke places one 1-contract bid at 1c, checks it rests,
and cancels it, to test the write path (data/kalshi/orders/smoke.csv).

--track (and every --execute, once it has placed) saves what became of each
order this module placed, in data/kalshi/orders/:

    orders.csv           one row per order, its latest state next to the numbers
                         it was placed on: RAPPA, the book and its depth, the
                         queue ahead of it and the volume expected, fair value, edge; contracts filled, average fill price, fees,
                         the result and profit once settled, a mark to the market
                         until then
    order_snapshots.csv  every --track's state of every order and its market, with
                         the order's place in the queue while it rests
    fills.csv            each fill: price, count, fee, maker or taker
    settlements.csv      each market's result and payout

Run --track after the games (and whenever else) to keep it current.

    uv run --with cryptography -m cfb.kalshi_orders              # plan only
    uv run --with cryptography -m cfb.kalshi_orders --execute    # place, after confirming
    uv run --with cryptography -m cfb.kalshi_orders --cancel     # cancel ours (asks first)
    uv run --with cryptography -m cfb.kalshi_orders --cancel-order Ohio --cancel-order "Southern Miss"
    uv run --with cryptography -m cfb.kalshi_orders --smoke      # test the write path
    uv run --with cryptography -m cfb.kalshi_orders --track      # save what became of our orders

Run cfb.week_parts first so the week's predictions are current.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from cfb import kalshi_display as display

from rappa import kalshi
from rappa.kalshi import TICK, Client, KalshiError, maker_fee, parse_book, similarity, ticker_date

log = logging.getLogger(__name__)

SERIES = "KXNCAAFGAME"
ET = ZoneInfo("America/New_York")

WEIGHT = 0.25          # how far toward RAPPA fair value moves from the market's mid, in log-odds
MIN_EDGE = 0.02        # dollars a contract fair value must clear, after the maker fee
# The lowest an order rests below the best bid. Below the bid an order fills only once the market has
# moved through it, mostly on news against the side bought, so by default it joins or improves the bid.
MAX_BELOW_BID = 0.0
MAX_WIDTH = 0.03       # widest book (ask - bid) priced off
MAX_BOOK_GAP = 0.03    # how far the two teams' mids may sum from 1
MIN_PRICE, MAX_PRICE = 0.10, 0.90
MAX_DISAGREE = 0.20    # largest |RAPPA - market| in win probability still traded
MIN_LEAD = timedelta(hours=2)
EXPIRE_BEFORE = timedelta(minutes=30)
MIN_MATCH = 1.6        # summed name similarity (2 is a perfect match) to pair a Kalshi event with a game

# Sizing. Each order stakes KELLY_FRACTION of the Kelly share of the bankroll, taking fair value as the chance it
# wins: for a contract costing c (price plus maker fee) that wins with probability q, Kelly stakes (q - c) / (1 - c)
# of the bankroll. It is full Kelly by choice (2026-10-07; it was half): the most growth if fair value is right, and
# the biggest swings. Fair value is uncertain (RAPPA's weight next to the market is 0.31 +/- 0.17 on 2025-26), and
# if the true edge is half the modelled one, full Kelly on it is twice the true Kelly, which grows nothing; half Kelly
# keeps about three quarters of the growth with far smaller swings. The bankroll is the balance plus what open
# positions cost.
#   MIN_CONTRACT_SHARE  a Kelly stake of at least this share of one contract buys one (whole contracts only); less
#                       is skipped. At 0.5, a single contract is at most twice its Kelly stake.
#   MAX_GAME_SHARE      no more than this share of the bankroll on one game: the biggest bet
#   MAX_TOTAL_SHARE     no more than this share of the bankroll spent in a run (cfb.kalshi_daily: in a week)
#   MAX_ORDERS          games a run places on. A count, not money, so it stays put as money is added: more money
#                       makes each bet bigger, not the week's good games more
# The money caps are shares of the bankroll, so sizes scale with it. Until 2026-10-09 they were dollars ($10 an
# order, $25 a run, 100 contracts), set on a $50 bankroll, where the shares come to the same. --max-contracts,
# --max-order-cost and --max-total-cost still set fixed caps, off by default.
KELLY_FRACTION = 1.0
MIN_CONTRACT_SHARE = 0.5
MAX_GAME_SHARE = 0.20
MAX_TOTAL_SHARE = 0.50
MAX_ORDERS = 10

# The queue. Kalshi fills a price level first come, first served, so a bid behind a big queue fills only once the
# price trades through it, which is mostly on news against the side bought. Expected volume is the market's last
# 24 hours of trading scaled to the hours until the order expires.
#   IMPROVE_SHARE   with more than this share of the expected volume already bid at the best bid, bid a cent
#                   above it (when the spread leaves room and the edge still clears) to go first
#   MAX_QUEUE_SHARE with more than this share of it still ahead of the order, skip the game
IMPROVE_SHARE = 0.10
MAX_QUEUE_SHARE = 0.50

# client_order_ids are uuid5s of the ticker, season, week and the run's time: new for every placement. Kalshi
# refuses an id it has seen before (409 order_already_exists), cancelled orders' included, so an id fixed for the
# week left a cancelled market unplaceable. What stops a second order on a game is account(): any resting order
# or position in its event.
NAMESPACE = uuid.UUID("5d1f6a52-3a43-4b8e-9a43-6f1d0c2b7e11")


def logit(p):
    return math.log(p / (1 - p))


def inv_logit(x):
    return 1 / (1 + math.exp(-x))


def fair_value(mid, rappa, weight=WEIGHT):
    """The market's mid moved `weight` of the way toward RAPPA, in log-odds."""
    rappa = min(max(rappa, 1e-4), 1 - 1e-4)
    return inv_logit(logit(mid) + weight * (logit(rappa) - logit(mid)))


def cents(x):
    return round(x / TICK) * TICK


def best_price(fair, book, min_edge=MIN_EDGE, max_below_bid=MAX_BELOW_BID, improve=True):
    """The highest price at which a resting bid clears min_edge, or None.

    At most a cent above the best bid (only the bid itself without improve), and always below the ask: the
    order must rest as a maker.
    """
    if book.bid is None or book.ask is None:
        return None
    top = cents(min(book.bid + (TICK if improve else 0), book.ask - TICK))
    floor = cents(book.bid - max_below_bid)
    price = top
    while price >= floor - 1e-9:
        if price > 0 and fair - price - maker_fee(price) >= min_edge - 1e-9:
            return round(price, 2)
        price = cents(price - TICK)
    return None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def expected_volume(market, hours):
    """Contracts the market should trade before the order expires: its last 24 hours' volume scaled to the
    hours left. None when the listing has no volume, and then the queue is not checked."""
    v = _num(market.get("volume_24h_fp"))
    return None if v is None else v * max(hours, 0) / 24


def kickoff_utc(game):
    """The game's kickoff, from parts' Eastern gameday and gametime; None if the time is TBD."""
    if pd.isna(game.get("gametime")) or pd.isna(game.get("gameday")):
        return None
    local = datetime.fromisoformat(f"{game['gameday']}T{game['gametime']}").replace(tzinfo=ET)
    return local.astimezone(timezone.utc)


def match_events(games, markets, min_match=MIN_MATCH):
    """Pair each Kalshi event (two team markets) with the parts game it is, by date and team names.

    Returns {game_id: {"event": ..., "home": market, "away": market, "score": ...}}. An event matches
    only a game within a day of its ticker's date, and each game at most once, to its best event.
    """
    by_event = {}
    for m in markets:
        by_event.setdefault(m["event_ticker"], []).append(m)
    best = {}
    for event, ms in by_event.items():
        if len(ms) != 2:
            continue
        day = ticker_date(event)
        if day is None:
            continue
        a, b = ms
        for g in games.itertuples():
            gday = datetime.fromisoformat(g.gameday).date()
            if abs((gday - day).days) > 1:
                continue
            straight = similarity(a["yes_sub_title"], g.home_team) + similarity(b["yes_sub_title"], g.away_team)
            crossed = similarity(b["yes_sub_title"], g.home_team) + similarity(a["yes_sub_title"], g.away_team)
            score = max(straight, crossed)
            if score < min_match:
                continue
            home, away = (a, b) if straight >= crossed else (b, a)
            if g.game_id not in best or score > best[g.game_id]["score"]:
                best[g.game_id] = {"event": event, "home": home, "away": away, "score": score}
    # One event per game, and one game per event: keep an event only for the game it matches best.
    owner = {}
    for gid, hit in best.items():
        if hit["event"] not in owner or hit["score"] > best[owner[hit["event"]]]["score"]:
            owner[hit["event"]] = gid
    return {gid: hit for gid, hit in best.items() if owner[hit["event"]] == gid}


def client_order_id(ticker, season, week, run):
    return str(uuid.uuid5(NAMESPACE, f"rappa-cfb:{ticker}:{season}:{week}:{run.isoformat()}"))


def kelly_share(fair, cost):
    """Kelly's share of the bankroll for a contract costing `cost` that pays $1 with probability `fair`."""
    return max(0.0, (fair - cost) / (1 - cost))


def contracts(fair, cost, bankroll, kelly=KELLY_FRACTION, max_game_share=MAX_GAME_SHARE,
              max_contracts=None, max_order_cost=None):
    """Whole contracts to bid for, and the stake Kelly asked for in dollars. At most max_game_share of the
    bankroll; max_contracts and max_order_cost are fixed caps on top, none when None."""
    stake = kelly * kelly_share(fair, cost) * bankroll
    n = math.floor(stake / cost + 1e-9)
    if n == 0 and stake >= MIN_CONTRACT_SHARE * cost - 1e-9:
        n = 1
    dollars = max_game_share * bankroll if max_order_cost is None else min(max_order_cost, max_game_share * bankroll)
    n = min(n, math.floor(dollars / cost + 1e-9))
    return max(0, n if max_contracts is None else min(n, max_contracts)), stake


def plan(games, matches, books, busy_events, now, season, week, weight=WEIGHT, min_edge=MIN_EDGE,
         max_contracts=None, max_order_cost=None, max_total_cost=None, max_orders=MAX_ORDERS, balance=None,
         bankroll=None, kelly=KELLY_FRACTION, max_game_share=MAX_GAME_SHARE, max_total_share=MAX_TOTAL_SHARE,
         held=None):
    """One row per game: the order to place, or why there is none.

    games is the parts slate; matches is match_events'; books maps a ticker to its Book; busy_events
    are the event tickers to leave alone (a set, or a dict of event to the reason). balance is what new
    orders may spend; bankroll is what Kelly sizes against and the caps are shares of. The run spends at
    most max_total_cost dollars, max_total_share of the bankroll when None. Pure: nothing is sent.

    held maps an event already held to {"ticker", "contracts", "cost"}: the YES market held, its contracts
    (filled and resting) and what they cost or commit. Such a game is topped up instead of skipped: on the
    same team only (never a hedge), to its Kelly size at today's fair value and price, less what it holds,
    within max_game_share of the bankroll counting what it holds. A top-up ("topup", with "held") competes
    with new orders for the money by edge, and uses none of max_orders, which counts games.
    """
    if bankroll is None:
        raise ValueError("plan needs a bankroll to size against")
    if max_total_cost is None:
        max_total_cost = max_total_share * bankroll
    rows, sized = [], []
    for g in games.to_dict("records"):
        row = {"game_id": g["game_id"], "game": f"{g['away_team']} at {g['home_team']}", "rappa_home": g["rappa_wp"],
               "decision": "skip"}
        rows.append(row)
        hit = matches.get(g["game_id"])
        if hit is None:
            row["reason"] = "no Kalshi market matched"
            continue
        row["event"] = hit["event"]
        kick = kickoff_utc(g)
        if kick is None:
            row["reason"] = "kickoff time TBD"
            continue
        row["kickoff"] = kick.isoformat()
        if kick - now < MIN_LEAD:
            row["reason"] = "kickoff too soon"
            continue
        if hit["event"] in busy_events:
            row["reason"] = (busy_events[hit["event"]] if isinstance(busy_events, dict)
                             else "already have an order or position in this game")
            continue
        hold = (held or {}).get(hit["event"])
        hb, ab = books.get(hit["home"]["ticker"]), books.get(hit["away"]["ticker"])
        if hb is None or ab is None or hb.mid is None or ab.mid is None:
            row["reason"] = "a book is empty"
            continue
        row["mid_home"] = round(hb.mid, 4)
        if max(hb.width, ab.width) > MAX_WIDTH + 1e-9:
            row["reason"] = f"book too wide ({max(hb.width, ab.width):.2f})"
            continue
        if abs(hb.mid + ab.mid - 1) > MAX_BOOK_GAP + 1e-9:
            row["reason"] = "the two books disagree"
            continue
        market_home = (hb.mid + 1 - ab.mid) / 2
        if abs(g["rappa_wp"] - market_home) > MAX_DISAGREE + 1e-9:
            row["reason"] = f"RAPPA too far from the market ({g['rappa_wp'] - market_home:+.2f}), likely news"
            continue
        options, deep = [], []
        hours = (kick - EXPIRE_BEFORE - now).total_seconds() / 3600
        for side, market, book, p in (("home", hit["home"], hb, g["rappa_wp"]),
                                      ("away", hit["away"], ab, 1 - g["rappa_wp"])):
            team_mid = book.mid
            fair = fair_value(team_mid, p, weight)
            flow = expected_volume(market, hours)
            improve = flow is None or book.bid_size > IMPROVE_SHARE * flow
            price = best_price(fair, book, min_edge, improve=improve)
            if price is None or not MIN_PRICE <= price <= MAX_PRICE:
                continue
            ahead = book.bid_size if price <= book.bid + 1e-9 else 0.0
            if flow is not None and ahead > MAX_QUEUE_SHARE * flow:
                deep.append(f"{market['yes_sub_title']}: {ahead:,.0f} ahead, ~{flow:,.0f} expected to trade")
                continue
            options.append((fair - price - maker_fee(price), side, market, book, fair, price, ahead, flow))
        if hold is not None:
            row.update(topup=True, held=hold["contracts"])
            if options and not any(o[2]["ticker"] == hold["ticker"] for o in options):
                row["reason"] = "the edge is now on the other team (no hedging)"
                continue
            options = [o for o in options if o[2]["ticker"] == hold["ticker"]]
        if not options:
            row["reason"] = f"queue too deep ({'; '.join(deep)})" if deep else "no edge at a maker price"
            continue
        edge, side, market, book, fair, price, ahead, flow = max(options, key=lambda o: o[0])
        row.update(queue_ahead=ahead, expected_volume=None if flow is None else round(flow, 1),
                   volume_24h=_num(market.get("volume_24h_fp")), improved=price > book.bid + 1e-9)
        per = price + maker_fee(price)
        n, stake = contracts(fair, per, bankroll, kelly, max_game_share, max_contracts, max_order_cost)
        if hold is not None:
            target = n
            room = math.floor((max_game_share * bankroll - hold["cost"]) / per + 1e-9)
            n = max(0, min(target - math.floor(hold["contracts"] + 1e-9), room))
        row.update(side=side, team=market["yes_sub_title"], ticker=market["ticker"], fair=round(fair, 4),
                   bid=book.bid, ask=book.ask, bid_size=book.bid_size, ask_size=book.ask_size, price=price,
                   edge=round(edge, 4), kelly_share=round(kelly_share(fair, per), 4), bankroll=round(bankroll, 2),
                   kelly_stake=round(stake, 4), count=n, cost=round(n * per, 4),
                   expires=(kick - EXPIRE_BEFORE).isoformat(),
                   client_order_id=client_order_id(market["ticker"], season, week, now))
        if n < 1 and hold is not None:
            row["reason"] = (f"holds {hold['contracts']:g}, its Kelly size or more" if target <= hold["contracts"]
                             else "holds the most one game may have")
        elif n < 1:
            row["reason"] = (f"Kelly stake ${stake:.2f} is under {MIN_CONTRACT_SHARE:g} of a contract (${per:.2f})"
                             if stake < MIN_CONTRACT_SHARE * per else "the caps leave less than one contract")
        else:
            sized.append(row)
    # The caps go to the biggest edges first, not to whichever games come first in the slate.
    placed, spent = 0, 0.0
    for row in sorted(sized, key=lambda r: -r["edge"]):
        new_game = not row.get("topup")
        if new_game and placed >= max_orders:
            row["reason"] = "order count cap reached"
        elif spent + row["cost"] > max_total_cost + 1e-9:
            row["reason"] = "run's total cost cap reached"
        elif balance is not None and spent + row["cost"] > balance + 1e-9:
            row["reason"] = "not enough balance"
        else:
            row["decision"], row["reason"] = "order", ""
            placed += new_game
            spent += row["cost"]
    return pd.DataFrame(rows)


def order_body(row):
    """The V2 order (POST /portfolio/events/orders) for a planned row: a post-only YES bid that expires."""
    return {
        "ticker": row["ticker"],
        "client_order_id": row["client_order_id"],
        "side": "bid",
        "count": f"{int(row['count'])}.00",
        "price": f"{row['price']:.4f}",
        "time_in_force": "good_till_canceled",
        "expiration_time": int(datetime.fromisoformat(row["expires"]).timestamp()),
        "post_only": True,
        "self_trade_prevention_type": "maker",
        "cancel_order_on_pause": True,
    }


def execute(planned, client):
    """Place the planned orders one by one. A duplicate (409) is skipped; any other error stops the run."""
    results = planned.copy()
    results["status"], results["order_id"], results["error"] = "", "", ""
    for i, row in results[results.decision == "order"].iterrows():
        try:
            resp = client.post("/portfolio/events/orders", order_body(row))
        except KalshiError as e:
            results.at[i, "error"] = str(e)
            if e.status == 409:
                results.at[i, "status"] = "duplicate"
                continue
            results.at[i, "status"] = "failed"
            log.error("stopping: %s", e)
            break
        results.at[i, "status"] = "placed"
        results.at[i, "order_id"] = resp.get("order_id", "")
        filled = float(resp.get("fill_count") or 0)
        if filled:
            results.at[i, "status"] = f"placed, {filled:g} filled"
        log.info("placed %s %s x%s at %.2f", row["team"], row["ticker"], row["count"], row["price"])
    return results


def account(client, tickers):
    """The games already taken among these markets, the money resting orders have committed (any market),
    and what open positions cost: (event tickers, committed dollars, exposure dollars)."""
    events = {t.rsplit("-", 1)[0] for t in tickers}
    taken, committed, exposure = set(), 0.0, 0.0
    for o in client.pages("/portfolio/orders", "orders", status="resting", limit=200):
        if o["ticker"].rsplit("-", 1)[0] in events:
            taken.add(o["ticker"].rsplit("-", 1)[0])
        left = float(o.get("remaining_count_fp") or 0)
        yes = float(o.get("yes_price_dollars") or 0)
        price = yes if o.get("book_side", "bid") == "bid" else 1 - yes
        committed += left * (price + maker_fee(price))
    for p in client.pages("/portfolio/positions", "market_positions", limit=200):
        if float(p.get("position_fp") or 0) != 0:
            exposure += abs(float(p.get("market_exposure_dollars") or 0))
            if p["ticker"].rsplit("-", 1)[0] in events:
                taken.add(p["ticker"].rsplit("-", 1)[0])
    return taken, committed, exposure


def busy(client, tickers):
    """Event tickers among these markets with a resting order or an open position."""
    events = {t.rsplit("-", 1)[0] for t in tickers}
    out = set()
    for o in client.pages("/portfolio/orders", "orders", status="resting", limit=200):
        if o["ticker"].rsplit("-", 1)[0] in events:
            out.add(o["ticker"].rsplit("-", 1)[0])
    for p in client.pages("/portfolio/positions", "market_positions", limit=200):
        if float(p.get("position_fp") or 0) != 0 and p["ticker"].rsplit("-", 1)[0] in events:
            out.add(p["ticker"].rsplit("-", 1)[0])
    return out


def kicked_off(game, now):
    """Whether the game has started by `now`: its kickoff has passed or, with the time TBD, its Eastern day has."""
    kick = kickoff_utc(game)
    if kick is not None:
        return kick <= now
    return pd.notna(game.get("gameday")) and datetime.fromisoformat(game["gameday"]).date() < now.astimezone(ET).date()


def load_week(season, week, out, now=None):
    """The week's games still to play. Without a score is not enough: parts are built from the last pull, so a
    midweek game played since has none, and its Kalshi market has closed. With `now`, games already kicked off
    are dropped too."""
    path = out / str(season) / f"parts_wk{week:02d}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run -m cfb.week_parts {season}` first")
    games = pd.read_csv(path)
    games = games[games.home_score.isna()]
    if now is not None:
        started = pd.Series([kicked_off(g, now) for g in games.to_dict("records")], index=games.index, dtype=bool)
        if started.any():
            log.info("%d game(s) already kicked off, left out", started.sum())
        games = games[~started]
    for name in ("plays.csv", "games.csv"):
        source = out / str(season) / name
        if source.exists() and source.stat().st_mtime > path.stat().st_mtime:
            log.warning("%s is newer than %s: rerun cfb.week_parts for current predictions", source, path.name)
    return games


def log_path(season, week, out):
    return out.parent / "kalshi" / "orders" / f"{season}_wk{week:02d}.csv"


def append_log(frame, path, mode):
    """Add this run's rows to the week's log. Runs differ in columns, so the file is rewritten whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.assign(run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), mode=mode)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def show(frame):
    """The orders as a table (with how placing went, once it has), then the games skipped by reason."""
    orders = frame[frame.decision == "order"]
    if len(orders):
        placed = "status" in orders and orders.status.astype(str).str.len().gt(0).any()
        print(display.orders_table(orders, status=placed))
    skipped = frame[frame.decision == "skip"]
    if len(skipped):
        print(("\n" if len(orders) else "") + display.skipped_lines(skipped, games=True))


def cancel_ours(client, season, week, out, which=None, confirm=input, yes=False):
    """Cancel resting orders this module placed this week: all of them, or those `which` names.

    Each name in `which` is an order id, a ticker, or part of a ticker or team name ("Ohio", "OHIO",
    "CMUOHIO"), and must pick out exactly one of our resting orders, or nothing is cancelled. Asks for a
    typed CANCEL unless yes. Writes data/kalshi/orders/cancels.csv.
    """
    path = log_path(season, week, out)
    if not path.exists():
        print(f"no log at {path}, nothing to cancel")
        return 0
    logged = pd.read_csv(path)
    ours = set(logged.client_order_id.dropna())
    team = (logged.dropna(subset=["client_order_id"]).drop_duplicates("client_order_id", keep="last")
            .set_index("client_order_id").team.to_dict() if "team" in logged else {})
    resting = [o for o in client.pages("/portfolio/orders", "orders", status="resting", limit=200)
               if o.get("client_order_id") in ours]
    if which:
        chosen, problems = [], []
        for name in which:
            key = name.strip().lower()
            exact = [o for o in resting if key in (o["order_id"].lower(), o["ticker"].lower())
                     or key == str(team.get(o["client_order_id"], "")).lower()]
            hits = exact or [o for o in resting if key in o["ticker"].lower()
                             or key in str(team.get(o["client_order_id"], "")).lower()]
            if len(hits) != 1:
                found = ", ".join(f"{team.get(o['client_order_id'], '?')} ({o['ticker']})" for o in hits) or "none"
                problems.append(f"{name!r} matches {len(hits)} resting order(s): {found}")
            elif hits[0] not in chosen:
                chosen.append(hits[0])
        if problems:
            print("nothing cancelled:\n  " + "\n  ".join(problems))
            return 1
        resting = chosen
    print(f"{len(resting)} resting order(s) placed by this module to cancel:")
    for o in resting:
        print(f"  {team.get(o['client_order_id'], '?'):16s} {o['ticker']}  {o.get('yes_price_dollars')}"
              f" x{o.get('remaining_count_fp')}")
    if not resting:
        return 0
    if not yes and ask(confirm, f"Type CANCEL to cancel {len(resting)} order(s): ") != "CANCEL":
        print("not confirmed: nothing cancelled")
        return 1
    done = []
    for o in resting:
        try:
            resp = client.delete(f"/portfolio/events/orders/{o['order_id']}", market_ticker=o["ticker"])
            done.append({"order_id": o["order_id"], "ticker": o["ticker"], "client_order_id": o["client_order_id"],
                         "team": team.get(o["client_order_id"]), "reduced_by": (resp or {}).get("reduced_by"),
                         "ok": True, "error": ""})
            print("cancelled", o["ticker"], o["order_id"])
        except KalshiError as e:
            done.append({"order_id": o["order_id"], "ticker": o["ticker"], "client_order_id": o["client_order_id"],
                         "team": team.get(o["client_order_id"]), "reduced_by": None, "ok": False, "error": str(e)})
            print("FAILED to cancel", o["ticker"], e)
    _append(pd.DataFrame(done).assign(cancelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds")),
            out.parent / "kalshi" / "orders" / "cancels.csv")
    return 0 if all(d["ok"] for d in done) else 1


def status(client, season, week, out):
    bal = client.get("/portfolio/balance")
    print(f"balance ${float(bal.get('balance_dollars') or 0):.2f}")
    resting = [o for o in client.pages("/portfolio/orders", "orders", status="resting", limit=200)
               if o["ticker"].startswith(SERIES)]
    print(f"{len(resting)} resting {SERIES} order(s)")
    for o in resting:
        print(" ", o["ticker"], o.get("yes_price_dollars"), o.get("remaining_count_fp"), o.get("client_order_id"))
    held = [p for p in client.pages("/portfolio/positions", "market_positions", limit=200)
            if p["ticker"].startswith(SERIES) and float(p.get("position_fp") or 0) != 0]
    print(f"{len(held)} open {SERIES} position(s)")
    for p in held:
        print(" ", p["ticker"], p["position_fp"], p.get("market_exposure_dollars"))
    return 0


def look_up(client, order_id, want, tries=10, wait=0.5, sleep=None):
    """GET an order until its status is `want`. Kalshi's reads lag its writes: an order placed a quarter
    second earlier can still be 404, and a cancelled one still resting."""
    import time
    sleep = sleep or time.sleep
    order = {}
    for i in range(tries):
        try:
            order = client.get(f"/portfolio/orders/{order_id}").get("order", {})
        except KalshiError as e:
            if e.status != 404 or i == tries - 1:
                raise
        if order.get("status") == want:
            return order
        sleep(wait)
    return order


def smoke(client, markets, books, now, out, sleep=None):
    """Place one 1-contract post-only bid at 1c on a liquid market, check it rests, cancel it, check it is gone.

    1c is far under any market traded here, so it cannot fill in the minute it rests, and it expires in
    ten minutes if the cancel fails. Every step is written to data/kalshi/orders/smoke.csv.
    """
    ticker = next((t for t, b in books.items() if b.bid is not None and b.bid >= 0.10 and b.width is not None
                   and b.width <= MAX_WIDTH), None)
    if ticker is None:
        raise SystemExit("no liquid market to smoke-test on")
    body = {"ticker": ticker, "client_order_id": str(uuid.uuid4()), "side": "bid", "count": "1.00", "price": "0.0100",
            "time_in_force": "good_till_canceled", "expiration_time": int((now + timedelta(minutes=10)).timestamp()),
            "post_only": True, "self_trade_prevention_type": "maker", "cancel_order_on_pause": True}
    steps, order_id = [], None

    def step(name, ok, detail=""):
        steps.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "step": name, "ok": ok,
                      "ticker": ticker, "order_id": order_id, "client_order_id": body["client_order_id"],
                      "detail": str(detail)[:300]})
        print(f"  {'ok ' if ok else 'BAD'} {name} {detail}")
        return ok

    try:
        resp = client.post("/portfolio/events/orders", body)
        order_id = resp.get("order_id")
        step("place", bool(order_id) and float(resp.get("fill_count") or 0) == 0, resp)
        o = look_up(client, order_id, "resting", sleep=sleep)
        step("rests", o.get("status") == "resting" and float(o.get("yes_price_dollars") or 0) == 0.01
             and float(o.get("remaining_count_fp") or 0) == 1 and o.get("client_order_id") == body["client_order_id"],
             {k: o.get(k) for k in ("status", "yes_price_dollars", "remaining_count_fp", "book_side", "expiration_time")})
    except KalshiError as e:
        step("place", False, e)
    finally:
        if order_id:
            try:
                resp = client.delete(f"/portfolio/events/orders/{order_id}", market_ticker=ticker)
                step("cancel", True, resp)
                o = look_up(client, order_id, "canceled", sleep=sleep)
                step("cancelled", o.get("status") == "canceled" and float(o.get("fill_count_fp") or 0) == 0,
                     {k: o.get(k) for k in ("status", "fill_count_fp", "remaining_count_fp")})
            except KalshiError as e:
                step("cancel", False, e)
    path = out.parent / "kalshi" / "orders" / "smoke.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(steps)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)
    passed = all(x["ok"] for x in steps) and [x["step"] for x in steps] == ["place", "rests", "cancel", "cancelled"]
    print("smoke test", "passed" if passed else "FAILED", f"(logged to {path})")
    return passed


def our_orders(out):
    """Every order this module placed, from the week logs: the plan row it came from, keyed by Kalshi's order_id.

    Not by client_order_id: that is the same for a market all week, and Kalshi takes it again once the first
    order with it is cancelled, so a re-placed order shares it with the one it replaced."""
    rows = []
    for path in sorted((out.parent / "kalshi" / "orders").glob("*_wk*.csv")):
        log_ = pd.read_csv(path)
        if "status" not in log_:
            continue
        placed = log_[(log_["mode"] == "execute") & log_.status.astype(str).str.startswith("placed")]
        rows.append(placed.assign(week_log=path.stem))
    if not rows:
        return pd.DataFrame()
    placed = pd.concat(rows, ignore_index=True)
    placed = placed[placed.order_id.astype(str).str.len() > 0]
    return placed.drop_duplicates("order_id", keep="last").set_index("order_id")


def track(client, out, now=None):
    """Save the state of every order this module placed: a snapshot of each order and its market, new fills,
    and settlements, under data/kalshi/orders/. Prints the resting, just-placed and recently filled orders
    (cfb.kalshi_display.shown_orders), with profit where games are settled and totals over every order.

        orders.csv           one row per order, latest state, with the plan's numbers (RAPPA, book, fair, edge)
        order_snapshots.csv  every run's state of every order and its market, appended
        fills.csv            every fill of our orders (price, count, fee, maker or taker), one row per fill
        settlements.csv      the result of every market we held when it settled
    """
    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="seconds")
    folder = out.parent / "kalshi" / "orders"
    plans = our_orders(out)
    if plans.empty:
        print("no orders placed by this module yet")
        return pd.DataFrame()
    ids = set(plans.index)
    orders = {o["order_id"]: o for o in client.pages("/portfolio/orders", "orders", limit=200) if o["order_id"] in ids}
    try:
        for o in client.pages("/historical/orders", "orders", limit=200):
            if o["order_id"] in ids and o["order_id"] not in orders:
                orders[o["order_id"]] = o
    except KalshiError as e:
        log.warning("historical orders unavailable: %s", e)
    order_ids = set(orders)
    tickers = sorted(set(plans.ticker))
    fills = [f for f in client.pages("/portfolio/fills", "fills", limit=200) if f.get("order_id") in order_ids]
    settlements = [x for x in client.pages("/portfolio/settlements", "settlements", limit=200) if x["ticker"] in tickers]
    positions = {p["ticker"]: p for p in client.pages("/portfolio/positions", "market_positions", limit=200)}
    markets = {}
    for t in tickers:
        m = client.get(f"/markets/{t}").get("market", {})
        markets[t] = {"market_status": m.get("status"), "market_result": m.get("result"),
                      "market_bid": m.get("yes_bid_dollars"), "market_ask": m.get("yes_ask_dollars"),
                      "market_last": m.get("last_price_dollars"), "market_volume": m.get("volume_fp"),
                      "market_volume_24h": m.get("volume_24h_fp")}
    queue = {}
    resting_tickers = sorted({o["ticker"] for o in orders.values() if o.get("status") == "resting"})
    if resting_tickers:
        try:
            q = client.get("/portfolio/orders/queue_positions", market_tickers=",".join(resting_tickers))
            queue = {x["order_id"]: x.get("queue_position_fp") for x in q.get("queue_positions") or []}
        except KalshiError as e:
            log.warning("queue positions unavailable: %s", e)

    snap = []
    for oid, plan_row in plans.iterrows():
        o = orders.get(oid, {})
        pos = positions.get(plan_row.ticker, {})
        snap.append({"snapshot_at": stamp, "client_order_id": plan_row.client_order_id, "order_id": oid,
                     "ticker": plan_row.ticker, "status": o.get("status", "not found"),
                     "initial_count": o.get("initial_count_fp"), "fill_count": o.get("fill_count_fp"),
                     "remaining_count": o.get("remaining_count_fp"), "price": o.get("yes_price_dollars"),
                     "maker_fill_cost": o.get("maker_fill_cost_dollars"), "taker_fill_cost": o.get("taker_fill_cost_dollars"),
                     "maker_fees": o.get("maker_fees_dollars"), "taker_fees": o.get("taker_fees_dollars"),
                     "created_time": o.get("created_time"), "last_update_time": o.get("last_update_time"),
                     "expiration_time": o.get("expiration_time"), "queue_position": queue.get(o.get("order_id")),
                     "position": pos.get("position_fp"),
                     "position_exposure": pos.get("market_exposure_dollars"), **markets.get(plan_row.ticker, {})})
    snap = pd.DataFrame(snap)
    _append(snap, folder / "order_snapshots.csv")
    _merge(pd.DataFrame(fills), folder / "fills.csv", "fill_id")
    _merge(pd.DataFrame(settlements), folder / "settlements.csv", "ticker")

    keep = ["week_log", "run_at", "game_id", "game", "kickoff", "team", "side", "rappa_home", "mid_home", "fair", "bid",
            "ask", "bid_size", "ask_size", "price", "improved", "queue_ahead", "volume_24h", "expected_volume", "edge",
            "kelly_share", "bankroll", "kelly_stake", "count", "cost", "expires"]
    latest = snap.set_index("order_id").join(plans[[c for c in keep if c in plans]].rename(
        columns={"price": "plan_price", "run_at": "placed_at"}))
    settled = {x["ticker"]: x for x in settlements}
    latest["result"] = latest.ticker.map(lambda t: settled.get(t, {}).get("market_result") or markets[t]["market_result"])
    filled = pd.to_numeric(latest.fill_count, errors="coerce").fillna(0)
    cost = (pd.to_numeric(latest.maker_fill_cost, errors="coerce").fillna(0)
            + pd.to_numeric(latest.taker_fill_cost, errors="coerce").fillna(0))
    fees = (pd.to_numeric(latest.maker_fees, errors="coerce").fillna(0)
            + pd.to_numeric(latest.taker_fees, errors="coerce").fillna(0))
    latest["avg_fill_price"] = (cost / filled).where(filled > 0)
    won = latest.result.eq("yes")
    latest["pnl"] = (filled * won - cost - fees).where(latest.result.isin(["yes", "no"]))
    mid = (pd.to_numeric(latest.market_bid, errors="coerce") + pd.to_numeric(latest.market_ask, errors="coerce")) / 2
    latest["mark_to_market"] = (filled * mid - cost - fees).where(latest.result.isna() | ~latest.result.isin(["yes", "no"]))
    latest.reset_index().to_csv(folder / "orders.csv", index=False)

    print(display.track_table(latest, now))
    return latest


def _append(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def _merge(frame, path, key):
    """Add rows not already in the file, by key."""
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame[~frame[key].astype(str).isin(old[key].astype(str))]], ignore_index=True)
    frame.to_csv(path, index=False)


def ask(confirm, prompt):
    """The typed answer, or "" where there is no keyboard (a script, or Claude Code's ! prompt): not confirmed."""
    try:
        return confirm(prompt).strip()
    except EOFError:
        print("\nno input to confirm with: run it in a terminal, or add --yes to confirm on the command line")
        return ""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, help="default: the latest season with predictions")
    p.add_argument("--week", type=int, help="default: cfb.week_parts' next week")
    p.add_argument("--out", type=Path, default=Path("data/cfb"))
    p.add_argument("--env", type=Path, default=Path(".env"), help="where KALSHI_ID and KALSHI_PRIVATE_KEY are")
    p.add_argument("--weight", type=float, default=WEIGHT, help=f"fair value's step toward RAPPA (default {WEIGHT})")
    p.add_argument("--min-edge", type=float, default=MIN_EDGE, help=f"dollars a contract (default {MIN_EDGE})")
    p.add_argument("--kelly", type=float, default=KELLY_FRACTION,
                   help=f"share of Kelly to stake (default {KELLY_FRACTION})")
    p.add_argument("--max-game-share", type=float, default=MAX_GAME_SHARE,
                   help=f"most of the bankroll on one game, the biggest bet (default {MAX_GAME_SHARE})")
    p.add_argument("--max-total-share", type=float, default=MAX_TOTAL_SHARE,
                   help=f"most of the bankroll the run spends (default {MAX_TOTAL_SHARE})")
    p.add_argument("--bankroll", type=float, help="size against this many dollars instead of the account's")
    p.add_argument("--max-contracts", type=int, help="a fixed cap on contracts an order (default none)")
    p.add_argument("--max-order-cost", type=float, help="a fixed cap in dollars an order (default none)")
    p.add_argument("--max-total-cost", type=float, help="a fixed cap in dollars for the run, in place of the share")
    p.add_argument("--max-orders", type=int, default=MAX_ORDERS, help=f"games the run places on (default {MAX_ORDERS})")
    p.add_argument("--demo", action="store_true", help="Kalshi's demo exchange (needs a demo key)")
    p.add_argument("--execute", action="store_true", help="place the orders (asks first)")
    p.add_argument("--yes", action="store_true", help="skip the typed confirmation of --execute, --smoke or a cancel")
    p.add_argument("--status", action="store_true", help="balance, resting orders and positions, then stop")
    p.add_argument("--cancel", action="store_true", help="cancel the resting orders this module placed this week")
    p.add_argument("--cancel-order", action="append", metavar="NAME",
                   help="cancel one of them: an order id, a ticker, or part of a ticker or team name; repeatable")
    p.add_argument("--smoke", action="store_true",
                   help="place a 1-contract 1c bid on one market, check it, cancel it (asks first)")
    p.add_argument("--track", action="store_true",
                   help="save the state, fills and settlements of every order this module placed, then stop")
    return p.parse_args(argv)


def main(argv=None, client=None, now=None, confirm=input):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from cfb.week_parts import upcoming_week
    season = args.season or max(int(p.name) for p in args.out.iterdir() if p.name.isdigit()
                                and any(p.glob("parts_wk*.csv")))
    week = args.week or upcoming_week(season, args.out)
    if week is None and not args.track:  # tracking still has settlements to collect once the season is over
        raise SystemExit(f"no week ahead in {season}")
    writes = args.execute or args.cancel or args.smoke or bool(args.cancel_order)
    if client is None:
        client = Client.from_env(args.env, host=kalshi.DEMO if args.demo else kalshi.PROD, read_only=not writes)
    if (writes or args.status or args.track) and not client.signed:
        raise SystemExit(f"no KALSHI_ID / KALSHI_PRIVATE_KEY in {args.env}")
    if args.status:
        return status(client, season, week, args.out)
    if args.cancel or args.cancel_order:
        result = cancel_ours(client, season, week, args.out, args.cancel_order, confirm, args.yes)
        track(client, args.out, now)
        return result
    if args.track:
        track(client, args.out, now)
        return 0

    now = now or datetime.now(timezone.utc)
    games = load_week(season, week, args.out, now)
    markets = client.pages("/markets", "markets", series_ticker=SERIES, status="open", limit=1000)
    matches = match_events(games, markets)
    log.info("%d of %d games matched to Kalshi markets", len(matches), len(games))
    books = {}
    for hit in matches.values():
        for m in (hit["home"], hit["away"]):
            books[m["ticker"]] = parse_book(client.get(f"/markets/{m['ticker']}/orderbook", depth=5))
    taken, available, bankroll = set(), None, args.bankroll
    if client.signed:
        taken, committed, exposure = account(client, books)
        balance = float(client.get("/portfolio/balance").get("balance_dollars") or 0)
        # Kalshi's balance did not fall when orders were resting, so what they commit is taken off here. If it
        # does already exclude them, this only spends less.
        available = max(0.0, balance - committed)
        bankroll = bankroll if bankroll is not None else balance + exposure
        log.info("balance $%.2f, $%.2f committed to resting orders, $%.2f in positions: $%.2f to spend, "
                 "bankroll $%.2f; %d game(s) already with an order or position",
                 balance, committed, exposure, available, bankroll, len(taken))
    else:
        log.warning("no Kalshi key: not checking existing orders, positions or balance")
    if args.smoke:
        if not args.yes and ask(confirm, "Type PLACE to place and cancel one 1-contract 1c test order: ") != "PLACE":
            print("not confirmed: nothing sent")
            return 1
        return 0 if smoke(client, markets, books, now, args.out, sleep=getattr(client, "sleep", None)) else 1
    if bankroll is None:
        raise SystemExit("no Kalshi key to read the bankroll from: give --bankroll to size the plan against")
    planned = plan(games, matches, books, taken, now, season, week, args.weight, args.min_edge, args.max_contracts,
                   args.max_order_cost, args.max_total_cost, args.max_orders, available, bankroll, args.kelly,
                   args.max_game_share, args.max_total_share)
    show(planned)
    orders = planned[planned.decision == "order"]
    cap = args.max_total_cost if args.max_total_cost is not None else args.max_total_share * bankroll
    print(f"\n{len(orders)} order(s), ${orders.cost.sum() if len(orders) else 0:.2f} if all fill (fees included), "
          f"of the run's ${cap:.2f}; week {week}, {season}")
    path = log_path(season, week, args.out)
    if not args.execute or orders.empty:
        append_log(planned, path, "plan")
        print("dry run: nothing sent" if not args.execute else "nothing to place")
        return 0
    if not args.yes and ask(confirm, f"Type PLACE to send {len(orders)} order(s) to Kalshi: ") != "PLACE":
        append_log(planned, path, "declined")
        print("not confirmed: nothing sent")
        return 1
    results = execute(planned, client)
    append_log(results, path, "execute")
    show(results[results.decision == "order"])
    print()
    track(client, args.out)
    return 0 if not (results.status == "failed").any() else 1


if __name__ == "__main__":
    sys.exit(main())
