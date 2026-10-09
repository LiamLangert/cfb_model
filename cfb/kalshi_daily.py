"""The daily Kalshi run: the week's plan, placed a game at a time, 2-30 hours before each kickoff.

cfb.kalshi_orders plans the whole week at once, and run on Tuesday its resting
bids were picked off: in backtests a bid placed Tuesday at 10am made +0.9c a
contract on the fills it got, against +3 to +8c for one placed 6-72 hours
before kickoff (each about +/- 3-4c), while the bids that never filled would
have won. Prices that sit for days get hit by whoever learns something first.
So this runs every day and only places on games kicking off between MIN_LEAD
(2h) and MAX_LEAD (30h) from now. Run at 4pm, Friday's run takes Friday's
games and Saturday's up to 10pm; Saturday's takes the rest.

Each run, with cfb.kalshi_orders' pricing, guards and sizing:

  1. Reviews every resting order this module or cfb.kalshi_orders placed on
     the week's games, against the book without the order itself in it:
       keep     the edge still clears KEEP_EDGE, and nobody bids above it
       cancel   the edge is under KEEP_EDGE, or a guard now fails (a wide or
                inconsistent book, RAPPA too far from the market): the
                market has moved, most likely on news, and the bid is what
                the news would fill
       replace  others now bid above it: below the bid it fills only once the
                price falls through it, so it is cancelled and re-placed at
                the best price that still clears MIN_EDGE, for no more than
                it had left, or just cancelled if there is none
  2. Plans the week's remaining games at today's prices, to a budget for
     the week (MAX_WEEK_SHARE of the bankroll and MAX_WEEK_ORDERS games,
     counting what is already filled or resting on this week's games,
     manual trades included; the bankroll is the balance plus what open
     positions cost, so the budget grows with the account). The caps go to the biggest edges across the whole week, so
     a Thursday game cannot spend money a bigger Saturday edge would get:
       order    kicks off within MAX_LEAD: place it now
       later    the budget is held for it, but it is placed in a later run,
                at that day's prices
     A game already held (a position, or a resting order the review keeps)
     is topped up to its Kelly size at today's fair value and price: on the
     same team only, never a hedge, and within MAX_GAME_SHARE of the
     bankroll counting what it holds. A top-up competes for the budget by
     edge like a new order, but takes none of the week's games. A game held
     on both teams or as NO contracts, or whose resting order is cancelled
     or replaced in the run, is left alone.
  3. Prints the week so far, the review, today's orders and the later ones,
     and logs the run.

Nothing is sent without --execute, which replans from scratch at the
prices of that moment, shows it, and asks for a typed PLACE (or --yes):
cancels first, then replacements and new orders, stopping at the first
error it does not expect, then a --track snapshot. A plan is never placed
from an earlier run, so what is confirmed is what is sent.

--notify (the scheduled run) pops up a Windows notification when there is
something to place or cancel, or when the run fails. The run builds the
week's predictions (cfb.week_parts) first when they are missing or older
than the season's games and plays.

Logs, in data/kalshi/orders/: the plan and execute rows go to the week's
<season>_wk<NN>.csv beside cfb.kalshi_orders' (planner "daily", with
lead_hours, the hours from the run to kickoff), so --track follows the
orders placed here; reviews.csv has every review; cancels.csv every cancel.

    uv run --with cryptography -m cfb.kalshi_daily              # plan only
    uv run --with cryptography -m cfb.kalshi_daily --execute    # place, after confirming
    cfb/kalshi_daily.sh                                         # the scheduled run: track, plan, notify
    cfb/kalshi_daily.sh --execute                               # the same as the second line, logged
"""

from __future__ import annotations

import argparse
import base64
import logging
import math
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from cfb import kalshi_display as display
from cfb import kalshi_orders as ko
from rappa import kalshi
from rappa.kalshi import Client, KalshiError, maker_fee, parse_book, ticker_date

log = logging.getLogger(__name__)

MIN_LEAD = ko.MIN_LEAD
MAX_LEAD = timedelta(hours=30)
# A resting order is kept while its edge clears KEEP_EDGE, below the MIN_EDGE it was placed at, so a
# half-cent wiggle in the mid does not cancel it and cost its place in the queue.
KEEP_EDGE = 0.01
# The week's budget, in place of cfb.kalshi_orders' per-run MAX_TOTAL_SHARE and MAX_ORDERS: the same caps, for a
# week of daily runs instead of one Tuesday run. The money is a share of the bankroll, so it grows with it.
MAX_WEEK_SHARE = ko.MAX_TOTAL_SHARE
MAX_WEEK_ORDERS = ko.MAX_ORDERS

REASONS = {"run's total cost cap reached": "the week's budget is held for bigger edges",
           "order count cap reached": "the week's order cap is held for bigger edges"}

# Windows' PowerShell, which can show a notification from a WSL process; its app id is the one toasts show under.
POWERSHELL = "powershell.exe"
TOAST_APP = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


# ---------- the books, without our own orders ----------

def book_without(payload, own=None):
    """A Book from an orderbook payload with our resting bids ({price: contracts}) taken out of it."""
    if not own:
        return parse_book(payload)
    book = dict((payload or {}).get("orderbook_fp") or {})
    yes = []
    for p, n in book.get("yes_dollars") or []:
        left = float(n) - sum(c for q, c in own.items() if abs(q - float(p)) < 1e-6)
        if left > 1e-9:
            yes.append([p, str(left)])
    book["yes_dollars"] = yes
    return parse_book({"orderbook_fp": book})


def own_bids(resting):
    """{ticker: {price: contracts}} of resting YES bids."""
    out = {}
    for o in resting:
        if o.get("book_side", "bid") != "bid":
            continue
        price = round(float(o.get("yes_price_dollars") or 0), 4)
        levels = out.setdefault(o["ticker"], {})
        levels[price] = levels.get(price, 0.0) + float(o.get("remaining_count_fp") or 0)
    return out


def event_of(ticker):
    return ticker.rsplit("-", 1)[0]


def guard(g, hb, ab):
    """Why the game's books cannot be priced off, as cfb.kalshi_orders.plan's guards say it, or None."""
    if hb is None or ab is None or hb.mid is None or ab.mid is None:
        return "a book is empty"
    if max(hb.width, ab.width) > ko.MAX_WIDTH + 1e-9:
        return f"book too wide ({max(hb.width, ab.width):.2f})"
    if abs(hb.mid + ab.mid - 1) > ko.MAX_BOOK_GAP + 1e-9:
        return "the two books disagree"
    market_home = (hb.mid + 1 - ab.mid) / 2
    if abs(g["rappa_wp"] - market_home) > ko.MAX_DISAGREE + 1e-9:
        return f"RAPPA too far from the market ({g['rappa_wp'] - market_home:+.2f}), likely news"
    return None


# ---------- 1. the review ----------

def replacement_id(order, season, week, run):
    """A new client_order_id for the order replacing `order`: Kalshi refuses any id it has seen (see
    cfb.kalshi_orders.NAMESPACE), so it names the order replaced as well as the run."""
    return str(uuid.uuid5(ko.NAMESPACE, f"rappa-cfb:{order['ticker']}:{season}:{week}:{run.isoformat()}:"
                                        f"replaces:{order['order_id']}"))


def review(resting, games, matches, payloads, now, season, week, bankroll, weight=ko.WEIGHT, min_edge=ko.MIN_EDGE,
           keep_edge=KEEP_EDGE, kelly=ko.KELLY_FRACTION, max_game_share=ko.MAX_GAME_SHARE,
           max_contracts=None, max_order_cost=None):
    """keep, cancel or replace each of our resting orders on a matched game. Pure: nothing is sent.

    resting is our resting orders (Kalshi's order dicts), games the parts slate, matches match_events',
    payloads each matched ticker's orderbook response. Returns (reviews, replacements): one review row per
    order, and the plan rows (cfb.kalshi_orders' columns) of the orders that replace those being replaced.
    """
    by_event = {hit["event"]: (gid, hit) for gid, hit in matches.items()}
    slate = {g["game_id"]: g for g in games.to_dict("records")}
    own = own_bids(resting)
    rows, replacements = [], []
    for o in resting:
        found = by_event.get(event_of(o["ticker"]))
        if found is None or found[0] not in slate:
            continue  # not one of this week's games, or its market has closed: it expires on its own
        gid, hit = found
        g = slate[gid]
        side = "home" if o["ticker"] == hit["home"]["ticker"] else "away"
        market = hit[side]
        price = round(float(o.get("yes_price_dollars") or 0), 4)
        left = float(o.get("remaining_count_fp") or 0)
        hb = book_without(payloads.get(hit["home"]["ticker"]), own.get(hit["home"]["ticker"]))
        ab = book_without(payloads.get(hit["away"]["ticker"]), own.get(hit["away"]["ticker"]))
        book = hb if side == "home" else ab
        kick = ko.kickoff_utc(g)
        row = {"order_id": o["order_id"], "client_order_id": o.get("client_order_id"), "game_id": gid,
               "game": f"{g['away_team']} at {g['home_team']}", "side": side, "team": market["yes_sub_title"],
               "ticker": o["ticker"],
               "kickoff": kick.isoformat() if kick else None, "price": price, "remaining": left,
               "rappa_home": g["rappa_wp"], "bid": book.bid, "ask": book.ask, "bid_size": book.bid_size}
        rows.append(row)
        why = guard(g, hb, ab)
        if why:
            row.update(decision="cancel", reason=f"can't price it now: {why}")
            continue
        p = g["rappa_wp"] if side == "home" else 1 - g["rappa_wp"]
        fair = ko.fair_value(book.mid, p, weight)
        edge = fair - price - maker_fee(price)
        row.update(mid=round(book.mid, 4), fair=round(fair, 4), edge=round(edge, 4))
        if edge < keep_edge - 1e-9:
            row.update(decision="cancel", reason=f"edge gone ({100 * edge:+.1f}c a contract)")
            continue
        if book.bid is None or price >= book.bid - 1e-9:
            row.update(decision="keep", reason="")
            continue
        # Outbid: re-price as plan would, for no more than the order had left.
        why = None
        if kick is None or kick - now < MIN_LEAD:
            why = "too close to kickoff to re-place"
        else:
            hours = (kick - ko.EXPIRE_BEFORE - now).total_seconds() / 3600
            flow = ko.expected_volume(market, hours)
            new = ko.best_price(fair, book, min_edge, improve=flow is None or book.bid_size > ko.IMPROVE_SHARE * flow)
            ahead = 0.0 if new is None or new > book.bid + 1e-9 else book.bid_size
            per = None if new is None else new + maker_fee(new)
            n = 0 if new is None else min(math.floor(left + 1e-9), ko.contracts(
                fair, per, bankroll, kelly, max_game_share, max_contracts, max_order_cost)[0])
            if new is None or not ko.MIN_PRICE <= new <= ko.MAX_PRICE:
                why = "no price above the bid clears the edge"
            elif flow is not None and ahead > ko.MAX_QUEUE_SHARE * flow:
                why = f"queue too deep ({ahead:,.0f} ahead, ~{flow:,.0f} expected to trade)"
            elif n < 1:
                why = "Kelly now sizes it under one contract"
        if why:
            row.update(decision="cancel", reason=f"outbid at {book.bid:.2f}; {why}")
            continue
        row.update(decision="replace", reason=f"outbid at {book.bid:.2f}", new_price=new, new_count=n)
        replacements.append({
            "game_id": gid, "game": row["game"], "rappa_home": g["rappa_wp"], "decision": "order", "event": hit["event"],
            "kickoff": kick.isoformat(), "mid_home": round(hb.mid, 4), "side": side, "team": market["yes_sub_title"],
            "ticker": o["ticker"], "fair": round(fair, 4), "bid": book.bid, "ask": book.ask, "bid_size": book.bid_size,
            "ask_size": book.ask_size, "price": new, "edge": round(fair - per, 4), "count": n, "cost": round(n * per, 4),
            "queue_ahead": ahead, "expected_volume": None if flow is None else round(flow, 1),
            "volume_24h": ko._num(market.get("volume_24h_fp")), "improved": new > book.bid + 1e-9,
            "kelly_share": round(ko.kelly_share(fair, per), 4), "bankroll": round(bankroll, 2),
            "expires": (kick - ko.EXPIRE_BEFORE).isoformat(),
            "client_order_id": replacement_id(o, season, week, now), "replaces": o["order_id"],
            "reason": ""})
    return pd.DataFrame(rows), pd.DataFrame(replacements)


# ---------- 2. the week's budget and plan ----------

def week_days(parts):
    """The first and last day of the week's games."""
    days = pd.to_datetime(parts.gameday).dt.date
    return days.min(), days.max()


def in_week(ticker, first, last):
    """A game market whose event is dated within a day of the week's games."""
    day = ticker_date(event_of(ticker)) if ticker.startswith(ko.SERIES) else None
    return day is not None and first - timedelta(days=1) <= day <= last + timedelta(days=1)


def fill_cost(f):
    """What a fill cost, fee included; a sale counts nothing."""
    if (f.get("action") or "buy") != "buy":
        return 0.0
    price = f.get("yes_price_dollars") if (f.get("side") or "yes") == "yes" else f.get("no_price_dollars")
    return float(f.get("count_fp") or 0) * float(price or 0) + float(f.get("fee_cost") or 0)


def committed(o):
    """What a resting order holds back: its contracts left at its price and the maker fee."""
    yes = float(o.get("yes_price_dollars") or 0)
    price = yes if o.get("book_side", "bid") == "bid" else 1 - yes
    return float(o.get("remaining_count_fp") or 0) * (price + maker_fee(price))


def holdings(resting, positions):
    """What each game already holds, for plan's top-ups: ({event: {"ticker", "contracts", "cost"}}, odd), the
    first for games held on one team's YES only (positions and resting bids together), odd the events held any
    other way (both teams, NO contracts, a resting sell), which are left alone."""
    held, odd = {}, set()

    def add(ticker, n, cost):
        e = event_of(ticker)
        h = held.setdefault(e, {"ticker": ticker, "contracts": 0.0, "cost": 0.0})
        if h["ticker"] != ticker:
            odd.add(e)
        h["contracts"] += n
        h["cost"] += cost

    for p in positions:
        n = float(p.get("position_fp") or 0)
        if n < 0:
            odd.add(event_of(p["ticker"]))
        elif n > 0:
            add(p["ticker"], n, abs(float(p.get("market_exposure_dollars") or 0)))
    for o in resting:
        if o.get("book_side", "bid") != "bid":
            odd.add(event_of(o["ticker"]))
        else:
            add(o["ticker"], float(o.get("remaining_count_fp") or 0), committed(o))
    return {e: h for e, h in held.items() if e not in odd}, odd


def week_spent(fills, resting, first, last):
    """Dollars filled or resting on the week's games, and the games (event tickers) they are on."""
    spent, events = 0.0, set()
    for f in fills:
        if in_week(f["ticker"], first, last) and fill_cost(f) > 0:
            spent += fill_cost(f)
            events.add(event_of(f["ticker"]))
    for o in resting:
        if in_week(o["ticker"], first, last):
            spent += committed(o)
            events.add(event_of(o["ticker"]))
    return spent, events


def split(planned, now, max_lead=MAX_LEAD):
    """Orders kicking off beyond max_lead become 'later': budgeted for, placed in a later run."""
    planned = planned.copy()
    if "kickoff" not in planned:
        planned["kickoff"] = None
    kick = pd.to_datetime(planned.kickoff, utc=True, errors="coerce")
    planned["lead_hours"] = ((kick - now).dt.total_seconds() / 3600).round(1)
    later = planned.decision.eq("order") & (kick - now > max_lead)
    planned.loc[later, "decision"] = "later"
    planned.loc[later, "reason"] = [f"kicks off in {h:.0f}h: placed {max_lead.total_seconds() / 3600:.0f}h out or less"
                                    for h in planned.loc[later, "lead_hours"]]
    planned["reason"] = planned.reason.replace(REASONS)
    return planned


# ---------- the notification ----------

def _ps_string(text):
    return "'" + str(text).replace("'", "''") + "'"


def toast(title, body, run=subprocess.run):
    """A Windows notification, through PowerShell from WSL. Logs and carries on when there is none to show."""
    script = "\n".join([
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null",
        "$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02)",
        "$t = $xml.GetElementsByTagName('text')",
        f"$t.Item(0).AppendChild($xml.CreateTextNode({_ps_string(title)})) > $null",
        f"$t.Item(1).AppendChild($xml.CreateTextNode({_ps_string(body)})) > $null",
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        f"{_ps_string(TOAST_APP)}).Show([Windows.UI.Notifications.ToastNotification]::new($xml))",
    ])
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    try:
        done = run([POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                   capture_output=True, timeout=60)
        if done.returncode:
            log.warning("notification failed: %s", (done.stderr or b"")[:300])
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("no notification: %s", e)


def summary(reviews, now_orders):
    """The notification's title and body, or None when there is nothing to do."""
    acts = reviews[reviews.decision.isin(["cancel", "replace"])] if len(reviews) else reviews
    if now_orders.empty and acts.empty:
        return None
    parts = []
    if len(now_orders):
        parts.append(f"{len(now_orders)} to place")
    for d in ("cancel", "replace"):
        n = int((acts.decision == d).sum()) if len(acts) else 0
        if n:
            parts.append(f"{n} to {d}")
    lines = [f"{r.team} {'+' if display.is_topup(r.get('topup')) else ''}{int(r['count'])} @ {r.price:.2f}"
             for _, r in now_orders.iterrows()]
    lines += [f"{d} {r.team} @ {r.price:.2f}" for d in ("cancel", "replace")
              for _, r in (acts[acts.decision == d].iterrows() if len(acts) else [])]
    cost = f" (${now_orders.cost.sum():.2f})" if len(now_orders) else ""
    return (f"RAPPA Kalshi: {', '.join(parts)}",
            f"{'; '.join(lines)}{cost}. Run cfb/kalshi_daily.sh --execute")


# ---------- the run ----------

# A game is taken to be over this long after kickoff, for whether a pull has its plays.
GAME_LENGTH = timedelta(hours=4)


def pulled_through(season, week, out):
    """Whether the season's last pull (games.csv) came after every earlier week's games were over."""
    from cfb.ratings import continuous_week
    path = out / str(season) / "games.csv"
    games = pd.read_csv(path, low_memory=False)
    start = pd.to_datetime(games.start_date, utc=True)
    before = start[continuous_week(games) < week]
    pulled = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return before.empty or pulled >= before.max() + GAME_LENGTH


def ensure_parts(season, week, out):
    """The week's predictions, built with cfb.week_parts when missing or older than games or plays: "ok",
    "built", or "waiting" when missing and the last pull predates the week before's games (on the Sunday
    and Monday before Tuesday's pull, ratings built then would leave the last week out)."""
    path = out / str(season) / f"parts_wk{week:02d}.csv"
    sources = [out / str(season) / name for name in ("games.csv", "plays.csv")]
    if path.exists() and all(not s.exists() or s.stat().st_mtime <= path.stat().st_mtime for s in sources):
        return "ok"
    if not path.exists() and not pulled_through(season, week, out):
        return "waiting"
    log.info("%s is %s: building it with cfb.week_parts", path, "stale" if path.exists() else "missing")
    from cfb import week_parts
    week_parts.main([str(season), str(week), "--out", str(out)])
    return "built"


def show_week(spent, events, budget, max_games, reviews, planned, max_lead=MAX_LEAD, basis=""):
    left, games_left = max(0.0, budget - spent), max(0, max_games - len(events))
    print("\n" + display.wrap(f"This week: ${spent:.2f} filled or resting on {len(events)} games; ${left:.2f} and "
                              f"{games_left} games left of the ${budget:.2f} budget{basis}"))
    if len(reviews):
        print(f"\nResting orders: {len(reviews)}")
        print(display.reviews_table(reviews))
    hours = f"{max_lead.total_seconds() / 3600:g}h"
    for decision, head in (("order", f"Place now (kickoff within {hours})"),
                           ("later", f"Later this week (at today's prices; placed once within {hours})")):
        rows = planned[planned.decision == decision]
        print(f"\n{head}: " + (f"{len(rows)}, ${rows.cost.sum():.2f}" if len(rows) else "none"))
        if len(rows):
            print(display.orders_table(rows))
    skipped = planned[planned.decision == "skip"]
    if len(skipped):
        print("\n" + display.skipped_lines(skipped))


def run_day(client, season, week, out, now, args, confirm=input, notify=toast):
    games = ko.load_week(season, week, out, now)
    first, last = week_days(pd.read_csv(out / str(season) / f"parts_wk{week:02d}.csv"))
    markets = client.pages("/markets", "markets", series_ticker=ko.SERIES, status="open", limit=1000)
    matches = ko.match_events(games, markets)
    log.info("%d of %d games matched to Kalshi markets", len(matches), len(games))
    payloads = {}
    for hit in matches.values():
        for m in (hit["home"], hit["away"]):
            payloads[m["ticker"]] = client.get(f"/markets/{m['ticker']}/orderbook", depth=10)

    balance = float(client.get("/portfolio/balance").get("balance_dollars") or 0)
    resting = [o for o in client.pages("/portfolio/orders", "orders", status="resting", limit=200)]
    positions = [p for p in client.pages("/portfolio/positions", "market_positions", limit=200)
                 if float(p.get("position_fp") or 0) != 0]
    fills = client.pages("/portfolio/fills", "fills", limit=200)
    exposure = sum(abs(float(p.get("market_exposure_dollars") or 0)) for p in positions)
    bankroll = args.bankroll if args.bankroll is not None else balance + exposure

    ours = set(ko.our_orders(out).index)
    reviews, replacements = review([o for o in resting if o["order_id"] in ours], games, matches, payloads, now,
                                   season, week, bankroll, args.weight, args.min_edge, args.keep_edge, args.kelly,
                                   args.max_game_share, args.max_contracts, args.max_order_cost)
    going = set(reviews.order_id[reviews.decision.isin(["cancel", "replace"])]) if len(reviews) else set()
    kept = [o for o in resting if o["order_id"] not in going]
    extra = float(replacements.cost.sum()) if len(replacements) else 0.0
    spent, events = week_spent(fills, kept, first, last)
    spent += extra
    events |= {event_of(t) for t in (replacements.ticker if len(replacements) else [])}
    if args.max_week_cost is not None:
        budget, basis = args.max_week_cost, ""
    else:
        budget, basis = args.max_week_share * bankroll, f" ({args.max_week_share:.0%} of the ${bankroll:.2f} bankroll)"
    budget_left = max(0.0, budget - spent)
    orders_left = max(0, args.max_week_orders - len(events))
    # Kalshi's balance does not fall while orders rest, so what they hold back is taken off.
    available = max(0.0, balance - sum(committed(o) for o in kept) - extra)
    held, odd = holdings(kept, positions)
    busy = {e: "held on both teams or as NO contracts" for e in odd}
    moving = reviews.ticker[reviews.decision.isin(["cancel", "replace"])] if len(reviews) else []
    busy |= {event_of(t): "its resting order is cancelled or replaced in this run" for t in moving}
    log.info("balance $%.2f, $%.2f to spend, bankroll $%.2f", balance, available, bankroll)

    planned = ko.plan(games, matches, {t: book_without(p, own_bids(resting).get(t)) for t, p in payloads.items()},
                      busy, now, season, week, args.weight, args.min_edge, args.max_contracts, args.max_order_cost,
                      budget_left, orders_left, available, bankroll, args.kelly, args.max_game_share,
                      held=held)
    planned = split(planned, now, timedelta(hours=args.max_lead)).assign(planner="daily")
    now_orders = planned[planned.decision == "order"]
    show_week(spent, events, budget, args.max_week_orders, reviews, planned, timedelta(hours=args.max_lead), basis)

    path = ko.log_path(season, week, out)
    reviews_path = out.parent / "kalshi" / "orders" / "reviews.csv"
    acts = reviews[reviews.decision.isin(["cancel", "replace"])] if len(reviews) else reviews
    if not args.execute:
        ko.append_log(planned, path, "plan")
        if len(reviews):
            ko.append_log(reviews, reviews_path, "plan")
        print("\ndry run: nothing sent")
        message = summary(reviews, now_orders)
        if args.notify and message:
            notify(*message)
        return 0
    if now_orders.empty and acts.empty:
        ko.append_log(planned, path, "plan")
        if len(reviews):
            ko.append_log(reviews, reviews_path, "plan")
        print("\nnothing to place or cancel")
        return 0
    n_cancel = int((acts.decision == "cancel").sum()) if len(acts) else 0
    what = ", ".join(f"{n} {w}" for n, w in ((n_cancel, "to cancel"), (len(replacements), "to replace"),
                                             (len(now_orders), "to place")) if n)
    if not args.yes and ko.ask(confirm, f"\nType PLACE to send these to Kalshi ({what}): ") != "PLACE":
        ko.append_log(planned, path, "declined")
        print("not confirmed: nothing sent")
        return 1

    reviews = reviews.copy().assign(status="") if len(reviews) else pd.DataFrame(columns=["order_id", "status"])
    cancelled, failed = [], False
    for i, r in acts.iterrows():
        try:
            resp = client.delete(f"/portfolio/events/orders/{r.order_id}", market_ticker=r.ticker)
            reviews.at[i, "status"] = "cancelled"
            cancelled.append({"order_id": r.order_id, "ticker": r.ticker, "client_order_id": r.client_order_id,
                              "team": r.team, "reduced_by": (resp or {}).get("reduced_by"), "ok": True, "error": "",
                              "reason": r.reason})
            print("cancelled", r.team, r.ticker, f"@ {r.price:.2f}:", r.reason)
        except KalshiError as e:
            reviews.at[i, "status"] = "failed"
            cancelled.append({"order_id": r.order_id, "ticker": r.ticker, "client_order_id": r.client_order_id,
                              "team": r.team, "reduced_by": None, "ok": False, "error": str(e), "reason": r.reason})
            print("FAILED to cancel", r.team, r.ticker, e)
            failed = True
            break
    if cancelled:
        ko._append(pd.DataFrame(cancelled).assign(cancelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds")),
                   out.parent / "kalshi" / "orders" / "cancels.csv")
    if len(acts):
        ko.append_log(reviews, reviews_path, "execute")
    if failed:  # an order that may still rest: place nothing more until a rerun sees the account as it is
        ko.append_log(planned, path, "declined")
        print("stopping: a cancel failed, so nothing is placed; rerun to see where things stand")
        return 1
    done = set(reviews.order_id[reviews.status == "cancelled"])
    if len(replacements):
        replacements = replacements[replacements.replaces.isin(done)].assign(planner="daily")
        replacements = split(replacements, now, timedelta(days=365))  # placed now whatever the lead
    results = ko.execute(pd.concat([planned, replacements], ignore_index=True), client)
    ko.append_log(results, path, "execute")
    ko.show(results[results.decision == "order"])
    print()
    ko.track(client, out)
    return 0 if not (results.status == "failed").any() else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, help="default: the latest season with predictions")
    p.add_argument("--week", type=int, help="default: cfb.week_parts' next week")
    p.add_argument("--out", type=Path, default=Path("data/cfb"))
    p.add_argument("--env", type=Path, default=Path(".env"), help="where KALSHI_ID and KALSHI_PRIVATE_KEY are")
    p.add_argument("--max-lead", type=float, default=MAX_LEAD.total_seconds() / 3600,
                   help=f"place games kicking off within this many hours (default {MAX_LEAD.total_seconds() / 3600:g})")
    p.add_argument("--max-week-share", type=float, default=MAX_WEEK_SHARE,
                   help=f"most of the bankroll filled or resting on the week's games (default {MAX_WEEK_SHARE})")
    p.add_argument("--max-week-cost", type=float, help="a fixed weekly budget in dollars, in place of the share")
    p.add_argument("--max-week-orders", type=int, default=MAX_WEEK_ORDERS,
                   help=f"games with an order or position in the week, at most (default {MAX_WEEK_ORDERS})")
    p.add_argument("--keep-edge", type=float, default=KEEP_EDGE,
                   help=f"cancel a resting order whose edge falls under this (default {KEEP_EDGE})")
    p.add_argument("--weight", type=float, default=ko.WEIGHT)
    p.add_argument("--min-edge", type=float, default=ko.MIN_EDGE)
    p.add_argument("--kelly", type=float, default=ko.KELLY_FRACTION)
    p.add_argument("--max-game-share", type=float, default=ko.MAX_GAME_SHARE,
                   help=f"most of the bankroll on one game, the biggest bet (default {ko.MAX_GAME_SHARE})")
    p.add_argument("--bankroll", type=float, help="size against this many dollars instead of the account's")
    p.add_argument("--max-contracts", type=int, help="a fixed cap on contracts an order (default none)")
    p.add_argument("--max-order-cost", type=float, help="a fixed cap in dollars an order (default none)")
    p.add_argument("--demo", action="store_true", help="Kalshi's demo exchange (needs a demo key)")
    p.add_argument("--execute", action="store_true", help="cancel, replace and place, after confirming")
    p.add_argument("--yes", action="store_true", help="skip the typed confirmation of --execute")
    p.add_argument("--notify", action="store_true",
                   help="a Windows notification when there is something to do, or the run fails")
    p.add_argument("--no-build", action="store_true", help="don't build missing or stale predictions")
    return p.parse_args(argv)


def main(argv=None, client=None, now=None, confirm=input, notify=toast):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from cfb.week_parts import upcoming_week
        season = args.season or max(int(p.name) for p in args.out.iterdir() if p.name.isdigit()
                                    and (p / "games.csv").exists())
        week = args.week or upcoming_week(season, args.out)
        if week is None:
            print(f"no week ahead in {season}: nothing to plan")
            return 0
        if not args.no_build and ensure_parts(season, week, args.out) == "waiting":
            print(f"week {week}'s predictions wait for the pull after week {week - 1}'s games (site/weekly.sh)")
            return 0
        if client is None:
            client = Client.from_env(args.env, host=kalshi.DEMO if args.demo else kalshi.PROD,
                                     read_only=not args.execute)
        if not client.signed:
            raise SystemExit(f"no KALSHI_ID / KALSHI_PRIVATE_KEY in {args.env}: the budget needs the account")
        return run_day(client, season, week, args.out, now or datetime.now(timezone.utc), args, confirm, notify)
    except BaseException as e:
        if args.notify and not isinstance(e, KeyboardInterrupt):
            notify("RAPPA Kalshi: the daily run failed", f"{type(e).__name__}: {str(e)[:150]}. See data/logs.")
        raise


if __name__ == "__main__":
    sys.exit(main())
