"""The daily Kalshi run: the week's plan, placed a game at a time, 2-30 hours before each kickoff.

betting.orders plans the whole week at once, and run on Tuesday its resting
bids were picked off: in backtests a bid placed Tuesday at 10am made +0.9c a
contract on the fills it got, against +3 to +8c for one placed 6-72 hours
before kickoff (each about +/- 3-4c), while the bids that never filled would
have won. Prices that sit for days get hit by whoever learns something first.
So it only places on games kicking off between MIN_LEAD (2h) and MAX_LEAD
(30h) from now, and is run again on the days after for the rest: run at 4pm,
Friday's run takes Friday's games and Saturday's up to 10pm; Saturday's takes
the rest.

Each run, with betting.orders' pricing, guards and sizing (kalshi.maker's), and kalshi.review's review:

  1. Reviews every resting order this module or betting.orders placed on
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

--notify (for a run nobody watches) pops up a Windows notification when there is
something to place or cancel, or when the run fails. The run builds the
week's predictions (cfb.week_parts) first when they are missing or older
than the season's games and plays.

Logs, in data/kalshi/orders/: the plan and execute rows go to the week's
<season>_wk<NN>.csv beside betting.orders' (planner "daily", with
lead_hours, the hours from the run to kickoff), so --track follows the
orders placed here; reviews.csv has every review; cancels.csv every cancel.

    uv run -m betting.daily              # plan only
    uv run -m betting.daily --execute    # place, after confirming
    ./bets.sh                            # --track, then --execute, logged
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from betting import display
from betting import orders as ko
from kalshi import maker
from kalshi import review as shared
from kalshi.client import DEMO, PROD, Client, ticker_date
from kalshi.review import KEEP_EDGE, MAX_LEAD, book_without, event_of, own_bids, toast  # noqa: F401

log = logging.getLogger(__name__)

MIN_LEAD = ko.MIN_LEAD
# The week's budget, in place of betting.orders' per-run MAX_TOTAL_SHARE and MAX_ORDERS: the same caps, for a
# week of daily runs instead of one Tuesday run. The money is a share of the bankroll, so it grows with it.
MAX_WEEK_SHARE = ko.MAX_TOTAL_SHARE
MAX_WEEK_ORDERS = ko.MAX_ORDERS

REASONS = {"run's total cost cap reached": "the week's budget is held for bigger edges",
           "order count cap reached": "the week's order cap is held for bigger edges"}


# ---------- 1. the review ----------

def review(resting, games, matches, payloads, now, season, week, bankroll, weight=None, min_edge=ko.MIN_EDGE,
           keep_edge=KEEP_EDGE, kelly=ko.KELLY_FRACTION, max_game_share=ko.MAX_GAME_SHARE,
           max_contracts=None, max_order_cost=None):
    """kalshi.review.review on the week's parts slate: keep, cancel or replace each of our resting orders."""
    return shared.review(resting, ko.slate(games, week), matches, payloads, now, season, bankroll, league=ko.LEAGUE,
                         weight=ko.WEIGHT if weight is None else weight, min_edge=min_edge, keep_edge=keep_edge,
                         kelly=kelly, max_game_share=max_game_share, max_contracts=max_contracts,
                         max_order_cost=max_order_cost)


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
    return shared.committed(o, ko.LEAGUE.fee_type)


def holdings(resting, positions):
    return shared.holdings(resting, positions, ko.LEAGUE.fee_type)


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
    return shared.split(planned, now, max_lead, league=ko.LEAGUE, reasons=REASONS)


def summary(reviews, now_orders):
    return shared.summary(reviews, now_orders, league=ko.LEAGUE)


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
    if games.empty:  # Saturday night to Sunday: the week is played out and the next one not pulled yet
        print(f"every game of week {week} has kicked off: nothing to plan")
        return 0
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
    bankroll = maker.funds(balance, 0.0, exposure, args.reserve, args.bankroll)[1]

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
    # What the kept resting orders and the replacements hold back is taken off, and the reserve.
    available = maker.funds(balance, sum(committed(o) for o in kept) + extra, exposure, args.reserve)[0]
    held, odd = holdings(kept, positions)
    busy = {e: "held on both teams or as NO contracts" for e in odd}
    moving = reviews.ticker[reviews.decision.isin(["cancel", "replace"])] if len(reviews) else []
    busy |= {event_of(t): "its resting order is cancelled or replaced in this run" for t in moving}
    log.info("balance $%.2f ($%.2f held back), $%.2f to spend, bankroll $%.2f", balance, args.reserve, available,
             bankroll)

    planned = ko.plan(games, matches, {t: book_without(p, own_bids(resting).get(t)) for t, p in payloads.items()},
                      busy, now, season, week, args.weight, args.min_edge, args.max_contracts, args.max_order_cost,
                      budget_left, orders_left, available, bankroll, args.kelly, args.max_game_share,
                      held=held)
    planned = split(planned, now, timedelta(hours=args.max_lead)).assign(planner="daily")
    show_week(spent, events, budget, args.max_week_orders, reviews, planned, timedelta(hours=args.max_lead), basis)

    return shared.finish(client, planned, reviews, replacements, now, ko.orders_folder(out), season, week,
                         league=ko.LEAGUE, execute=args.execute, yes=args.yes, confirm=confirm,
                         notify=notify if args.notify else None)


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
    p.add_argument("--reserve", type=float, default=maker.RESERVE,
                   help=f"dollars in the account never traded: the bankroll is the account less this "
                        f"(default {maker.RESERVE:g}, kalshi.maker.RESERVE)")
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
            print(f"week {week}'s predictions wait for the pull after week {week - 1}'s games (./update.sh)")
            return 0
        if client is None:
            client = Client.from_env(args.env, host=DEMO if args.demo else PROD, read_only=not args.execute)
        if not client.signed:
            raise SystemExit(f"no KALSHI_ID / KALSHI_PRIVATE_KEY in {args.env}: the budget needs the account")
        return run_day(client, season, week, args.out, now or datetime.now(timezone.utc), args, confirm, notify)
    except BaseException as e:
        if args.notify and not isinstance(e, KeyboardInterrupt):
            notify("RAPPA Kalshi: the daily run failed", f"{type(e).__name__}: {str(e)[:150]}. See data/logs.")
        raise


if __name__ == "__main__":
    sys.exit(main())
