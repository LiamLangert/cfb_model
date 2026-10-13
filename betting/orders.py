"""Maker orders on Kalshi's college game markets, priced off RAPPA's win probability for the coming week.

The method is kalshi.maker's (../kalshi, shared with cbb_model): fair value, the guards, Kelly sizing, the
plan, placing, cancelling and tracking, and the logs. This holds what is football's: the League (series
KXNCAAFGAME, whose makers pay a fee), WEIGHT, the week's games and how Kalshi's team names are matched to
them, and the command line.

For every FBS-vs-FBS game in cfb.week_parts' predictions for the week
(data/cfb/<season>/parts_wk<NN>.csv) not yet kicked off, this finds Kalshi's two winner markets (one YES
contract per team), reads both books, and decides whether to rest a buy on one side, as kalshi.maker.plan
decides it.

Fair value is not RAPPA's probability itself. Next to Kalshi's Tuesday price, RAPPA's walk-forward win
probability got a weight of about 0.26 +/- 0.17 in predicting the winner (935 FBS games, 2025-26), and the
price drifted toward it by about 5% of the gap by kickoff. So fair value moves the market's mid WEIGHT of the
way toward RAPPA, in log-odds. The edge here has not been shown to be profitable; see the numbers above.
Events are matched to games by date and betting.names' similarity of the two teams' names, summed (MIN_MATCH
of 2).

By default nothing is sent: the plan is printed and logged. --execute asks for a typed confirmation (or
--yes), then places the orders one at a time, stopping at the first error it does not expect. Every run
appends to data/kalshi/orders/<season>_wk<NN>.csv. --status lists the resting orders and positions;
--cancel cancels the resting orders this week's log says were placed here, and --cancel-order NAME just
the one NAME picks out (an order id, ticker, or part of a ticker or team name; it must match exactly one).
Both ask for a typed CANCEL, log to data/kalshi/orders/cancels.csv, and take a --track snapshot after.
--smoke places one 1-contract bid at 1c, checks it rests, and cancels it, to test the write path. --track
(and every --execute, once it has placed) saves what became of each order placed (kalshi.maker.track):
run it after the games, and whenever else, to keep it current. betting.daily is the run ./bets.sh uses,
placing a game at a time in the 30 hours before it.

    uv run -m betting.orders              # plan only
    uv run -m betting.orders --execute    # place, after confirming
    uv run -m betting.orders --cancel     # cancel ours (asks first)
    uv run -m betting.orders --cancel-order Ohio --cancel-order "Southern Miss"
    uv run -m betting.orders --smoke      # test the write path
    uv run -m betting.orders --track      # save what became of our orders

Run cfb.week_parts first so the week's predictions are current.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from betting.names import similarity
from kalshi import maker
from kalshi.client import DEMO, PROD, Client
from kalshi.league import League
from kalshi.maker import (EXPIRE_BEFORE, IMPROVE_SHARE, KELLY_FRACTION, MAX_BELOW_BID, MAX_BOOK_GAP,  # noqa: F401
                          MAX_DISAGREE, MAX_GAME_SHARE, MAX_ORDERS, MAX_PRICE, MAX_QUEUE_SHARE, MAX_TOTAL_SHARE,
                          MAX_WIDTH, MIN_CONTRACT_SHARE, MIN_EDGE, MIN_LEAD, MIN_PRICE, _append, _merge, append_log,
                          ask, contracts, execute, expected_volume, kelly_share, look_up, order_body)

log = logging.getLogger(__name__)

SERIES = "KXNCAAFGAME"
ET = ZoneInfo("America/New_York")
LEAGUE = League(series=SERIES, fee_type="quadratic_with_maker_fees", model="RAPPA", model_col="rappa_home",
                start="kickoff", the_start="kickoff", starts="kicks off", id_prefix="rappa-cfb",
                namespace=uuid.UUID("5d1f6a52-3a43-4b8e-9a43-6f1d0c2b7e11"))

WEIGHT = 0.25          # how far toward RAPPA fair value moves from the market's mid, in log-odds
MIN_MATCH = 1.6        # summed name similarity (2 is a perfect match) to pair a Kalshi event with a game


def maker_fee(price):
    return LEAGUE.maker_fee(price)


def fair_value(mid, rappa, weight=None):
    """The market's mid moved `weight` (WEIGHT if None) of the way toward RAPPA, in log-odds."""
    return maker.fair_value(mid, rappa, WEIGHT if weight is None else weight)


def best_price(fair, book, min_edge=MIN_EDGE, max_below_bid=MAX_BELOW_BID, improve=True):
    return maker.best_price(fair, book, min_edge, max_below_bid, improve, LEAGUE.fee_type)


def kickoff_utc(game):
    """The game's kickoff, from parts' Eastern gameday and gametime; None if the time is TBD."""
    if pd.isna(game.get("gametime")) or pd.isna(game.get("gameday")):
        return None
    local = datetime.fromisoformat(f"{game['gameday']}T{game['gametime']}").replace(tzinfo=ET)
    return local.astimezone(timezone.utc)


def slate(games, week):
    """The parts slate in kalshi.maker.plan's columns."""
    rows = games.to_dict("records")
    return pd.DataFrame({"game_id": [g["game_id"] for g in rows],
                         "game": [f"{g['away_team']} at {g['home_team']}" for g in rows],
                         "p_home": [g["rappa_wp"] for g in rows],
                         "start": pd.to_datetime([kickoff_utc(g) for g in rows], utc=True),
                         "week": week})


def match_events(games, markets, min_match=MIN_MATCH):
    """Pair each Kalshi event with the parts game it is (kalshi.maker.match_events): within a day of its
    ticker's date, by the two teams' summed name similarity."""
    def day_of(g):
        return None if pd.isna(g.get("gameday")) else datetime.fromisoformat(g["gameday"]).date()

    def scores(a, b, g):
        return (similarity(a["yes_sub_title"], g["home_team"]) + similarity(b["yes_sub_title"], g["away_team"]),
                similarity(b["yes_sub_title"], g["home_team"]) + similarity(a["yes_sub_title"], g["away_team"]))

    return maker.match_events(games, markets, day_of, scores, min_match)


def client_order_id(ticker, season, week, run):
    return maker.client_order_id(ticker, season, week, run, league=LEAGUE)


def plan(games, matches, books, busy_events, now, season, week, weight=None, min_edge=MIN_EDGE,
         max_contracts=None, max_order_cost=None, max_total_cost=None, max_orders=MAX_ORDERS, balance=None,
         bankroll=None, kelly=KELLY_FRACTION, max_game_share=MAX_GAME_SHARE, max_total_share=MAX_TOTAL_SHARE,
         held=None):
    """kalshi.maker.plan on the week's parts slate, at WEIGHT unless weight is given."""
    return maker.plan(slate(games, week), matches, books, busy_events, now, season, league=LEAGUE,
                      weight=WEIGHT if weight is None else weight, min_edge=min_edge, max_contracts=max_contracts,
                      max_order_cost=max_order_cost, max_total_cost=max_total_cost, max_orders=max_orders,
                      balance=balance, bankroll=bankroll, kelly=kelly, max_game_share=max_game_share,
                      max_total_share=max_total_share, held=held)


def account(client, tickers):
    return maker.account(client, tickers, LEAGUE.fee_type)


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


# ---------- the logs, under data/kalshi/orders/ ----------

def orders_folder(out):
    return out.parent / "kalshi" / "orders"


def log_path(season, week, out):
    return maker.log_path(season, week, orders_folder(out))


def show(frame):
    maker.show(frame, league=LEAGUE)


def cancel_ours(client, season, week, out, which=None, confirm=input, yes=False):
    """Cancel resting orders this week's log says were placed here (kalshi.maker.cancel_ours)."""
    return maker.cancel_ours(client, orders_folder(out), which, confirm, yes, logs=[log_path(season, week, out)])


def status(client, season, week, out):
    return maker.status(client, SERIES)


def smoke(client, markets, books, now, out, sleep=None):
    return maker.smoke(client, books, now, orders_folder(out), sleep)


def our_orders(out):
    return maker.our_orders(orders_folder(out))


def track(client, out, now=None):
    return maker.track(client, orders_folder(out), now, league=LEAGUE)


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
        client = Client.from_env(args.env, host=DEMO if args.demo else PROD, read_only=not writes)
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
    books = maker.books_for(client, matches)
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
    if games.empty:
        print(f"every game of week {week} has kicked off: nothing to plan")
        return 0
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
