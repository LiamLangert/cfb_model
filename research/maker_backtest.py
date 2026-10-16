"""A backtest of maker orders on Kalshi's college football winner markets, priced off RAPPA as kalshi.maker prices
them: when to place them, and what the daily run's caps and sizing make of them.

    uv run -m research.maker_backtest                                  # 2025 and 2026 so far
    uv run -m research.maker_backtest --variants daily hourly:6 hourly:12 hourly:18 --weights 0.25
    uv run -m research.maker_backtest --seasons 2025 --run-hours 10 16

A port of ../cbb_model's research.maker_backtest (WCBB, off T-Rank) to football. Every game of the seasons
with a Kalshi winner market (research.pull_kalshi's 1-minute candles and trades) and a RAPPA probability
(the website's, ~/my_website/fb/data/seasons/cfb/<season>.json: cfb.week_parts.walk's walk-forward
rappa_wp, which cfb.handoff hands over) is put through the plan at each placement time, with the live
constants (kalshi.maker, kalshi.review, kalshi.cfb.orders):
  fair      the team's mid moved WEIGHT of the way toward RAPPA in log-odds (--weights tries others, which
            these games then pick between, so only the live WEIGHT is not fitted on them)
  price     the highest of the best bid plus a cent and the best bid that is below the ask and leaves
            fair - price - maker fee >= MIN_EDGE (football charges makers kalshi.client.maker_fee)
  guards    both teams' books two-sided and at most MAX_WIDTH wide, their mids summing to within
            MAX_BOOK_GAP of 1, the price within [MIN_PRICE, MAX_PRICE], RAPPA within MAX_DISAGREE of the
            market, the kickoff at least MIN_LEAD away; one order a game, on the side with the bigger edge,
            expiring EXPIRE_BEFORE kickoff
The live planner bids a cent above the best bid only when the queue there is deep (kalshi.maker.IMPROVE_SHARE);
Kalshi keeps no history of the queue, so this always improves when it can, as the WCBB backtest does.
RAPPA's probability is known from 10am Eastern on the Tuesday of the game's week (the handoff, after the
Tuesday pull), and fixed after: the week is rated once.

When the order is placed:
  open        the first hour after the market opens and RAPPA is known at which both books pass the guards
  tuesday     10am Eastern on the Tuesday of the game's week, once (the old Tuesday plan)
  T-72h ... T-2h
              that long before kickoff, once
  daily       kalshi.cfb.daily: a run at 4pm Eastern every day (--run-hours for others, or several), placing
              on games kicking off 2 to 30 hours later (daily:H for H hours instead), and at each later run
              before the order expires, reviewing it: kept while fair - price - fee >= KEEP_EDGE and nobody
              bids above it, cancelled when the edge is under KEEP_EDGE, and replaced at the best price still
              clearing MIN_EDGE (or cancelled, if none) when outbid. A game without an edge at its first run
              is looked at again at the next one, if that is still 2 hours out.
  hourly:H    the same, run every hour, placing on games kicking off 2 to H hours later

Fills, from Kalshi's trades in the order's market while it rests: a YES bid fills when a taker sells YES (buys
NO) at or below its price. A bid a cent above the best bid is alone at the top, so any such trade at or below
it reaches it first. A bid that joins the best bid has a queue ahead of it that Kalshi's history does not
record, so two bounds: `at` fills it on any trade at its price (first in the queue), `through` only on one
below it (last). The market's response to a new best bid is not modelled. A filled contract makes its result
less its price and the maker fee.

Two views:
  orders   every order the plan would place, one contract each, no caps: how many, how often filled, and P&L
           a filled contract (mean and its standard error), by placement time, weight and fill bound
  plan     the live sizing on a bankroll (--bankroll, $1000), with kalshi.cfb.daily's caps on what is active
           at once: half Kelly on fair value, at most MAX_GAME_SHARE on a game, at most MAX_ACTIVE_ORDERS games
           and cfb's share of the bankroll resting or held on games not yet settled. The orders go in the order
           they are placed, the biggest edges first within a run; one the caps have no room for is skipped. A
           game is active from its order until the order is cancelled or expires unfilled, or, once any of it
           fills, until its market closes on the result; the bankroll is the start plus what settled games made.
           An order fills up to the contracts traded through it.

Writes data/cfb/maker/: orders.csv (every game, variant and weight: the decision or why not, the order, its
fills and P&L), summary.csv, plan.csv and plan_ledger.csv (the plan's every bet), with --suffix on each.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from kalshi import maker, strategies
from kalshi.cfb import orders as ko
from kalshi.client import maker_fee as kalshi_maker_fee
from kalshi.review import KEEP_EDGE, MAX_LEAD

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
KALSHI = Path("data/kalshi")
SITE = Path.home() / "my_website" / "fb" / "data" / "seasons" / "cfb"
OUT = Path("data/cfb/maker")
SERIES = "KXNCAAFGAME"
SEASONS = (2025, 2026)

# The live constants.
WEIGHT = ko.WEIGHT
MIN_EDGE = maker.MIN_EDGE
TICK = 0.01
MAX_WIDTH = maker.MAX_WIDTH
MAX_BOOK_GAP = maker.MAX_BOOK_GAP
MIN_PRICE, MAX_PRICE = maker.MIN_PRICE, maker.MAX_PRICE
MAX_DISAGREE = maker.MAX_DISAGREE
MIN_LEAD = maker.MIN_LEAD
EXPIRE_BEFORE = maker.EXPIRE_BEFORE
KELLY_FRACTION = maker.KELLY_FRACTION
MIN_CONTRACT_SHARE = maker.MIN_CONTRACT_SHARE
MAX_GAME_SHARE = maker.MAX_GAME_SHARE
MAX_ACTIVE_SHARE = strategies.get("cfb").share
MAX_ACTIVE_ORDERS = ko.MAX_ORDERS
FEE_TYPE = ko.LEAGUE.fee_type
DAILY_RUN = time(16, 0)
DAILY_RUNS = (DAILY_RUN,)   # --run-hours: several runs a day
HOURLY_RUNS = tuple(time(h, 0) for h in range(24))
# The handoff: the week rated after the Tuesday pull.
KNOWN_AT = time(10, 0)
# A market closes once the winner is declared; this long after kickoff where it has no close time.
SETTLE_AFTER = timedelta(hours=4)

VARIANTS = ("open", "tuesday", "T-72h", "T-48h", "T-24h", "T-12h", "T-6h", "T-2h", "daily")   # and daily:H, hourly:H
LEADS = {f"T-{h}h": timedelta(hours=h) for h in (72, 48, 24, 12, 6, 2)}


def logit(p):
    return math.log(p / (1 - p))


def inv_logit(x):
    return 1 / (1 + math.exp(-x))


def fair_value(mid, model, weight):
    model = min(max(model, 1e-4), 1 - 1e-4)
    return inv_logit(logit(mid) + weight * (logit(model) - logit(mid)))


def cents(x):
    return round(round(x / TICK) * TICK, 2)


def fee(price):
    return kalshi_maker_fee(price, FEE_TYPE)


def best_price(fair, bid, ask, min_edge=MIN_EDGE):
    """The highest of bid + a cent and bid that is below the ask and clears min_edge after the maker fee, or None."""
    for price in (cents(min(bid + TICK, ask - TICK)), cents(bid)):
        if 0 < price < ask - 1e-9 and fair - price - fee(price) >= min_edge - 1e-9:
            return price
    return None


# ---------- data ----------

class Market:
    """One market's quotes (the close of each 1-minute candle) and trades, as arrays."""

    def __init__(self, candles, trades):
        c = np.array([[r[0], np.nan if r[4] is None else r[4], np.nan if r[8] is None else r[8]] for r in candles],
                     dtype=float).reshape(-1, 3)
        self.ts, self.bid, self.ask = c[:, 0], c[:, 1], c[:, 2]
        t = np.array(trades, dtype=float).reshape(-1, 4)
        self.t_ts, self.t_price, self.t_count, self.t_taker_yes = t[:, 0] / 1000, t[:, 1], t[:, 2], t[:, 3]

    def quote(self, ts):
        """(bid, ask) at ts; None for an empty side."""
        i = np.searchsorted(self.ts, ts, side="right") - 1
        if i < 0:
            return None, None
        bid, ask = self.bid[i], self.ask[i]
        return (bid if np.isfinite(bid) and bid > 0 else None), (ask if np.isfinite(ask) and ask < 1 else None)

    def sold_into(self, start, end, price, strict):
        """(contracts, first time) sold by takers at or below price (below it, if strict) between start and end."""
        m = (self.t_ts >= start) & (self.t_ts < end) & (self.t_taker_yes == 0)
        m &= (self.t_price < price - 1e-9) if strict else (self.t_price <= price + 1e-9)
        if not m.any():
            return 0.0, None
        return float(self.t_count[m].sum()), float(self.t_ts[m][0])


def read_jsonl(path):
    with path.open() as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def load_markets(season, tickers):
    d = KALSHI / str(season)
    store = {"candles": {}, "trades": {}}
    for name in store:
        for rec in read_jsonl(d / f"{SERIES}_{name}.jsonl"):
            if rec["ticker"] in tickers:
                store[name][rec["ticker"]] = rec[name]
    return {t: Market(store["candles"].get(t, []), store["trades"].get(t, [])) for t in tickers}


def known_from(kickoff):
    """When RAPPA's probability for a game kicking off then was handed over: 10am Eastern on its week's Tuesday."""
    day = kickoff.astimezone(ET).date()
    tuesday = day - timedelta(days=(day.weekday() - 1) % 7)
    return datetime.combine(tuesday, KNOWN_AT, ET)


def site_games(season):
    """The website's RAPPA games for the season with a kickoff time, in kalshi.cfb.orders' columns."""
    data = json.loads((SITE / f"{season}.json").read_text())
    rows = [{"game_id": g["id"], "week": w["week"], "home_team": g["home"]["team"], "away_team": g["away"]["team"],
             "p_home": g["home_wp"], "gameday": g["gameday"], "gametime": g["gametime"]}
            for w in data["games"] for g in w["games"]]
    return pd.DataFrame(rows)


def load_games(season):
    """Each game matched to its two winner markets (kalshi.cfb.orders.match_events), with its kickoff, when RAPPA
    was known, when the markets opened and when they settled, and each side's result."""
    games = site_games(season)
    games = games[games.gametime.notna()].reset_index(drop=True)
    markets = [m for m in read_jsonl(KALSHI / str(season) / f"{SERIES}_markets.jsonl")
               if m.get("result") in ("yes", "no")]
    by_ticker = {m["ticker"]: m for m in markets}
    matches = ko.match_events(games, markets)
    rows = []
    for g in games.to_dict("records"):
        hit = matches.get(g["game_id"])
        if hit is None:
            continue
        home, away = by_ticker[hit["home"]["ticker"]], by_ticker[hit["away"]["ticker"]]
        if home["result"] == away["result"]:
            continue   # a tie or a void: no winner to grade
        kick = ko.kickoff_utc(g)
        close = max(pd.Timestamp(m["close_time"]).timestamp() for m in (home, away) if m.get("close_time"))
        rows.append({**g, "event": hit["event"], "home_ticker": home["ticker"], "away_ticker": away["ticker"],
                     "kick_ts": kick.timestamp(), "known_ts": known_from(kick).timestamp(),
                     "open_ts": max(pd.Timestamp(m["open_time"]).timestamp() for m in (home, away)),
                     "settle_ts": close if close > kick.timestamp() else kick.timestamp()
                     + SETTLE_AFTER.total_seconds(),
                     "home_win": int(home["result"] == "yes"), "season": season})
    log.info("%d: %d games on the website with a kickoff, %d matched to settled markets", season, len(games),
             len(rows))
    return pd.DataFrame(rows)


# ---------- the plan at one moment ----------

def books_ok(hb, ab):
    """Why the two books cannot be priced off at this moment, or None."""
    if None in hb or None in ab:
        return "a book is empty"
    if max(hb[1] - hb[0], ab[1] - ab[0]) > MAX_WIDTH + 1e-9:
        return "book too wide"
    if abs((hb[0] + hb[1]) / 2 + (ab[0] + ab[1]) / 2 - 1) > MAX_BOOK_GAP + 1e-9:
        return "the two books disagree"
    return None


def decide(g, ts, markets, weight):
    """The order the plan would place at ts, as a dict, or (None, reason)."""
    if g.kick_ts - ts < MIN_LEAD.total_seconds():
        return None, "kickoff too soon"
    if ts < g.known_ts:
        return None, "RAPPA not handed over yet"
    hb, ab = markets[g.home_ticker].quote(ts), markets[g.away_ticker].quote(ts)
    why = books_ok(hb, ab)
    if why:
        return None, why
    market_home = ((hb[0] + hb[1]) / 2 + 1 - (ab[0] + ab[1]) / 2) / 2
    if abs(g.p_home - market_home) > MAX_DISAGREE + 1e-9:
        return None, "RAPPA too far from the market"
    options = []
    for side, ticker, (bid, ask), q in (("home", g.home_ticker, hb, g.p_home), ("away", g.away_ticker, ab,
                                                                                1 - g.p_home)):
        fair = fair_value((bid + ask) / 2, q, weight)
        price = best_price(fair, bid, ask)
        if price is None or not MIN_PRICE <= price <= MAX_PRICE:
            continue
        options.append({"side": side, "ticker": ticker, "bid": bid, "ask": ask, "fair": fair, "price": price,
                        "edge": fair - price - fee(price), "improved": price > bid + 1e-9, "p_model": q,
                        "y": g.home_win if side == "home" else 1 - g.home_win})
    if not options:
        return None, "no edge at a maker price"
    return max(options, key=lambda o: o["edge"]), ""


def runs(g, max_lead=MAX_LEAD, hours=None):
    """The runs (DAILY_RUNS each day, or hours) from max_lead before kickoff to MIN_LEAD before it."""
    kick = datetime.fromtimestamp(g.kick_ts, timezone.utc).astimezone(ET)
    day, out = (kick - max_lead).date(), []
    while day <= kick.date():
        for h in hours or DAILY_RUNS:
            r = datetime.combine(day, h, ET)
            if MIN_LEAD <= kick - r <= max_lead:
                out.append(r.timestamp())
        day += timedelta(days=1)
    return sorted(out)


def variant_lead(variant):
    """daily -> 30h, daily:48 or hourly:12 -> that many hours."""
    return timedelta(hours=float(variant.split(":")[1])) if ":" in variant else MAX_LEAD


def reviewed(variant):
    return variant.startswith(("daily", "hourly"))


def place_times(g, variant):
    """The moments the variant would look at the game: the first that places is the order's."""
    if variant in LEADS:
        return [g.kick_ts - LEADS[variant].total_seconds()]
    if variant == "tuesday":
        return [g.known_ts]
    if variant.startswith("daily"):
        return runs(g, variant_lead(variant))
    if variant.startswith("hourly"):
        return runs(g, variant_lead(variant), HOURLY_RUNS)
    if variant == "open":
        start = math.ceil(max(g.open_ts, g.known_ts) / 3600) * 3600
        return list(np.arange(start, g.kick_ts - MIN_LEAD.total_seconds(), 3600.0))
    raise ValueError(variant)


def review(g, order, price, ts, markets, weight):
    """A run's review of a resting order at price: (new price or None, what was done)."""
    hb, ab = markets[g.home_ticker].quote(ts), markets[g.away_ticker].quote(ts)
    if books_ok(hb, ab):
        return None, "cancelled: the book could not be priced off"
    book = hb if order["side"] == "home" else ab
    fair = fair_value((book[0] + book[1]) / 2, order["p_model"], weight)
    if fair - price - fee(price) < KEEP_EDGE - 1e-9:
        return None, "cancelled: edge under KEEP_EDGE"
    if book[0] > price + 1e-9:
        new = best_price(fair, book[0], book[1])
        return new, (f"replaced at {new:.2f}" if new else "cancelled: outbid, no price clears MIN_EDGE")
    return price, "kept"


def simulate(g, variant, markets, weight):
    """One row: the order the variant places on the game (or why none), and its fills under both bounds."""
    row = {"season": g.season, "week": g.week, "game_id": g.game_id, "variant": variant, "weight": weight,
           "kickoff": datetime.fromtimestamp(g.kick_ts, ET).isoformat(timespec="minutes")}
    expire = g.kick_ts - EXPIRE_BEFORE.total_seconds()
    order, reason, placed_at = None, "never looked at", None
    times = place_times(g, variant)
    for ts in times:
        o, why = decide(g, ts, markets, weight)
        if variant == "open" and why in ("a book is empty", "book too wide", "the two books disagree"):
            reason = why
            continue  # not yet a market to price off: try the next hour
        order, reason, placed_at = o, why, ts
        if not reviewed(variant) or o is not None:
            break
    if order is None:
        row |= {"decision": "skip", "reason": reason}
        return row
    row |= {"decision": "order", "reason": "", "placed": placed_at, "settle": g.settle_ts,
            "lead_h": (g.kick_ts - placed_at) / 3600,
            **{k: order[k] for k in ("side", "ticker", "bid", "ask", "fair", "price", "edge", "improved", "p_model",
                                     "y")}}
    m = markets[order["ticker"]]
    # A reviewed order's life: segments between its reviews, each at its price then, ending at a cancel or expiry.
    segments, price, start = [], order["price"], placed_at
    if reviewed(variant):
        for r in [t for t in times if placed_at < t < expire]:
            segments.append((start, r, price))
            price, row["review"] = review(g, order, price, r, markets, weight)
            if price is None:
                break
            start = r
    if price is not None:
        segments.append((start, expire, price))
    row["segments"], row["rest_end"] = len(segments), segments[-1][1] if segments else placed_at
    for bound, strict in (("at", False), ("through", True)):
        # Each segment's contracts sold into the order: what it could have filled, at that segment's price.
        fills = []
        for s, e, pr in segments:
            improved = pr > (m.quote(s)[0] or 0) + 1e-9
            qty, when = m.sold_into(s, e, pr, strict=strict and not improved)
            fills.append((qty, pr, when))
        first = next((f for f in fills if f[0] > 0), None)
        row[f"fills_{bound}"] = [(q, pr) for q, pr, _ in fills]
        row[f"filled_{bound}"] = 1.0 if first else 0.0
        row[f"fill_price_{bound}"] = first[1] if first else None
        row[f"pnl_{bound}"] = (order["y"] - first[1] - fee(first[1])) if first else 0.0
        row[f"fill_lead_h_{bound}"] = (g.kick_ts - first[2]) / 3600 if first else np.nan
        row[f"volume_{bound}"] = sum(q for q, _, _ in fills)
    return row


# ---------- the views ----------

def orders_table(df):
    rows = []
    for (variant, weight), g in df.groupby(["variant", "weight"], sort=False):
        o = g[g["decision"] == "order"]
        for bound in ("at", "through"):
            f = o[o[f"filled_{bound}"] > 0]
            pnl = f[f"pnl_{bound}"]
            rows.append({"variant": variant, "weight": weight, "fills": bound, "games": len(g), "orders": len(o),
                         "improved": o["improved"].mean() if len(o) else np.nan,
                         "lead_h": o["lead_h"].median() if len(o) else np.nan,
                         "edge_placed": o["edge"].mean() if len(o) else np.nan,
                         "filled": len(f), "fill_rate": len(f) / len(o) if len(o) else np.nan,
                         "price_filled": f[f"fill_price_{bound}"].mean() if len(f) else np.nan,
                         "won": f["y"].mean() if len(f) else np.nan,
                         "pnl_mean": pnl.mean() if len(f) else np.nan,
                         "pnl_se": pnl.std(ddof=1) / np.sqrt(len(f)) if len(f) > 1 else np.nan,
                         "pnl_total": pnl.sum(),
                         "unfilled_won": o.loc[o[f"filled_{bound}"] == 0, "y"].mean() if len(o) else np.nan})
    return pd.DataFrame(rows)


def kelly_contracts(fair, price, bankroll):
    cost = price + fee(price)
    stake = KELLY_FRACTION * max(0.0, (fair - cost) / (1 - cost)) * bankroll
    n = math.floor(stake / cost + 1e-9)
    if n == 0 and stake >= MIN_CONTRACT_SHARE * cost - 1e-9:
        n = 1
    return min(n, math.floor(MAX_GAME_SHARE * bankroll / cost + 1e-9))


def plan_table(df, start_bankroll, max_orders=MAX_ACTIVE_ORDERS, max_share=MAX_ACTIVE_SHARE):
    """The live sizing on a bankroll with the caps on what is active at once, for each variant and weight and both
    fill bounds: the orders in the order they are placed, the biggest edges first at the same moment."""
    rows, ledgers = [], []
    for (variant, weight), g in df[df["decision"] == "order"].groupby(["variant", "weight"], sort=False):
        g = g.assign(_neg_edge=-g["edge"]).sort_values(["placed", "_neg_edge"])
        for bound in ("at", "through"):
            bankroll, peak, drawdown, staked, pnl_total = start_bankroll, start_bankroll, 0.0, 0.0, 0.0
            n_orders = n_filled = most = capped_games = capped_money = 0
            most_held = 0.0
            # When what is active ends: (time, order, change in games, change in dollars held, P&L realised then).
            ends, games, held, seq = [], 0, 0.0, itertools.count()

            def settle(until):
                nonlocal ends, games, held, bankroll, pnl_total, peak, drawdown
                for _, _, d_games, d_held, pnl in sorted(e for e in ends if e[0] <= until):
                    games, held = games + d_games, held + d_held
                    if pnl is not None:
                        bankroll, pnl_total = bankroll + pnl, pnl_total + pnl
                        peak = max(peak, bankroll)
                        drawdown = max(drawdown, (peak - bankroll) / peak)
                ends = [e for e in ends if e[0] > until]

            for r in g.itertuples():
                settle(r.placed)
                n = kelly_contracts(r.fair, r.price, bankroll)
                cost = n * (r.price + fee(r.price))
                if n < 1:
                    continue
                if games >= max_orders:
                    capped_games += 1
                    continue
                if held + cost > max_share * bankroll + 1e-9:
                    capped_money += 1
                    continue
                # Up to n contracts, from the order's segments in turn, each at its own price.
                left, filled, pnl, paid = n, 0.0, 0.0, 0.0
                for qty, price in getattr(r, f"fills_{bound}"):
                    take = min(left, qty)
                    each = price + fee(price)
                    filled, pnl, paid, left = filled + take, pnl + take * (r.y - each), paid + take * each, left - take
                games, held = games + 1, held + cost
                most, most_held = max(most, games), max(most_held, held / bankroll)
                if filled > 0:
                    # Resting until it stops, then what filled is held until the market closes on the result.
                    ends += [(r.rest_end, next(seq), 0, paid - cost, None), (r.settle, next(seq), -1, -paid, pnl)]
                else:
                    ends.append((r.rest_end, next(seq), -1, -cost, None))
                staked += paid
                n_orders += 1
                n_filled += filled > 0
                ledgers.append({"variant": variant, "weight": weight, "fills": bound,
                                "placed": datetime.fromtimestamp(r.placed, ET).isoformat(timespec="minutes"),
                                "game_id": r.game_id, "ticker": r.ticker, "price": r.price, "contracts": n,
                                "filled": filled, "y": r.y, "pnl": pnl, "bankroll_before": bankroll,
                                "active_before": games - 1})
            settle(math.inf)
            rows.append({"variant": variant, "weight": weight, "fills": bound, "orders": n_orders, "filled": n_filled,
                         "capped_games": capped_games, "capped_money": capped_money, "most_active": most,
                         "most_held": most_held,
                         "staked": staked, "pnl": pnl_total, "roi_on_stake": pnl_total / staked if staked else np.nan,
                         "bankroll_end": bankroll, "growth": bankroll / start_bankroll - 1,
                         "max_drawdown": drawdown})
    return pd.DataFrame(rows), pd.DataFrame(ledgers)


def show(title, df, floats="{:.3f}"):
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", floats.format):
        print(df.to_string(index=False))


def main(argv=None):
    global DAILY_RUNS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", type=int, nargs="+", default=list(SEASONS))
    p.add_argument("--weights", type=float, nargs="+", default=[WEIGHT, 0.5, 1.0])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   help=f"any of {', '.join(VARIANTS)}, daily:H and hourly:H (H hours of lookahead)")
    p.add_argument("--run-hours", type=int, nargs="+", default=[DAILY_RUN.hour],
                   help="the hours of each day's daily runs, Eastern (default 16)")
    p.add_argument("--suffix", default="", help="added to the output files' names")
    p.add_argument("--bankroll", type=float, default=1000.0)
    a = p.parse_args(argv)
    for v in a.variants:
        if v not in VARIANTS and not (v.split(":")[0] in ("daily", "hourly") and v.count(":") == 1):
            p.error(f"unknown variant {v}")
    DAILY_RUNS = tuple(time(h, 0) for h in sorted(a.run_hours))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for season in a.seasons:
        games = load_games(season)
        tickers = set(games["home_ticker"]) | set(games["away_ticker"])
        markets = load_markets(season, tickers)
        rows += [simulate(g, v, markets, w) for w in a.weights for v in a.variants for g in games.itertuples()]
    df = pd.DataFrame(rows)
    df.drop(columns=["fills_at", "fills_through"]).to_csv(OUT / f"orders{a.suffix}.csv", index=False)
    n_games = df.drop_duplicates(["season", "game_id"]).groupby("season").size()
    print(f"Daily runs at {', '.join(f'{t:%H:%M}' for t in DAILY_RUNS)} Eastern; games: "
          + ", ".join(f"{s} {n}" for s, n in n_games.items()))
    skip = df[(df["decision"] == "skip") & (df["weight"] == a.weights[0])]
    show(f"Why games got no order (weight {a.weights[0]:g})",
         skip.groupby(["variant", "reason"]).size().unstack(0).fillna(0).astype(int).reset_index(), "{:.0f}")
    ot = orders_table(df)
    ot.to_csv(OUT / f"summary{a.suffix}.csv", index=False)
    show("Every order, one contract each, no caps (P&L a filled contract, after the maker fee)", ot)
    pt, ledger = plan_table(df, a.bankroll)
    pt.to_csv(OUT / f"plan{a.suffix}.csv", index=False)
    ledger.to_csv(OUT / f"plan_ledger{a.suffix}.csv", index=False)
    show(f"The plan: {KELLY_FRACTION:g} Kelly, {MAX_GAME_SHARE:.0%} a game, at most {MAX_ACTIVE_ORDERS} games and "
         f"{MAX_ACTIVE_SHARE:.0%} of the bankroll active at once, from ${a.bankroll:,.0f}", pt)


if __name__ == "__main__":
    main()
