"""Kalshi's college football winner markets for a season: every market, its 1-minute candles and every trade.

    uv run -m research.pull_kalshi 2025
    uv run -m research.pull_kalshi 2026 --refresh     # a season in progress: re-list to pick up new markets

A port of ../cbb_model's research.pull_kalshi, for research.maker_backtest. The series is KXNCAAFGAME,
the winner: one market a team, two a game (--series for others, such as KXNCAAFSPREAD). data/kalshi/
already has every market with hourly candles (cfb_markets.jsonl, from the old Tuesday study); a maker
backtest needs the minute's quote and every trade, which this pulls.

Seasons are named by the year they start (2025 is the 2025 season, its bowls and playoff in January
2026); a market is in a season when its event ticker's date is from August 1 to July 31 after. Markets
settled before Kalshi's historical cutoff are served only from /historical, along with their candles and
trades; later ones from the live endpoints. Both are read, so a season in progress pulls too, its
settled markets only (one still trading would be written short and then skipped next time).

Written to data/kalshi/<season>/, one JSON object a line:
  <series>_markets.jsonl  every market as Kalshi lists it
  <series>_candles.jsonl  {"ticker", "event_ticker", "candles": [[...CANDLE_FIELDS], ...]}: one a minute in
                          which the market traded or its quote moved, from its open to its close
  <series>_trades.jsonl   {"ticker", "event_ticker", "trades": [[...TRADE_FIELDS], ...]}, oldest first

Prices are dollars on the YES side; a candle's bid and ask are the best YES bid and YES ask in that minute.
Kalshi keeps no history of the book's depth. Re-running skips markets already written.

Signed requests (KALSHI_ID and KALSHI_PRIVATE_KEY in .env) go out at --rate a second; unsigned, Kalshi
throttles to about 4.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
from datetime import date
from pathlib import Path

from kalshi.client import Client, KalshiError, parse_ts, ticker_date

log = logging.getLogger(__name__)

DATA = Path("data/kalshi")
SERIES = ["KXNCAAFGAME"]

CANDLE_FIELDS = ["end_ts", "bid_open", "bid_low", "bid_high", "bid_close", "ask_open", "ask_low", "ask_high",
                 "ask_close", "price_open", "price_low", "price_high", "price_close", "price_mean", "volume",
                 "open_interest"]
TRADE_FIELDS = ["ts_ms", "yes_price", "count", "taker_yes"]

# Candles a request: Kalshi caps a window at 5000 periods.
CANDLE_WINDOW = 4000 * 60


def in_season(event_ticker, season):
    d = ticker_date(event_ticker)
    return d is not None and date(season, 8, 1) <= d <= date(season + 1, 7, 31)


def _num(x):
    return None if x is None or x == "" else float(x)


def _field(d, name, suffix):
    """d[name], or the live endpoint's spelling of it, d[name + suffix]: /historical candles say "close" and
    "volume", the live ones "close_dollars" and "volume_fp"."""
    return _num(d.get(name, d.get(name + suffix)))


def parse_candle(c):
    bid, ask, px = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
    q = lambda d, k: _field(d, k, "_dollars")
    return [c["end_period_ts"], q(bid, "open"), q(bid, "low"), q(bid, "high"), q(bid, "close"),
            q(ask, "open"), q(ask, "low"), q(ask, "high"), q(ask, "close"),
            q(px, "open"), q(px, "low"), q(px, "high"), q(px, "close"), q(px, "mean"),
            _field(c, "volume", "_fp"), _field(c, "open_interest", "_fp")]


def parse_trade(t):
    ts = parse_ts(t["created_time"])
    return [int(ts.timestamp() * 1000), float(t["yes_price_dollars"]), float(t["count_fp"]),
            1 if t["taker_side"] == "yes" else 0]


def list_markets(client, series, season):
    """Every market of the series in the season, each marked _historical when /historical serves it."""
    seen, out = set(), []
    for path, hist in (("/historical/markets", True), ("/markets", False)):
        for m in client.pages(path, "markets", series_ticker=series, limit=1000):
            if in_season(m["event_ticker"], season) and m["ticker"] not in seen:
                seen.add(m["ticker"])
                out.append({**m, "_historical": hist})
    return out


def candles(client, market):
    path = (f"/historical/markets/{market['ticker']}/candlesticks" if market["_historical"]
            else f"/series/{market['series']}/markets/{market['ticker']}/candlesticks")
    start = int(parse_ts(market["open_time"]).timestamp())
    end = int(parse_ts(market["close_time"]).timestamp()) + 60
    rows = {}
    for s in range(start, end, CANDLE_WINDOW):
        r = client.get(path, start_ts=s, end_ts=min(s + CANDLE_WINDOW, end), period_interval=1)
        for c in r.get("candlesticks") or []:
            rows[c["end_period_ts"]] = parse_candle(c)
    return [rows[k] for k in sorted(rows)]


def trades(client, market):
    path = "/historical/trades" if market["_historical"] else "/markets/trades"
    ts = client.pages(path, "trades", ticker=market["ticker"], limit=1000)
    return sorted((parse_trade(t) for t in ts), key=lambda t: t[0])


def write_jsonl(path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    """Every record in a JSON-lines file, less a last line cut short by a pull that was stopped mid-write."""
    out = []
    with path.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                log.warning("%s: skipping a line that is not whole JSON", path)
    return out


def done_tickers(path):
    return {r["ticker"] for r in read_jsonl(path)} if path.exists() else set()


def pull_series(client, series, season, want_trades=True, refresh=False, workers=8):
    out = DATA / str(season)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / f"{series}_markets.jsonl"
    if refresh or not mpath.exists():
        write_jsonl(mpath, list_markets(client, series, season))
    markets = read_jsonl(mpath)
    for m in markets:
        m["series"] = series
    log.info("%s %d: %d markets in %d events", series, season, len(markets),
             len({m["event_ticker"] for m in markets}))

    jobs = [("candles", candles, out / f"{series}_candles.jsonl")]
    if want_trades:
        jobs.append(("trades", trades, out / f"{series}_trades.jsonl"))
    for kind, fetch, path in jobs:
        have = done_tickers(path)
        todo = [m for m in markets if m["ticker"] not in have and m.get("status") in ("settled", "finalized")]
        log.info("%s %s: %d markets to fetch (%d already)", series, kind, len(todo), len(have))
        failed = 0
        with path.open("a") as f, cf.ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(fetch, client, m): m for m in todo}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                m = futs[fut]
                try:
                    rows = fut.result()
                except KalshiError as e:
                    failed += 1
                    log.warning("%s %s: %s", kind, m["ticker"], e)
                    continue
                f.write(json.dumps({"ticker": m["ticker"], "event_ticker": m["event_ticker"], kind: rows}) + "\n")
                if i % 250 == 0:
                    f.flush()
                    log.info("%s %s: %d/%d", series, kind, i, len(todo))
        if failed:
            log.warning("%s %s: %d markets failed; re-run to retry them", series, kind, failed)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("season", type=int)
    p.add_argument("--series", nargs="+", default=SERIES)
    p.add_argument("--no-trades", action="store_true", help="markets and candles only")
    p.add_argument("--refresh", action="store_true", help="re-list the markets")
    p.add_argument("--rate", type=float, default=12.0, help="signed requests a second (default 12)")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    client = Client.from_env(rate=a.rate)
    if not client.signed:
        log.warning("no Kalshi key in .env: unsigned requests are throttled to about 4 a second")
        client = Client(rate=min(a.rate, 4.0))
    for s in a.series:
        pull_series(client, s, a.season, not a.no_trades, a.refresh, a.workers)


if __name__ == "__main__":
    main()
