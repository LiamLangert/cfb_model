"""Terminal tables for betting.orders and betting.daily.

Every table fits the terminal: text columns (games, teams, reasons) are cut, widest first, until the row fits,
so nothing wraps. Prices are in cents, probabilities in percent, kickoffs in Eastern time. The order a row
buys is shown as the team bought and its opponent ("@ Washington" away, "vs Texas" at home).

The --track table shows only the orders worth a look: resting ones, ones just placed (Kalshi lists an order a
moment after it is placed), and filled ones until SHOW_SETTLED after their kickoff. Cancelled and expired
orders that never filled are counted under the table, not listed; orders.csv keeps every order.
"""

from __future__ import annotations

import os
import textwrap
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
DEFAULT_WIDTH = 100
GAP = "  "
MIN_TEXT = 8               # a text column is never cut narrower than this (or its header)
SHOW_SETTLED = timedelta(days=7)
JUST_PLACED = timedelta(minutes=15)


def term_width(default=DEFAULT_WIDTH):
    """Columns to fit to: COLUMNS if set, else the terminal on stdout, stderr or stdin (bets.sh pipes stdout
    and stderr through tee, but stdin is still the terminal), else default (a run with no terminal)."""
    if os.environ.get("COLUMNS", "").isdigit():
        return int(os.environ["COLUMNS"])
    for fd in (1, 2, 0):
        try:
            return os.get_terminal_size(fd).columns
        except OSError:
            continue
    return default


def wrap(text, width=None, indent="  "):
    return textwrap.fill(text, width or term_width(), subsequent_indent=indent, break_on_hyphens=False)


def table(headers, rows, align, width=None):
    """rows of cell strings as an aligned table no wider than width. align has a letter a column: 'l' left and
    cut (with an ellipsis) to fit, widest first; 'L' left and never cut; 'r' right and never cut."""
    width = width or term_width()
    rows = [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    over = sum(widths) + len(GAP) * (len(widths) - 1) - width
    while over > 0:
        cuttable = [i for i, a in enumerate(align) if a == "l" and widths[i] > max(MIN_TEXT, len(headers[i]))]
        if not cuttable:
            break
        i = max(cuttable, key=lambda i: widths[i])
        widths[i] -= 1
        over -= 1

    def line(cells):
        out = []
        for c, w, a in zip(cells, widths, align):
            c = c if len(c) <= w else c[:w - 1] + "…"
            out.append(c.rjust(w) if a == "r" else c.ljust(w))
        return GAP.join(out).rstrip()

    return "\n".join([line(headers), GAP.join("─" * w for w in widths)] + [line(r) for r in rows])


# ---------- cells ----------

def _missing(x):
    return x is None or (isinstance(x, float) and pd.isna(x))


def cents(x, digits=0):
    return "" if _missing(x) else f"{100 * float(x):.{digits}f}¢"


def signed_cents(x):
    return "" if _missing(x) else f"{100 * float(x):+.1f}¢"


def pct(x):
    return "" if _missing(x) else f"{100 * float(x):.0f}%"


def money(x, signed=False):
    if _missing(x):
        return ""
    x = float(x)
    if not signed or abs(x) < 0.005:
        return f"${abs(x) if abs(x) < 0.005 else x:.2f}"
    return f"{'-' if x < 0 else '+'}${abs(x):.2f}"


def _num(x):
    x = pd.to_numeric(x, errors="coerce")
    return None if _missing(x) else float(x)


def is_topup(x):
    """Whether a plan row adds to a game already held (its "topup" is True, not missing)."""
    return not _missing(x) and bool(x)


def count(x):
    x = _num(x)
    return "" if x is None else f"{x:g}"


def book(bid, ask):
    """'43/44¢', or nothing when a side is empty."""
    return "" if _missing(bid) or _missing(ask) else f"{100 * float(bid):.0f}/{cents(ask)}"


def kickoff(iso, date=False):
    """'Sat 3:30pm' in Eastern time, or 'Sat 10/10' with date; 'TBD' when there is none."""
    t = pd.to_datetime(iso, utc=True, errors="coerce")
    if pd.isna(t):
        return "TBD"
    t = t.tz_convert(ET)
    if date:
        return f"{t:%a} {t.month}/{t.day}"
    return f"{t:%a} {t.hour % 12 or 12}:{t:%M}{'am' if t.hour < 12 else 'pm'}"


def matchup(game, side, team=None):
    """(team bought, its opponent) from 'Away at Home' and the side bought; the Kalshi team name and no
    opponent when the side is not known."""
    if not isinstance(game, str) or " at " not in game or side not in ("home", "away"):
        return (team if isinstance(team, str) else game), ""
    away, home = game.split(" at ", 1)
    return (home, f"vs {away}") if side == "home" else (away, f"@ {home}")


def side_prob(r):
    """RAPPA's win probability for the team bought."""
    p = r.get("rappa_home")
    if _missing(p) or r.get("side") not in ("home", "away"):
        return None
    return p if r["side"] == "home" else 1 - p


# ---------- tables ----------

def by_kickoff(frame):
    if frame.empty or "kickoff" not in frame:
        return frame
    kick = pd.to_datetime(frame.kickoff, utc=True, errors="coerce")
    return frame.iloc[kick.reset_index(drop=True).sort_values(kind="stable", na_position="last").index]


def orders_table(frame, width=None, status=False):
    """Planned or placed orders: one row per order, by kickoff. With top-ups among them, "Held" is what each
    of those games already holds, and Qty what the order adds."""
    headers = ["Buy", "Opponent", "Kickoff", "RAPPA", "Fair", "Bid/Ask", "Price", "Edge", "Qty", "Cost"]
    align = "llLrrrrrrr"
    topups = "topup" in frame and any(is_topup(x) for x in frame.topup)
    if topups:
        headers, align = headers[:8] + ["Held"] + headers[8:], align[:8] + "r" + align[8:]
    if status:
        headers, align = headers + ["Status"], align + "l"
    rows = []
    for r in by_kickoff(frame).to_dict("records"):
        team, opp = matchup(r.get("game"), r.get("side"), r.get("team"))
        row = [team, opp, kickoff(r.get("kickoff")), pct(side_prob(r)), cents(r.get("fair"), 1),
               book(r.get("bid"), r.get("ask")), cents(r.get("price")), signed_cents(r.get("edge"))]
        if topups:
            row.append(count(r.get("held")) if is_topup(r.get("topup")) else "")
        row += [("+" if is_topup(r.get("topup")) else "") + count(r.get("count")), money(r.get("cost"))]
        if status:
            err = r.get("error")
            row.append(r.get("status", "") + (f": {err}" if isinstance(err, str) and err else ""))
        rows.append(row)
    return table(headers, rows, align, width)


def reviews_table(reviews, width=None):
    """The daily review of resting orders, by kickoff, and under it why each cancel or replace."""
    headers = ["Team", "Opponent", "Kickoff", "Order", "Bid/Ask", "Fair", "Edge", "Action"]
    rows, notes = [], []
    for r in by_kickoff(reviews).to_dict("records"):
        team, opp = matchup(r.get("game"), r.get("side"), r.get("team"))
        action = r.get("decision", "")
        if action == "replace":
            action = f"replace {cents(r.get('new_price'))} x{count(r.get('new_count'))}"
        if isinstance(r.get("status"), str) and r["status"]:
            action += f" ({r['status']})"
        rows.append([team, opp, kickoff(r.get("kickoff")), f"{cents(r.get('price'))} x{count(r.get('remaining'))}",
                     book(r.get("bid"), r.get("ask")), cents(r.get("fair"), 1), signed_cents(r.get("edge")), action])
        if r.get("decision") in ("cancel", "replace") and isinstance(r.get("reason"), str) and r["reason"]:
            notes.append(wrap(f"  {team}: {r['reason']}", width, indent="    "))
    return "\n".join([table(headers, rows, "llLrrrrL", width)] + notes)


def shown_orders(latest, now, show_settled=SHOW_SETTLED, just_placed=JUST_PLACED):
    """The tracked orders worth printing: resting, just placed, or filled with kickoff under show_settled ago."""
    if latest.empty:
        return latest
    filled = pd.to_numeric(latest.get("fill_count"), errors="coerce").fillna(0)
    kick = pd.to_datetime(latest.get("kickoff"), utc=True, errors="coerce")
    placed = pd.to_datetime(latest.get("placed_at"), utc=True, errors="coerce")
    resting = latest.status.eq("resting")
    new = latest.status.eq("not found") & (now - placed < just_placed)
    recent = kick.isna() | (kick > now - show_settled)
    return latest[resting | new | ((filled > 0) & recent)]


def track_table(latest, now, width=None):
    """The --track summary: the orders shown_orders keeps, then totals over every order."""
    width = width or term_width()
    shown = shown_orders(latest, now)
    rows = []
    for r in by_kickoff(shown).to_dict("records"):
        team, opp = matchup(r.get("game"), r.get("side"), r.get("team"))
        filled = _num(r.get("fill_count")) or 0.0
        of = _num(r.get("initial_count"))
        of = _num(r.get("count")) if of is None else of
        result = r.get("result") if isinstance(r.get("result"), str) else ""
        if r.get("status") == "not found":
            status, pnl = "placed", ""            # Kalshi lists a new order a moment later
        elif filled > 0 and result in ("yes", "no"):
            status, pnl = ("won" if result == "yes" else "lost"), money(r.get("pnl"), signed=True)
        elif r.get("status") == "resting":
            status = "resting" if filled == 0 else "part filled"
            pnl = f"({money(r.get('mark_to_market'), signed=True)})" if filled > 0 else ""
        else:
            status, pnl = "filled", f"({money(r.get('mark_to_market'), signed=True)})"
        rows.append([team, opp, kickoff(r.get("kickoff"), date=True), cents(r.get("plan_price", r.get("price"))),
                     f"{filled:g}/{count(of)}", status, pnl])
    lines = []
    if rows:
        lines.append(table(["Team", "Opponent", "Game", "Price", "Filled", "Status", "P&L"], rows, "llLrrLr", width))
    else:
        lines.append("no resting or recently filled orders")

    filled = pd.to_numeric(latest.get("fill_count"), errors="coerce").fillna(0)
    fees = sum(pd.to_numeric(latest.get(c), errors="coerce").fillna(0) for c in ("maker_fees", "taker_fees"))
    hidden = ~latest.index.isin(shown.index)
    lines.append("\n" + wrap(f"Settled P&L {money(latest.pnl.sum(), signed=True)}; open "
                             f"{money(latest.mark_to_market.sum(), signed=True)} (bracketed: marked at the mid); "
                             f"fees ${fees.sum():.2f}", width))
    left_out = [f"{n} {what}" for n, what in (
        (int((hidden & (filled == 0)).sum()), "cancelled or expired without a fill"),
        (int((hidden & (filled > 0)).sum()), f"settled over {SHOW_SETTLED.days} days ago")) if n]
    lines.append(wrap(f"{int((filled > 0).sum())} of {len(latest)} orders filled ({filled.sum():g} contracts)"
                      + (f"; not shown: {', '.join(left_out)}" if left_out else ""), width))
    return "\n".join(lines)


def skipped_lines(skipped, width=None, games=False):
    """Skipped games by reason (the reason's detail in brackets dropped): counts, or with games the games too."""
    if skipped.empty:
        return ""
    reason = skipped.reason.fillna("").str.replace(r" \(.*", "", regex=True)
    if not games:
        counts = reason.value_counts()
        return wrap(f"Skipped {len(skipped)} games: " + "; ".join(f"{n} {r}" for r, n in counts.items()), width)
    out = [f"Skipped {len(skipped)} games:"]
    for r in reason.value_counts().index:
        group = skipped[reason == r]
        out.append(wrap(f"  {r} ({len(group)}): " + ", ".join(group.game), width, indent="    "))
    return "\n".join(out)
