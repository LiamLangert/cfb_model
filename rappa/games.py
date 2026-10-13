"""Turning RAPPA ratings into game margins, and grading them against the market.

Shared by both leagues. Margins are from the home team's point of view, and
`spread` is in CFBD's convention: negative when the home team is favoured, so
the market's expected margin is -spread. nfl.predict_games converts nflverse's
spread_line to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# What the points conversion is fitted on: each side's per-play edge, and
# whether the game is at a neutral site.
FEATURES = ["home_edge", "away_edge", "neutral_site"]

# Win rate needed to break even betting at -110.
BREAK_EVEN = 110 / 210


def add_edges(slate, ratings):
    """Each side's expected per-play edge: its offence against their defence.

    A team with no rating yet gets NaN, so its game drops out downstream.
    """
    return slate.assign(
        home_edge=slate.home_team.map(ratings["off"]) - slate.away_team.map(ratings["def"]),
        away_edge=slate.away_team.map(ratings["off"]) - slate.home_team.map(ratings["def"]),
        neutral_site=slate.neutral_site.astype(int),
    )


def win_probability(preds, min_games):
    """RAPPA's home win probability, from the spread of its own earlier misses."""
    slot = preds.season * 100 + preds.week
    miss = (preds.margin - preds.predicted).dropna()
    sigma = {}
    for s in sorted(slot.unique()):
        earlier = miss[slot[miss.index] < s]
        sigma[s] = earlier.std() if len(earlier) >= min_games else np.nan
    sigma = slot.map(sigma)
    return preds.assign(sigma=sigma, rappa_wp=stats.norm.cdf(preds.predicted / sigma))


def market_wp(games):
    """The home side's closing moneyline as a probability, with the vig taken out."""
    def implied(ml):
        return np.where(ml < 0, -ml / (100 - ml), 100 / (ml + 100))
    home, away = implied(games.home_moneyline), implied(games.away_moneyline)
    return home / (home + away)


def score(preds):
    """Accuracy of the model and of the market, on the same games."""
    graded = preds.dropna(subset=["spread"]).copy()
    graded["market"] = -graded.spread
    out = {}
    for name, col in (("rappa", "predicted"), ("market", "market")):
        err = graded.margin - graded[col]
        out[name] = {
            "games": len(graded),
            "mae": err.abs().mean(),
            "rmse": np.sqrt((err**2).mean()),
            "corr": np.corrcoef(graded[col], graded.margin)[0, 1],
            "bias": err.mean(),
        }
    return pd.DataFrame(out).T, graded


def against_the_spread(graded, edge):
    """Bet the side the model disagrees with the market about, by at least `edge`.

    A win rate above 52.4% only means something if it clears the noise, so the
    binomial p-value against a coin flip is reported next to it. Several edges
    are tried at once, which makes a lone small p-value cheaper than it looks.
    """
    picks = graded[(graded.predicted - graded.market).abs() >= edge].copy()
    if picks.empty:
        return None
    # Home covers when the actual margin beats the market's expectation.
    home_covered = picks.margin > picks.market
    push = picks.margin == picks.market
    took_home = picks.predicted > picks.market
    won = (took_home == home_covered) & ~push
    n = int(len(picks) - push.sum())
    wins = int(won.sum())
    if n == 0:
        return None
    pct = wins / n
    # One-sided: probability of this many wins or more from a fair coin.
    p_value = float(stats.binomtest(wins, n, 0.5, alternative="greater").pvalue)
    return {"edge": edge, "bets": n, "wins": wins, "win_pct": pct,
            "p_value": p_value, "roi_110": pct * 0.909 - (1 - pct),
            "pushes": int(push.sum())}


def grade(rows):
    """Each game's bet against the spread: win, loss, push or no bet (model on the line).

    Needs margin, predicted and spread, the last in CFBD's convention:
    negative when the home team is favoured.
    """
    rows = rows.dropna(subset=["spread", "margin", "predicted"]).copy()
    market = -rows.spread
    edge = rows.predicted - market
    home_covered = rows.margin > market
    rows["result"] = np.select(
        [edge == 0, rows.margin == market, (edge > 0) == home_covered],
        ["no bet", "push", "win"], "loss",
    )
    return rows


def record(results):
    wins, losses = int((results == "win").sum()), int((results == "loss").sum())
    if wins + losses == 0:
        return ""
    return f"{wins}-{losses} ({wins / (wins + losses):.1%})"


def week_table(games, weeks):
    """W-L for each week label (rows, in the order given) and season, with totals."""
    seasons = sorted(games.season.unique())
    table = pd.DataFrame(
        {s: [record(games.result[(games.season == s) & (games.label == w)]) for w in weeks]
         for s in seasons},
        index=weeks,
    )
    table["all"] = [record(games.result[games.label == w]) for w in weeks]
    table.loc["season"] = [record(games.result[games.season == s]) for s in seasons] + [
        record(games.result)
    ]
    table.index.name = "week"
    return table


def overall(games):
    """The pooled record in one line, with its 95% interval and a test against break-even."""
    wins, losses = int((games.result == "win").sum()), int((games.result == "loss").sum())
    n = wins + losses
    test = stats.binomtest(wins, n, BREAK_EVEN, alternative="greater")
    low, high = stats.binomtest(wins, n).proportion_ci(0.95)
    return (f"{wins}-{losses} over {n} bets ({wins / n:.1%}, 95% CI {low:.1%}-{high:.1%}); "
            f"one-sided p = {test.pvalue:.3f} against a true rate of {BREAK_EVEN:.1%} (break-even). "
            f"{int((games.result == 'push').sum())} pushes.")
