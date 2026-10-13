"""College stage 2: rating edges into points, scaled by how many plays the game will have, and win probability.

cfb.predict_games' stage 2 regresses the final margin on the two rating
edges, in points per play. A point per play is worth more in a game with
more plays, and college has lost plays: 71.0 run and pass plays per team a
game in 2015, 65.2 in 2026, most of it to the 2023 clock rules. Fitted on
every earlier game at the same weight, one coefficient lagged behind:
within each season a unit of edge was worth 70-75 points a game in 2015-2020
and 55-60 from 2022, so predicted margins came out too big (the final margin
ran 0.93 of the predicted one, the market's 1.00), the most in mismatches.
FBS teams against Group of Five ones, the biggest edges, were over-predicted
by 1.3 points a game.

So each edge is scaled by the game's expected plays before the fit (scale):

    league   run and pass plays per team a game over the LEAGUE_GAMES games
             before the week (league_plays), against REFERENCE_PLAYS
    tempo    REFERENCE_PACE against the two teams' seconds between snaps,
             summed (cfb.pace: the season so far, last season's mixed in)

The coefficients then read as points a game per unit of edge in a game of
REFERENCE_PLAYS plays at REFERENCE_PACE; any other reference would only
rescale them. Walked forward over 2015-2026 on 8,084 FBS-vs-FBS games
(2026-10-08), the final margin ran 0.99 of the predicted one and MAE fell
by 0.031 +/- 0.014, better in 8 of 12 seasons: about 0.01 in 2015-2021 and
0.06 in 2022-2026. Group of Five games' over-prediction went to +0.02.
The league term alone was worth 0.020 (0.95 of the margin), tempo alone
0.023 (0.96), and weighting recent seasons instead 0.019-0.030.

Win probability is Phi(predicted / sigma), with sigma the one that gave the
best log-loss on every FBS-vs-FBS game predicted before the week
(win_probability). The sd of past misses, which research.nfl_benchmark uses and
college used before, had been about right only because it was as much too
wide as the margins were too big: with the margins fixed it cost 0.0004 of
log-loss. With the fitted sigma, log-loss was level (+0.0001 +/- 0.0005).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from cfb import pace
from cfb.ratings import PASS_TYPES, RUSH_TYPES, continuous_week
from rappa.games import FEATURES

REFERENCE_PLAYS = 65.0
REFERENCE_PACE = 56.0

# About a season of games: the league's plays per team-game is their average.
LEAGUE_GAMES = 1700


def game_plays(season, out):
    """Each played game's run and pass plays per team, with its week (cfb.ratings.continuous_week), by game id."""
    p = pd.read_csv(out / str(season) / "plays.csv", usecols=["game_id", "offense", "play_type"], low_memory=False)
    p = p[p.play_type.isin(RUSH_TYPES | PASS_TYPES) & p.offense.notna()]
    games = pd.read_csv(out / str(season) / "games.csv", low_memory=False)
    week = continuous_week(games).set_axis(games.id)
    per_team = p.groupby("game_id").size() / 2
    return pd.DataFrame({"season": season, "week": week.reindex(per_team.index), "plays": per_team}).dropna()


def league_plays(seasons, out):
    """Plays per team-game over the LEAGUE_GAMES games before each week, by season * 100 + week.

    The season before the first is read too when it is on disk, so the first
    week has a league figure; a week with none before it gets REFERENCE_PLAYS.
    """
    years = [s for s in range(min(seasons) - 1, max(seasons) + 1) if (out / str(s) / "plays.csv").exists()]
    games = pd.concat([game_plays(s, out) for s in years])
    games = games.assign(slot=games.season * 100 + games.week)
    by_slot = games.groupby("slot").plays.agg(["sum", "count"]).sort_index()
    wanted = sorted({s * 100 + w for s in seasons for w in range(40)})
    out_ = {}
    for slot in wanted:
        before = by_slot[by_slot.index < slot][::-1]
        if before.empty:
            out_[slot] = REFERENCE_PLAYS
            continue
        keep = before[before["count"].cumsum().shift(fill_value=0) < LEAGUE_GAMES]
        out_[slot] = keep["sum"].sum() / keep["count"].sum()
    return pd.Series(out_, name="league_plays")


def with_scale(rows, seasons, out):
    """rows (one per game, by id) with cfb.pace's features if missing and each game's plays_scale."""
    if "pace" not in rows:
        rows = rows.join(pace.features(seasons, out)[pace.FEATURES], on="id")
    league = (rows.season * 100 + rows.week).map(league_plays(seasons, out))
    return rows.assign(plays_scale=league / REFERENCE_PLAYS * REFERENCE_PACE / rows.pace)


def design(rows):
    """Stage 2's columns: rappa.games.FEATURES with both edges scaled by the game's plays_scale."""
    if "plays_scale" not in rows:
        raise ValueError("rows need plays_scale - pass them through cfb.stage2.with_scale first")
    return rows[FEATURES].assign(home_edge=rows.home_edge * rows.plays_scale,
                                 away_edge=rows.away_edge * rows.plays_scale)


def best_sigma(predicted, margin):
    """The sigma whose Phi(predicted / sigma) gives these decided games the lowest log-loss."""
    won = (margin > 0).to_numpy()
    x = predicted.to_numpy(dtype=float)

    def loss(sigma):
        return -np.mean(np.where(won, stats.norm.logcdf(x / sigma), stats.norm.logcdf(-x / sigma)))

    return float(optimize.minimize_scalar(loss, bounds=(5.0, 30.0), method="bounded").x)


def win_probability(preds, min_games):
    """RAPPA's home win probability, sigma fitted each week on the decided games predicted before it.

    preds are FBS-vs-FBS predictions with season, week, predicted and margin
    (missing until played); a week with fewer than min_games decided games
    before it gets none. Adds sigma and rappa_wp.
    """
    slot = preds.season * 100 + preds.week
    decided = preds.margin.notna() & preds.margin.ne(0)
    sigma = {}
    for s in sorted(slot.unique()):
        earlier = decided & (slot < s)
        if earlier.sum() >= min_games:
            sigma[s] = best_sigma(preds.predicted[earlier], preds.margin[earlier])
    sigma = slot.map(sigma)
    return preds.assign(sigma=sigma, rappa_wp=stats.norm.cdf(preds.predicted / sigma))
