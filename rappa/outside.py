"""Special teams and penalties: the points a game RAPPA's play ratings leave out, rated by team, for either league.

RAPPA's four units and its quarterbacks are fitted on runs and passes, each
worth its value in expected points (rappa.expected_points) and capped at the
league's CLIP. Everything else in a game moves the score too, valued the
same way (rappa.expected_points.node_values: the points before the next node
plus that node's expected points, less its own). Per game, home minus away:

    kicks         punts and kickoffs, returns included
    penalties     penalty snaps (no-plays), which the play fits drop
    field goals   field goal attempts
    big plays     the part of a run's or pass's value past the cap
    the rest      other snaps the fits drop, overtime past its first period,
                  and plays missing from the data

Together they are the game's margin less its fitted plays' (capped) values
and less its kneels and spikes: clock management by a team that is already
winning, which the fits drop for the same reason (game_points). The site
shows the rating as special teams and penalties, which carry most of what
repeats. These points repeat from game to game far less than play values do
(split-half reliability about .3 in college and .2 in the NFL, against about
.65 for play values), so the rating is heavily shrunk and goes onto the
margin at well under face value.

Each team's rating is the points a game it gains this way against an average
team, from a ridge regression on games (fit): each game is home rating -
away rating + a home edge, and each team is pulled toward CARRY of its
rating at the end of last season, with the strength of PRIOR_GAMES games
(chain runs the seasons in order, the first on disk from zero). A predicted
margin adds WEIGHT times the gap between the two teams (value).

It replaced each league's punt and kickoff rating (rappa.special_teams, at
face value) on 2026-10-09. Walked forward over 2015-2026, against it: college
log-loss -0.0015 +/- 0.0008 (8,082 games, better in 8 of 12 seasons, MAE
-0.027); NFL -0.0001 +/- 0.0013 (2,890 games, level), taken for the same
treatment in both leagues. The constants are college's: stage 2, given the
gap as one more input, put about 0.7 on it in every period; 0.6 and 0.7 at
face value predicted the same, 0.5 a little worse. In the NFL no setting
helped (pulls of 4-16 games and weights of 0.4-1.0, merged with or kept apart
from the kick rating), and weights near 1 hurt. Kept apart from the kick
rating, a rating of the rest did about as well in college (-0.0015 +/-
0.0005) with two ratings instead of one. Of the pieces, penalties carried the
most and big plays, punts and kickoffs less; field goals barely repeat.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from rappa import expected_points

# The prior's pull, in games: each team's rating is shrunk toward CARRY of its
# final rating last season as if it had played PRIOR_GAMES average games.
# Walked forward in college, 2-16 games and a carry of .5-1 all predicted
# about the same.
PRIOR_GAMES = 8.0
CARRY = 0.8
# Points on the predicted margin for each point of rating gap.
WEIGHT = 0.7


def game_points(frame, kneel, fitted, games, clip):
    """Each completed game's points outside the fitted plays, home minus away (see the module docstring).

    frame is the season's full play-by-play as rappa.expected_points reads it,
    in game order; kneel marks its kneels and spikes. fitted are the plays the
    league's fit uses (offense, home, game_id and ppa, the value it fits).
    games have game_id, week, home_team, away_team, neutral_site and margin.
    Games with no fitted plays are left out.
    """
    values = expected_points.node_values(frame)
    side = np.where(frame.offense == frame.home, 1.0, -1.0)
    kneels = (values.where(kneel, 0.0).fillna(0.0) * side).groupby(frame.game_id).sum()
    fitted_side = np.where(fitted.offense == fitted.home, 1.0, -1.0)
    inside = (fitted.ppa.clip(-clip, clip) * fitted_side).groupby(fitted.game_id).sum()
    played = games[games.margin.notna() & games.game_id.isin(inside.index)].set_index("game_id")
    points = played.margin - inside.reindex(played.index) - kneels.reindex(played.index).fillna(0.0)
    return pd.DataFrame({"game_id": played.index, "week": played.week.to_numpy(),
                         "home_team": played.home_team.to_numpy(), "away_team": played.away_team.to_numpy(),
                         "neutral_site": played.neutral_site.astype(int).to_numpy(), "points": points.to_numpy()})


def cached(path, inputs, compute):
    """compute()'s game_points frame, kept at path and rebuilt when any of inputs is newer.

    Written whole (to a temporary file, then moved) and always read back, so
    parallel runs never see half a file and a fresh run and a cached one
    return the same thing.
    """
    if not path.exists() or any(p.exists() and p.stat().st_mtime > path.stat().st_mtime for p in inputs):
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        compute().to_csv(tmp, index=False)
        os.replace(tmp, path)
    return pd.read_csv(path)


def fit(games, prior=None, prior_games=PRIOR_GAMES, carry=CARRY, pool=None):
    """Each team's rating from these games (points a game against an average team) and its value (WEIGHT x it).

    games are game_points rows; prior is a rating by team (last season's final
    ones) or None. A team in prior with no games keeps carry times its prior.
    Ratings are centred on the average team, leaving out pool (the college
    FCS pool), which is rated too.
    """
    teams = sorted(set(games.home_team) | set(games.away_team) | (set(prior.index) if prior is not None else set()))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rows = np.arange(len(games))
    X = np.zeros((len(games), n + 1))
    X[rows, games.home_team.map(index).to_numpy()] = 1.0
    X[rows, games.away_team.map(index).to_numpy()] = -1.0
    X[:, n] = 1 - games.neutral_site.to_numpy()
    b0 = np.zeros(n + 1)
    if prior is not None:
        b0[:n] = carry * prior.reindex(teams).fillna(0.0).to_numpy()
    if len(games):
        penalty = np.eye(n + 1) * prior_games
        penalty[n, n] = 1e-6  # the home edge is barely shrunk
        b = np.linalg.solve(X.T @ X + penalty, X.T @ (games.points.to_numpy() - X @ b0)) + b0
    else:
        b = b0
    rating = pd.Series(b[:n], index=teams)
    rating = rating - (rating if pool is None else rating.drop(pool, errors="ignore")).mean()
    played = pd.concat([games.home_team, games.away_team]).value_counts().reindex(teams).fillna(0).astype(int)
    return pd.DataFrame({"rating": rating, "value": WEIGHT * rating, "games": played}).rename_axis("team")


def chain(seasons, season_points, pool=None):
    """The final ratings after these seasons, in order: each season fitted with the one before as its prior.

    season_points(season) gives a season's game_points frame. None for no seasons.
    """
    rating = None
    for s in seasons:
        rating = fit(season_points(s), rating, pool=pool).rating
    return rating
