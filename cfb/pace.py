"""Pace and scoring for college games: what RAPPA's ratings miss about a game's total.

The college counterpart of nfl.pace, for cfb.week_parts' projected total.
For each team, from its games this season before the week:

    pace     seconds between its offence's snaps in neutral states: the first
             three quarters, the score within 7, outside the last two minutes
             of a half, on the same drive and 1 to 60 seconds apart. CFBD's
             clock runs down within each quarter. The league averaged 26.4 in
             2015 and 30.0 from 2024, after the clock rules changed.
    scoring  points a game in its games, both teams'.

Each is the season so far with last season's mixed in, as nfl.pace mixes it
(nfl.pace.priors, PRIOR_SNAPS and PRIOR_GAMES). A team's games against FCS
opponents count too.

Walked forward on 6,983 FBS-vs-FBS games from 2016-2026, the projected total
missed the actual one by:

    edges only               13.562
    + pace                   13.280
    + scoring                13.140
    + both, as used          13.102   (2016-2021 13.823 to 13.458, 2022-2026 13.245 to 12.668)
    closing total            12.835

Both help far more than in the NFL (10.741 to 10.672): college tempo varies
more from team to team.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.ratings import ORDER, PASS_TYPES, RUSH_TYPES, continuous_week
from nfl.pace import PRIOR_GAMES, PRIOR_SNAPS, priors

COLUMNS = ["game_id", "drive_id", "drive_number", "play_number", "offense", "offense_score", "defense_score",
           "period", "clock_minutes", "clock_seconds", "play_type"]

# The columns this adds to a game, each the two teams' sum, for the total's regression.
FEATURES = ["pace", "scoring"]


def season_games(season, out):
    """The season's games, weeks numbered as cfb.ratings numbers them."""
    games = pd.read_csv(out / str(season) / "games.csv", low_memory=False)
    return games.assign(week=continuous_week(games))


def team_games(season, out):
    """Each team's played games: week, neutral seconds and snaps on offence, and the game's points."""
    path = out / str(season) / "plays.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run -m cfb.pull {season}` first")
    p = pd.read_csv(path, usecols=COLUMNS, low_memory=False).sort_values(ORDER, kind="stable")
    p = p[p.play_type.isin(RUSH_TYPES | PASS_TYPES) & p.offense.notna()].copy()
    clock = p.clock_minutes * 60 + p.clock_seconds
    p["game_seconds"] = (4 - p.period).clip(lower=0) * 900 + clock
    half = clock + np.where(p.period.isin([1, 3]), 900, 0)
    gap = p.groupby(["game_id", "drive_id"]).game_seconds.shift() - p.game_seconds
    neutral = p.period.le(3) & (p.offense_score - p.defense_score).abs().le(7) & (half > 120) & gap.between(1, 60)
    tg = (pd.DataFrame({"game_id": p.game_id, "team": p.offense, "seconds": gap.where(neutral, 0.0),
                        "snaps": neutral.astype(float)})
          .groupby(["game_id", "team"], as_index=False).sum())
    games = season_games(season, out)
    played = games[games.completed.eq(True)].set_index("id")
    tg = tg[tg.game_id.isin(played.index)].copy()
    tg["week"] = tg.game_id.map(played.week)
    tg["points"] = tg.game_id.map(played.home_points + played.away_points)
    return tg


def season_features(season, out, this, last=None):
    """Each of the season's games' pace and scoring for both sides, and their sums, from the weeks before it.

    this and last are team_games for the season and the one before it (None
    if there is none). Indexed by the game's id.
    """
    games = season_games(season, out)
    if last is not None:
        pace_prior, scoring_prior, pace_default, scoring_default = priors(last)
    frames = []
    for week in sorted(games.week.dropna().unique()):
        before = this[this.week < week]
        if last is None:
            if before.empty:
                continue
            pace_prior, scoring_prior, pace_default, scoring_default = priors(before)
        so_far = before.groupby("team").agg(seconds=("seconds", "sum"), snaps=("snaps", "sum"),
                                            points=("points", "sum"), games=("points", "size"))
        slate = games[games.week == week]
        cols = {"id": slate.id.to_numpy()}
        for side in ("home", "away"):
            teams = slate[f"{side}_team"]
            s = so_far.reindex(teams).fillna(0.0)
            pp = teams.map(pace_prior).fillna(pace_default).to_numpy()
            sp = teams.map(scoring_prior).fillna(scoring_default).to_numpy()
            cols[f"{side}_pace"] = (s.seconds.to_numpy() + PRIOR_SNAPS * pp) / (s.snaps.to_numpy() + PRIOR_SNAPS)
            cols[f"{side}_scoring"] = (s.points.to_numpy() + PRIOR_GAMES * sp) / (s.games.to_numpy() + PRIOR_GAMES)
        frames.append(pd.DataFrame(cols))
    out_df = pd.concat(frames, ignore_index=True).set_index("id")
    out_df["pace"] = out_df.home_pace + out_df.away_pace
    out_df["scoring"] = out_df.home_scoring + out_df.away_scoring
    return out_df


def features(seasons, out):
    """season_features for each season, each season's plays read once."""
    seasons = sorted(seasons)
    loaded = {s: team_games(s, out) for s in [seasons[0] - 1, *seasons] if (out / str(s) / "plays.csv").exists()}
    return pd.concat([season_features(s, out, loaded[s], loaded.get(s - 1)) for s in seasons])
