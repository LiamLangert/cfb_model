"""Pace and scoring: what RAPPA's ratings miss about a game's total.

RAPPA rates teams by the play, so it says how well each side plays but not
how many plays a game will have, or how many points come with them.
nfl.predict_week projects a game's total on the rating edges plus two things
for each team, from its games this season before the week:

    pace     seconds between its offence's snaps in neutral states: the first
             three quarters, the score within 7, outside the last two minutes
             of a half, on the same drive and 1 to 60 seconds apart. A slower
             team's games have fewer plays. In 2025 it ran from 30.3 (NO) to
             35.5 (MIA), and the league has slowed from 30.8 in 2014 to 33.0.
    scoring  points a game in its games, both teams'.

Each is the season so far with last season's mixed in, as PRIOR_SNAPS
neutral snaps or PRIOR_GAMES games of it. Last season's is regressed partway
to that season's league average first (PRIOR_KEEP): a team's pace
correlates 0.47 from one season to the next. With no season before, the
prior is the league's this season so far.

Walked forward over 2015-2026, on the 2,675 games nfl.predict_week would have
predicted, the projected total missed the actual one by:

    edges only               10.741
    + pace                   10.707
    + scoring                10.687
    + both, as used          10.672   (2015-2022 10.880 to 10.801, 2023-2026 10.438 to 10.391)
    closing total            10.416

The totals spread out more, an sd of 3.73 points against 3.28 (the closing
total's is 4.46), and track the closing total more closely: a correlation of
0.84 against 0.71. Neither version beats the over/under. Where the two
disagree by 3 or more, the game went RAPPA's way 48.1% of the time. Halving
or doubling PRIOR_SNAPS, PRIOR_GAMES or PRIOR_KEEP moved MAE by under 0.01.

nflverse's plays give every season today's team abbreviations. The games
come through nfl.predict_games.load_games, which maps the schedules' old ones
(STL, SD, OAK) to them, so relocated teams' games match their plays.
"""

from __future__ import annotations

import pandas as pd

from nfl.predict_games import load_games

COLUMNS = ["play_id", "game_id", "posteam", "game_seconds_remaining", "half_seconds_remaining", "qtr",
           "score_differential", "drive", "pass", "rush", "two_point_attempt", "qb_kneel", "qb_spike"]

# The columns this adds to a game, each the two teams' sum, for the total's regression.
FEATURES = ["pace", "scoring"]

# How much of last season goes in: as this many neutral snaps of pace (about
# six games' worth), and this many games of scoring.
PRIOR_SNAPS = 150.0
PRIOR_GAMES = 4.0

# The share of a team's own last season kept in its prior; the rest is that
# season's league average.
PRIOR_KEEP = 0.5


def team_games(season, out):
    """Each team's played games: week, neutral seconds and snaps on offence, and the game's points."""
    path = out / str(season) / "plays.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run nfl/pull.py {season}` first")
    p = pd.read_csv(path, usecols=COLUMNS, low_memory=False).sort_values(["game_id", "play_id"], kind="stable")
    snap = p["pass"].eq(1) | p["rush"].eq(1) | p.qb_kneel.eq(1) | p.qb_spike.eq(1)
    p = p[snap & p.two_point_attempt.ne(1) & p.posteam.notna()]
    gap = p.groupby(["game_id", "drive"]).game_seconds_remaining.shift() - p.game_seconds_remaining
    neutral = p.qtr.le(3) & p.score_differential.abs().le(7) & p.half_seconds_remaining.gt(120) & gap.between(1, 60)
    tg = (pd.DataFrame({"game_id": p.game_id, "team": p.posteam, "seconds": gap.where(neutral, 0.0),
                        "snaps": neutral.astype(float)})
          .groupby(["game_id", "team"], as_index=False).sum())
    games = load_games(season, out)
    played = games[games.margin.notna()].set_index("game_id")
    tg = tg[tg.game_id.isin(played.index)].copy()
    tg["week"] = tg.game_id.map(played.week)
    tg["points"] = tg.game_id.map(played.home_score + played.away_score)
    return tg


def priors(last):
    """Each team's pace and scoring prior from last season's team games, and the league's for a team without one."""
    by = last.groupby("team").agg(seconds=("seconds", "sum"), snaps=("snaps", "sum"), points=("points", "mean"))
    league_pace, league_scoring = last.seconds.sum() / last.snaps.sum(), last.points.mean()
    pace = PRIOR_KEEP * by.seconds / by.snaps + (1 - PRIOR_KEEP) * league_pace
    scoring = PRIOR_KEEP * by.points + (1 - PRIOR_KEEP) * league_scoring
    return pace, scoring, league_pace, league_scoring


def season_features(season, out, this, last=None):
    """Each of the season's games' pace and scoring for both sides, and their sums, from the weeks before it.

    this and last are team_games for the season and the one before it (None
    if there is none). Indexed by game_id.
    """
    games = load_games(season, out)
    if last is not None:
        pace_prior, scoring_prior, pace_default, scoring_default = priors(last)
    frames = []
    for week in sorted(games.week.unique()):
        before = this[this.week < week]
        if last is None:
            if before.empty:
                continue
            pace_prior, scoring_prior, pace_default, scoring_default = priors(before)
        so_far = before.groupby("team").agg(seconds=("seconds", "sum"), snaps=("snaps", "sum"),
                                            points=("points", "sum"), games=("points", "size"))
        slate = games[games.week == week]
        cols = {"game_id": slate.game_id.to_numpy()}
        for side in ("home", "away"):
            teams = slate[f"{side}_team"]
            s = so_far.reindex(teams).fillna(0.0)
            pp = teams.map(pace_prior).fillna(pace_default).to_numpy()
            sp = teams.map(scoring_prior).fillna(scoring_default).to_numpy()
            cols[f"{side}_pace"] = (s.seconds.to_numpy() + PRIOR_SNAPS * pp) / (s.snaps.to_numpy() + PRIOR_SNAPS)
            cols[f"{side}_scoring"] = (s.points.to_numpy() + PRIOR_GAMES * sp) / (s.games.to_numpy() + PRIOR_GAMES)
        frames.append(pd.DataFrame(cols))
    out_df = pd.concat(frames, ignore_index=True).set_index("game_id")
    out_df["pace"] = out_df.home_pace + out_df.away_pace
    out_df["scoring"] = out_df.home_scoring + out_df.away_scoring
    return out_df


def features(seasons, out):
    """season_features for each season, each season's plays read once."""
    seasons = sorted(seasons)
    loaded = {}
    for s in [seasons[0] - 1, *seasons]:
        if (out / str(s) / "plays.csv").exists():
            loaded[s] = team_games(s, out)
    return pd.concat([season_features(s, out, loaded[s], loaded.get(s - 1)) for s in seasons])
