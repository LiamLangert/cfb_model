"""Special-teams ratings for the NFL: what each team's punt and kickoff units are worth, in points a game.

RAPPA's scrimmage ratings only see runs and passes. This rates the kicking
game the same way - a ridge fit on nflverse epa, opponent-adjusted, walked
forward - and nfl.predict_games adds the gap between the two teams straight
onto the predicted margin.

    plays   every punt and kickoff, with epa credited to the kicking team
            (nflverse credits a kickoff to the receiver, so those are
            flipped) and clipped to +/-CLIP. Onside kicks are left out: the
            score calls them, not how good a team is.
    units   four per team - punt (punter and coverage), punt_return, kickoff
            (kicker and coverage) and kick_return. Each play is the kicking
            unit against the returning one, and punts and kickoffs get an
            intercept each, since their average epa differs.
    value   each unit's rating times how many plays a game a team's unit is
            on the field for, summed: epa per game, which is points, so it
            goes onto the margin at face value (WEIGHT).

Every choice was made walking forward over 2015-2022 on top of the scrimmage
model, and checked on 2023-2026:

  - Punts carry almost all of it: -0.039 +/- 0.013 points of MAE on
    2015-2022, and better in 10 of 11 seasons. Kickoffs add a little
    (most were touchbacks before the 2024 rule change). Field goals and
    extra points are left out: they added nothing on 2015-2022 and hurt on
    2023-2026 - a kick's epa swings hard, and kicker accuracy does not hold
    up well enough to predict.
  - Fitting WEIGHT as a feature of the points conversion instead was worse:
    it swung between 0.3 and 2.4 from season to season.
  - Shrinking toward last season's ratings changed MAE by under 0.005, so
    there is no prior and each season stands alone.

`uv run -m nfl.special_teams <season>` prints a season's ratings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["game_id", "week", "posteam", "defteam", "epa", "desc", "kickoff_attempt", "punt_attempt"]

UNITS = ("punt", "punt_return", "kickoff", "kick_return")

# Ridge penalty. Ratings from a handful of punts are mostly noise, but
# 100-1000 shrank the signal away; 10 did better still on 2015-2022 and lost
# it on 2023-2026.
ALPHA = 30.0

# Each play's epa is clipped to +/-CLIP. Return touchdowns and blocked punts
# reach +/-7; 2 predicted best, and a harder clip did better only on 2015-2022.
CLIP = 2.0

# Points of margin per point of special-teams value gap - epa at face value.
WEIGHT = 1.0


def load(season, out):
    """One season's punts and kickoffs, from the kicking team's side."""
    path = out / str(season) / "plays.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run nfl/pull.py {season}` first")
    raw = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
    kickoff, punt = raw.kickoff_attempt.eq(1), raw.punt_attempt.eq(1)
    onside = raw.desc.str.contains("onside", case=False, na=False)
    keep = (kickoff | punt) & ~onside & raw.epa.notna() & raw.posteam.notna() & raw.defteam.notna()
    df, kickoff = raw[keep], kickoff[keep]
    return pd.DataFrame({
        "game_id": df.game_id,
        "week": df.week.astype(float),
        "kind": np.where(kickoff, "kickoff", "punt"),
        "kicking": np.where(kickoff, df.defteam, df.posteam),
        "receiving": np.where(kickoff, df.posteam, df.defteam),
        "epa": np.where(kickoff, -df.epa, df.epa),
    }).reset_index(drop=True)


def fit(plays, alpha=ALPHA, clip=CLIP):
    """Each team's four unit ratings (epa per play) and its value (epa per game).

    Ridge with an unpenalised intercept per kind of play, solved in closed
    form: there are only 4 x 32 + 1 columns.
    """
    teams = sorted(set(plays.kicking) | set(plays.receiving))
    col = {t: i for i, t in enumerate(teams)}
    n, p = len(plays), 4 * len(teams) + 1
    punt = (plays.kind == "punt").to_numpy()
    X = np.zeros((n, p))
    rows = np.arange(n)
    X[rows, plays.kicking.map(col).to_numpy() * 4 + np.where(punt, 0, 2)] = 1.0
    X[rows, plays.receiving.map(col).to_numpy() * 4 + np.where(punt, 1, 3)] = -1.0
    X[:, -1] = punt  # the punt intercept, on top of the kickoff one
    y = plays.epa.clip(-clip, clip).to_numpy() if clip else plays.epa.to_numpy()

    Xc, yc = X - X.mean(axis=0), y - y.mean()
    penalty = np.full(p, alpha)
    penalty[-1] = 0.0
    coef = np.linalg.solve(Xc.T @ Xc + np.diag(penalty), Xc.T @ yc)
    ratings = pd.DataFrame(coef[:-1].reshape(len(teams), 4), index=teams, columns=UNITS)

    # How often a team's unit is on the field, per team-game in these plays.
    team_games = len(pd.concat([plays[["game_id", "kicking"]].set_axis(["g", "t"], axis=1),
                                plays[["game_id", "receiving"]].set_axis(["g", "t"], axis=1)]).drop_duplicates())
    per_game = np.array([punt.sum(), punt.sum(), (~punt).sum(), (~punt).sum()]) / team_games
    ratings["value"] = ratings[list(UNITS)].to_numpy() @ per_game
    ratings.index.name = "team"
    return ratings


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season(s) to rate, fitted separately")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"ridge penalty (default: {ALPHA:g})")
    p.add_argument("--clip", type=float, default=CLIP, help=f"clip each play's epa (default: {CLIP:g}; 0 = off)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for season in args.seasons:
        plays = load(season, args.out)
        ratings = fit(plays, args.alpha, args.clip).sort_values("value", ascending=False)
        print(f"\n{season} special teams: {len(plays)} punts and kickoffs. Units in epa per play, value in "
              "epa (points) per game vs average:")
        print(ratings.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
