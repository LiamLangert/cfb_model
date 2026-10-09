"""Special-teams ratings for college: what each team's punt and kickoff units are worth, in points a game.

The method is rappa.special_teams', shared with the NFL - see there for how
each kick is valued (CFBD gives punts and kickoffs no PPA, so their value
comes from an expected-points curve fitted on the season's own plays) and
how the units are rated. This module only reads CFBD's plays into the frame
it takes: FCS teams pooled, in game order, with the score at each snap.

Walked forward over 2023-2024 and checked on 2025-2026 (see
cfb.predict_games), this cut MAE by 0.066 +/- 0.027 points a game in both,
about twice what it did in the NFL: college kicking units differ more, by
0.74 points a game (sd) in 2025. The NFL's ALPHA, CLIP and WEIGHT held up:
other values moved MAE by under 0.04.

`uv run -m cfb.special_teams <season>` prints a season's ratings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.ratings import FCS, ORDER, attach_game_context, pool_fcs
from rappa import special_teams as shared
from rappa.special_teams import (ALPHA, CLIP, EP_DEGREE, KICKOFFS, NOT_SNAPS, PUNTS, SPOT_DEGREE, UNITS,  # noqa: F401
                                 WEIGHT, expected_points)

log = logging.getLogger(__name__)


def scores(plays):
    """Each play's home and away score at the snap (CFBD reports them after it; see cfb.ratings)."""
    home_off = plays.offense == plays.home
    home = pd.Series(np.where(home_off, plays.offense_score, plays.defense_score), index=plays.index)
    away = pd.Series(np.where(home_off, plays.defense_score, plays.offense_score), index=plays.index)
    home, away = home.groupby(plays.game_id).cummax(), away.groupby(plays.game_id).cummax()
    pre = pd.DataFrame({"h": home, "a": away}).groupby(plays.game_id).shift(1).fillna(0)
    return pre.h, pre.a


def load(season, out):
    """One season's punts and kickoffs, from the kicking team's side, valued against their spot."""
    plays_path, games_path = out / str(season) / "plays.csv", out / str(season) / "games.csv"
    if not plays_path.exists() or not games_path.exists():
        raise SystemExit(f"need {plays_path} and {games_path} - run `uv run -m cfb.pull {season}` first")
    plays, games = pool_fcs(pd.read_csv(plays_path, low_memory=False), pd.read_csv(games_path))
    plays = attach_game_context(plays.sort_values(ORDER).reset_index(drop=True), games)
    plays["h"], plays["a"] = scores(plays)
    return shared.value_kicks(plays)


def fit(kicks, alpha=ALPHA, clip=CLIP):
    """rappa.special_teams.fit, with the college constants as defaults."""
    return shared.fit(kicks, alpha, clip)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season(s) to rate, fitted separately")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"ridge penalty (default: {ALPHA:g})")
    p.add_argument("--clip", type=float, default=CLIP, help=f"clip each kick's value (default: {CLIP:g}; 0 = off)")
    p.add_argument("--top", type=int, default=15, help="how many teams to print at each end (default: 15)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for season in args.seasons:
        kicks = load(season, args.out)
        ratings = fit(kicks, args.alpha, args.clip).drop(index=FCS, errors="ignore")
        ratings = ratings.sort_values("value", ascending=False)
        print(f"\n{season} special teams: {len(kicks)} punts and kickoffs. Units in points per kick, value in "
              f"points per game vs average ({', '.join(UNITS)}):")
        print(pd.concat([ratings.head(args.top), ratings.tail(args.top)]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
