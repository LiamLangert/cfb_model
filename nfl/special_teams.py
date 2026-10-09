"""Special-teams ratings for the NFL: what each team's punt and kickoff units are worth, in points a game.

The method is rappa.special_teams', shared with college - see there for how
each kick is valued and how the units are rated - and nfl.predict_games
adds the gap between the two teams straight onto the predicted margin. This
module only reads nflverse's plays into the frame it takes, in CFBD's terms:

  - a kickoff's offence is the kicking team (nflverse gives the receiving
    team the ball) and its yards to goal are the kicking team's, 100 less
    nflverse's yardline_100
  - nflverse's play types are mapped onto CFBD's names; a kick that a
    penalty wiped out counts as a penalty, not a kick
  - the score at each snap is nflverse's running total from the play before

Until 2026-10-06 a kick's value here was nflverse's own epa. Valued the
college way instead, the NFL predicted as well: MAE +0.002 +/- 0.005 and
log-loss +0.0004 +/- 0.0003 over 2015-2026 (rappa.special_teams has the
rest), so both leagues now value kicks one way.

`uv run -m nfl.special_teams <season>` prints a season's ratings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rappa import special_teams as shared
from rappa.special_teams import ALPHA, CLIP, UNITS, WEIGHT  # noqa: F401

log = logging.getLogger(__name__)

# Only what the valuation needs; the full play-by-play file has 372 columns.
COLUMNS = ["game_id", "play_id", "week", "qtr", "down", "yardline_100", "posteam", "defteam", "home_team",
           "total_home_score", "total_away_score", "play_type", "desc", "kickoff_attempt", "punt_attempt"]

# nflverse play types in CFBD's names, as rappa.special_teams reads them. A
# missing play type is a timeout, quarter end or the like: not a snap.
PLAY_TYPES = {"punt": "Punt", "kickoff": "Kickoff", "run": "Rush", "pass": "Pass Reception",
              "field_goal": "Field Goal Good", "extra_point": "Extra Point Good", "no_play": "Penalty",
              "qb_kneel": "Rush", "qb_spike": "Pass Incompletion"}


def plays_frame(raw, no_play_needs_penalty=False):
    """nflverse play-by-play as rappa.special_teams.value_kicks takes it, and rappa.expected_points with it.

    It carries each snap's distance (ydstogo, where raw has it) and row, raw's index for each row. With
    no_play_needs_penalty, a no-play without a penalty (a review, a timeout logged as one) is not a snap, as
    nfl.ratings' play values have it; special teams were measured without that.
    """
    raw = raw.sort_values(["game_id", "play_id"])
    rows = raw.index.to_numpy()
    raw = raw.reset_index(drop=True)
    kick = raw.kickoff_attempt.eq(1) & raw.play_type.eq("kickoff")
    punt = raw.punt_attempt.eq(1) & raw.play_type.eq("punt")
    kind = raw.play_type.map(PLAY_TYPES).fillna("Timeout")
    # A kick replayed after a penalty is flagged as an attempt with play type no_play.
    wiped = (raw.kickoff_attempt.eq(1) & ~kick) | (raw.punt_attempt.eq(1) & ~punt)
    kind = kind.where(~wiped, "Penalty")
    if no_play_needs_penalty:
        kind = kind.where(~(raw.play_type.eq("no_play") & raw.penalty.ne(1)), "Timeout")
    # nflverse's totals are the score after the play; at the snap is the play before's.
    home = raw.total_home_score.groupby(raw.game_id).cummax()
    away = raw.total_away_score.groupby(raw.game_id).cummax()
    pre = pd.DataFrame({"h": home, "a": away}).groupby(raw.game_id).shift(1).fillna(0)
    plays = pd.DataFrame({
        "game_id": raw.game_id, "week": raw.week.astype(float), "period": raw.qtr, "down": raw.down,
        "offense": np.where(kick, raw.defteam, raw.posteam), "defense": np.where(kick, raw.posteam, raw.defteam),
        "home": raw.home_team,
        "yards_to_goal": np.where(kick, 100 - raw.yardline_100, raw.yardline_100),
        "play_type": kind, "play_text": raw.desc.fillna(""),
        "h": pre.h.to_numpy(), "a": pre.a.to_numpy(),
        "distance": raw.ydstogo if "ydstogo" in raw else np.nan, "row": rows,
    })
    return plays[plays.offense.notna() | plays.play_type.eq("Timeout")].reset_index(drop=True)


def load(season, out):
    """One season's punts and kickoffs, from the kicking team's side, valued against their spot."""
    path = out / str(season) / "plays.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run nfl/pull.py {season}` first")
    return shared.value_kicks(plays_frame(pd.read_csv(path, usecols=COLUMNS, low_memory=False)))


def fit(kicks, alpha=ALPHA, clip=CLIP):
    """rappa.special_teams.fit, with the NFL constants as defaults."""
    return shared.fit(kicks, alpha, clip)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season(s) to rate, fitted separately")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"ridge penalty (default: {ALPHA:g})")
    p.add_argument("--clip", type=float, default=CLIP, help=f"clip each kick's value (default: {CLIP:g}; 0 = off)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for season in args.seasons:
        kicks = load(season, args.out)
        ratings = fit(kicks, args.alpha, args.clip).sort_values("value", ascending=False)
        print(f"\n{season} special teams: {len(kicks)} punts and kickoffs. Units in points per kick, value in "
              "points per game vs average:")
        print(ratings.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
