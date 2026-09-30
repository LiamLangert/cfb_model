"""Backtest the NFL predictions against the closing spread, week by week.

The predictions are nfl.predict_games', made the same way: each week rated
from the plays before it, shrunk toward last season's ratings, with the
points conversion fitted on every game played before that week, earlier
seasons included. Grading is the same as cfb.backtest_week (rappa.games.grade):
every game with a line is a bet on the side the model likes more than the
market, and pushes are left out of the win rate. Break-even at -110 is 52.4%.

The first season has no prior on disk, and its early weeks are not graded:
the conversion waits until it has --min-train games to learn from. Week 1 of
every season has no plays to rate from.

Writes the week-by-season table to <out>/ats_by_week.csv.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nfl.predict_games import ALPHA, season_features, walk_forward_predict
from rappa.games import grade, overall, week_table

log = logging.getLogger(__name__)

# nflverse numbers the playoff rounds on as weeks 19-22.
ROUNDS = {"WC": "wild card", "DIV": "divisional", "CON": "conference", "SB": "super bowl"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="seasons to grade, e.g. 2023 2024 2025 2026")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    # The same defaults as nfl.predict_games, so this grades what it predicts.
    p.add_argument("--min-train", type=int, default=100,
                   help="games the conversion needs before it predicts (default: 100)")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--prior-weight", type=float, help="one weight for every unit (default: nfl.ratings's, by side)")
    p.add_argument("--save", type=Path, help="write every graded game here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("rappa", "nfl"):
        logging.getLogger(name).setLevel(logging.WARNING)

    rows = pd.concat([
        season_features(s, args.out, 2, args.alpha, args.prior_weight, True) for s in sorted(args.seasons)
    ], ignore_index=True)
    games = grade(walk_forward_predict(rows, args.min_train))
    games["label"] = np.where(
        games.game_type == "REG", games.week.astype(int).astype(str), games.game_type.map(ROUNDS)
    )
    for season, group in games.groupby("season"):
        log.info("%d: graded weeks %d-%d", season, group.week.min(), group.week.max())

    weeks = games.sort_values("week").label.drop_duplicates().tolist()
    table = week_table(games, weeks)
    print("\n=== NFL against the closing spread, betting every game (W-L, pushes excluded) ===")
    print(table.to_string())
    dest = args.out / "ats_by_week.csv"
    table.to_csv(dest)
    log.info("wrote %s", dest)
    print("\n" + overall(games))

    if args.save:
        cols = ["season", "week", "game_type", "game_id", "home_team", "away_team", "margin", "predicted",
                "spread_line", "result"]
        games[cols].to_csv(args.save, index=False)
        log.info("wrote %s (%d games)", args.save, len(games))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
