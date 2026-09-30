"""Backtest the college predictions against the closing spread, week by week.

The predictions are cfb.predict_games', made the same way: each week rated
from the plays before it, shrunk toward last season's ratings, with
quarterbacks and special teams, and the points conversion fitted on every
game played before that week, earlier seasons included. Grading is
rappa.games.grade, as in nfl.backtest_week: every game with a line is a bet
on the side the model likes more than the market, and pushes are left out of
the win rate. Break-even at -110 is 52.4%.

The --warmup seasons before the first one given are fitted but not graded,
so the conversion has its --min-train games by week 2. Week 1 of every
season has no plays to rate from. Bowls and playoff games are graded as one
week, "post".

Writes the week-by-season table to <out>/ats_by_week.csv.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from cfb.predict_games import ALPHA, predict
from rappa.games import grade, overall, week_table

log = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="seasons to grade, e.g. 2023 2024 2025 2026")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--warmup", type=int, default=1,
                   help="seasons before the first one given to fit but not grade (default: 1)")
    # The same defaults as cfb.predict_games, so this grades what it predicts.
    p.add_argument("--min-train", type=int, default=200,
                   help="games the conversion needs before it predicts (default: 200)")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--prior-weight", type=float, help="one weight for every unit (default: cfb.ratings's, by side)")
    p.add_argument("--save", type=Path, help="write every graded game here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("rappa", "cfb"):
        logging.getLogger(name).setLevel(logging.WARNING)
    seasons = sorted(args.seasons)

    warmup = [s for s in range(seasons[0] - args.warmup, seasons[0])
              if (args.out / str(s) / "plays.csv").exists()]
    preds = predict(warmup + seasons, args.out, 2, args.alpha, args.prior_weight, min_train=args.min_train)
    games = grade(preds[preds.season.isin(seasons)])
    games["label"] = np.where(games.season_type == "regular", games.week.astype(int).astype(str), "post")
    for season, group in games.groupby("season"):
        log.info("%d: graded weeks %d-%d", season, group.week.min(), group.week.max())

    weeks = sorted(games.label.unique(), key=lambda w: (w == "post", int(w) if w != "post" else 0))
    table = week_table(games, weeks)
    print("\n=== college against the closing spread, betting every game (W-L, pushes excluded) ===")
    print(table.to_string())
    dest = args.out / "ats_by_week.csv"
    table.to_csv(dest)
    log.info("wrote %s", dest)
    print("\n" + overall(games))

    if args.save:
        cols = ["season", "label", "id", "home_team", "away_team", "margin", "predicted", "spread", "result"]
        games[cols].rename(columns={"label": "week"}).to_csv(args.save, index=False)
        log.info("wrote %s (%d games)", args.save, len(games))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
