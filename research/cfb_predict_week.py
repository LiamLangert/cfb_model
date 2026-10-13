"""Predict one college week from RAPPA ratings to date.

The predictions are cfb.predict_games', for the target week only: ratings
from the season's plays before it (quarterbacks and special teams
included), shrunk toward last season's, and the points conversion fitted
on every game already played in this season and the one before. The week
can be one already played, to see what the model would have said.

On 2023-2026 FBS-vs-FBS games, research.cfb_benchmark has these predictions about
0.8 points a game worse than the closing line in MAE, and research.cfb_backtest_week
has never shown an edge against the spread - so read "lean" as
disagreement, not as a bet.

Predicted margin is from the home team's point of view; model_spread is its
negative, on the same scale as the market spread. Writes
<out>/<season>/predictions_wk<NN>.csv.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cfb.predict_games import ALPHA, predict, slate_table

log = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("season", type=int)
    p.add_argument("week", type=int, help="regular-season week to predict")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"ridge penalty (default: {ALPHA:g})")
    p.add_argument("--prior-weight", type=float,
                   help="carry this much of last season for every unit (default: cfb.ratings's, by side)")
    p.add_argument("--min-train", type=int, default=200,
                   help="games the conversion needs before it predicts (default: 200)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # One line per refit from the fit and the college modules is noise here.
    for name in ("rappa", "cfb"):
        logging.getLogger(name).setLevel(logging.WARNING)

    seasons = [s for s in (args.season - 1, args.season) if (args.out / str(s) / "plays.csv").exists()]
    preds = predict(seasons, args.out, alpha=args.alpha, prior_weight=args.prior_weight, min_train=args.min_train)
    slate = preds[(preds.season == args.season) & (preds.week == args.week) & (preds.season_type == "regular")]
    if slate.empty:
        last = preds.week[preds.season == args.season].max()
        raise SystemExit(f"no predictions for {args.season} week {args.week}: it needs plays from the weeks "
                         f"before it (predictions run through week {last:.0f})")
    slate = slate.sort_values("start_date").assign(model_spread=-slate.predicted)
    # Positive: the model likes the home side more than the market does.
    diff = slate.spread - slate.model_spread
    table = slate_table(slate)
    slate = slate.assign(lean=table.lean.to_numpy(), lean_pts=diff.abs().to_numpy())

    dest = args.out / str(args.season) / f"predictions_wk{args.week:02d}.csv"
    cols = ["id", "start_date", "home_team", "away_team", "neutral_site", "home_qb", "away_qb", "home_edge",
            "away_edge", "st_edge", "predicted", "model_spread", "spread", "lean", "lean_pts", "margin"]
    slate[cols].to_csv(dest, index=False)
    log.info("wrote %s (%d games)", dest, len(slate))

    played = slate.margin.notna().sum()
    print(f"\n=== {args.season} week {args.week}: {len(slate)} games"
          f"{f', {played} already played' if played else ''} ===")
    print(table.to_string(index=False, na_rep=""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
