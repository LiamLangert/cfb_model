"""Season-by-season accuracy of college RAPPA, ESPN's FPI and the closing market, on the same games.

The college counterpart of nfl.benchmark, scored the same way (its row and
season_table do the scoring):

    MAE       mean absolute error of the projected home margin, in points
    log-loss  of the home team's win probability. Ties are left out of it;
              a coin flip scores ln 2 = 0.693.

RAPPA's margins are cfb.predict_games's, walked forward the same way,
conferences, offseason shifts, quarterbacks and special teams and penalties
(cfb.outside) included unless --no-conferences, --no-offseason, --no-qbs or
--no-special-teams is given. It has no win probability of its own, so it gets a normal one:
P(home win) = Phi(predicted / sigma), where sigma is the one with the best
log-loss on every FBS-vs-FBS game it predicted before that week
(cfb.stage2.win_probability). ESPN's are
its matchup predictor's espn_margin and espn_home_wp. The market's are one
sportsbook's closing spread and moneylines, the vig taken out (games.csv's,
picked by cfb.pull.book_lines). CFBD has moneylines only from 2021 (and not for every game since), so
where there are none the market's win probability comes from its spread the
way RAPPA's does: Phi(-spread / sigma), sigma the sd of the spread's misses
on every earlier FBS-vs-FBS game. The table's market log-loss is then partly
or (before 2021) wholly the spread's, and the run says how many games.

Only FBS-vs-FBS games are graded - RAPPA rates every FCS opponent as one
pooled team, so those games say little about it - and only those all three
have. That leaves out week 1 (RAPPA has no plays to rate yet). The --warmup
seasons before the first one given are fitted but not graded, so the
points conversion already has its --min-train games by week 2 of the first
graded season.

Run cfb.pull and cfb.pull_espn for the seasons, and cfb.ratings for each
season before one (its prior), first. Writes the table to
<out>/benchmark.csv.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cfb.predict_games import ALPHA, predict
from cfb.stage2 import win_probability
from nfl.benchmark import MODELS, market_wp, paired, season_table

log = logging.getLogger(__name__)


def load_espn(seasons, out):
    frames = []
    for season in seasons:
        path = out / str(season) / "espn.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found - run `uv run -m cfb.pull_espn {season}` first")
        frames.append(pd.read_csv(path, usecols=["game_id", "espn_margin", "espn_home_wp"]))
    return pd.concat(frames, ignore_index=True).rename(columns={"game_id": "id"})


def spread_wp(preds, min_games):
    """The market's home win probability from its closing spread, walked forward like RAPPA's."""
    slot = preds.season * 100 + preds.week
    miss = (preds.margin + preds.spread).dropna()
    sigma = {}
    for s in sorted(slot.unique()):
        earlier = miss[slot[miss.index] < s]
        sigma[s] = earlier.std() if len(earlier) >= min_games else np.nan
    return stats.norm.cdf(-preds.spread / slot.map(sigma))


def show(table, games):
    labels = {"games": "games", "weeks": "weeks"}
    for name in MODELS:
        labels[f"mae_{name}"] = f"MAE {name}"
    for name in MODELS:
        labels[f"ll_{name}"] = f"LL {name}"
    printed = table.rename(columns=labels)[["season", *labels.values()]]
    for metric in ("mae", "ll"):
        digits = 2 if metric == "mae" else 4
        printed[f"{metric.upper()} R-E"] = [f"{d:+.{digits}f} ± {se:.{digits}f}" for d, se in
                                           zip(table[f"{metric}_rappa_espn"], table[f"{metric}_rappa_espn_se"])]
    span = f"{games.season.min()}-{games.season.max()}"
    print(f"\n=== {span}: the same {len(games)} FBS-vs-FBS games for all three (weeks 2 on, bowls included) ===")
    print("MAE: points off the actual home margin. LL: log-loss of home win probability (ties left out;"
          " a coin flip is 0.693).\nR-E: RAPPA minus ESPN per game, ± its standard error - negative means RAPPA is"
          " better. Bowls and playoff games are one week, numbered after the regular season.")
    print(printed.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== against the market, per game (negative = better than the closing line) ===")
    for name in ("rappa", "espn"):
        d = paired(games, name, "market")
        print(f"{name:>5} - market: MAE {d['mae'][0]:+.3f} ± {d['mae'][1]:.3f}   LL {d['ll'][0]:+.4f} ± {d['ll'][1]:.4f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="seasons to grade, e.g. 2023 2024 2025 2026")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--warmup", type=int, default=1,
                   help="seasons before the first one given to fit but not grade (default: 1)")
    # The same defaults as cfb.predict_games, so RAPPA is what it predicts.
    p.add_argument("--min-train", type=int, default=200,
                   help="games the conversion, and sigma, need before RAPPA predicts (default: 200)")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--prior-weight", type=float, help="one weight for every unit (default: cfb.ratings's, by side)")
    p.add_argument("--no-special-teams", action="store_true",
                   help="leave special teams and penalties out of RAPPA's margin")
    p.add_argument("--no-qbs", action="store_true", help="leave quarterbacks in their teams' pass offence")
    p.add_argument("--no-conferences", action="store_true",
                   help="rate teams without conference ratings (needs priors rated with cfb.ratings "
                        "--no-conferences)")
    p.add_argument("--no-offseason", action="store_true",
                   help="leave each prior unshifted by the offseason (needs priors rated with cfb.ratings "
                        "--no-offseason)")
    p.add_argument("--save", type=Path, help="write every graded game here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("rappa", "cfb.ratings", "cfb.outside", "cfb.predict_games"):
        logging.getLogger(name).setLevel(logging.WARNING)
    seasons = sorted(args.seasons)

    warmup = [s for s in range(seasons[0] - args.warmup, seasons[0])
              if (args.out / str(s) / "plays.csv").exists()]
    if len(warmup) < args.warmup:
        log.warning("only %d of %d warm-up seasons on disk - the first graded weeks may be missing",
                    len(warmup), args.warmup)
    preds = predict(warmup + seasons, args.out, 2, args.alpha, args.prior_weight, True,
                    not args.no_special_teams, not args.no_qbs, args.min_train, not args.no_conferences,
                    not args.no_offseason)
    preds = win_probability(preds[preds.fbs].reset_index(drop=True), args.min_train)
    preds = preds.assign(spread_wp=spread_wp(preds, args.min_train))

    games = preds[preds.season.isin(seasons) & preds.margin.notna()]
    games = games.merge(load_espn(seasons, args.out), on="id", how="left")
    moneyline = market_wp(games)
    games = games.assign(spread_line=-games.spread, market_wp=np.where(np.isnan(moneyline), games.spread_wp, moneyline),
                         week=games.week.astype(int))
    from_spread = np.isnan(moneyline) & games.spread_wp.notna()
    need = [col for pair in MODELS.values() for col in pair]
    missing = games[need].isna().any(axis=1)
    if missing.any():
        log.warning("%d games lack a projection from one of the three - left out (%s)", missing.sum(),
                    ", ".join(f"{s}: {n}" for s, n in games[missing].groupby("season").size().items()))
    games, from_spread = games[~missing], from_spread[~missing.to_numpy()]

    table = season_table(games)
    show(table, games)
    if from_spread.any():
        counts = games[from_spread].groupby("season").size()
        print("\nmarket win probability from the spread (no moneylines) for "
              + ", ".join(f"{s}: {n} of {(games.season == s).sum()}" for s, n in counts.items()))
    dest = args.out / "benchmark.csv"
    table.to_csv(dest, index=False)
    log.info("wrote %s", dest)

    if args.save:
        cols = ["season", "week", "id", "home_team", "away_team", "margin", *need]
        games[cols].to_csv(args.save, index=False)
        log.info("wrote %s (%d games)", args.save, len(games))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
