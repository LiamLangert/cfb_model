"""Season-by-season accuracy of RAPPA, ESPN's FPI and the closing market, on the same games.

Where compare_espn grades against the spread, this scores each projection
directly:

    MAE       mean absolute error of the projected home margin, in points
    log-loss  of the home team's win probability. Ties are left out of it;
              a coin flip scores ln 2 = 0.693.

RAPPA's margins are nfl.predict_games's, walked forward the same way,
quarterbacks, special teams and penalties (nfl.outside), rest and the
returning-snap shifts of the prior (nfl.offseason) included unless --no-qbs,
--no-special-teams, --no-rest or --no-offseason is given. Games are predicted with the
quarterback who actually started them, which is known before kickoff. It has
no win probability of its own, so it gets the usual normal one: P(home win)
= Phi(predicted / sigma), where sigma is the sd of RAPPA's misses on every
game it predicted before that week, so that is walked forward too. ESPN's
are its matchup predictor's espn_margin and espn_home_wp. The market's are
the closing spread_line, and the closing moneylines with the vig taken out.

Only games all three have are graded. That leaves out week 1 (RAPPA has no
plays to rate yet) and anything ESPN never projected, such as 2017's
wild-card and divisional rounds. The --warmup seasons before the first one
given are fitted but not graded, so the points conversion already has its
--min-train games by week 2 of the first graded season.

ESPN's projections start in 2015. Run nfl/pull.py and nfl.pull_espn for the
seasons, and nfl.ratings for each season before one (its prior), first.
Writes the table to <out>/benchmark.csv.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nfl.predict_games import ALPHA, season_features, walk_forward_predict
from rappa.games import market_wp, win_probability

log = logging.getLogger(__name__)

# Each source's home margin and home win probability.
MODELS = {
    "rappa": ("predicted", "rappa_wp"),
    "espn": ("espn_margin", "espn_home_wp"),
    "market": ("spread_line", "market_wp"),
}


def load_espn(seasons, out):
    frames = []
    for season in seasons:
        path = out / str(season) / "espn.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found - run `uv run -m nfl.pull_espn {season}` first")
        frames.append(pd.read_csv(path, usecols=["game_id", "espn_margin", "espn_home_wp"]))
    return pd.concat(frames, ignore_index=True)


def losses(games, wp):
    """Each decided game's log-loss for one source's win probability."""
    decided = games[games.margin != 0]
    p = decided[wp].clip(1e-6, 1 - 1e-6)
    return -np.where(decided.margin > 0, np.log(p), np.log(1 - p))


def row(g):
    """One line of the table: every source's MAE and log-loss, and RAPPA minus ESPN game by game."""
    out = {"games": len(g), "weeks": f"{g.week.min()}-{g.week.max()}"}
    for name, (margin, wp) in MODELS.items():
        out[f"mae_{name}"] = (g.margin - g[margin]).abs().mean()
    for name, (margin, wp) in MODELS.items():
        out[f"ll_{name}"] = losses(g, wp).mean()
    for metric, diff in paired(g, "rappa", "espn").items():
        out[f"{metric}_rappa_espn"], out[f"{metric}_rappa_espn_se"] = diff
    return out


def paired(g, a, b):
    """Mean per-game difference a - b in absolute error and in log-loss, each with its standard error."""
    (ma, wa), (mb, wb) = MODELS[a], MODELS[b]
    diffs = {"mae": (g.margin - g[ma]).abs() - (g.margin - g[mb]).abs(), "ll": losses(g, wa) - losses(g, wb)}
    return {k: (np.mean(d), np.std(d, ddof=1) / np.sqrt(len(d))) for k, d in diffs.items()}


def season_table(games):
    rows = [{"season": str(s), **row(g)} for s, g in games.groupby("season")]
    return pd.DataFrame(rows + [{"season": "all", **row(games)}])


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
    print(f"\n=== {span}: the same {len(games)} games for all three (weeks 2 on, playoffs included) ===")
    print("MAE: points off the actual home margin. LL: log-loss of home win probability (ties left out;"
          " a coin flip is 0.693).\nR-E: RAPPA minus ESPN per game, ± its standard error - negative means RAPPA is"
          " better.")
    print(printed.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== against the market, per game (negative = better than the closing line) ===")
    for name in ("rappa", "espn"):
        d = paired(games, name, "market")
        print(f"{name:>5} - market: MAE {d['mae'][0]:+.3f} ± {d['mae'][1]:.3f}   LL {d['ll'][0]:+.4f} ± {d['ll'][1]:.4f}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="seasons to grade, e.g. 2015 2016 ... 2026")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    p.add_argument("--warmup", type=int, default=1,
                   help="seasons before the first one given to fit but not grade (default: 1)")
    # The same defaults as nfl.predict_games, so RAPPA is what it predicts.
    p.add_argument("--min-train", type=int, default=100,
                   help="games the conversion, and sigma, need before RAPPA predicts (default: 100)")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--prior-weight", type=float, help="one weight for every unit (default: nfl.ratings's, by side)")
    p.add_argument("--no-special-teams", action="store_true",
                   help="leave special teams and penalties out of RAPPA's margin")
    p.add_argument("--no-qbs", action="store_true", help="leave quarterbacks in their teams' pass offence")
    p.add_argument("--no-rest", action="store_true", help="leave the rest gap out of RAPPA's margin")
    p.add_argument("--no-offseason", action="store_true", help="leave the prior unshifted by returning snaps")
    p.add_argument("--save", type=Path, help="write every graded game here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("rappa").setLevel(logging.WARNING)
    logging.getLogger("nfl.predict_games").setLevel(logging.WARNING)
    seasons = sorted(args.seasons)

    warmup = [s for s in range(seasons[0] - args.warmup, seasons[0])
              if (args.out / str(s) / "plays.csv").exists()]
    if len(warmup) < args.warmup:
        log.warning("only %d of %d warm-up seasons on disk - the first graded weeks may be missing",
                    len(warmup), args.warmup)
    rows = pd.concat([
        season_features(s, args.out, 2, args.alpha, args.prior_weight, True, not args.no_special_teams,
                        not args.no_qbs, not args.no_rest, not args.no_offseason)
        for s in warmup + seasons
    ], ignore_index=True)
    preds = win_probability(walk_forward_predict(rows, args.min_train), args.min_train)

    games = preds[preds.season.isin(seasons) & preds.margin.notna()]
    games = games.merge(load_espn(seasons, args.out), on="game_id", how="left")
    games["market_wp"] = market_wp(games)
    need = [col for pair in MODELS.values() for col in pair]
    missing = games[need].isna().any(axis=1)
    if missing.any():
        log.warning("%d games lack a projection from one of the three - left out (%s)", missing.sum(),
                    ", ".join(f"{s}: {n}" for s, n in games[missing].groupby("season").size().items()))
    games = games[~missing]

    table = season_table(games)
    show(table, games)
    dest = args.out / "benchmark.csv"
    table.to_csv(dest, index=False)
    log.info("wrote %s", dest)

    if args.save:
        cols = ["season", "week", "game_id", "home_team", "away_team", "margin", *need]
        games[cols].to_csv(args.save, index=False)
        log.info("wrote %s (%d games)", args.save, len(games))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
