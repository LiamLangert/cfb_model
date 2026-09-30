"""Grade ESPN's college FPI projections against the closing spread, next to RAPPA's.

The college counterpart of nfl.compare_espn, which does most of the work
here (pregame_check, season_table, agreement). RAPPA's predictions are
cfb.predict_games', and both are graded by rappa.games.grade on the same
games: every game with a line is a bet on the side each likes more than the
market, pushes are left out, and break-even at -110 is 52.4%. ESPN's number
is espn_margin from its matchup predictor.

Run `uv run -m cfb.pull_espn` for the same seasons first. When fpi.csv is there too,
this first checks that ESPN's game projections are pre-game: they should
track the team snapshot taken before each game more closely than the one
after it. Only snapshots dated inside their week are used for that.

ESPN is also graded alone on every game it has a projection and a line for,
including the ones RAPPA does not predict (week 1).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from cfb.predict_games import ALPHA, EDGES, predict
from cfb.ratings import load
from nfl.compare_espn import agreement, load_fpi, pregame_check, season_table
from rappa.games import against_the_spread, grade, overall, record

log = logging.getLogger(__name__)


def load_espn(seasons, out):
    """Every season's games (FCS pooled, as RAPPA sees them) with ESPN's projection alongside."""
    frames = []
    for season in seasons:
        path = out / str(season) / "espn.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found - run `uv run -m cfb.pull_espn {season}` first")
        espn = pd.read_csv(path, usecols=["game_id", "espn_margin"]).rename(columns={"game_id": "id"})
        _, games = load(season, out)
        frames.append(games.assign(season=season).merge(espn, on="id", how="left"))
    return pd.concat(frames, ignore_index=True)


def by_edge(both):
    """Each model's record when it disagrees with the line by at least each edge in EDGES."""
    graded = both.assign(market=-both.spread)
    rows = []
    for name, col in (("RAPPA", "predicted"), ("ESPN", "espn_margin")):
        for edge in EDGES:
            row = against_the_spread(graded.assign(predicted=graded[col]), edge)
            if row:
                rows.append({"model": name, **row})
    print("\n=== by size of disagreement with the line ===")
    cols = ["model", "edge", "bets", "wins", "win_pct", "p_value", "pushes"]
    print(pd.DataFrame(rows)[cols].to_string(index=False, float_format="%.3f"))
    print(f"p_value is one-sided vs a coin flip; {len(EDGES)} edges were tried per model, so treat it loosely.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="seasons to grade, e.g. 2023 2024 2025 2026")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--warmup", type=int, default=1,
                   help="seasons before the first one given to fit but not grade (default: 1)")
    # The same defaults as cfb.backtest_week, so RAPPA's side is what it grades.
    p.add_argument("--min-train", type=int, default=200,
                   help="games the conversion needs before it predicts (default: 200)")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--prior-weight", type=float, help="one weight for every unit (default: cfb.ratings's, by side)")
    p.add_argument("--save", type=Path, help="write every game both graded here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("rappa", "cfb"):
        logging.getLogger(name).setLevel(logging.WARNING)
    seasons = sorted(args.seasons)

    schedule = load_espn(seasons, args.out)
    played = schedule[schedule.margin.notna()]
    fpi = load_fpi(seasons, args.out)
    if fpi is None:
        log.warning("no fpi.csv for these seasons - skipping the pre-game check")
    else:
        pregame_check(played[played.season_type == "regular"].reset_index(drop=True), fpi)

    warmup = [s for s in range(seasons[0] - args.warmup, seasons[0])
              if (args.out / str(s) / "plays.csv").exists()]
    preds = predict(warmup + seasons, args.out, 2, args.alpha, args.prior_weight, min_train=args.min_train)
    rappa = grade(preds[preds.season.isin(seasons)])
    both = rappa.merge(schedule[["id", "espn_margin"]], on="id")
    if both.espn_margin.isna().any():
        log.warning("%d of RAPPA's games have no ESPN projection - left out", both.espn_margin.isna().sum())
        both = both.dropna(subset=["espn_margin"])
    both["espn_result"] = grade(both.assign(predicted=both.espn_margin)).result

    span = ", ".join(f"{s} wk {int(g.week.min())}-{int(g.week.max())}" for s, g in both.groupby("season"))
    print(f"\n=== RAPPA and ESPN against the closing spread, same games ({span}) ===")
    print(season_table(both).to_string(index=False, float_format="%.2f"))
    print("\nRAPPA: " + overall(both))
    print("ESPN:  " + overall(both.assign(result=both.espn_result)))
    agreement(both)
    by_edge(both)

    alone = grade(played.assign(predicted=played.espn_margin))
    print(f"\n=== ESPN alone, every game it has a projection and a line for ({len(alone)} games) ===")
    print("  ".join(f"{s}: {record(g.result)}" for s, g in alone.groupby("season")))
    print(overall(alone))

    if args.save:
        cols = ["season", "week", "id", "home_team", "away_team", "margin", "spread", "predicted", "espn_margin",
                "result", "espn_result"]
        both[cols].to_csv(args.save, index=False)
        log.info("wrote %s (%d games)", args.save, len(both))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
