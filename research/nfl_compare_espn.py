"""Grade ESPN's FPI projections against the closing spread, next to RAPPA's.

RAPPA's predictions are rebuilt the way research.nfl_backtest_week makes them, and
both are graded by its rule on the same games: every game with a line is a
bet on the side each likes more than the market, pushes are left out, and
break-even at -110 is 52.4%. ESPN's number is espn_margin from its matchup
predictor. It is a benchmark, not a like-for-like test of the ratings: ESPN
also folds in quarterbacks, rest and travel, which RAPPA does not see.

Run `uv run -m nfl.pull_espn` for the same seasons first. When fpi.csv is there too,
this first checks that ESPN's game projections are pre-game: they should
track the team snapshot taken before each game more closely than the one
after it. Only snapshots dated inside their week are used for that.

ESPN is also graded alone on every game it has, including the ones RAPPA
cannot predict (week 1, and weeks before --min-train is reached).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nfl.predict_games import ALPHA, EDGES, load_games, season_features, walk_forward_predict
from rappa.games import against_the_spread, grade, overall, record

log = logging.getLogger(__name__)


def load_espn(seasons, out):
    """Every season's schedule with ESPN's projection alongside."""
    frames = []
    for season in seasons:
        path = out / str(season) / "espn.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found - run `uv run -m nfl.pull_espn {season}` first")
        espn = pd.read_csv(path, usecols=["game_id", "espn_margin"])
        frames.append(load_games(season, out).merge(espn, on="game_id", how="left"))
    return pd.concat(frames, ignore_index=True)


def load_fpi(seasons, out):
    """The in-window snapshots of every season with an fpi.csv, or None if none has one."""
    frames = [pd.read_csv(p) for p in (out / str(s) / "fpi.csv" for s in seasons) if p.exists()]
    if not frames:
        return None
    fpi = pd.concat(frames, ignore_index=True)
    return fpi[fpi.in_window]


def pregame_check(games, fpi):
    """How closely ESPN's projection follows the FPI gap from just before each game, and just after."""
    rating = fpi.set_index(["season", "through_week", "team"]).fpi

    def gap(week):
        def at(team):
            return rating.reindex(pd.MultiIndex.from_arrays([games.season, week, team])).to_numpy()
        return at(games.home_team) - at(games.away_team)

    before, after = gap(games.week - 1), gap(games.week)
    # The same games for both, so the two fits are comparable.
    ok = ~np.isnan(before) & ~np.isnan(after) & games.espn_margin.notna().to_numpy()
    y = games.espn_margin.to_numpy()[ok]
    rows = []
    for label, x in (("before the game", before[ok]), ("after the game", after[ok])):
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({"FPI snapshot": label, "r": np.corrcoef(x, y)[0, 1], "slope": slope,
                     "home field": intercept, "resid sd": np.std(y - (intercept + slope * x))})
    print(f"\n=== is ESPN's projection pre-game? ({ok.sum()} games, home margin vs FPI gap) ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.2f"))


def season_table(both):
    """Both records and both errors, season by season, on the same games."""
    rows = []
    for season, g in list(both.groupby("season")) + [("all", both)]:
        rows.append({
            "season": season, "games": len(g),
            "RAPPA ATS": record(g.result), "ESPN ATS": record(g.espn_result),
            "MAE market": (g.margin + g.spread).abs().mean(),
            "MAE RAPPA": (g.margin - g.predicted).abs().mean(),
            "MAE ESPN": (g.margin - g.espn_margin).abs().mean(),
        })
    return pd.DataFrame(rows)


def agreement(both):
    """How often the two take the same side, and how each does when they split."""
    rappa_edge = both.predicted + both.spread
    espn_edge = both.espn_margin + both.spread
    same = np.sign(rappa_edge) == np.sign(espn_edge)
    print(f"\ncorrelation of the two edges against the line: {np.corrcoef(rappa_edge, espn_edge)[0, 1]:.2f}")
    print(f"same side in {same.mean():.0%} of games: {record(both.result[same])} (the same bets for both)")
    print(f"opposite sides: RAPPA {record(both.result[~same])}, ESPN {record(both.espn_result[~same])}")


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
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    # The same defaults as research.nfl_backtest_week, so RAPPA's side is what it grades.
    p.add_argument("--min-train", type=int, default=100,
                   help="games the conversion needs before it predicts (default: 100)")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--prior-weight", type=float, help="one weight for every unit (default: nfl.ratings's, by side)")
    p.add_argument("--save", type=Path, help="write every game both graded here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("rappa", "nfl"):
        logging.getLogger(name).setLevel(logging.WARNING)
    seasons = sorted(args.seasons)

    schedule = load_espn(seasons, args.out)
    played = schedule[schedule.margin.notna()]
    fpi = load_fpi(seasons, args.out)
    if fpi is None:
        log.warning("no fpi.csv for these seasons - skipping the pre-game check")
    else:
        pregame_check(played.reset_index(drop=True), fpi)

    rows = pd.concat([
        season_features(s, args.out, 2, args.alpha, args.prior_weight, True) for s in seasons
    ], ignore_index=True)
    rappa = grade(walk_forward_predict(rows, args.min_train))
    both = rappa.merge(schedule[["game_id", "espn_margin"]], on="game_id")
    if both.espn_margin.isna().any():
        log.warning("%d of RAPPA's games have no ESPN projection - left out", both.espn_margin.isna().sum())
        both = both.dropna(subset=["espn_margin"])
    both["espn_result"] = grade(both.assign(predicted=both.espn_margin)).result

    span = ", ".join(f"{s} wk {g.week.min()}-{g.week.max()}" for s, g in both.groupby("season"))
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
        cols = ["season", "week", "game_type", "game_id", "home_team", "away_team", "margin", "spread_line",
                "predicted", "espn_margin", "result", "espn_result"]
        both[cols].to_csv(args.save, index=False)
        log.info("wrote %s (%d games)", args.save, len(both))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
