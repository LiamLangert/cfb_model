"""Backtest college RAPPA: does it predict football that has not happened yet?

Every experiment fits on early weeks and scores on later ones, so no holdout
play can leak into the ratings.

  predictors  Compare RAPPA against the obvious baselines. A team's expected
              offensive PPA is its offence rating minus the opponent's defence
              rating; the raw baseline is season-to-date mean PPA for and
              against. Reported as correlation with what actually happened.

  garbage     Refit RAPPA with each garbage-time rule applied to the training
              plays and see whether dropping them helps or hurts.

  prior       Shrink toward last season's ratings (--prior) at several weights.

  decay       Weight recent weeks more heavily, at several half-lives.

Scoring unit is one team's offence in one game, keeping only units with enough
plays for the average to mean anything (see research/validation.py).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from cfb.ratings import (FCS, HFA, PRIOR_WEIGHT, add_pre_play_margin, attach_game_context, classify,
                         fit, pool_fcs)
from rappa.model import GARBAGE_RULES, garbage_time, summarize
from research import validation
from research.validation import correlate, predict_units, weighted_mse


def load(season, out):
    """Plays with a PPA, a rush/pass label, a week and a pre-play margin."""
    path = out / str(season) / "plays.csv"
    games_path = out / str(season) / "games.csv"
    if not path.exists() or not games_path.exists():
        raise SystemExit(f"need {path} and {games_path} - run `uv run -m cfb.pull {season}` first")

    plays, games = pool_fcs(pd.read_csv(path, low_memory=False), pd.read_csv(games_path))
    plays = attach_game_context(add_pre_play_margin(plays), games)
    df = plays[plays.ppa.notna() & plays.offense.notna() & plays.defense.notna()]
    return classify(df).dropna(subset=["week"])


def holdout_units(df, known_teams):
    """Actual mean PPA per (game, offence), for FBS units we have a rating for.

    Units facing the pooled FCS team are left out. They are easy calls that
    lift correlation without saying anything about telling FBS teams apart.
    """
    return validation.holdout_units(df, known_teams - {FCS})


def rate(train, prior=None, prior_weight=PRIOR_WEIGHT, half_life=None, hfa=HFA):
    ratings, alpha, intercept = fit(
        train, prior=prior, prior_weight=prior_weight, half_life=half_life, hfa=hfa,
    )
    # Hand hfa back so predictions add the same home field the fit took off.
    return summarize(ratings, train), alpha, intercept, hfa


def experiment_predictors(df, cuts):
    """RAPPA vs season-to-date raw PPA vs predicting the mean."""
    rows = []
    for cut in cuts:
        train, test = df[df.week <= cut], df[df.week > cut]
        if train.empty or test.empty:
            continue
        ratings, _, intercept, hfa = rate(train)
        units = holdout_units(test, set(ratings.index))

        # Raw baseline: PPA allowed is a defensive weakness, so it ADDS to what
        # the offence should expect. Subtracting it here silently inverts the
        # baseline and flatters RAPPA.
        raw_off = train.groupby("offense").ppa.mean()
        raw_def = train.groupby("defense").ppa.mean()

        rappa_pred = predict_units(units, ratings, intercept, hfa)
        # Raw PPA already carries the league level in both terms, so it needs
        # the mean subtracted back off rather than added.
        raw_pred = units.offense.map(raw_off) + units.defense.map(raw_def) - train.ppa.mean()
        # Predicting the training mean for everyone: no information at all.
        flat = pd.Series(train.ppa.mean(), index=units.index)

        rows.append({
            "train_weeks": f"1-{cut}",
            "train_plays": len(train),
            "units": len(units),
            "corr_rappa": correlate(units, rappa_pred),
            "corr_raw": correlate(units, raw_pred),
            "mse_rappa": weighted_mse(units, rappa_pred),
            "mse_raw": weighted_mse(units, raw_pred),
            "mse_const": weighted_mse(units, flat),
        })
    return pd.DataFrame(rows)


def experiment_garbage(df, cut):
    """Each garbage-time rule applied to training, scored both ways."""
    train_all, test_all = df[df.week <= cut], df[df.week > cut]
    rows = []
    for rule in sorted(GARBAGE_RULES):
        train = train_all[~garbage_time(train_all, rule)]
        ratings, alpha, intercept, hfa = rate(train)
        known = set(ratings.index)
        scores = {}
        # Score twice: against clean holdout football, and against everything
        # that happened. A filter should not need a filtered target to look good.
        for label, test in (
            ("vs_clean", test_all[~garbage_time(test_all, "connelly")]),
            ("vs_all", test_all),
        ):
            units = holdout_units(test, known)
            scores[label] = correlate(units, predict_units(units, ratings, intercept, hfa))
        rows.append({
            "rule": rule, "train_plays": len(train),
            "dropped_pct": round(100 * (1 - len(train) / len(train_all)), 1),
            "alpha": alpha, **scores,
        })
    # Best rule first, so the answer is the top row.
    return pd.DataFrame(rows).sort_values("vs_all", ascending=False)


def experiment_prior(df, prior, cuts, weights):
    """Does shrinking toward last season beat shrinking toward zero?

    The payoff should be biggest at the earliest cutoffs, where a team has
    played too little for its own plays to say much. Weight 0 is plain ridge.
    """
    rows = []
    for cut in cuts:
        train, test = df[df.week <= cut], df[df.week > cut]
        if train.empty or test.empty:
            continue
        row = {"train_weeks": f"1-{cut}", "train_plays": len(train)}
        for weight in weights:
            ratings, _, intercept, hfa = rate(train, prior if weight > 0 else None, weight)
            units = holdout_units(test, set(ratings.index))
            pred = predict_units(units, ratings, intercept, hfa)
            row[f"corr@{weight}"] = correlate(units, pred)
            row[f"mse@{weight}"] = weighted_mse(units, pred)
        rows.append(row)
    return pd.DataFrame(rows)


def experiment_decay(df, cuts, half_lives, prior=None):
    """Does weighting recent weeks more heavily beat treating all plays alike?

    Decay trades sample size for freshness, so it should only pay once there
    are enough weeks for "recent" to mean anything.
    """
    rows = []
    for cut in cuts:
        train, test = df[df.week <= cut], df[df.week > cut]
        if train.empty or test.empty:
            continue
        row = {"train_weeks": f"1-{cut}", "train_plays": len(train)}
        for hl in half_lives:
            ratings, _, intercept, hfa = rate(train, prior, PRIOR_WEIGHT, hl)
            units = holdout_units(test, set(ratings.index))
            pred = predict_units(units, ratings, intercept, hfa)
            label = "none" if hl is None else f"{hl:g}w"
            row[f"corr@{label}"] = correlate(units, pred)
            row[f"mse@{label}"] = weighted_mse(units, pred)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("season", type=int, help="season to backtest, e.g. 2024")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--cuts", type=int, nargs="+", default=[4, 6, 9], help="train/test week splits")
    p.add_argument("--garbage-cut", type=int, default=9, help="split for the garbage-time test")
    p.add_argument(
        "--experiment", choices=["predictors", "garbage", "prior", "decay", "all"], default="all",
    )
    p.add_argument("--prior", type=Path, help="previous season's rappa.csv, for the prior experiment")
    p.add_argument("--prior-weights", type=float, nargs="+", default=[0.0, 0.3, 0.57, 0.85],
                   help="prior strengths to compare (0 = plain ridge)")
    p.add_argument("--prior-cuts", type=int, nargs="+", default=[1, 2, 3, 5, 8],
                   help="cutoffs for the prior experiment (early weeks matter most)")
    p.add_argument("--half-lives", type=float, nargs="+", default=[2, 4, 6, 10],
                   help="decay half-lives in weeks to compare against no decay")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    df = load(args.season, args.out)
    print(f"{args.season}: {len(df)} classified plays with PPA, weeks {int(df.week.min())}-{int(df.week.max())}")

    if args.experiment in ("predictors", "all"):
        print("\n=== Predicting later-week team PPA (correlation, higher is better) ===")
        print(experiment_predictors(df, args.cuts).to_string(index=False, formatters={
            "corr_rappa": "{:+.4f}".format, "corr_raw": "{:+.4f}".format,
            "mse_rappa": "{:.5f}".format, "mse_raw": "{:.5f}".format, "mse_const": "{:.5f}".format,
        }))
        print("mse is play-weighted; lower is better. A predictor that cannot beat"
              " mse_const is worse than useless.")

    if args.experiment in ("garbage", "all"):
        print(f"\n=== Garbage-time filters, trained on weeks 1-{args.garbage_cut} ===")
        table = experiment_garbage(df, args.garbage_cut)
        print(table.to_string(index=False, formatters={
            "dropped_pct": "{:.1f}%".format,
            "alpha": "{:.0f}".format,
            "vs_clean": "{:+.4f}".format,
            "vs_all": "{:+.4f}".format,
        }))
    if args.experiment in ("prior", "all"):
        if not args.prior:
            print("\n(skipping prior experiment: pass --prior <previous season rappa.csv>)")
        else:
            prior = pd.read_csv(args.prior, index_col="team")
            print(f"\n=== Shrinking toward {args.prior} (weight 0 = plain ridge) ===")
            table = experiment_prior(df, prior, args.prior_cuts, args.prior_weights)
            fmt = {c: "{:+.4f}".format for c in table.columns if c.startswith("corr")}
            fmt.update({c: "{:.5f}".format for c in table.columns if c.startswith("mse")})
            print(table.to_string(index=False, formatters=fmt))

    if args.experiment in ("decay", "all"):
        prior = pd.read_csv(args.prior, index_col="team") if args.prior else None
        tag = " (on top of the prior)" if prior is not None else ""
        print(f"\n=== Time decay across weeks{tag}; 'none' is uniform weighting ===")
        table = experiment_decay(df, args.cuts, [None] + list(args.half_lives), prior)
        fmt = {c: "{:+.4f}".format for c in table.columns if c.startswith("corr")}
        fmt.update({c: "{:.5f}".format for c in table.columns if c.startswith("mse")})
        print(table.to_string(index=False, formatters=fmt))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
