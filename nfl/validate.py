"""Backtest NFL RAPPA: does it predict football that has not happened yet?

The NFL counterpart of cfb.validate, and where the constants in nfl.ratings
come from. Every experiment fits on weeks 1-N of a season and
scores on the weeks after, so no holdout play can leak into the ratings.

  predictors  RAPPA against season-to-date raw epa and against predicting
              the mean for everyone
  hfa         fixed home-field values, from none up past the market's
  clip        how hard to clip each play's epa before fitting
  penalties   keep penalty no-plays in the fit, or drop them as college does
  garbage     each garbage-time rule applied to the training plays
  decay       time-decay half-lives against weighting every week alike
  prior       shrinking toward last season's rappa.csv, at several weights

One NFL season is too short to settle any of these, so pass several: each
(season, cut) is a row, and the mean across them is printed last. The prior
experiment uses <out>/<season - 1>/rappa.csv and skips seasons without one.

Scoring unit is one team's offence in one game, as in cfb.validate (see
rappa/validation.py): correlation with the unit's actual mean epa (higher is
better) and play-weighted squared error (lower is better).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from nfl.ratings import CLIP, HFA, PRIOR_WEIGHT, fit, load
from rappa.model import GARBAGE_RULES, garbage_time, summarize
from rappa.validation import correlate, holdout_units, predict_units, weighted_mse

log = logging.getLogger(__name__)

EXPERIMENTS = ("predictors", "hfa", "clip", "penalties", "garbage", "decay", "prior")

# 0.013 is about what the market's 1.7 points of home field comes to per play.
HFAS = (0.0, 0.003, 0.008, 0.013, 0.02)
CLIPS = (0.0, 2.0, 3.0, 4.0, 6.0)
PRIOR_WEIGHTS = (0.0, 0.3, 0.45, 0.6, 0.85)
HALF_LIVES = (None, 4.0, 8.0, 16.0)


def splits(data, cuts, priors=None):
    """(season, cut, train, test, prior) for every cut that leaves a holdout."""
    for season, df in data.items():
        prior = priors.get(season) if priors is not None else None
        if priors is not None and prior is None:
            continue
        for cut in cuts:
            train, test = df[df.week <= cut], df[df.week > cut]
            if not train.empty and not test.empty:
                yield season, cut, train, test, prior


def rappa_prediction(train, units, alpha, hfa=HFA, clip=CLIP, prior=None,
                     prior_weight=PRIOR_WEIGHT, half_life=None):
    """Fit on train and predict each holdout unit's mean epa."""
    ratings, _, intercept = fit(train, alpha, prior=prior, prior_weight=prior_weight,
                                half_life=half_life, hfa=hfa, clip=clip)
    ratings = summarize(ratings, train)
    # Hand hfa back so predictions add the same home field the fit took off.
    return predict_units(units, ratings, intercept, hfa)


def compare(split_list, variants):
    """Score every variant on every split. A variant is (train, units, prior) -> prediction."""
    rows = []
    for season, cut, train, test, prior in split_list:
        units = holdout_units(test, set(train.offense) | set(train.defense))
        row = {"season": season, "train_weeks": f"1-{cut}", "units": len(units)}
        for label, predict in variants.items():
            pred = predict(train, units, prior)
            # Correlation is undefined for a flat prediction such as const.
            row[f"corr@{label}"] = correlate(units, pred) if pred.nunique() > 1 else float("nan")
            row[f"mse@{label}"] = weighted_mse(units, pred)
        rows.append(row)
        log.info("%d weeks 1-%d done", season, cut)
    return pd.DataFrame(rows)


def variants_for(experiment, alpha, hfas=HFAS, clips=CLIPS, weights=PRIOR_WEIGHTS,
                 half_lives=HALF_LIVES):
    def rappa(**kw):
        return lambda train, units, prior: rappa_prediction(train, units, alpha, **kw)

    if experiment == "predictors":
        def raw(train, units, prior):
            # Epa allowed is a defensive weakness, so it adds to what the
            # offence should expect; the league mean is in both terms once.
            by_off, by_def = train.groupby("offense").ppa.mean(), train.groupby("defense").ppa.mean()
            return units.offense.map(by_off) + units.defense.map(by_def) - train.ppa.mean()

        def const(train, units, prior):
            return pd.Series(train.ppa.mean(), index=units.index)

        return {"rappa": rappa(), "raw": raw, "const": const}
    if experiment == "hfa":
        return {f"{h:g}": rappa(hfa=h) for h in hfas}
    if experiment == "clip":
        return {f"{c:g}": rappa(clip=c) for c in clips}
    if experiment == "penalties":
        def dropped(train, units, prior):
            return rappa_prediction(train[~train.no_play], units, alpha)

        return {"kept": rappa(), "dropped": dropped}
    if experiment == "garbage":
        def filtered(rule):
            return lambda train, units, prior: rappa_prediction(
                train[~garbage_time(train, rule)], units, alpha)

        return {rule: filtered(rule) for rule in sorted(GARBAGE_RULES)}
    if experiment == "decay":
        return {"none" if hl is None else f"{hl:g}w": rappa(half_life=hl) for hl in half_lives}
    if experiment == "prior":
        def shrunk(weight):
            return lambda train, units, prior: rappa_prediction(
                train, units, alpha, prior=prior if weight else None, prior_weight=weight)

        return {f"{w:g}": shrunk(w) for w in weights}
    raise ValueError(experiment)


def with_mean(table):
    """The per-split table with a mean row appended, which is the one to read."""
    mean = table.drop(columns=["season", "train_weeks"]).mean().to_frame().T
    mean["season"], mean["train_weeks"] = "mean", ""
    return pd.concat([table, mean[table.columns]], ignore_index=True)


def show(title, table):
    fmt = {c: "{:+.4f}".format for c in table.columns if c.startswith("corr")}
    fmt.update({c: "{:.5f}".format for c in table.columns if c.startswith("mse")})
    fmt["units"] = "{:.0f}".format
    print(f"\n=== {title} ===")
    print(with_mean(table).to_string(index=False, formatters=fmt))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season(s) to backtest, e.g. 2023 2024 2025")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    p.add_argument("--experiment", choices=EXPERIMENTS + ("all",), nargs="+", default=["all"])
    p.add_argument("--cuts", type=int, nargs="+", default=[4, 8, 12], help="train/test week splits")
    p.add_argument("--prior-cuts", type=int, nargs="+", default=[1, 2, 3, 5, 8],
                   help="cutoffs for the prior experiment (early weeks matter most)")
    p.add_argument("--alpha", type=float,
                   help="fix the ridge penalty to run faster (default: cross-validate every fit)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    experiments = EXPERIMENTS if "all" in args.experiment else args.experiment

    data = {season: load(season, args.out) for season in args.seasons}
    for season, df in data.items():
        print(f"{season}: {len(df)} plays, weeks {int(df.week.min())}-{int(df.week.max())}")

    for experiment in experiments:
        variants = variants_for(experiment, args.alpha)
        if experiment == "prior":
            paths = {s: args.out / str(s - 1) / "rappa.csv" for s in data}
            priors = {s: pd.read_csv(p, index_col="team") for s, p in paths.items() if p.exists()}
            if not priors:
                print(f"\n(skipping prior: no {paths[min(paths)]} - run `uv run -m nfl.ratings` on the season before)")
                continue
            table = compare(list(splits(data, args.prior_cuts, priors)), variants)
            show(f"Shrinking toward last season's rappa.csv (weight 0 = plain ridge; {PRIOR_WEIGHT:g} is"
                 " the default)", table)
        elif experiment == "penalties":
            # Score twice: against everything that happened, and against snaps
            # alone, so keeping penalties cannot win just by predicting them.
            all_splits = list(splits(data, args.cuts))
            show("Penalty no-plays kept or dropped from the fit, scored on all holdout plays",
                 compare(all_splits, variants))
            snaps = [(s, c, tr, te[~te.no_play], pr) for s, c, tr, te, pr in all_splits]
            show("... scored on holdout snaps only", compare(snaps, variants))
        else:
            table = compare(list(splits(data, args.cuts)), variants)
            titles = {
                "predictors": "RAPPA vs season-to-date raw epa vs the mean (mse_const is the floor to beat)",
                "hfa": f"Fixed home field in epa/play per side ({HFA:g} is the default)",
                "clip": f"Clip each play's epa to +/- this (0 = off; {CLIP:g} is the default)",
                "garbage": "Garbage-time rules applied to training plays",
                "decay": "Time decay across weeks; 'none' is uniform weighting",
            }
            show(titles[experiment], table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
