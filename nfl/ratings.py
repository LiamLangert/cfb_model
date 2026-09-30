"""RAPPA for the NFL, fitted on nflverse play-by-play.

The model is in rappa/model.py - see there for how it works. This module
only feeds it NFL plays and swaps in constants re-measured on NFL data:

    target      nflverse epa, renamed ppa so the fit reads it as is
    plays       every play nflverse flags as a pass or a rush (see load)
    home field  HFA below, about a fifth of the college value
    garbage     GARBAGE below: plays at a 21-point lead or more are dropped
    clip        CLIP below: none, once garbage time is gone
    prior       PRIOR_WEIGHT and PRIOR_WEIGHT_DEF below, lower than college
                and lower for defence - NFL teams carry less of their
                rating from one season to the next, defences least
    passers     quarterbacks are rated on their own (fit_qbs and the QB_
                constants below), so a team's pass_off is its supporting cast

All 32 teams are rated from every game, playoffs included. Run
`uv run nfl/pull.py <season>` first.

Writes <out>/<season>/rappa.csv (default data/nfl/<season>/rappa.csv), one row
per team, and qbs.csv next to it, one row per passer. In rappa.csv, qb is the
team's passers' rating weighted by their dropbacks, and off and net count it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rappa import model
from rappa.model import GARBAGE_RULES, conference_home_edge, garbage_time, summarize

log = logging.getLogger(__name__)

# Only what the fit needs; the full play-by-play file has 372 columns.
COLUMNS = [
    "game_id", "week", "posteam", "defteam", "home_team", "location", "div_game",
    "qtr", "score_differential", "play_type", "pass", "rush", "two_point_attempt", "epa",
    "passer_id", "passer",
]

# Home-field advantage in epa per play, per side, held fixed - the same
# convention as cfb.ratings.HFA. Measured the same way too, on division games: each
# division pair meets once at each stadium, so that schedule is exactly
# balanced. The raw edge was +0.0013, +0.0014 and +0.0048 in 2023-2025. On
# 2023-2025 holdout weeks 0.003 predicted best, with 0 and 0.008 about level.
# Home teams did win by 2.4 points a game and the market priced 1.7 (about
# 0.013 a play), but that much predicted clearly worse: the scoreboard edge
# does not show up in scrimmage epa.
HFA = 0.003

# The constants below were re-tuned on game margins rather than holdout plays:
# nfl.predict_games walked forward over 2015-2022, choosing on MAE and
# log-loss, then checked on 2023-2026 (see nfl.benchmark). Together they cut
# MAE by 0.066 +/- 0.026 points a game against the play-tuned values (clip 4,
# no garbage filter, one prior weight of 0.45), and by 0.11 on 2023-2026.
# The optimum is broad for all of them.

# Plays by a team leading by at least this much, in any quarter, are dropped
# (rappa.model.GARBAGE_RULES). Every rule helped a little on game margins,
# flat21 most; on holdout plays none did.
GARBAGE = "flat21"

# Each play's epa is clipped to +/-CLIP before fitting; 0 is no clip. 4.4% of
# NFL plays are beyond +/-3, against a tail that reaches -12, and 4 was
# marginally best on holdout plays. With garbage time dropped, though, no clip
# predicted game margins best (0.05 better than 4 on 2023-2026).
CLIP = 0.0

# How much of last season's rating to carry forward when there is a prior,
# offence (PRIOR_WEIGHT) and defence (PRIOR_WEIGHT_DEF) apart. One weight of
# 0.45 predicted holdout plays best in 2024-25; on game margins offence
# carried over more than defence, 0.45-0.5 against 0.25-0.35 - offence is
# steadier from one season to the next.
PRIOR_WEIGHT = 0.5
PRIOR_WEIGHT_DEF = 0.3

# Quarterbacks get a rating of their own (rappa.model.fit_passers): every pass
# play, sacks and scrambles included, is the team's pass unit plus its passer,
# and a game uses the quarterback who starts it. Walking forward on game
# margins over 2015-2022 and checked on 2023-2026, this cut MAE by 0.11 and
# log-loss by 0.009-0.010 in both, most of it in the third of games where a
# team started someone other than its usual quarterback.
#
# QB_PRIOR_WEIGHT is how much of last season's rating a quarterback carries,
# wherever he plays: far more than a team does.
QB_PRIOR_WEIGHT = 0.9

# QB_REPLACEMENT is the rating, in epa per dropback, of a quarterback we know
# nothing about, and what every passer is shrunk toward in a season's own
# fit. It sits where the model's quarterback-driven moves match the size of
# the market's in games with a changed starter: -0.15 moved a third too
# little and -0.7 too much. Anywhere from -0.4 to -0.7 predicted about as well.
QB_REPLACEMENT = -0.5

# PASS_PRIOR_WEIGHT is the team pass unit's carry-over once the quarterback is
# out of it: the supporting cast carries over like a defence.
PASS_PRIOR_WEIGHT = 0.3

# The penalty for a season's own fit when quarterbacks are rated, instead of
# cross-validating. Shrinking every passer toward QB_REPLACEMENT offsets the
# target, and cross-validation then picks 100, which predicted the next
# season's games worse (log-loss +0.003 on 2023-2026); 300 is what it picks
# without passers.
SEASON_ALPHA = 300.0

# Because every passer is shrunk toward QB_REPLACEMENT, the fitted ratings
# carry offsets: quarterbacks sit well below zero and some team units above
# it. They are common to every team and cancel in predictions, so the files
# keep them (they are next season's prior), and centered() takes them out
# for reading.


def load(season, out):
    """One season's plays, in the columns rappa.fit expects.

    A play counts if nflverse flags it as a pass or a rush. That puts sacks and
    scrambles with passes, and it keeps penalty no-plays such as pass
    interference, which the college fit leaves out. Keeping them predicted
    holdout epa slightly better, and no worse once the holdout had its
    penalties taken out too. Kneels and spikes carry neither flag. Two-point
    tries do, and are dropped because they are not played from a down and
    distance.
    """
    path = out / str(season) / "plays.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run nfl/pull.py {season}` first")
    raw = pd.read_csv(path, usecols=COLUMNS, low_memory=False)

    keep = (raw["pass"].eq(1) | raw["rush"].eq(1)) & raw.two_point_attempt.ne(1)
    keep &= raw.epa.notna() & raw.posteam.notna() & raw.defteam.notna()
    df = raw[keep]
    return pd.DataFrame({
        "game_id": df.game_id,
        # nflverse weeks already run straight through the playoffs.
        "week": df.week.astype(float),
        "offense": df.posteam,
        "defense": df.defteam,
        "home": df.home_team,
        "neutral_site": df.location.eq("Neutral"),
        # Division games play the part conference games do in
        # conference_home_edge: the balanced schedule to measure home field on.
        "conference_game": df.div_game.eq(1),
        "period": df.qtr,
        # Already the offence's lead at the snap, as garbage_time wants.
        "margin": df.score_differential,
        "play_kind": np.where(df["pass"].eq(1), "pass", "rush"),
        "ppa": df.epa,
        # For fit_qbs: nflverse's passer_id covers sacks and scrambles too.
        "passer": df.passer_id.where(df["pass"].eq(1)),
        "passer_name": df.passer.where(df["pass"].eq(1)),
        # Not used by the fit; lets nfl.validate test dropping them.
        "no_play": df.play_type.eq("no_play"),
    }).reset_index(drop=True)


def fit(df, alpha=None, seed=0, prior=None, prior_weight=None, half_life=None, hfa=HFA,
        clip=CLIP):
    """rappa.model.fit, with the NFL constants as defaults.

    prior_weight is one weight for every unit, or an (offence, defence) pair;
    None is (PRIOR_WEIGHT, PRIOR_WEIGHT_DEF). Quarterbacks are not rated; see
    fit_qbs.
    """
    if prior is not None:
        if "qb" in prior:
            # A quarterback-aware rappa.csv leaves the passers out of pass_off:
            # put them back for a fit that does not rate them, and take out its
            # offsets (see SEASON_ALPHA).
            prior = centered(prior.assign(pass_off=prior.pass_off + prior.qb))
        if prior_weight is None:
            prior_weight = (PRIOR_WEIGHT, PRIOR_WEIGHT_DEF)
        off, dfn = prior_weight if isinstance(prior_weight, tuple) else (prior_weight, prior_weight)
        # rappa.model.fit takes one weight, so each side's goes into the prior itself.
        prior = prior.assign(rush_off=prior.rush_off * off, pass_off=prior.pass_off * off,
                             rush_def=prior.rush_def * dfn, pass_def=prior.pass_def * dfn)
    return model.fit(df, alpha, hfa=hfa, clip=clip, prior=prior, prior_weight=1.0,
                     half_life=half_life, seed=seed)


def fit_qbs(df, alpha=None, seed=0, prior=None, qb_prior=None, prior_weight=None, half_life=None,
            hfa=HFA, clip=CLIP):
    """rappa.model.fit_passers with the NFL constants: team and quarterback ratings.

    prior is last season's rappa.csv and qb_prior its qbs.csv, either of which
    may be None. prior_weight, if given, is one weight for every team unit.
    Returns the team ratings, the quarterback ratings (a Series by passer id),
    the penalty and the intercept.
    """
    if prior is not None:
        weights = {"rush_off": PRIOR_WEIGHT, "pass_off": PASS_PRIOR_WEIGHT,
                   "rush_def": PRIOR_WEIGHT_DEF, "pass_def": PRIOR_WEIGHT_DEF}
        if prior_weight is not None:
            weights = dict.fromkeys(weights, prior_weight)
        prior = prior.assign(**{unit: prior[unit] * w for unit, w in weights.items()})
    return model.fit_passers(df, alpha, hfa=hfa, clip=clip, prior=prior, prior_weight=1.0,
                             half_life=half_life, seed=seed, passer_prior=qb_prior_means(qb_prior),
                             passer_default=QB_REPLACEMENT)


def qb_prior_means(qb_prior):
    """What each quarterback rated last season is shrunk toward: QB_PRIOR_WEIGHT of his gap to replacement."""
    if qb_prior is None:
        return None
    return QB_REPLACEMENT + QB_PRIOR_WEIGHT * (qb_prior.rating - QB_REPLACEMENT)


def qb_table(qbs, df):
    """One row per passer: name, the team he threw most for, dropbacks, rating."""
    thrown = df[df.passer.notna()]
    table = pd.DataFrame({
        "name": thrown.groupby("passer").passer_name.agg(lambda s: s.mode().iat[0]),
        "team": thrown.groupby("passer").offense.agg(lambda s: s.mode().iat[0]),
        "dropbacks": thrown.passer.value_counts(),
        "rating": qbs,
    })
    # For reading: against the average dropback this season.
    table["vs_average"] = table.rating - np.average(table.rating, weights=table.dropbacks)
    table.index.name = "passer_id"
    return table.sort_values("rating", ascending=False).round({"rating": 4, "vs_average": 4})


def centered(ratings):
    """rappa.csv's ratings against the average team, for reading (see SEASON_ALPHA's note)."""
    cols = [c for c in ("rush_off", "pass_off", "qb", "rush_def", "pass_def", "off", "def", "net") if c in ratings]
    return ratings.assign(**{c: ratings[c] - ratings[c].mean() for c in cols}).round(4)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season(s) to rate, fitted separately")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    p.add_argument("--alpha", type=float, help="fix the ridge penalty (default: cross-validated, or "
                                                f"{SEASON_ALPHA:g} when rating quarterbacks)")
    p.add_argument(
        "--garbage-time",
        choices=sorted(GARBAGE_RULES),
        default=GARBAGE,
        help=f"drop plays in garbage time (default: {GARBAGE})",
    )
    p.add_argument("--top", type=int, default=32, help="how many teams to print (default: all 32)")
    p.add_argument("--hfa", type=float, default=HFA,
                   help=f"home-field edge in epa/play per side, held fixed (default: {HFA}; 0 = none)")
    p.add_argument("--clip", type=float, default=CLIP,
                   help=f"clip each play's epa to +/- this before fitting (default: {CLIP}; 0 = off)")
    p.add_argument("--half-life", type=float,
                   help="weeks after which a play counts half as much (default: no decay)")
    p.add_argument("--prior", type=Path, help="previous season's rappa.csv to shrink toward")
    p.add_argument("--prior-weight", type=float,
                   help="how much of the prior to carry forward, for every unit (default: "
                        f"{PRIOR_WEIGHT} for offence, {PRIOR_WEIGHT_DEF} for defence)")
    p.add_argument("--no-qbs", action="store_true", help="leave quarterbacks in their teams' pass_off")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for season in args.seasons:
        df = load(season, args.out)
        log.info("=== %d: %d plays from %d games", season, len(df), df.game_id.nunique())

        if args.garbage_time != "none":
            junk = garbage_time(df, args.garbage_time)
            log.info("dropping %d garbage-time plays (%.1f%%, rule=%s)",
                     junk.sum(), 100 * junk.mean(), args.garbage_time)
            df = df[~junk]

        # A sanity check on the fixed home field, not an input to the fit.
        log.info("division-game home edge this season %+.4f (fitting with %+.4f)",
                 conference_home_edge(df, args.clip), args.hfa)
        prior = pd.read_csv(args.prior, index_col="team") if args.prior else None
        dest = args.out / str(season) / "rappa.csv"
        if args.no_qbs:
            ratings, _, _ = fit(
                df, args.alpha, args.seed, prior, args.prior_weight, args.half_life, args.hfa,
                args.clip,
            )
            ratings = summarize(ratings, df)
        else:
            qb_path = args.prior.parent / "qbs.csv" if args.prior else None
            qb_prior = pd.read_csv(qb_path, index_col="passer_id") if qb_path and qb_path.exists() else None
            alpha = SEASON_ALPHA if args.alpha is None else args.alpha
            ratings, qbs, _, _ = fit_qbs(df, alpha, args.seed, prior, qb_prior, args.prior_weight,
                                         args.half_life, args.hfa, args.clip)
            ratings = summarize(ratings, df, qbs)
            table = qb_table(qbs, df)
            table.to_csv(dest.parent / "qbs.csv")
            log.info("wrote %s (%d passers)", dest.parent / "qbs.csv", len(table))
            print(f"\nQuarterbacks with 200+ dropbacks, {season} (vs_average: epa per dropback against the average "
                  "dropback, on top of the team's pass_off):")
            print(table[table.dropbacks >= 200].drop(columns="rating").to_string())
        ratings.to_csv(dest)
        log.info("wrote %s (%d teams)", dest, len(ratings))
        print(f"\nTop {args.top} by net RAPPA, {season} (epa per play vs the average team):")
        print(centered(ratings).head(args.top).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
