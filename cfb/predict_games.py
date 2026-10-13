"""Turn college RAPPA ratings into game predictions, and score them against the market.

The college counterpart of nfl.predict_games, with the same two stages, both
walked forward so nothing is fitted on a game it predicts:

  stage 1  For each week W, fit RAPPA on the season's plays from every week
           before W, shrunk toward last season's ratings when they exist.
  stage 2  A regression maps the two rating edges onto a final margin and
           picks up home field, in points, as its intercept. Each edge is
           first scaled by the game's expected plays (cfb.stage2).

Stage 1 gives every conference a shared rating (cfb.ratings.conferences), so
a team is its conference's part plus its own, and rates quarterbacks on their
own (cfb.ratings.fit_qbs), their carries counted as dropbacks. Each
side's pass offence in a game is its supporting cast plus the quarterback who
starts it: the one who threw its first pass, or for a game not yet played,
the team's most recent starter. A starter who has not thrown yet this season
counts at his prior - last season's rating carried over, from his old team
if he transferred (cfb.ratings.qb_prior_means) - or at replacement level.
Each team's prior is shifted by what changed over the offseason - its
recruiting, returning production and coaching (cfb.offseason). A team that has
not played yet is rated at its prior, plus its conference's rating so far.

Special teams and penalties - every point in a game outside the plays
stage 1 rates (cfb.outside) - are rated alongside it from the season's games
so far, and cfb.outside.WEIGHT times the gap between the two teams goes onto
the margin; stage 2 is fitted on what is left.

As in the NFL version:

  - Several seasons run together, and stage 2 trains on every game already
    played, earlier seasons included, so it is ready by week 2.
  - The prior is <out>/<season - 1>/rappa.csv and qbs.csv, picked up automatically.
  - Week 1 has no plays to rate, so predictions start in week 2. Bowls and
    playoff games are all one week (cfb.ratings.continuous_week), rated on
    the whole regular season.
  - For a season still in progress, the next unplayed week is predicted too.

The constants were tuned walking forward on game margins over 2023-2024
and checked on 2025-2026, on FBS-vs-FBS games from week 2 on, with 2022 as
the warm-up (see cfb.ratings and cfb.benchmark).

Predicted margin is from the home team's point of view, matching -spread.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cfb import outside, stage2
from cfb.offseason import prior_shift
from cfb.ratings import (FCS, QB_REPLACEMENT, centered, conferences, fit, fit_qbs, load, prior_weights,
                         qb_carries_as_dropbacks, qb_prior_means)
from nfl.predict_games import qb_edges
from rappa.games import FEATURES, add_edges, against_the_spread, score
from rappa.model import UNITS, summarize

log = logging.getLogger(__name__)

# College spreads run wider than the NFL's, and so do disagreements with them.
EDGES = (0.0, 3.0, 6.0, 10.0)

# Ridge penalty of the weekly fits. Walked forward on game margins, 200 beat
# 300 (log-loss -0.004 on 2025-2026) and 500 lost clearly; 150 was level.
ALPHA = 200.0


def starters(plays, games):
    """Each game's home and away starting quarterback: the passer on each side's first pass play.

    A game with none - one not played yet, or a side that never threw -
    gets the team's starter from its latest earlier game.
    """
    first = plays[plays.passer.notna()].groupby(["game_id", "offense"]).passer.first()
    long = pd.concat([
        games[["id", "week", f"{side}_team"]].set_axis(["id", "week", "team"], axis=1).assign(side=side)
        for side in ("home", "away")
    ])
    long["qb"] = first.reindex(pd.MultiIndex.from_arrays([long.id, long.team])).to_numpy()
    long = long.sort_values("week", kind="stable")
    long["qb"] = long.groupby("team").qb.ffill()
    wide = long.pivot_table(index="id", columns="side", values="qb", aggfunc="first")
    return wide.home.reindex(games.id).to_numpy(), wide.away.reindex(games.id).to_numpy()


def unplayed(prior, teams, prior_weight, use_qbs, use_conferences=True, shift=None):
    """What the fit would rate teams with no plays yet: their prior, at the weights it carries.

    Only the prior's teams among `teams` come back; the rest stay unrated.
    With conferences the fit rates every member itself, so this is only for
    independents, whose whole rating is their gap.
    """
    weights = prior_weights(prior_weight, use_qbs, use_conferences)
    if not use_qbs and "qb" in prior:
        # As cfb.ratings.fit reads a quarterback-aware prior.
        prior = centered(prior.assign(pass_off=prior.pass_off + prior.qb))
    rows = prior.reindex([t for t in teams if t in prior.index])
    rows = rows.assign(**{u: rows[u] * w for u, w in weights.items()})[list(UNITS)]
    if shift is not None:
        rows = rows + shift.reindex(rows.index).fillna(0.0)[list(UNITS)]
    return rows


def season_features(season, out, start_week, alpha, prior_weight, use_prior, use_special_teams=True,
                    use_qbs=True, use_conferences=True, use_offseason=True):
    """One row per game, using only ratings that predate it.

    Covers every week that has been played, plus the one after it.
    prior_weight is as cfb.ratings.fit takes it (None for its defaults).
    """
    plays, games = load(season, out)
    members = conferences(games) if use_conferences else None
    games = games.assign(season=season, neutral_site=games.neutral_site.astype(int),
                         fbs=games.home_team.ne(FCS) & games.away_team.ne(FCS))
    units = outside.game_points(season, out) if use_special_teams else None
    units_prior = outside.prior(season, out) if use_special_teams and use_prior else None
    prior_path = out / str(season - 1) / "rappa.csv"
    prior = qb_prior = None
    if use_prior and prior_path.exists():
        prior = pd.read_csv(prior_path, index_col="team")
        log.info("%d: shrinking toward %s", season, prior_path)
        qb_path = prior_path.parent / "qbs.csv"
        if use_qbs and qb_path.exists():
            qb_prior = pd.read_csv(qb_path, index_col="passer_id")
        elif use_qbs:
            log.warning("%d: no %s, every quarterback starts at replacement level", season, qb_path)
    elif use_prior:
        log.warning("%d: no %s, fitting without a prior", season, prior_path)
    shift = prior_shift(season, out) if prior is not None and use_offseason else None
    if use_qbs:
        games["home_qb"], games["away_qb"] = starters(plays, games)
        ids = pd.concat([plays.passer, games.home_qb, games.away_qb])
        start_at = qb_prior_means(qb_prior, ids)

    rows = []
    for week in sorted(games.week.unique()):
        if week < start_week or week > plays.week.max() + 1:
            continue
        before = plays[plays.week < week]
        if before.empty:
            continue
        if use_qbs:
            before = qb_carries_as_dropbacks(before)
        slate = games[games.week == week]
        rush_share = float((before.play_kind == "rush").mean())
        if use_qbs:
            ratings, qbs, _, _ = fit_qbs(before, alpha, prior=prior, qb_prior=start_at, prior_weight=prior_weight,
                                         members=members, shift=shift)
        else:
            ratings, _, _ = fit(before, alpha, prior=prior, prior_weight=prior_weight, members=members, shift=shift)
            qbs = None
        if prior is not None:
            waiting = set(slate.home_team) | set(slate.away_team)
            ratings = pd.concat([ratings, unplayed(prior, sorted(waiting - set(ratings.index)), prior_weight,
                                                   use_qbs, use_conferences, shift)])
        ratings = summarize(ratings, before, qbs)
        if use_qbs:
            if start_at is not None:
                qbs = pd.concat([qbs, start_at[~start_at.index.isin(qbs.index)]])

            # The fit's common offsets (see cfb.ratings.SEASON_ALPHA) move from
            # week to week; taking them out keeps them out of stage 2's intercept.
            level = before.passer[before.play_kind == "pass"].dropna().map(qbs).mean()

            def qb_rating(ids):
                return ids.map(qbs).fillna(QB_REPLACEMENT).to_numpy() - level

            slate = qb_edges(slate, centered(ratings), rush_share, qb_rating)
        else:
            slate = add_edges(slate, ratings)
        if units is not None:
            # Points on the margin, cfb.outside.WEIGHT times the rating gap; a team never rated counts as average.
            value = outside.fit(units[units.week < week], units_prior).value
            slate["st_edge"] = (slate.home_team.map(value) - slate.away_team.map(value)).fillna(0.0)
        rows.append(slate)
        log.info("%d week %2d: %3d games rated on %6d prior plays", season, week, len(slate), len(before))
    return pd.concat(rows, ignore_index=True).dropna(subset=FEATURES)


def walk_forward_predict(rows, min_train):
    """Fit the points conversion on every game played before this week, predict the week.

    rows need cfb.stage2.with_scale's plays_scale, which the edges are scaled
    by (cfb.stage2.design). Special teams and penalties, when rows has them,
    are already points (cfb.outside's value) and go onto the margin as they
    are, so the conversion is fitted on what they leave.
    """
    st = rows.st_edge if "st_edge" in rows else 0.0
    rows = rows.assign(slot=rows.season * 100 + rows.week, st_points=st)
    played = rows[rows.margin.notna()]
    preds = []
    for slot in sorted(rows.slot.unique()):
        train = played[played.slot < slot]
        if len(train) < min_train:
            continue
        test = rows[rows.slot == slot].copy()
        model = LinearRegression().fit(stage2.design(train), train.margin - train.st_points)
        test["predicted"] = model.predict(stage2.design(test)) + test.st_points
        test["hfa"] = model.intercept_
        test["n_train"] = len(train)
        preds.append(test)
    if not preds:
        raise SystemExit(f"never had {min_train} games to train on - lower --min-train or add a season")
    return pd.concat(preds, ignore_index=True).drop(columns=["slot", "st_points"])


def predict(seasons, out, start_week=2, alpha=ALPHA, prior_weight=None, use_prior=True, use_special_teams=True,
            use_qbs=True, min_train=200, use_conferences=True, use_offseason=True):
    """Walk-forward predictions for every game of these seasons that stage 2 has enough games for.

    The priors on disk must have been rated the same way: with conferences and
    offseason shifts (the default), or with cfb.ratings --no-conferences or
    --no-offseason to match use_conferences=False or use_offseason=False.
    """
    rows = pd.concat([
        season_features(s, out, start_week, alpha, prior_weight, use_prior, use_special_teams, use_qbs,
                        use_conferences, use_offseason)
        for s in sorted(seasons)
    ], ignore_index=True)
    return walk_forward_predict(stage2.with_scale(rows, sorted(seasons), out), min_train)


def report(preds):
    """Model and market accuracy by season and overall, then against the spread."""
    played = preds[preds.margin.notna()]
    tables = []
    for season, group in played.groupby("season"):
        table, _ = score(group)
        tables.append(table.assign(season=season))
    pooled, graded = score(played)
    tables.append(pooled.assign(season="all"))
    table = pd.concat(tables).rename_axis("model").reset_index().set_index(["season", "model"])
    table["games"] = table.games.astype(int)
    weeks = played.groupby("season").week.agg(["min", "max"])
    span = ", ".join(f"{s} wk {int(w['min'])}-{int(w['max'])}" for s, w in weeks.iterrows())
    print(f"\n=== walk-forward game predictions, every game with a line ({span}) ===")
    print(table.to_string(float_format="%.3f"))
    print(f"\nlearned home-field advantage: {preds.hfa.iloc[-1]:+.2f} points")

    print("\n=== against the closing spread (break-even at -110 is 52.4%) ===")
    ats = [against_the_spread(graded, e) for e in EDGES]
    print(pd.DataFrame([a for a in ats if a]).to_string(index=False, float_format="%.3f"))
    print(f"p_value is one-sided vs a coin flip; {len(EDGES)} edges were tried, so treat it loosely.")
    return graded


def line(home, away, spread):
    """A home-view spread as the favourite and the points, e.g. 'Ohio State -6.5'."""
    if pd.isna(spread):
        return ""
    if round(spread, 1) == 0:
        return "pick'em"
    return f"{home if spread < 0 else away} -{abs(spread):.1f}"


def slate_table(slate):
    """A week's predictions for reading: kickoff, matchup, both lines, and which side the model leans to."""
    slate = slate.sort_values("start_date")
    model_spread = -slate.predicted
    # Positive: the model likes the home side more than the market does.
    diff = slate.spread - model_spread
    kickoff = pd.to_datetime(slate.start_date, utc=True).dt.tz_convert("America/New_York")
    return pd.DataFrame({
        "kickoff (ET)": kickoff.dt.strftime("%a %I:%M%p"),
        "matchup": slate.away_team + np.where(slate.neutral_site == 1, " vs ", " @ ") + slate.home_team,
        "model": [line(h, a, s) for h, a, s in zip(slate.home_team, slate.away_team, model_spread)],
        "market": [line(h, a, s) for h, a, s in zip(slate.home_team, slate.away_team, slate.spread)],
        "lean": np.where(diff.isna(), "", np.where(diff > 0, slate.home_team, slate.away_team)),
        "by": diff.abs().round(1),
    })


def upcoming(preds):
    """Predictions for games not played yet, next to the current line."""
    # The latest season's last predicted week; unplayed games before it were
    # cancelled or postponed, not upcoming.
    latest = preds[preds.season == preds.season.max()]
    week = latest.week.max()
    ahead = latest[(latest.week == week) & latest.margin.isna()].copy()
    if ahead.empty:
        return ahead
    print(f"\n=== upcoming: {int(ahead.season.iloc[0])} week {int(week)} ({len(ahead)} games) ===")
    print(slate_table(ahead).to_string(index=False, na_rep=""))
    return ahead


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season(s) to predict, e.g. 2023 2024 2025 2026")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--start-week", type=int, default=2, help="first week to rate (default: 2)")
    p.add_argument("--min-train", type=int, default=200,
                   help="games stage 2 needs before it predicts (default: 200)")
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"ridge penalty (default: {ALPHA:g})")
    p.add_argument("--prior-weight", type=float,
                   help="carry this much of last season for every unit (default: cfb.ratings's, by side)")
    p.add_argument("--no-prior", action="store_true", help="fit every season from scratch")
    p.add_argument("--no-special-teams", action="store_true",
                   help="leave special teams and penalties (cfb.outside) out of the margin")
    p.add_argument("--no-qbs", action="store_true", help="leave quarterbacks in their teams' pass offence")
    p.add_argument("--no-conferences", action="store_true",
                   help="rate teams without conference ratings (needs priors rated with cfb.ratings "
                        "--no-conferences)")
    p.add_argument("--no-offseason", action="store_true",
                   help="leave each prior unshifted by the offseason (needs priors rated with cfb.ratings "
                        "--no-offseason)")
    p.add_argument("--save", type=Path, help="write per-game predictions, upcoming included, here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Four seasons are ~70 fits, and fit logs three lines for each.
    for name in ("rappa", "cfb.ratings", "cfb.outside"):
        logging.getLogger(name).setLevel(logging.WARNING)

    preds = predict(args.seasons, args.out, args.start_week, args.alpha, args.prior_weight, not args.no_prior,
                    not args.no_special_teams, not args.no_qbs, args.min_train, not args.no_conferences,
                    not args.no_offseason)
    graded = report(preds)
    ahead = upcoming(preds)

    if args.save:
        cols = ["season", "week", "id", "home_team", "away_team", "margin", "predicted", "spread"]
        pd.concat([graded, ahead])[cols].to_csv(args.save, index=False)
        log.info("wrote %s (%d games, %d upcoming)", args.save, len(graded), len(ahead))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
