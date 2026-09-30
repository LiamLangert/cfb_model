"""Turn NFL RAPPA ratings into game predictions, and score them against the market.

The NFL counterpart of cfb.predict_games, with the same two stages, both
walked forward so nothing is fitted on a game it predicts:

  stage 1  For each week W, fit RAPPA on the season's plays from every week
           before W, shrunk toward last season's ratings when they exist.
  stage 2  A regression maps the two rating edges onto a final margin and
           picks up home field, in points, as its intercept.

Stage 1 rates quarterbacks on their own (nfl.ratings.fit_qbs), and each
side's pass offence in a game is its supporting cast plus the quarterback who
starts it: the listed starter, or for a game not yet played without one, the
team's most recent starter. A starter who has not thrown yet this season
counts at his prior - last season's rating carried over, or replacement level.

Special teams are rated alongside stage 1 from the season's punts and
kickoffs (nfl.special_teams), and the gap between the two teams goes onto
the margin at face value; stage 2 is fitted on what is left.

Where it differs from the college version:

  - Several seasons run together, and stage 2 trains on every game already
    played, earlier seasons included. One season is 272 games, 16 a week -
    too few to learn the conversion from its own games before midseason.
  - The prior is <out>/<season - 1>/rappa.csv and qbs.csv, picked up automatically.
  - Week 1 has no plays to rate, so predictions start in week 2.
  - For a season still in progress, the next unplayed week is predicted too.

Predicted margin is from the home team's point of view, matching spread_line.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from nfl import special_teams
from nfl.ratings import GARBAGE, QB_REPLACEMENT, centered, fit, fit_qbs, load, qb_prior_means
from rappa.games import FEATURES, add_edges, against_the_spread, score
from rappa.model import garbage_time, summarize

log = logging.getLogger(__name__)

# Closing-line disagreements are smaller in the NFL than in college.
EDGES = (0.0, 1.5, 3.0, 5.0)

# Ridge penalty of the weekly fits. Cross-validation on holdout plays picks 300
# in almost every NFL fit, but walked forward on game margins over 2015-2022,
# 500 predicted better (see nfl.ratings for the other constants tuned there).
ALPHA = 500.0

# nflverse play-by-play gives every season today's abbreviations, but its
# schedules keep the old ones (STL to 2015, SD to 2016, OAK to 2019), so
# without this a relocated team's games find no rating and drop out.
RELOCATED = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def load_games(season, out):
    path = out / str(season) / "games.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run nfl/pull.py {season}` first")
    games = pd.read_csv(path)
    return games.assign(
        home_team=games.home_team.replace(RELOCATED),
        away_team=games.away_team.replace(RELOCATED),
        # nflverse result is home minus away, and empty until the game is played.
        margin=games.result,
        neutral_site=games.location.eq("Neutral").astype(int),
        # CFBD's sign convention (negative = home favoured), so that
        # rappa.games can grade against it unchanged.
        spread=-games.spread_line,
    )


def starters(games):
    """Each game's home and away starting quarterback.

    nflverse lists a starter once a game is played, and for only some games
    before. A game without one gets the team's starter from its latest
    earlier game that has one.
    """
    long = pd.concat([
        games[["game_id", "week", f"{side}_team", f"{side}_qb_id"]].set_axis(["game_id", "week", "team", "qb"], axis=1)
        .assign(side=side) for side in ("home", "away")
    ]).sort_values("week", kind="stable")
    long["qb"] = long.groupby("team").qb.ffill()
    wide = long.pivot(index="game_id", columns="side", values="qb")
    return wide.home.reindex(games.game_id).to_numpy(), wide.away.reindex(games.game_id).to_numpy()


def qb_edges(slate, ratings, rush_share, qb_rating):
    """rappa.games.add_edges, with each side's pass offence its cast plus its starter."""
    def offence(team, qb):
        r = ratings.reindex(team)
        return (rush_share * r.rush_off + (1 - rush_share) * (r.pass_off + qb)).to_numpy()
    home_off = offence(slate.home_team, qb_rating(slate.home_qb))
    away_off = offence(slate.away_team, qb_rating(slate.away_qb))
    return slate.assign(
        home_edge=home_off - slate.away_team.map(ratings["def"]).to_numpy(),
        away_edge=away_off - slate.home_team.map(ratings["def"]).to_numpy(),
        neutral_site=slate.neutral_site.astype(int),
    )


def season_features(season, out, start_week, alpha, prior_weight, use_prior, use_special_teams=True,
                    use_qbs=True):
    """One row per game, using only ratings that predate it.

    Covers every week that has been played, plus the one after it. prior_weight
    is as nfl.ratings.fit takes it (None for its defaults).
    """
    plays, games = load(season, out), load_games(season, out)
    plays = plays[~garbage_time(plays, GARBAGE)]
    kicks = special_teams.load(season, out) if use_special_teams else None
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
    if use_qbs:
        games["home_qb"], games["away_qb"] = starters(games)
        start_at = qb_prior_means(qb_prior)

    rows = []
    for week in sorted(games.week.unique()):
        if week < start_week or week > plays.week.max() + 1:
            continue
        before = plays[plays.week < week]
        if before.empty:
            continue
        slate = games[games.week == week]
        if use_qbs:
            ratings, qbs, _, _ = fit_qbs(before, alpha, prior=prior, qb_prior=qb_prior, prior_weight=prior_weight)
            ratings = summarize(ratings, before, qbs)
            if start_at is not None:
                qbs = pd.concat([qbs, start_at[~start_at.index.isin(qbs.index)]])

            # The fit's common offsets (see nfl.ratings.SEASON_ALPHA) move from
            # week to week; taking them out keeps them out of stage 2's intercept.
            level = before.passer[before.play_kind == "pass"].dropna().map(qbs).mean()

            def qb_rating(ids):
                return ids.map(qbs).fillna(QB_REPLACEMENT).to_numpy() - level

            slate = qb_edges(slate, centered(ratings), float((before.play_kind == "rush").mean()), qb_rating)
        else:
            ratings, _, _ = fit(before, alpha, prior=prior, prior_weight=prior_weight)
            ratings = summarize(ratings, before)
            slate = add_edges(slate, ratings)
        if kicks is not None:
            # A team with no punts or kickoffs yet counts as average.
            value = special_teams.fit(kicks[kicks.week < week]).value
            slate["st_edge"] = (slate.home_team.map(value) - slate.away_team.map(value)).fillna(0.0)
        rows.append(slate)
        log.info("%d week %2d: %2d games rated on %5d prior plays", season, week, len(slate), len(before))
    return pd.concat(rows, ignore_index=True).dropna(subset=FEATURES)


def walk_forward_predict(rows, min_train):
    """Fit the points conversion on every game played before this week, predict the week.

    Special teams, when rows has them, go onto the margin at face value, so the
    conversion is fitted on what they leave.
    """
    st = rows.st_edge * special_teams.WEIGHT if "st_edge" in rows else 0.0
    rows = rows.assign(slot=rows.season * 100 + rows.week, st_points=st)
    played = rows[rows.margin.notna()]
    preds = []
    for slot in sorted(rows.slot.unique()):
        train = played[played.slot < slot]
        if len(train) < min_train:
            continue
        test = rows[rows.slot == slot].copy()
        model = LinearRegression().fit(train[FEATURES], train.margin - train.st_points)
        test["predicted"] = model.predict(test[FEATURES]) + test.st_points
        test["hfa"] = model.intercept_
        test["n_train"] = len(train)
        preds.append(test)
    if not preds:
        raise SystemExit(f"never had {min_train} games to train on - lower --min-train or add a season")
    return pd.concat(preds, ignore_index=True).drop(columns=["slot", "st_points"])


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
    print(f"\n=== walk-forward game predictions ({span}) ===")
    print(table.to_string(float_format="%.3f"))
    print(f"\nlearned home-field advantage: {preds.hfa.iloc[-1]:+.2f} points")

    print("\n=== against the closing spread (break-even at -110 is 52.4%) ===")
    ats = [against_the_spread(graded, e) for e in EDGES]
    print(pd.DataFrame([a for a in ats if a]).to_string(index=False, float_format="%.3f"))
    print(f"p_value is one-sided vs a coin flip; {len(EDGES)} edges were tried, so treat it loosely.")
    return graded


def upcoming(preds):
    """Predictions for games not played yet, next to the current line."""
    ahead = preds[preds.margin.isna()].copy()
    if ahead.empty:
        return ahead
    ahead["market"] = ahead.spread_line
    ahead["edge"] = ahead.predicted - ahead.market
    # The side the model likes against the spread, where it has one.
    ahead["lean"] = np.where(ahead.edge > 0, ahead.home_team, ahead.away_team)
    ahead.loc[ahead.market.isna(), "lean"] = ""
    cols = ["week", "gameday", "away_team", "home_team", "predicted", "market", "edge", "lean"]
    print(f"\n=== upcoming: {int(ahead.season.iloc[0])} week {int(ahead.week.min())}"
          " (predicted and market are the home margin) ===")
    print(ahead[cols].to_string(index=False, float_format="%+.1f"))
    return ahead


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season(s) to predict, e.g. 2023 2024 2025 2026")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    p.add_argument("--start-week", type=int, default=2, help="first week to rate (default: 2)")
    p.add_argument("--min-train", type=int, default=100,
                   help="games stage 2 needs before it predicts (default: 100)")
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"ridge penalty (default: {ALPHA:g})")
    p.add_argument("--prior-weight", type=float,
                   help="carry this much of last season for every unit (default: nfl.ratings's, by side)")
    p.add_argument("--no-prior", action="store_true", help="fit every season from scratch")
    p.add_argument("--no-special-teams", action="store_true", help="leave special teams out of the margin")
    p.add_argument("--no-qbs", action="store_true", help="leave quarterbacks in their teams' pass offence")
    p.add_argument("--save", type=Path, help="write per-game predictions, upcoming included, here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Four seasons are ~80 fits, and fit logs three lines for each.
    logging.getLogger("rappa").setLevel(logging.WARNING)

    rows = pd.concat([
        season_features(s, args.out, args.start_week, args.alpha, args.prior_weight, not args.no_prior,
                        not args.no_special_teams, not args.no_qbs)
        for s in sorted(args.seasons)
    ], ignore_index=True)
    preds = walk_forward_predict(rows, args.min_train)
    graded = report(preds)
    ahead = upcoming(preds)

    if args.save:
        cols = ["season", "week", "game_id", "home_team", "away_team", "margin", "predicted", "spread_line"]
        pd.concat([graded, ahead])[cols].to_csv(args.save, index=False)
        log.info("wrote %s (%d games, %d upcoming)", args.save, len(graded), len(ahead))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
