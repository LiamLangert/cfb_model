"""Predict one NFL week, and take each prediction apart into what went into it.

The predictions are nfl.predict_games', for the target week only: ratings
from the season's plays before it (quarterbacks, special teams and rest
included), shrunk toward last season's, and the points conversion fitted on
every game played before the week since --since. The default, 2015, trains
it as research.nfl_benchmark grades it. The week defaults to the season's next one: the
first with a game not played yet. A week already played shows what the model
would have said.

Each predicted home margin is a sum of parts, all in points. Stage 2 is
linear, so each side's edge splits the way nfl.predict_games.qb_edges builds
it:

    home field     stage 2's intercept, plus its neutral-site term at a neutral site
    rush offence   rush_off at the league's share of rushes, times stage 2's
                   coefficient on that side's edge
    pass offence   pass_off at the share of dropbacks: the passing game less its quarterback
    quarterback    the starter's rating, on his dropbacks and designed runs
    rush defence   rush_def and pass_def at the same shares, times the
    pass defence   coefficient on the other side's edge
    special teams  special teams and penalties: nfl.outside's gap, at rappa.outside.WEIGHT
    rest           the rest gap, at nfl.predict_games.REST_WEIGHT points a day

Every team part is that team's unit against an average one, in points for
the team, so home parts less away parts is the predicted margin. Nothing in
RAPPA pairs a unit with the one it faces - a unit is worth the same against
any opponent - so the parts read as matchups: when a side has the ball, its
run game against the other's rush defence and its passing game, quarterback
included, against their pass defence, each matchup the offence's worth less
the defence's, with nothing extra for a mismatch.

Win probability is rappa.games.win_probability, as research.nfl_benchmark
grades it: Phi(margin / sigma), where sigma is the sd of RAPPA's misses on
every game it predicted before this week.

The line is the predicted margin as the market sets one: the half point the
home side is as likely to beat as to fall short of, from nfl.key_numbers'
margins around the prediction, which bunch on 3 and 7 as finals do. That is
the margins' median, not their average, the predicted margin: for a
favourite of a touchdown or more it is a point or two less. The same margins
give each side's chance to cover the market's spread, and of a push.

The projected score is a projected total split by the predicted margin, so
the two scores always differ by exactly the margin. The total comes from one
more regression like stage 2, on the same edges plus each side's pace and
points in its games so far (nfl.pace). RAPPA rates teams by the play, so
without those it knew nothing of how many plays or points a game would have.
It still knows nothing of weather, and the total is the weaker half. Walked
forward over 2015-2026, the projected total missed the actual one by 10.67
points a game (10.74 on the edges alone), against 10.42 for the closing total
and 11.01 for the league average to date. The score shown with the line is nfl.key_numbers'
most likely final at it, within 3 points of the projected total, so the two
read alike; it is no better a guess at the score.

Injuries count only through the starting quarterback. nflverse lists one for
most games a few days ahead; a game without one gets the team's latest
starter. A starter who has not played this season is rated at last season's
rating carried over (nfl.ratings.QB_PRIOR_WEIGHT), or at replacement level.

On 2015-2026 research.nfl_benchmark has these margins 0.28 points a game worse than
the closing line in MAE, and research.nfl_backtest_week has never shown an edge
against the spread, so read a gap to the market as disagreement, not a bet.

Writes <out>/<season>/predictions_wk<NN>.csv, one row per game, which
site/build.py shows on the ratings page.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from nfl import key_numbers, offseason, outside, pace
from nfl.predict_games import ALPHA, REST_CLIP, REST_WEIGHT, load_games, season_features, walk_forward_predict
from nfl.ratings import QB_REPLACEMENT, centered, fit_qbs, in_fit, load, qb_prior_means, with_qb_runs
from rappa.games import FEATURES, market_wp, win_probability
from rappa.model import summarize

log = logging.getLogger(__name__)

FIRST_SEASON = 2015

# The parts of a predicted margin, in the order they are printed and shown.
PARTS = ["qb", "rush_off", "pass_off", "rush_def", "pass_def", "special_teams", "rest", "home_field"]

# Every file the predictions read, by season: the season's own, and the one
# before's ratings as its prior, its plays and games for nfl.pace's, and its
# snaps for nfl.offseason's.
SEASON_INPUTS = ("plays.csv", "games.csv", "roster_week1.csv")
PRIOR_INPUTS = ("rappa.csv", "qbs.csv", "plays.csv", "games.csv", "snaps.csv")


def upcoming_week(season, out):
    """The season's first week with a game not played yet, or None."""
    games = load_games(season, out)
    ahead = games[games.margin.isna()]
    return int(ahead.week.min()) if len(ahead) else None


def sources(season, out, since=FIRST_SEASON):
    """The data and code the week's predictions depend on, for telling a stale file from a fresh one."""
    files = [out / str(s) / f for s in range(since, season + 1) for f in SEASON_INPUTS]
    files += [out / str(s) / f for s in range(since - 1, season) for f in PRIOR_INPUTS]
    code = Path(__file__).resolve().parent
    files += [code / f for f in
              ("predict_week.py", "predict_games.py", "ratings.py", "special_teams.py", "outside.py",
               "key_numbers.py", "pace.py", "offseason.py")]
    files += [code.parent / "rappa" / f for f in ("model.py", "games.py", "special_teams.py", "expected_points.py",
                                                   "outside.py")]
    return [p for p in files if p.exists()]


def week_ratings(season, week, out, alpha, prior_weight, plays=None, units=None):
    """The team, quarterback and special-teams ratings nfl.predict_games rates the week on.

    season_features fits these and keeps only the two edges; this repeats its
    fit for the one week so the edges can be taken apart. plays (in_fit's)
    and units (nfl.outside.game_points') are the season's, read here unless given.
    Returns the
    centred team ratings, each quarterback's rating against the average
    dropback, his dropbacks so far, the ids rated from last season, the
    rush share, the designed-run share and each team's special teams and
    penalties (nfl.outside's value, points a game).
    """
    plays = in_fit(load(season, out)) if plays is None else plays
    before = with_qb_runs(plays[plays.week < week])
    prior = qb_prior = None
    prior_path = out / str(season - 1) / "rappa.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path, index_col="team")
        if (prior_path.parent / "qbs.csv").exists():
            qb_prior = pd.read_csv(prior_path.parent / "qbs.csv", index_col="passer_id")
    shift = offseason.prior_shift(season, out) if prior is not None else None
    ratings, qbs, _, _ = fit_qbs(before, alpha, prior=prior, qb_prior=qb_prior, prior_weight=prior_weight,
                                 shift=offseason.faded(shift, week))
    ratings = summarize(ratings, before, qbs)
    start_at = qb_prior_means(qb_prior)
    carried = pd.Index([])
    if start_at is not None:
        carried = start_at.index[~start_at.index.isin(qbs.index)]
        qbs = pd.concat([qbs, start_at[carried]])
    level = before.passer[before.play_kind == "pass"].dropna().map(qbs).mean()
    dropbacks = before.passer[before.play_kind == "pass"].value_counts()
    units = outside.game_points(season, out) if units is None else units
    st = outside.fit(units[units.week < week], outside.prior(season, out)).value
    return (centered(ratings), qbs - level, QB_REPLACEMENT - level, dropbacks, carried,
            float((before.play_kind == "rush").mean()), float(before.qb_runner.notna().mean()), st)


def qb_names(season, out):
    """Full quarterback names by id, from the schedules' listed starters (nflverse's passer names are abbreviated)."""
    frames = []
    for s in (season - 1, season):
        path = out / str(s) / "games.csv"
        if path.exists():
            g = pd.read_csv(path)
            frames += [g[[f"{side}_qb_id", f"{side}_qb_name"]].set_axis(["id", "name"], axis=1)
                       for side in ("home", "away")]
    names = pd.concat(frames).dropna().drop_duplicates("id", keep="last")
    return names.set_index("id")["name"]


def season_starts(games, week):
    """Starts by team and quarterback in the season's games played before week, and each team's latest starter."""
    played = games[games.margin.notna() & (games.week < week)]
    long = pd.concat([played[["week", f"{side}_team", f"{side}_qb_id"]].set_axis(["week", "team", "qb"], axis=1)
                      for side in ("home", "away")]).dropna().sort_values("week", kind="stable")
    return long.groupby(["team", "qb"]).size(), long.groupby("team").qb.last()


def take_apart(slate, model, ratings, qb, rush_share, run_share, st):
    """Each side's parts of the predicted margin (<side>_<part>_pts), in points for that side."""
    b_home, b_away, b_neutral = model.coef_
    pass_share = 1 - rush_share
    parts = {}
    for side, own_b, their_b in (("home", b_home, b_away), ("away", -b_away, -b_home)):
        r = ratings.reindex(slate[f"{side}_team"])
        parts[f"{side}_rush_off_pts"] = own_b * rush_share * r.rush_off.to_numpy()
        parts[f"{side}_pass_off_pts"] = own_b * pass_share * r.pass_off.to_numpy()
        parts[f"{side}_qb_pts"] = own_b * (pass_share + run_share) * qb[side]
        # A defence counts on the other side's coefficient: it lowers their edge.
        defence = -their_b * r["def"].to_numpy()
        parts[f"{side}_rush_def_pts"] = -their_b * rush_share * r.rush_def.to_numpy()
        parts[f"{side}_pass_def_pts"] = defence - parts[f"{side}_rush_def_pts"]
    home_st, away_st = (slate[f"{side}_team"].map(st).to_numpy() for side in ("home", "away"))
    # As nfl.predict_games: values are points already, and a team never rated makes the gap zero.
    missing = np.isnan(home_st) | np.isnan(away_st)
    parts["home_special_teams_pts"] = np.where(missing, 0.0, home_st)
    parts["away_special_teams_pts"] = np.where(missing, 0.0, away_st)
    rest = REST_WEIGHT * slate.rest_edge.to_numpy()
    parts["home_rest_pts"], parts["away_rest_pts"] = np.maximum(rest, 0.0), np.maximum(-rest, 0.0)
    parts["home_home_field_pts"] = model.intercept_ + b_neutral * slate.neutral_site.to_numpy()
    parts["away_home_field_pts"] = np.zeros(len(slate))
    return pd.DataFrame(parts, index=slate.index)


def walk(season, out=Path("data/nfl"), since=FIRST_SEASON, alpha=ALPHA, prior_weight=None, min_train=100):
    """Every game's walk-forward features and prediction from since through season, the slow part.

    Each game's are fitted only on what came before it, so one walk would
    serve any week up to season. Returns the rows and the predictions.
    """
    seasons = [s for s in range(since, season + 1) if (out / str(s) / "plays.csv").exists()]
    rows = pd.concat([season_features(s, out, 2, alpha, prior_weight, True) for s in seasons], ignore_index=True)
    rows = rows.join(pace.features(seasons, out), on="game_id")
    return rows, win_probability(walk_forward_predict(rows, min_train), min_train)


def predict(season, week=None, out=Path("data/nfl"), since=FIRST_SEASON, alpha=ALPHA, prior_weight=None,
            min_train=100):
    """One row per game of the week: prediction, win probability, projected score and parts."""
    week = upcoming_week(season, out) if week is None else week
    if week is None:
        raise SystemExit(f"{season} has no games left to play - give a week to see one already played")
    rows, preds = walk(season, out, since, alpha, prior_weight, min_train)
    return week_slate(rows, preds, season, week, out, alpha, prior_weight)


def week_slate(rows, preds, season, week, out, alpha=ALPHA, prior_weight=None, plays=None, units=None):
    """One week of predict, from walk's rows and predictions; plays and units as week_ratings takes them."""
    slate = preds[(preds.season == season) & (preds.week == week)].copy()
    if slate.empty:
        raise SystemExit(f"no predictions for {season} week {week}: it needs plays from the week before "
                         f"(`uv run nfl/pull.py {season}`), and week 1 has none")

    # Stage 2 again, for its coefficients: fitted on the same games as walk_forward_predict.
    face = rows.st_edge + REST_WEIGHT * rows.rest_edge
    train = rows[rows.margin.notna() & (rows.season * 100 + rows.week < season * 100 + week)]
    face_train = face[train.index]
    model = LinearRegression().fit(train[FEATURES], train.margin - face_train)
    # The total, on the edges and both sides' pace and scoring, split by the predicted margin.
    total_features = [*FEATURES, *pace.FEATURES]
    total = LinearRegression().fit(train[total_features], train.home_score + train.away_score)
    slate_total = total.predict(slate[total_features])
    slate["home_points"] = (slate_total + slate.predicted) / 2
    slate["away_points"] = (slate_total - slate.predicted) / 2

    # The line, the chances against the market's, and the score shown at the line, fitted on the same games.
    before = preds[preds.margin.notna() & (preds.season * 100 + preds.week < season * 100 + week)]
    probs = key_numbers.fit_margins(before.predicted, before.margin).probs(slate.predicted)
    slate["rappa_line"] = key_numbers.lines(probs)
    slate["rappa_home_cover"], slate["rappa_push"] = key_numbers.at_spread(probs, slate.spread_line)
    train_total = total.predict(train[total_features])
    train_margin = model.predict(train[FEATURES]) + face_train
    scores = key_numbers.fit_scores((train_total + train_margin) / 2, (train_total - train_margin) / 2,
                                    train.home_score, train.away_score)
    slate["home_points_at_line"], slate["away_points_at_line"] = key_numbers.score_at_line(
        scores, slate.home_points, slate.away_points, slate.rappa_line)

    ratings, qbs, replacement, dropbacks, carried, rush_share, run_share, st = week_ratings(
        season, week, out, alpha, prior_weight, plays, units)
    qb = {side: slate[f"{side}_qb"].map(qbs).fillna(replacement).to_numpy() for side in ("home", "away")}
    parts = take_apart(slate, model, ratings, qb, rush_share, run_share, st)
    total = sum(parts[f"home_{p}_pts"] - parts[f"away_{p}_pts"] for p in PARTS)
    if not np.allclose(total, slate.predicted, atol=1e-6):
        raise RuntimeError("the parts do not add up to nfl.predict_games' margin - has its fit changed?")
    slate = slate.join(parts)

    games = load_games(season, out)
    listed = games.set_index("game_id")
    names = qb_names(season, out)
    starts, latest = season_starts(games, week)
    for side in ("home", "away"):
        ids, teams = slate[f"{side}_qb"], slate[f"{side}_team"]
        slate[f"{side}_qb_name"] = ids.map(names)
        slate[f"{side}_qb_rating"] = qb[side]
        slate[f"{side}_qb_dropbacks"] = ids.map(dropbacks).fillna(0).astype(int)
        # Where his rating comes from: this season's dropbacks, last season's, or neither.
        slate[f"{side}_qb_rated_on"] = np.select(
            [slate[f"{side}_qb_dropbacks"] > 0, ids.isin(carried)], ["season", "prior"], "replacement")
        # Listed: nflverse names him for this game; otherwise he is the team's latest starter.
        slate[f"{side}_qb_listed"] = listed[f"{side}_qb_id"].reindex(slate.game_id).notna().to_numpy()
        # His starts so far, whom he replaces if the team's last game had someone else, and who else started.
        slate[f"{side}_qb_starts"] = [int(starts.get((t, q), 0)) for t, q in zip(teams, ids)]
        last = teams.map(latest)
        slate[f"{side}_qb_replaces"] = last.map(names).where(last.notna() & ids.notna() & ids.ne(last))
        slate[f"{side}_qb_others"] = [
            ", ".join(f"{names.get(q, q)} {n}" for (tm, q), n in starts.items() if tm == t and q != i) or None
            for t, i in zip(teams, ids)]
        for unit in ("rush_off", "pass_off", "rush_def", "pass_def"):
            slate[f"{side}_{unit}_rating"] = teams.map(ratings[unit])
        slate[f"{side}_st_value"] = teams.map(st)
    slate["rest_days"] = (slate.home_rest - slate.away_rest).clip(-REST_CLIP, REST_CLIP)
    slate["market_home_wp"] = market_wp(slate)
    slate["rush_share"], slate["run_share"] = rush_share, run_share
    slate["coef_home"], slate["coef_away"], slate["coef_neutral"] = model.coef_
    return slate.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)


COLUMNS = [
    "season", "week", "game_type", "game_id", "gameday", "weekday", "gametime", "away_team", "home_team",
    "neutral_site", "location", "stadium", "roof", "div_game", "away_rest", "home_rest", "rest_days",
    "predicted", "sigma", "rappa_wp", "home_points", "away_points", "hfa", "n_train",
    "rappa_line", "rappa_home_cover", "rappa_push", "home_points_at_line", "away_points_at_line",
    "home_pace", "away_pace", "home_scoring", "away_scoring",
    "coef_home", "coef_away", "coef_neutral", "rush_share", "run_share",
    *(f"{side}_{p}_pts" for side in ("home", "away") for p in PARTS),
    *(f"{side}_qb{c}" for side in ("home", "away") for c in
      ("", "_name", "_rating", "_dropbacks", "_rated_on", "_listed", "_starts", "_replaces", "_others")),
    *(f"{side}_{u}_rating" for side in ("home", "away") for u in ("rush_off", "pass_off", "rush_def", "pass_def")),
    "home_st_value", "away_st_value",
    "spread_line", "total_line", "home_moneyline", "away_moneyline", "market_home_wp",
    "home_score", "away_score",
]


def report(slate):
    """The week at a glance, then each game's parts by matchup, in points to the home side."""
    season, week = int(slate.season.iloc[0]), int(slate.week.iloc[0])
    table = pd.DataFrame({
        "game": slate.away_team + " @ " + slate.home_team,
        "kickoff": slate.weekday.str[:3] + " " + slate.gametime,
        "away qb": slate.away_qb_name.fillna(slate.away_qb),
        "home qb": slate.home_qb_name.fillna(slate.home_qb),
        "score": [f"{a}-{h}" for a, h in zip(slate.away_points_at_line, slate.home_points_at_line)],
        "margin": slate.predicted,
        "line": slate.rappa_line,
        "home wp": slate.rappa_wp,
        "market": slate.spread_line,
        "covers": slate.rappa_home_cover,
        "mkt wp": slate.market_home_wp,
    })
    print(f"\n=== {season} week {week} (margin, line and market are the home side's, and covers its chance "
          "against the market; score is away-home, at the line) ===")
    print(table.to_string(index=False, float_format="%.2f"))
    # Each side with the ball: its offence's worth less the other's defence's.
    net = {"game": table.game}
    for side, other, sign in (("away", "home", -1), ("home", "away", 1)):
        net[f"{side} pass"] = sign * (slate[f"{side}_qb_pts"] + slate[f"{side}_pass_off_pts"]
                                      - slate[f"{other}_pass_def_pts"])
        net[f"{side} run"] = sign * (slate[f"{side}_rush_off_pts"] - slate[f"{other}_rush_def_pts"])
    for p in ("special_teams", "rest", "home_field"):
        net[p] = slate[f"home_{p}_pts"] - slate[f"away_{p}_pts"]
    net = pd.DataFrame(net).assign(margin=slate.predicted)
    print("\nParts of each margin, in points to the home side; away and home pass (quarterback included) and run "
          "are that side's offence against the other's defence:")
    print(net.to_string(index=False, float_format="%+.1f"))
    print(f"\nstage 2 trained on {int(slate.n_train.iloc[0])} games; home field {slate.hfa.iloc[0]:+.2f}, "
          f"sigma {slate.sigma.iloc[0]:.2f} points")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("season", type=int)
    p.add_argument("week", type=int, nargs="?", help="week to predict (default: the next one with a game to play)")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    p.add_argument("--since", type=int, default=FIRST_SEASON,
                   help=f"first season the points conversion trains on (default: {FIRST_SEASON})")
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"ridge penalty (default: {ALPHA:g})")
    p.add_argument("--prior-weight", type=float,
                   help="carry this much of last season for every unit (default: nfl.ratings's, by side)")
    p.add_argument("--min-train", type=int, default=100,
                   help="games the conversion, and sigma, need before it predicts (default: 100)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("rappa", "nfl.predict_games"):
        logging.getLogger(name).setLevel(logging.WARNING)
    slate = predict(args.season, args.week, args.out, args.since, args.alpha, args.prior_weight, args.min_train)
    report(slate)
    dest = args.out / str(args.season) / f"predictions_wk{int(slate.week.iloc[0]):02d}.csv"
    slate[COLUMNS].to_csv(dest, index=False)
    log.info("wrote %s (%d games)", dest, len(slate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
