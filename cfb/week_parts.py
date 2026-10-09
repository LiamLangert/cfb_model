"""Predict one college week, and take each prediction apart into what went into it.

The college counterpart of nfl.predict_week, which the site shows. The
predictions are cfb.predict_games', for the target week only: ratings from
the season's plays before it (conferences, offseason shifts, quarterbacks
and special teams included), shrunk toward last season's, and the points
conversion fitted on every game played before the week since --since. The
default, 2015, trains it as cfb.benchmark grades it. The week defaults to
the next one: the first with an FBS-vs-FBS game still to be played. A week
already played shows what the model would have said.

Only FBS-vs-FBS games are predicted, as cfb.benchmark grades them: every FCS
opponent is rated as one pooled team.

Each predicted home margin is a sum of parts, all in points, as
nfl.predict_week splits it:

    home field     stage 2's intercept, plus its neutral-site term at a neutral site
    rush offence   rush_off at the league's share of rushes, times stage 2's
                   coefficient on that side's edge, scaled by the game's
                   expected plays (cfb.stage2)
    pass offence   pass_off at the share of dropbacks: the passing game less its quarterback
    quarterback    the starter's rating, on the same share (his carries count as
                   dropbacks, cfb.ratings.qb_carries_as_dropbacks)

As on the sites, a quarter of each conference's average pass_off
(cfb.display.QB_SHARE) counts on its quarterbacks and the rest on its pass
offences, so both compare across conferences. The two parts add up to the
same.
    rush defence   rush_def and pass_def at the same shares, times the
    pass defence   coefficient on the other side's edge
    special teams  the punt and kickoff gap, at face value (cfb.special_teams.WEIGHT)

A team's units are its conference's rating plus its own gap from it
(cfb.ratings.conferences); the parts count both. Every team part is that
team's unit against an average one, in points for the team, so home parts
less away parts is the predicted margin. Nothing in RAPPA pairs a unit with
the one it faces, so the parts read as matchups, as nfl.predict_week's do.
There is no rest term: the college line prices rest at about 0.09 points a
day, and a fixed 0.09 did not predict better.

Win probability is cfb.stage2's: Phi(margin / sigma), where sigma is the one
with the best log-loss on every FBS-vs-FBS game predicted before this week.

The line, the chances against the market's and the score shown at the line
are nfl.key_numbers', fitted on college games: FBS-vs-FBS games before the
week, their predicted margins and their projected and final scores.

The total is projected on the edges and both sides' pace and scoring
(cfb.pace), fitted on FBS-vs-FBS games before the week. Each side's points
are (total +/- predicted margin) / 2, so the margin is exactly stage 2's.
Walked forward over 2016-2026, the total missed by 13.10 points a game,
against 12.84 for the closing total.

Starters are cfb.predict_games': the passer on each side's first pass, or for
a game not yet played, the team's latest starter. CFBD lists no starters
ahead of a game, so a quarterback change shows only once he has started.

Writes <out>/<season>/parts_wk<NN>.csv, one row per game, which
site/build.py shows on the ratings page.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cfb import pace, special_teams, stage2
from cfb.display import QB_SHARE, pass_level, with_pass_level
from cfb.offseason import prior_shift
from cfb.predict_games import ALPHA, season_features, starters, unplayed, walk_forward_predict
from cfb.ratings import (QB_REPLACEMENT, centered, conferences, continuous_week, fit_qbs, load,
                         qb_carries_as_dropbacks, qb_prior_means, qb_table)
from nfl import key_numbers
from nfl.benchmark import market_wp
from rappa.games import FEATURES
from rappa.model import summarize

log = logging.getLogger(__name__)

FIRST_SEASON = 2015

# cfb.predict_games' default: stage 2 waits for this many games.
MIN_TRAIN = 200

# The parts of a predicted margin, in the order they are printed and shown.
PARTS = ["qb", "rush_off", "pass_off", "rush_def", "pass_def", "special_teams", "home_field"]

# What the total is regressed on: the edges and both sides' pace and scoring
# (cfb.pace, which has how much each helps).
TOTAL_FEATURES = [*FEATURES, *pace.FEATURES]

# Every file the predictions read: each season's own, the one before's
# ratings as its prior, and what cfb.offseason shifts the prior by.
SEASON_INPUTS = ("plays.csv", "games.csv", "returning.csv")
PRIOR_INPUTS = ("rappa.csv", "qbs.csv")


def upcoming_week(season, out):
    """The season's first week with an FBS-vs-FBS game not played yet, or None.

    A game whose kickoff is more than a day gone without a result was
    cancelled or postponed, not upcoming.
    """
    path = out / str(season) / "games.csv"
    if not path.exists():
        return None
    games = pd.read_csv(path, low_memory=False)
    games = games.assign(week=continuous_week(games))
    fbs = games.home_classification.eq("fbs") & games.away_classification.eq("fbs")
    start = pd.to_datetime(games.start_date, utc=True)
    ahead = fbs & games.completed.ne(True) & (start >= pd.Timestamp.now("UTC") - pd.Timedelta(days=1))
    return int(games.week[ahead].min()) if ahead.any() else None


def readable(name):
    """A passer's name as play text gives it, with a space after an initial: "T.Mizzell" is "T. Mizzell"."""
    return pd.Series(name).str.replace(r"^([A-Z])\.(?=\S)", r"\1. ", regex=True)


def sources(season, out, since=FIRST_SEASON):
    """The data and code the week's predictions depend on, for telling a stale file from a fresh one."""
    files = [out / str(s) / f for s in range(since, season + 1) for f in SEASON_INPUTS]
    files += [out / str(s) / f for s in range(since - 1, season) for f in PRIOR_INPUTS]
    files += [out / str(s) / "recruiting.csv" for s in range(since - 3, season + 1)] + [out / "coaches.csv"]
    files += [out / str(s) / "roster.csv" for s in range(since, season + 1)]
    files += [out / str(s) / "player_defense.csv" for s in range(since - 1, season)]
    code = Path(__file__).resolve().parent
    files += [code / f for f in ("week_parts.py", "predict_games.py", "ratings.py", "special_teams.py",
                                 "offseason.py", "pace.py", "stage2.py")]
    files += [code.parent / "nfl" / f for f in ("key_numbers.py", "benchmark.py", "predict_games.py", "pace.py")]
    files += [code.parent / "rappa" / f for f in ("model.py", "games.py", "special_teams.py", "expected_points.py")]
    return [p for p in files if p.exists()]


def walk(season, out=Path("data/cfb"), since=FIRST_SEASON, alpha=ALPHA, prior_weight=None, min_train=MIN_TRAIN):
    """Every game's walk-forward features and prediction from since through season, the slow part.

    Returns the rows (every game, FCS ones too, as stage 2 trains on them)
    and the FBS-vs-FBS predictions with their win probabilities.
    """
    seasons = [s for s in range(since, season + 1) if (out / str(s) / "plays.csv").exists()]
    rows = pd.concat([season_features(s, out, 2, alpha, prior_weight, True) for s in seasons], ignore_index=True)
    rows = stage2.with_scale(with_total_features(rows, seasons, out), seasons, out)
    preds = walk_forward_predict(rows, min_train)
    return rows, stage2.win_probability(preds[preds.fbs].reset_index(drop=True), min_train)


def with_total_features(rows, seasons, out):
    """rows with cfb.pace's features, by game."""
    return rows.join(pace.features(seasons, out)[pace.FEATURES], on="id")


def week_ratings(season, week, out, alpha, prior_weight):
    """The team, quarterback and special-teams ratings cfb.predict_games rates the week on.

    season_features fits these and keeps only the two edges; this repeats its
    fit for the one week so the edges can be taken apart. Returns the
    centred team ratings, each quarterback's rating against the average
    dropback, a quarterback's rating with none, his dropbacks so far, the
    ids rated from last season, every passer's name, the rush share, each
    team's special-teams value and the season's games with their starters.
    """
    plays, games = load(season, out)
    members = conferences(games)
    prior = qb_prior = None
    prior_path = out / str(season - 1) / "rappa.csv"
    if prior_path.exists():
        prior = pd.read_csv(prior_path, index_col="team")
        if (prior_path.parent / "qbs.csv").exists():
            qb_prior = pd.read_csv(prior_path.parent / "qbs.csv", index_col="passer_id")
    shift = prior_shift(season, out) if prior is not None else None
    games["home_qb"], games["away_qb"] = starters(plays, games)
    start_at = qb_prior_means(qb_prior, pd.concat([plays.passer, games.home_qb, games.away_qb]))

    before = qb_carries_as_dropbacks(plays[plays.week < week])
    slate = games[games.week == week]
    ratings, qbs, _, _ = fit_qbs(before, alpha, prior=prior, qb_prior=start_at, prior_weight=prior_weight,
                                 members=members, shift=shift)
    if prior is not None:
        waiting = set(slate.home_team) | set(slate.away_team)
        ratings = pd.concat([ratings, unplayed(prior, sorted(waiting - set(ratings.index)), prior_weight, True,
                                               True, shift)])
    ratings = summarize(ratings, before, qbs)
    names = qb_table(qbs, before).name
    carried = pd.Index([])
    if start_at is not None:
        carried = start_at.index[~start_at.index.isin(qbs.index)]
        qbs = pd.concat([qbs, start_at[carried]])
    level = before.passer[before.play_kind == "pass"].dropna().map(qbs).mean()
    dropbacks = before.passer[before.play_kind == "pass"].value_counts()
    kicks = special_teams.load(season, out)
    st = special_teams.fit(kicks[kicks.week < week]).value
    return (centered(ratings), qbs - level, QB_REPLACEMENT - level, dropbacks, carried, names,
            float((before.play_kind == "rush").mean()), st, games)


def take_apart(slate, model, ratings, qb, rush_share, st):
    """Each side's parts of the predicted margin (<side>_<part>_pts), in points for that side.

    As nfl.predict_week.take_apart, without rest, and with the quarterback on
    the dropback share alone: his carries are dropbacks. The edges'
    coefficients are each game's: the fitted ones times its plays_scale
    (cfb.stage2).
    """
    b_home, b_away, b_neutral = model.coef_
    scale = slate.plays_scale.to_numpy()
    b_home, b_away = b_home * scale, b_away * scale
    pass_share = 1 - rush_share
    parts = {}
    for side, own_b, their_b in (("home", b_home, b_away), ("away", -b_away, -b_home)):
        r = ratings.reindex(slate[f"{side}_team"])
        parts[f"{side}_rush_off_pts"] = own_b * rush_share * r.rush_off.to_numpy()
        parts[f"{side}_pass_off_pts"] = own_b * pass_share * r.pass_off.to_numpy()
        parts[f"{side}_qb_pts"] = own_b * pass_share * qb[side]
        # A defence counts on the other side's coefficient: it lowers their edge.
        defence = -their_b * r["def"].to_numpy()
        parts[f"{side}_rush_def_pts"] = -their_b * rush_share * r.rush_def.to_numpy()
        parts[f"{side}_pass_def_pts"] = defence - parts[f"{side}_rush_def_pts"]
    home_st, away_st = (slate[f"{side}_team"].map(st).to_numpy() for side in ("home", "away"))
    # As cfb.predict_games: a team with no kicks yet makes the gap zero.
    missing = np.isnan(home_st) | np.isnan(away_st)
    parts["home_special_teams_pts"] = np.where(missing, 0.0, special_teams.WEIGHT * home_st)
    parts["away_special_teams_pts"] = np.where(missing, 0.0, special_teams.WEIGHT * away_st)
    parts["home_home_field_pts"] = model.intercept_ + b_neutral * slate.neutral_site.to_numpy()
    parts["away_home_field_pts"] = np.zeros(len(slate))
    return pd.DataFrame(parts, index=slate.index)


def predict(season, week=None, out=Path("data/cfb"), since=FIRST_SEASON, alpha=ALPHA, prior_weight=None,
            min_train=MIN_TRAIN):
    """One row per FBS-vs-FBS game of the week: prediction, win probability, line, projected score and parts."""
    week = upcoming_week(season, out) if week is None else week
    if week is None:
        raise SystemExit(f"{season} has no games left to play - give a week to see one already played")
    rows, preds = walk(season, out, since, alpha, prior_weight, min_train)
    return week_slate(rows, preds, season, week, out, alpha, prior_weight)


def week_slate(rows, preds, season, week, out, alpha=ALPHA, prior_weight=None):
    """One week of predict, from walk's rows and predictions."""
    # CFBD's home_points are the final score; here they are the projected one, as in nfl.predict_week.
    rows = rows.rename(columns={"home_points": "home_score", "away_points": "away_score"})
    preds = preds.rename(columns={"home_points": "home_score", "away_points": "away_score"})
    slate = preds[(preds.season == season) & (preds.week == week)].copy()
    if slate.empty:
        raise SystemExit(f"no predictions for {season} week {week}: it needs plays from the week before "
                         f"(`uv run -m cfb.pull {season}`), and week 1 has none")
    slot = season * 100 + week

    # Stage 2 again, for its coefficients: fitted on the same games as walk_forward_predict.
    st_points = rows.st_edge * special_teams.WEIGHT
    train = rows[rows.margin.notna() & (rows.season * 100 + rows.week < slot)]
    model = LinearRegression().fit(stage2.design(train), train.margin - st_points[train.index])
    # The total, on FBS-vs-FBS games, split by the predicted margin.
    fbs = train[train.fbs]
    total = LinearRegression().fit(fbs[TOTAL_FEATURES], fbs.home_score + fbs.away_score)
    slate_total = total.predict(slate[TOTAL_FEATURES])
    slate["home_points"] = (slate_total + slate.predicted) / 2
    slate["away_points"] = (slate_total - slate.predicted) / 2

    # The line, the chances against the market's, and the score shown at the line, fitted on the same games.
    slate["spread_line"] = -slate.spread
    before = preds[preds.margin.notna() & (preds.season * 100 + preds.week < slot)]
    probs = key_numbers.fit_margins(before.predicted, before.margin).probs(slate.predicted)
    slate["rappa_line"] = key_numbers.lines(probs)
    slate["rappa_home_cover"], slate["rappa_push"] = key_numbers.at_spread(probs, slate.spread_line)
    fbs_total = total.predict(fbs[TOTAL_FEATURES])
    fbs_margin = model.predict(stage2.design(fbs)) + st_points[fbs.index]
    scores = key_numbers.fit_scores((fbs_total + fbs_margin) / 2, (fbs_total - fbs_margin) / 2,
                                    fbs.home_score, fbs.away_score)
    slate["home_points_at_line"], slate["away_points_at_line"] = key_numbers.score_at_line(
        scores, slate.home_points, slate.away_points, slate.rappa_line)

    ratings, qbs, replacement, dropbacks, carried, names, rush_share, st, games = week_ratings(
        season, week, out, alpha, prior_weight)
    # A share of each conference's average passing game moves onto its quarterbacks (cfb.display).
    team_dropbacks = dropbacks.groupby(dropbacks.index.str.rsplit("|", n=1).str[1]).sum()
    level = pass_level(ratings.pass_off, conferences(games), team_dropbacks)
    ratings = with_pass_level(ratings, level)
    qb = {side: slate[f"{side}_qb"].map(qbs).fillna(replacement).to_numpy()
          + QB_SHARE * slate[f"{side}_team"].map(level).fillna(0.0).to_numpy() for side in ("home", "away")}
    parts = take_apart(slate, model, ratings, qb, rush_share, st)
    total_parts = sum(parts[f"home_{p}_pts"] - parts[f"away_{p}_pts"] for p in PARTS)
    if not np.allclose(total_parts, slate.predicted, atol=1e-6):
        raise RuntimeError("the parts do not add up to cfb.predict_games' margin - has its fit changed?")
    slate = slate.join(parts)

    # Starts so far: each played game's starter, by team.
    played = games[games.margin.notna() & (games.week < week)]
    long = pd.concat([played[["week", f"{side}_team", f"{side}_qb"]].set_axis(["week", "team", "qb"], axis=1)
                      for side in ("home", "away")]).dropna().sort_values("week", kind="stable")
    starts, latest = long.groupby(["team", "qb"]).size(), long.groupby("team").qb.last()

    def name(ids):
        # A passer with no dropbacks in the fit falls back to the name in his id, "j.milroe".
        return readable(ids.map(names).fillna(ids.str.rsplit("|", n=1).str[0])).set_axis(ids.index)

    for side in ("home", "away"):
        ids, teams = slate[f"{side}_qb"], slate[f"{side}_team"]
        slate[f"{side}_qb_name"] = name(ids)
        slate[f"{side}_qb_rating"] = qb[side]
        slate[f"{side}_qb_dropbacks"] = ids.map(dropbacks).fillna(0).astype(int)
        slate[f"{side}_qb_rated_on"] = np.select(
            [slate[f"{side}_qb_dropbacks"] > 0, ids.isin(carried)], ["season", "prior"], "replacement")
        slate[f"{side}_qb_starts"] = [int(starts.get((t, q), 0)) for t, q in zip(teams, ids)]
        last = teams.map(latest)
        slate[f"{side}_qb_replaces"] = name(last).where(last.notna() & ids.notna() & ids.ne(last))
        slate[f"{side}_qb_others"] = [
            ", ".join(f"{name(pd.Series([q])).iat[0]} {n}" for (tm, q), n in starts.items() if tm == t and q != i)
            or None for t, i in zip(teams, ids)]
        for unit in ("rush_off", "pass_off", "rush_def", "pass_def"):
            slate[f"{side}_{unit}_rating"] = teams.map(ratings[unit])
        slate[f"{side}_st_value"] = teams.map(st)

    kickoff = pd.to_datetime(slate.start_date, utc=True).dt.tz_convert("America/New_York")
    slate["gameday"], slate["weekday"] = kickoff.dt.strftime("%Y-%m-%d"), kickoff.dt.strftime("%A")
    slate["gametime"] = kickoff.dt.strftime("%H:%M").where(slate.start_time_tbd.ne(True))
    slate["label"] = np.where(slate.season_type.eq("postseason"), "Postseason", f"Week {week}")
    # A conference game's conference, for the card's header.
    slate["conference"] = slate.home_conference.where(slate.conference_game.eq(True))
    # The game's book's moneylines (cfb.pull.book_lines), with its vig taken out.
    slate["market_home_wp"] = market_wp(slate)
    slate["rush_share"], slate["run_share"] = rush_share, 0.0
    # Each game's coefficients on its edges, its plays_scale in them (cfb.stage2).
    slate["coef_home"], slate["coef_away"] = model.coef_[0] * slate.plays_scale, model.coef_[1] * slate.plays_scale
    slate["coef_neutral"] = model.coef_[2]
    return slate.rename(columns={"id": "game_id", "venue": "stadium", "over_under": "total_line"}).sort_values(
        ["start_date", "game_id"]).reset_index(drop=True)


COLUMNS = [
    "season", "week", "label", "game_id", "gameday", "weekday", "gametime", "away_team", "home_team",
    "neutral_site", "stadium", "conference",
    "predicted", "sigma", "rappa_wp", "home_points", "away_points", "hfa", "n_train",
    "rappa_line", "rappa_home_cover", "rappa_push", "home_points_at_line", "away_points_at_line",
    "coef_home", "coef_away", "coef_neutral", "rush_share", "run_share",
    *(f"{side}_{p}_pts" for side in ("home", "away") for p in PARTS),
    *(f"{side}_qb{c}" for side in ("home", "away") for c in
      ("", "_name", "_rating", "_dropbacks", "_rated_on", "_starts", "_replaces", "_others")),
    *(f"{side}_{u}_rating" for side in ("home", "away") for u in ("rush_off", "pass_off", "rush_def", "pass_def")),
    "home_st_value", "away_st_value",
    "spread_line", "total_line", "home_moneyline", "away_moneyline", "market_home_wp",
    "home_score", "away_score",
]


def report(slate):
    """The week at a glance, then each game's parts by matchup, in points to the home side."""
    table = pd.DataFrame({
        "game": slate.away_team + np.where(slate.neutral_site.eq(1), " vs ", " @ ") + slate.home_team,
        "kickoff": slate.weekday.str[:3] + " " + slate.gametime.fillna("TBD"),
        "score": [f"{a}-{h}" for a, h in zip(slate.away_points_at_line, slate.home_points_at_line)],
        "margin": slate.predicted,
        "line": slate.rappa_line,
        "home wp": slate.rappa_wp,
        "market": slate.spread_line,
        "mkt wp": slate.market_home_wp,
    })
    print(f"\n=== {int(slate.season.iloc[0])} {slate.label.iloc[0]} "
          "(margin, line and market are the home side's; score is away-home at the line) ===")
    print(table.to_string(index=False, float_format="%.2f"))
    net = {"game": table.game}
    for side, other, sign in (("away", "home", -1), ("home", "away", 1)):
        net[f"{side} pass"] = sign * (slate[f"{side}_qb_pts"] + slate[f"{side}_pass_off_pts"]
                                      - slate[f"{other}_pass_def_pts"])
        net[f"{side} run"] = sign * (slate[f"{side}_rush_off_pts"] - slate[f"{other}_rush_def_pts"])
    for p in ("special_teams", "home_field"):
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
    p.add_argument("week", type=int, nargs="?",
                   help="week to predict, as cfb.ratings.continuous_week numbers them (default: the next one)")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--since", type=int, default=FIRST_SEASON,
                   help=f"first season the points conversion trains on (default: {FIRST_SEASON})")
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"ridge penalty (default: {ALPHA:g})")
    p.add_argument("--prior-weight", type=float,
                   help="carry this much of last season for every unit (default: cfb.ratings's, by side)")
    p.add_argument("--min-train", type=int, default=MIN_TRAIN,
                   help=f"games the conversion, and sigma, need before it predicts (default: {MIN_TRAIN})")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("rappa", "cfb"):
        logging.getLogger(name).setLevel(logging.WARNING)
    slate = predict(args.season, args.week, args.out, args.since, args.alpha, args.prior_weight, args.min_train)
    report(slate)
    dest = args.out / str(args.season) / f"parts_wk{int(slate.week.iloc[0]):02d}.csv"
    slate[COLUMNS].to_csv(dest, index=False)
    log.info("wrote %s (%d games)", dest, len(slate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
