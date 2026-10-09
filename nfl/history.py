"""NFL ratings and predictions week by week, as the site browses them: each from what came before it.

The NFL counterpart of cfb.history, for site/build.py:

  ratings_by_week  a season's team ratings through each of its weeks: the fit
                   that reproduces the season's rappa.csv (nfl.previous_week's
                   check; the repo's NFL seasons are rated without a prior), on
                   the plays up to and including the week, with special teams
                   on the same weeks' kicks. The last week's is rappa.csv itself.
  predictions      every game nfl.predict_week's walk predicts, each week from
                   ratings on the weeks before it: win probability and the score
                   at RAPPA's line, projected as nfl.predict_week.week_slate
                   projects the week it shows, with stage 2, the total and the
                   key-number fits all trained on games before the week.

points_per_play gives the Teams page's conversion for a week: stage 2's
coefficient fitted on every game through it, the one that predicts the week
after.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from nfl import key_numbers, pace, special_teams
from nfl.predict_games import REST_WEIGHT, load_games
from nfl.previous_week import fit_plays, priors, same
from nfl.ratings import in_fit, load
from rappa.games import FEATURES
from rappa.weeks import map_weeks

log = logging.getLogger(__name__)

# Playoff rounds by nflverse game_type; a regular-season week is "Week N".
ROUNDS = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference", "SB": "Super Bowl"}

TOTAL_FEATURES = [*FEATURES, *pace.FEATURES]


def week_labels(season, out):
    """Each week's name, by week: "Week N", or its playoff round."""
    games = load_games(season, out)
    kind = games.groupby("week").game_type.first()
    return {int(w): ROUNDS.get(t, f"Week {int(w)}") for w, t in kind.items()}


def ratings_by_week(season, out=Path("data/nfl")):
    """[(week, label, ratings)], a week at a time, or None when no fit reproduces the season's rappa.csv.

    ratings are as rappa.csv holds them, plus st (points a game) on the
    kicks through the week.
    """
    plays = in_fit(load(season, out))
    saved = pd.read_csv(out / str(season) / "rappa.csv", index_col="team")
    labels = week_labels(season, out)
    for prior in priors(season, out):
        if not same(fit_plays(plays, *prior), saved):
            continue
        kicks = special_teams.load(season, out)
        weeks = []
        for week in sorted(plays.week.unique()):
            ratings = fit_plays(plays[plays.week <= week], *prior)
            ratings["st"] = special_teams.fit(kicks[kicks.week <= week]).value
            weeks.append((int(week), labels.get(int(week), f"Week {int(week)}"), ratings))
        return weeks
    log.warning("%d: no fit of plays.csv reproduces rappa.csv - re-rate it to browse its weeks", season)
    return None


def stage2(rows, train):
    """Stage 2 as nfl.predict_week.week_slate fits it, special teams and rest taken off the margin first: the model,
    and each row's points at face value."""
    face = rows.st_edge * special_teams.WEIGHT + REST_WEIGHT * rows.rest_edge
    return LinearRegression().fit(train[FEATURES], train.margin - face[train.index]), face


def points_per_play(rows, season, week, min_train=100):
    """Stage 2's points a game for each unit of net, fitted on every game through the week: None before it has
    min_train (walk_forward_predict's wait)."""
    train = rows[rows.margin.notna() & (rows.season * 100 + rows.week <= season * 100 + week)]
    if len(train) < min_train:
        return None
    b_home, b_away, _ = stage2(rows, train)[0].coef_
    return float(b_home - b_away) / 2


def week_scores(slot, rows, preds):
    """One week's games (slot is season * 100 + week) with each side's points at the line; None before there are
    games to fit on."""
    slots = preds.season * 100 + preds.week
    slate = preds[slots == slot].copy()
    train = rows[rows.margin.notna() & (rows.season * 100 + rows.week < slot)]
    before = preds[preds.margin.notna() & (slots < slot)]
    if train.empty or before.empty:
        return None
    model, face = stage2(rows, train)
    total = LinearRegression().fit(train[TOTAL_FEATURES], train.home_score + train.away_score)
    slate_total = total.predict(slate[TOTAL_FEATURES])
    home_points, away_points = (slate_total + slate.predicted) / 2, (slate_total - slate.predicted) / 2
    line = key_numbers.lines(key_numbers.fit_margins(before.predicted, before.margin).probs(slate.predicted))
    train_total = total.predict(train[TOTAL_FEATURES])
    train_margin = model.predict(train[FEATURES]) + face[train.index]
    scores = key_numbers.fit_scores((train_total + train_margin) / 2, (train_total - train_margin) / 2,
                                    train.home_score, train.away_score)
    slate["home_points_at_line"], slate["away_points_at_line"] = key_numbers.score_at_line(
        scores, home_points, away_points, line)
    return slate


def predictions(rows, preds, skip=(), workers=None):
    """Every week's predictions from nfl.predict_week.walk's rows and preds, with the score at the line.

    One row per game: season, week, label, game_id, kickoff, teams, neutral
    site, predicted margin, rappa_wp, each side's points at the line and its
    final score (missing until played). Weeks in skip (season * 100 + week)
    are left out, and the rest run in parallel (rappa.weeks). None when no
    week is left.
    """
    slots = sorted(set(preds.season * 100 + preds.week) - set(skip))
    weeks = [w for w in map_weeks(week_scores, slots, (rows, preds), workers) if w is not None]
    if not weeks:
        return None
    games = pd.concat(weeks, ignore_index=True)
    games["label"] = [ROUNDS.get(t, f"Week {int(w)}") for t, w in zip(games.game_type, games.week)]
    return games.sort_values(["season", "week", "gameday", "gametime", "game_id"])[COLUMNS].reset_index(drop=True)


COLUMNS = ["season", "week", "label", "game_id", "gameday", "gametime", "away_team", "home_team", "neutral_site",
           "predicted", "rappa_wp", "hfa", "sigma", "home_points_at_line", "away_points_at_line", "home_score", "away_score"]
