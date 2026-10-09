"""College ratings and predictions week by week, as the site browses them: each from what came before it.

Two things, both for site/build.py:

  ratings_by_week  a season's team ratings through each of its weeks: the fit
                   that reproduces the season's rappa.csv (cfb.previous_week's
                   check: conferences, offseason shifts and quarterbacks, with or
                   without last season's prior), on the plays up to and
                   including the week, with special teams on the same games'
                   kicks. The last week's is rappa.csv itself.
  predictions      every FBS-vs-FBS game cfb.week_parts' walk predicts, each
                   week from ratings on the weeks before it: win probability
                   and the score at RAPPA's line, projected as
                   cfb.week_parts.week_slate projects the week it shows, with
                   stage 2, the total and the key-number fits all trained on
                   games before the week. The parts are left out.

points_per_play gives the Teams page's conversion for a week: stage 2's
coefficient fitted on every game through it, the one that predicts the week
after, at the week's games' expected plays (cfb.stage2).

Like the predictions, the ratings value plays with expected points fitted on
the whole season (rappa.expected_points), the one thing a week sees of the
weeks after it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cfb import special_teams, stage2
from cfb.display import pass_level
from cfb.previous_week import fit_plays, priors, same
from cfb.ratings import FCS, GARBAGE, conferences, load, qb_carries_as_dropbacks
from cfb.week_parts import TOTAL_FEATURES
from nfl import key_numbers
from rappa.model import garbage_time
from rappa.weeks import map_weeks

log = logging.getLogger(__name__)


def ratings_by_week(season, out=Path("data/cfb")):
    """[(week, label, ratings, level)], a week at a time, or None when no fit reproduces the season's rappa.csv.

    ratings are as rappa.csv holds them, plus st (points a game); level is
    cfb.display.pass_level on the plays through the week, for the page to
    move each conference's passing level onto its quarterbacks. Bowls and
    playoff games are one week (cfb.ratings.continuous_week), "Postseason".
    """
    plays, games = load(season, out)
    if GARBAGE != "none":
        plays = plays[~garbage_time(plays, GARBAGE)]
    saved = pd.read_csv(out / str(season) / "rappa.csv", index_col="team")
    members = conferences(games)
    last_regular = games.week[games.season_type == "regular"].max()
    for prior in priors(season, out):
        if not same(fit_plays(plays, games, *prior), saved):
            continue
        kicks = special_teams.load(season, out)
        weeks = []
        for week in sorted(plays.week.unique()):
            through = plays[plays.week <= week]
            ratings = fit_plays(through, games, *prior)
            ratings["st"] = special_teams.fit(kicks[kicks.game_id.isin(set(through.game_id))]).value
            passes = qb_carries_as_dropbacks(through)
            dropbacks = passes[passes.play_kind == "pass"].offense.value_counts()
            fbs = ratings.drop(index=FCS, errors="ignore")
            level = pass_level(fbs.pass_off - fbs.pass_off.mean(), members, dropbacks)
            label = f"Week {int(week)}" if week <= last_regular else "Postseason"
            weeks.append((int(week), label, ratings, level))
        return weeks
    log.warning("%d: no fit of plays.csv reproduces rappa.csv - re-rate it to browse its weeks", season)
    return None


def points_per_play(rows, season, week, min_train=200):
    """Stage 2's points a game for each unit of net, fitted on every game through the week: None before it has
    min_train (walk_forward_predict's wait). As week_slate fits it, special teams taken off the margin first.

    Stage 2 scales each game's edges by its expected plays (cfb.stage2), so
    this is at the week's FBS-vs-FBS games' average plays_scale.
    """
    st_points = rows.st_edge * special_teams.WEIGHT
    slot = rows.season * 100 + rows.week
    train = rows[rows.margin.notna() & (slot <= season * 100 + week)]
    if len(train) < min_train:
        return None
    model = LinearRegression().fit(stage2.design(train), train.margin - st_points[train.index])
    b_home, b_away, _ = model.coef_
    week_games = rows[(slot == season * 100 + week) & rows.fbs]
    scale = (week_games if len(week_games) else train[train.fbs]).plays_scale.mean()
    return float(b_home - b_away) / 2 * float(scale)


def week_scores(slot, rows, preds):
    """One week's FBS-vs-FBS games (slot is season * 100 + week) with each side's points at the line; None before
    there are games to fit on. rows and preds as predictions renames them."""
    st_points = rows.st_edge * special_teams.WEIGHT
    slots = preds.season * 100 + preds.week
    slate = preds[slots == slot].copy()
    train = rows[rows.margin.notna() & (rows.season * 100 + rows.week < slot)]
    before = preds[preds.margin.notna() & (slots < slot)]
    if train.empty or before.empty:
        return None
    # As week_slate: stage 2 and the total on the games before the week, the total split by the margin.
    model = LinearRegression().fit(stage2.design(train), train.margin - st_points[train.index])
    fbs = train[train.fbs]
    total = LinearRegression().fit(fbs[TOTAL_FEATURES], fbs.home_score + fbs.away_score)
    slate_total = total.predict(slate[TOTAL_FEATURES])
    home_points, away_points = (slate_total + slate.predicted) / 2, (slate_total - slate.predicted) / 2
    line = key_numbers.lines(key_numbers.fit_margins(before.predicted, before.margin).probs(slate.predicted))
    fbs_total = total.predict(fbs[TOTAL_FEATURES])
    fbs_margin = model.predict(stage2.design(fbs)) + st_points[fbs.index]
    scores = key_numbers.fit_scores((fbs_total + fbs_margin) / 2, (fbs_total - fbs_margin) / 2,
                                    fbs.home_score, fbs.away_score)
    slate["home_points_at_line"], slate["away_points_at_line"] = key_numbers.score_at_line(
        scores, home_points, away_points, line)
    return slate


def predictions(rows, preds, skip=(), workers=None):
    """Every week's FBS-vs-FBS predictions from cfb.week_parts.walk's rows and preds, with the score at the line.

    One row per game: season, week, label, game_id, kickoff, teams, neutral
    site, predicted margin, rappa_wp, each side's points at the line and its
    final score (missing until played). Weeks in skip (season * 100 + week)
    are left out, and the rest run in parallel (rappa.weeks). None when no
    week is left.
    """
    # CFBD's home_points are the final score; week_slate's are the projected one.
    rows = rows.rename(columns={"home_points": "home_score", "away_points": "away_score"})
    preds = preds.rename(columns={"home_points": "home_score", "away_points": "away_score"})
    slots = sorted(set(preds.season * 100 + preds.week) - set(skip))
    weeks = [w for w in map_weeks(week_scores, slots, (rows, preds), workers) if w is not None]
    if not weeks:
        return None
    games = pd.concat(weeks, ignore_index=True)
    kickoff = pd.to_datetime(games.start_date, utc=True).dt.tz_convert("America/New_York")
    games["gameday"] = kickoff.dt.strftime("%Y-%m-%d")
    games["gametime"] = kickoff.dt.strftime("%H:%M").where(games.start_time_tbd.ne(True))
    games["label"] = np.where(games.season_type.eq("postseason"), "Postseason",
                              "Week " + games.week.astype(int).astype(str))
    games = games.sort_values(["season", "week", "start_date", "id"])
    return games.rename(columns={"id": "game_id"})[COLUMNS].reset_index(drop=True)


COLUMNS = ["season", "week", "label", "game_id", "gameday", "gametime", "away_team", "home_team", "neutral_site",
           "predicted", "rappa_wp", "hfa", "sigma", "home_points_at_line", "away_points_at_line", "home_score", "away_score"]
