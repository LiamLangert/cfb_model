"""College ratings as they stood a week ago: cfb.ratings' fit on a season's plays before its latest week.

site/build.py ranks these beside the season's rappa.csv, to show how many
places each team has moved since the week before. The fit is cfb.ratings'
at its defaults - conferences, offseason shifts and quarterbacks included -
with or without last season's prior, whichever reproduces rappa.csv from
the season's plays. So the latest week's plays are the only difference from
rappa.csv, and a rappa.csv that neither reproduces (rated before the latest
plays were pulled, or with other options) gets nothing. Special teams are
rated on the same games' kicks (cfb.special_teams), as the rank counts them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cfb import special_teams
from cfb.offseason import prior_shift
from cfb.ratings import GARBAGE, SEASON_ALPHA, conferences, fit_qbs, load, qb_carries_as_dropbacks, qb_prior_means
from rappa.model import UNITS, garbage_time, summarize

log = logging.getLogger(__name__)


def sources(season, out=Path("data/cfb")):
    """The data and code the ratings depend on, for telling a stale file from a fresh one.

    The season's rappa.csv stands in for the prior and the offseason inputs:
    re-rating after any of them changes rewrites it.
    """
    files = [out / str(season) / f for f in ("plays.csv", "games.csv", "rappa.csv")]
    code = Path(__file__).resolve().parent
    files += [code / f for f in ("previous_week.py", "ratings.py", "offseason.py", "special_teams.py")]
    files += [code.parent / "rappa" / f for f in ("model.py", "special_teams.py", "expected_points.py")]
    return [p for p in files if p.exists()]


def priors(season, out):
    """The (prior, qb_prior, shift) cfb.ratings may have rated the season with: last season's, then none."""
    path = out / str(season - 1) / "rappa.csv"
    if path.exists():
        qb_path = path.parent / "qbs.csv"
        qb_prior = pd.read_csv(qb_path, index_col="passer_id") if qb_path.exists() else None
        yield pd.read_csv(path, index_col="team"), qb_prior, prior_shift(season, out)
    yield None, None, None


def fit_plays(plays, games, prior, qb_prior, shift):
    """cfb.ratings' fit at its defaults, on these plays: the ratings as rappa.csv holds them, offsets and all."""
    plays = qb_carries_as_dropbacks(plays)
    ratings, qbs, _, _ = fit_qbs(plays, SEASON_ALPHA, prior=prior, qb_prior=qb_prior_means(qb_prior, plays.passer),
                                 members=conferences(games), shift=shift)
    return summarize(ratings, plays, qbs)


def same(ratings, saved):
    """Whether a fit is the one saved, to the four places rappa.csv keeps."""
    cols = [*UNITS, "net"]
    return ratings.index.sort_values().equals(saved.index.sort_values()) and \
        np.allclose(ratings.loc[saved.index, cols], saved[cols], atol=2e-4)


def rate(season, out=Path("data/cfb")):
    """(week, ratings): the season's latest week with plays, and the team ratings on the plays before it.

    ratings also holds st, each team's special teams (points a game) on the
    kicks in the same games. Those are picked by game rather than by week, as
    cfb.special_teams keeps CFBD's weeks, which start again at 1 for the bowls.
    None when the latest week is the first, with nothing before it, or when
    no fit reproduces the season's rappa.csv.
    """
    plays, games = load(season, out)
    if GARBAGE != "none":
        plays = plays[~garbage_time(plays, GARBAGE)]
    week = plays.week.max()
    before = plays[plays.week < week]
    if before.empty:
        return None
    saved = pd.read_csv(out / str(season) / "rappa.csv", index_col="team")
    for prior in priors(season, out):
        if same(fit_plays(plays, games, *prior), saved):
            ratings = fit_plays(before, games, *prior)
            kicks = special_teams.load(season, out)
            ratings["st"] = special_teams.fit(kicks[kicks.game_id.isin(set(before.game_id))]).value
            return int(week), ratings
    log.warning("%d: no fit of plays.csv reproduces rappa.csv - re-rate with `uv run -m cfb.ratings %d --prior "
                "%s` to rank against last week", season, season, out / str(season - 1) / "rappa.csv")
    return None
