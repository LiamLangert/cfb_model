"""NFL ratings as they stood a week ago: nfl.ratings' fit on a season's plays before its latest week.

site/build.py ranks these beside the season's rappa.csv, to show how many
places each team has moved since the week before. The fit is nfl.ratings'
at its defaults - quarterbacks and their designed runs included - with or
without last season's prior, whichever reproduces rappa.csv from the
season's plays (the repo's NFL seasons are rated without one). So the
latest week's plays are the only difference from rappa.csv, and a rappa.csv
that neither reproduces (rated before the latest plays were pulled, or with
other options) gets nothing. Special teams are rated on the same weeks'
kicks (nfl.special_teams), as the rank counts them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nfl import special_teams
from nfl.ratings import SEASON_ALPHA, fit_qbs, in_fit, load, with_qb_runs
from rappa.model import UNITS, summarize

log = logging.getLogger(__name__)


def sources(season, out=Path("data/nfl")):
    """The data and code the ratings depend on, for telling a stale file from a fresh one.

    The season's rappa.csv stands in for the prior: re-rating after it
    changes rewrites it.
    """
    files = [out / str(season) / f for f in ("plays.csv", "rappa.csv")]
    code = Path(__file__).resolve().parent
    files += [code / f for f in ("previous_week.py", "ratings.py", "special_teams.py")]
    files += [code.parent / "rappa" / f for f in ("model.py", "special_teams.py", "expected_points.py")]
    return [p for p in files if p.exists()]


def priors(season, out):
    """The (prior, qb_prior) nfl.ratings may have rated the season with: none, then last season's."""
    yield None, None
    path = out / str(season - 1) / "rappa.csv"
    if path.exists():
        qb_path = path.parent / "qbs.csv"
        qb_prior = pd.read_csv(qb_path, index_col="passer_id") if qb_path.exists() else None
        yield pd.read_csv(path, index_col="team"), qb_prior


def fit_plays(plays, prior, qb_prior):
    """nfl.ratings' fit at its defaults, on these plays: the ratings as rappa.csv holds them, offsets and all."""
    # Marked here rather than inside fit_qbs so summarize counts the runs in off, as nfl.ratings does.
    plays = with_qb_runs(plays)
    ratings, qbs, _, _ = fit_qbs(plays, SEASON_ALPHA, prior=prior, qb_prior=qb_prior)
    return summarize(ratings, plays, qbs)


def same(ratings, saved):
    """Whether a fit is the one saved, to the four places rappa.csv keeps."""
    cols = [*UNITS, "net"]
    return ratings.index.sort_values().equals(saved.index.sort_values()) and \
        np.allclose(ratings.loc[saved.index, cols], saved[cols], atol=2e-4)


def rate(season, out=Path("data/nfl")):
    """(week, ratings): the season's latest week with plays, and the team ratings on the plays before it.

    ratings also holds st, each team's special teams (points a game) on the
    kicks before the week. None when the latest week is the first, with
    nothing before it, or when no fit reproduces the season's rappa.csv.
    """
    plays = in_fit(load(season, out))
    week = plays.week.max()
    before = plays[plays.week < week]
    if before.empty:
        return None
    saved = pd.read_csv(out / str(season) / "rappa.csv", index_col="team")
    for prior in priors(season, out):
        if same(fit_plays(plays, *prior), saved):
            ratings = fit_plays(before, *prior)
            kicks = special_teams.load(season, out)
            ratings["st"] = special_teams.fit(kicks[kicks.week < week]).value
            return int(week), ratings
    log.warning("%d: no fit of plays.csv reproduces rappa.csv - re-rate with `uv run -m nfl.ratings %d` "
                "to rank against last week", season, season)
    return None
