"""College ratings as the sites show them: each quarterback with a share of his conference's passing level.

The fit (cfb.ratings) splits a passing game three ways: its conference's
pass_off, the team's own gap from that, and the quarterback. Within a season
the three move together, so the penalties decide the split, not the plays.
A conference-wide edge lands mostly on the conference: its one column is
penalised at CONF_ALPHA, against two at alpha for each of its teams, so it
takes about 60% of a league-wide surprise and each passer about 20%. The
conference prior also carries in full, while every new quarterback starts at
QB_REPLACEMENT. So the fitted quarterbacks average about the same in every
conference: an SEC passer is rated against the SEC's passing game, and a
Mountain West one against the Mountain West's.

Whether that level belongs to the quarterbacks or to their surroundings,
the data split two ways.
  - A transferring passer's own PPA barely follows his conference. Over
    2015-2026, 150 passers changed teams with 100+ dropbacks in both
    seasons. Controlling for the season before, their opponent-adjusted PPA
    a dropback picked up only 0.19 +/- 0.11 of the change in their
    conference's average pass_off, and the season fits' team pass_off plus
    quarterback picked up 0.33 +/- 0.10.
  - Game margins don't show the model missing on those games. Shifting a
    transfer's prior by the change made predictions worse: walked forward
    over 2015-2026, MAE rose 0.012-0.020, and 0.15-0.21 in weeks 2-5 of the
    games he started.

So this is a choice for reading, not a finding. QB_SHARE of each team's
conference average pass_off moves from its pass_off to its quarterbacks,
the share the transfers suggest, and the rest stays on pass_off. Both then
compare across conferences: a column that took all of it would show the
whole conference edge (0.27 a dropback between the SEC and the MAC in 2025,
about five SDs of quarterback ratings) on that column alone. Until
2026-10-08 the quarterbacks took all of it, and pass_off read against its
conference's average. A team's pass_off plus its quarterback is unchanged,
and so is every composite and prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cfb.ratings import FCS, conferences

# The share of a conference's passing level shown on its quarterbacks: transfers' own PPA followed 0.19 +/-
# 0.11 of the change in their conference's, and the fits' pass_off plus quarterback 0.33 +/- 0.10.
QB_SHARE = 0.25


def pass_level(pass_off, members, dropbacks):
    """Each team's conference's average pass_off; with_pass_level moves QB_SHARE of it to its quarterbacks.

    pass_off is the teams' centred pass_off (a Series by team); members is
    cfb.ratings.conferences, and a team in none (an independent) moves
    nothing but the centring. dropbacks is each team's pass plays (a Series
    by team), which the level is centred on so quarterbacks stay against the
    average dropback. Returns a Series by team.
    """
    conference = pd.Series(pass_off.index.map(members), index=pass_off.index)
    level = pass_off.groupby(conference).transform("mean").reindex(pass_off.index).fillna(0.0)
    weights = dropbacks.reindex(level.index).fillna(0.0)
    return level - np.average(level, weights=weights)


def season_pass_level(season_dir: Path):
    """pass_level for one season's files: FBS teams' rappa.csv, qbs.csv and games.csv; None without qbs.csv."""
    if not (season_dir / "qbs.csv").exists():
        return None
    ratings = pd.read_csv(season_dir / "rappa.csv", index_col="team").drop(index=FCS, errors="ignore")
    games = pd.read_csv(season_dir / "games.csv", low_memory=False)
    dropbacks = pd.read_csv(season_dir / "qbs.csv").groupby("team").dropbacks.sum()
    return pass_level(ratings.pass_off - ratings.pass_off.mean(), conferences(games), dropbacks)


def with_pass_level(ratings, level):
    """Centred team ratings (by team) with QB_SHARE of level moved from pass_off to qb."""
    level = QB_SHARE * level.reindex(ratings.index).fillna(0.0)
    return ratings.assign(pass_off=ratings.pass_off - level, qb=ratings.qb + level)
