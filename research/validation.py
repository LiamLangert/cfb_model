"""Scoring RAPPA on plays it was not fitted on, shared by research.cfb_validate and research.nfl_validate.

Scoring unit is one team's offence in one game, keeping only units with enough
plays for the average to mean anything. A prediction is scored by its
correlation with the unit's actual mean PPA (higher is better) and by
play-weighted squared error (lower is better).
"""

from __future__ import annotations

import numpy as np

from rappa.model import home_field_column

MIN_UNIT_PLAYS = 20


def holdout_units(df, teams):
    """Actual mean PPA per (game, offence), for units where both sides are in `teams`."""
    df = df.assign(home_sign=home_field_column(df))
    units = df.groupby(["game_id", "offense", "defense"]).agg(
        **{"mean": ("ppa", "mean"), "size": ("ppa", "size"), "home_sign": ("home_sign", "first")}
    ).reset_index()
    return units[
        (units["size"] >= MIN_UNIT_PLAYS)
        & units.offense.isin(teams)
        & units.defense.isin(teams)
    ]


def correlate(units, prediction):
    ok = prediction.notna()
    if ok.sum() < 2:
        return float("nan")
    return float(np.corrcoef(prediction[ok], units["mean"][ok])[0, 1])


def weighted_mse(units, prediction):
    """Squared error, weighted by plays so noisy short samples count for less.

    Correlation is blind to scale and offset, so ratings that are twice as
    spread out as they should be score exactly the same. Getting the size
    right is what the ridge penalty is for, and only an error metric sees it.
    """
    ok = prediction.notna()
    if not ok.any():
        return float("nan")
    return float(np.average((units["mean"][ok] - prediction[ok]) ** 2, weights=units["size"][ok]))


def predict_units(units, ratings, intercept, hfa=0.0):
    """Expected PPA for each unit: league average, adjusted for both sides.

    The intercept is not optional. Team coefficients are deviations from
    average and centre on zero, so leaving it out biases every prediction low
    by league-average PPA - enough to lose to predicting a flat constant.
    """
    base = intercept + units.offense.map(ratings["off"]) - units.defense.map(ratings["def"])
    # Same signed convention as the design matrix: +1 home, -1 away, 0 neutral.
    return base + hfa * units.get("home_sign", 0)
