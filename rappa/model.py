"""RAPPA - Regularized Adjusted Predicted Points Added. The fit itself, for any league.

A RAPM-style ridge regression on play-by-play. Every play is one row; the
target is PPA from the offense's point of view. Each team gets four columns -
rush offense, pass offense, rush defense, pass defense - and only the two
units actually on the field are non-zero for a given play:

    rush play, A on offense vs B on defense
        x[A, rush_off] = +1      x[B, rush_def] = -1
        ppa ~= beta[A, rush_off] - beta[B, rush_def]

Defense enters as -1 so that a higher coefficient is better for every column:
good offenses add points, good defenses subtract them. Ridge shrinks units
with few plays toward league average, which is what makes the ratings usable
for teams that have not played much.

fit_passers also gives every passer a column of his own, +1 on each pass
play he is the passer on, so a pass play is the offence's pass unit plus its
passer against the defence's pass unit. The team's pass_off is then what the
rest of the offence adds, and a quarterback's rating goes with him when he
changes teams. The passers share the teams' penalty; their prior is set by
the caller.

Groups (college conferences, see Groups) optionally add four shared columns
per group, so a team's rating is its group's part plus its own:

    rush play, A (SEC) on offense vs B (ACC) on defense
        ppa ~= (beta[A, rush_off] + g[SEC, rush_off]) - (beta[B, rush_def] + g[ACC, rush_def])

Within a group's own games a shift of every member cancels, so only games
against other groups say how good the group is as a whole. With team columns
alone, the penalty charges that shift once per member while only those few
games argue for it, and the fit leaves most of it out. The group columns
carry it at one penalty for the whole group.

Home field is a fixed per-play constant taken off the target, not a fitted
column, and each play's PPA is clipped before fitting. Both are measured per
league, so fit has no defaults for them: cfb.ratings and nfl.ratings each wrap
it with their own constants.

Each league's loader hands over one row per play with these columns:

    game_id              groups plays for cross-validation
    week                 for time decay; weeks must run straight through the postseason
    offense, defense     team names
    home                 the home team, to sign home field
    neutral_site         optional; True means no home field
    play_kind            "rush" or "pass"
    ppa                  the target (NFL epa is renamed to this)
    period, margin       for garbage_time: quarter, and the offence's lead at the snap
    conference_game      for conference_home_edge
    passer               for fit_passers: an id for the passer on pass plays, else missing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

log = logging.getLogger(__name__)

UNITS = ("rush_off", "pass_off", "rush_def", "pass_def")

ALPHAS = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)

# Garbage-time rules, as a lead big enough to stop telling you about team quality.
# "connelly" is the Football Outsiders / SP+ convention: a margin threshold that
# falls as the clock runs down. Measured against the score BEFORE the play.
GARBAGE_RULES = {
    "none": None,
    "connelly": {1: 43, 2: 37, 3: 27, 4: 21},
    "flat28": {1: 28, 2: 28, 3: 28, 4: 28},
    "flat21": {1: 21, 2: 21, 3: 21, 4: 21},
    "q4_21": {4: 21},
}


def home_field_column(df):
    """+1 when the offence is at home, -1 when away, 0 at a neutral site.

    Signed rather than a 0/1 flag so the intercept stays league-average PPA
    and the team ratings stay centred on zero. HFA is therefore half the
    home-to-away swing.
    """
    signed = np.where(df.offense == df.home, 1.0, -1.0)
    if "neutral_site" in df:
        signed = np.where(df.neutral_site.to_numpy(dtype=bool), 0.0, signed)
    return signed


def conference_home_edge(df, clip):
    """Half the home-minus-away gap in offensive PPA, from conference games only.

    This is how HFA was measured. Conference schedules are close to balanced -
    everyone plays about half at home, against the same opponents - so unlike
    the non-conference gap it is not also measuring who scheduled whom.
    """
    conf = df[df.conference_game]
    sign = home_field_column(conf)
    ppa = conf.ppa.clip(-clip, clip) if clip else conf.ppa
    return float((ppa[sign > 0].mean() - ppa[sign < 0].mean()) / 2)


def garbage_time(df, rule):
    """Boolean mask of plays played in garbage time under the given rule."""
    thresholds = GARBAGE_RULES[rule]
    if thresholds is None:
        return pd.Series(False, index=df.index)
    # Periods with no threshold (and overtime) are never garbage time.
    limit = df.period.map(thresholds)
    return limit.notna() & (df.margin.abs() >= limit)


def build_design(df, teams, passers=()):
    """Sparse design matrix: one offence and one defence entry per play.

    With passers, one more column each after the teams', +1 on the pass plays
    whose df.passer it is.
    """
    # Four consecutive columns per team, one per unit.
    index = {(team, unit): 4 * i + unit for i, team in enumerate(teams) for unit in range(4)}
    n = len(df)
    is_rush = (df.play_kind == "rush").to_numpy()

    off_unit = np.where(is_rush, UNITS.index("rush_off"), UNITS.index("pass_off"))
    def_unit = np.where(is_rush, UNITS.index("rush_def"), UNITS.index("pass_def"))
    off_col = np.array([index[(t, u)] for t, u in zip(df.offense, off_unit)])
    def_col = np.array([index[(t, u)] for t, u in zip(df.defense, def_unit)])

    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=int)
    cols[0::2], cols[1::2] = off_col, def_col
    vals = np.empty(2 * n)
    vals[0::2], vals[1::2] = 1.0, -1.0

    width = 4 * len(teams)
    if len(passers):
        column = {p: width + j for j, p in enumerate(passers)}
        passer_col = df.passer.where(~is_rush).map(column).to_numpy(dtype=float)
        has = ~np.isnan(passer_col)
        rows = np.r_[rows, np.flatnonzero(has)]
        cols = np.r_[cols, passer_col[has].astype(int)]
        vals = np.r_[vals, np.ones(has.sum())]
        width += len(passers)

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n, width))
    return X, index


@dataclass
class Groups:
    """Teams that share a rating of their own, fitted with its own penalty: college conferences.

    members  team -> group. Every member gets a rating, whether or not it has
             plays yet (then it is its prior plus its group's fitted part);
             a team left out has no group part.
    alpha    the group columns' ridge penalty, in place of the teams'.
    prior    what each group's four units are shrunk toward, one row per group
             and a column per unit; None, or a group missing from it, is zero.
             The teams' own prior is then their gap from their group.
    """
    members: dict
    alpha: float
    prior: pd.DataFrame | None = None


def group_design(df, groups, names):
    """Sparse columns for the groups, four per group in UNITS order.

    A play puts +1 on its offence's group unit and -1 on its defence's, as
    build_design does for the teams.
    """
    column = {g: 4 * k for k, g in enumerate(names)}
    is_rush = (df.play_kind == "rush").to_numpy()
    off_unit = np.where(is_rush, UNITS.index("rush_off"), UNITS.index("pass_off"))
    def_unit = np.where(is_rush, UNITS.index("rush_def"), UNITS.index("pass_def"))
    off = df.offense.map(groups).map(column).to_numpy(dtype=float)
    dfn = df.defense.map(groups).map(column).to_numpy(dtype=float)
    has_off, has_def = ~np.isnan(off), ~np.isnan(dfn)
    rows = np.r_[np.flatnonzero(has_off), np.flatnonzero(has_def)]
    cols = np.r_[off[has_off] + off_unit[has_off], dfn[has_def] + def_unit[has_def]].astype(int)
    vals = np.r_[np.ones(has_off.sum()), -np.ones(has_def.sum())]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(df), 4 * len(names)))


def decay_weights(weeks, half_life):
    """Per-play sample weights that fade older football out of the fit.

    A play from `half_life` weeks before the edge of the data counts half as
    much as one from the most recent week. None means every play counts the
    same, which is the default.
    """
    if half_life is None:
        return None
    if weeks.isna().any():
        raise ValueError("time decay needs a week for every play")
    reference = weeks.max() + 1  # we are always aiming at the week after the data
    return (0.5 ** ((reference - weeks) / half_life)).to_numpy(dtype=float)


def prior_vector(prior, teams, index, weight, width):
    """Coefficients to shrink toward, taken from a previous season's ratings.

    Teams change between seasons, so last year is carried forward at less than
    full strength: `weight`, which each league sets to its measured
    year-over-year slope (PRIOR_WEIGHT in cfb.ratings and nfl.ratings).
    Teams with no prior (such as programs new to FBS) keep a prior of zero,
    which is the league average.
    """
    beta0 = np.zeros(width)
    if prior is None:
        return beta0
    hits = 0
    for team in teams:
        if team not in prior.index:
            continue
        hits += 1
        for unit_idx, unit in enumerate(UNITS):
            beta0[index[(team, unit_idx)]] = weight * float(prior.at[team, unit])
    log.info("prior covers %d/%d teams at weight %.2f", hits, len(teams), weight)
    return beta0


def pick_alpha(X, y, groups, seed=0, sample_weight=None):
    """Choose the ridge penalty by cross-validation, holding out whole games.

    Decay weights are applied when fitting each fold but not when scoring it:
    the penalty should suit the weighted fit, while the error it is judged on
    stays an honest average over held-out plays.
    """
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(X, y, groups))
    best = (None, np.inf)
    for alpha in ALPHAS:
        errors = []
        for train, test in splits:
            model = Ridge(alpha=alpha, solver="sparse_cg", random_state=seed)
            fold_w = None if sample_weight is None else sample_weight[train]
            model.fit(X[train], y[train], sample_weight=fold_w)
            errors.append(np.mean((y[test] - model.predict(X[test])) ** 2))
        mse = float(np.mean(errors))
        log.info("  alpha %8.0f  cv mse %.5f", alpha, mse)
        if mse < best[1]:
            best = (alpha, mse)
    var = float(np.var(y))
    log.info("best alpha %.0f | cv mse %.5f vs variance %.5f (cv r2 %.4f)",
             best[0], best[1], var, 1 - best[1] / var)
    return best[0]


def fit(df, alpha=None, *, hfa, clip, prior=None, prior_weight, half_life=None, seed=0, groups=None):
    """Fit the ridge model, cross-validating the penalty unless one is given.

    Rates teams only, whether or not df has a passer column; fit_passers rates
    passers too.

    Home field is not fitted: `hfa` is taken off the home offence's target and
    added to the away offence's before the fit, so the ratings come out as
    neutral-site ratings. Pass 0 to leave home field out; `clip` of 0 or None
    fits raw PPA.

    With a prior, the penalty pulls each team toward its previous-season
    rating instead of toward league average. Minimising
    ||y - X.b||^2 + alpha*||b - b0||^2 is ordinary ridge on the residual
    target y - X.b0, with b0 added back afterwards.

    With groups (a Groups), the teams' prior is their gap from their group,
    and each team's rating comes back with its group's part added in. The
    penalty cannot be cross-validated then, so alpha must be given.

    Returns the ratings, the penalty used and the intercept.
    """
    ratings, _, alpha, intercept = _fit(df, alpha, hfa, clip, prior, prior_weight, half_life, seed, None, groups)
    return ratings, alpha, intercept


def fit_passers(df, alpha=None, *, hfa, clip, prior=None, prior_weight, half_life=None, seed=0,
                passer_prior=None, passer_default=0.0, groups=None):
    """fit, with a column for every passer in df.passer as well.

    passer_prior maps a passer to the rating to shrink him toward, already
    weighted by the caller; a passer missing from it is shrunk toward
    passer_default. groups is as for fit. Returns the team ratings, the
    passer ratings (a Series, epa per pass play on top of the team's
    pass_off), the penalty and the intercept.
    """
    passers = sorted(df.passer[df.play_kind != "rush"].dropna().unique())
    means = pd.Series(passer_default, index=passers, dtype=float)
    if passer_prior is not None:
        known = means.index.intersection(passer_prior.index)
        means[known] = passer_prior[known]
    return _fit(df, alpha, hfa, clip, prior, prior_weight, half_life, seed, means, groups)


def _fit(df, alpha, hfa, clip, prior, prior_weight, half_life, seed, passer_means, groups=None):
    teams = set(df.offense) | set(df.defense)
    if groups is not None:
        teams |= set(groups.members)
    teams = sorted(teams)
    passers = () if passer_means is None else list(passer_means.index)
    X, index = build_design(df, teams, passers)
    n_team_cols = 4 * len(teams)
    names = []
    if groups is not None:
        if alpha is None:
            raise ValueError("a fit with groups needs its penalty given")
        names = sorted({groups.members[t] for t in teams if t in groups.members})
        X = sparse.hstack([X, group_design(df, groups.members, names)]).tocsr()
    group_start = n_team_cols + len(passers)
    raw = df.ppa.to_numpy(dtype=float)
    y = np.clip(raw, -clip, clip) if clip else raw
    log.info("design matrix %d plays x %d columns (%d teams, %d passers, %d groups)", X.shape[0], X.shape[1],
             len(teams), len(passers), len(names))

    # Shrinking toward b0 is ridge on the residual, with b0 restored after.
    beta0 = prior_vector(prior, teams, index, prior_weight, X.shape[1])
    if len(passers):
        beta0[n_team_cols:group_start] = passer_means.to_numpy()
    if names and groups.prior is not None:
        known = groups.prior.reindex(names).fillna(0.0)
        beta0[group_start:] = known[list(UNITS)].to_numpy().ravel()
    target = y - hfa * home_field_column(df) - X @ beta0

    weights = decay_weights(df.week, half_life) if half_life else None
    if alpha is None:
        alpha = pick_alpha(X, target, df.game_id.to_numpy(), seed, weights)
    # Ridge takes one penalty. Scaling the group columns by sqrt(alpha /
    # groups.alpha) and their coefficients back by the same factor gives them
    # groups.alpha instead, and leaves the rest of the fit as it was.
    scale = np.ones(X.shape[1])
    if names:
        scale[group_start:] = np.sqrt(alpha / groups.alpha)
    fitted_on = X @ sparse.diags(scale) if names else X
    model = Ridge(alpha=alpha, solver="sparse_cg", random_state=seed)
    model.fit(fitted_on, target, sample_weight=weights)
    coef = model.coef_ * scale + beta0
    # The intercept is league-average PPA. Predictions need it: the team
    # coefficients are deviations from average and sum to roughly zero.
    # Clipping pulls the average in, so put back what it took off to keep
    # predictions on the raw PPA scale.
    intercept = float(model.intercept_ + np.average(raw - y, weights=weights))
    log.info("in-sample r2 %.4f | intercept %+.4f | home field %+.4f ppa/play (fixed)",
             model.score(fitted_on, target), intercept, hfa)

    ratings = pd.DataFrame(
        {unit: [coef[index[(t, u)]] for t in teams] for u, unit in enumerate(UNITS)},
        index=teams,
    )
    if names:
        parts = pd.DataFrame(coef[group_start:].reshape(len(names), 4), index=names, columns=list(UNITS))
        log.info("group parts (ppa per play):\n%s", parts.round(4).to_string())
        ratings = ratings + parts.reindex(ratings.index.map(groups.members)).fillna(0.0).to_numpy()
    passer_ratings = pd.Series(coef[n_team_cols:group_start], index=passers, name="rating", dtype=float)
    passer_ratings.index.name = "passer"
    return ratings, passer_ratings, alpha, intercept


def summarize(ratings, df, passers=None):
    """Add play counts and league-mix composites.

    With passer ratings from fit_passers, qb is each offence's passers'
    rating weighted by its pass plays in df, and off counts it with pass_off.
    """
    rush_share = float((df.play_kind == "rush").mean())
    pass_off = ratings.pass_off
    if passers is not None:
        thrown = df[(df.play_kind != "rush") & df.passer.notna()]
        rated = thrown.passer.map(passers)
        ratings["qb"] = rated.groupby(thrown.offense).mean().reindex(ratings.index).fillna(0.0)
        pass_off = pass_off + ratings.qb
    # Blend each team's rush and pass ratings at the league play mix so the
    # composite does not reward a team simply for being pass-heavy.
    ratings["off"] = rush_share * ratings.rush_off + (1 - rush_share) * pass_off
    ratings["def"] = rush_share * ratings.rush_def + (1 - rush_share) * ratings.pass_def
    ratings["net"] = ratings["off"] + ratings["def"]
    ratings["plays_off"] = df.offense.value_counts().reindex(ratings.index).fillna(0).astype(int)
    ratings["plays_def"] = df.defense.value_counts().reindex(ratings.index).fillna(0).astype(int)
    ratings.index.name = "team"
    return ratings.sort_values("net", ascending=False).round(4)
