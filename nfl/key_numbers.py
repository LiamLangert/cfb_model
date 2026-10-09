"""Key numbers: NFL games end on some margins, and some scores, far more often than others.

A predicted margin is an average. The market's line is not: it is the number
each side covers as often as not, the middle outcome, and since games bunch
on a few margins - 3 is 15% of finals since 2015, 7 is 9% - lines bunch on
them too. This gives RAPPA's predictions the same shape, so its line reads
like the market's, and the score shown with it is one a game could end on.

    margins  P(margin = k) is proportional to weight(|k|) times a normal curve
             around RAPPA's predicted margin: the curve, reweighted by how
             often each margin happens. Fitted by maximum likelihood, with a
             ridge penalty pulling each log weight toward 0 (the plain curve).
             On 2015-2026, 3 gets 3.5 times the curve's share, 7 2.2, 14 1.8
             and 6 1.6; 9, 11 and 13 get well under it, and a tie 0.15.
    line     the half point the home side is as likely to beat as to fall
             short of: the margins' median. Favourites win big more often than
             they lose big, so it sits below the average for a favourite of a
             touchdown or more. RAPPA's average of BUF by 8.1 is a line of 6.5.
    scores   both scores at once: weight(home score) times weight(away score)
             times weight(|margin|), times a normal curve around each
             projected score. The score shown is the most likely one whose
             margin is the line, or either side of a half-point line, never
             a tie, and whose total is within TOTAL_WITHIN of the projected
             total.

Walked forward over 2016-2026, each season fitted on the ones before it:

  - The line misses the final margin by 10.029 points a game, the average by
    10.068: the median is the better guess when misses are counted in points.
    nfl.benchmark grades the average all the same.
  - 14.8% of lines are 3, against 14.1% of the closing lines and 5.2% of the
    averages rounded to the half point.
  - Favourites by 8 to 12 beat the average 45.6% of the time and the line
    51.8%.
  - It would not improve win probability: P(margin > 0) from these margins
    predicted the winner a little worse than nfl.benchmark's normal curve
    (log-loss 0.6248 against 0.6243), so that stays.

The score at the line is for reading, not for accuracy, so it agrees with the
line beside it. It misses each team's score by 7.451 points, against 7.457
for the projected ones (nfl.predict_week's, pace and scoring included), and
its total is 1.5 points from the projected one on average. PIT 18.9 at CLE
20.6, a line of CLE by 2.5, shows as 17-20.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

# The margins (home less away) and scores the fits cover. Every NFL margin
# since 2015 is inside them. Of the scores, only Miami's 70 against Denver in
# 2023 is outside, so that game is left out of the score fit.
MARGINS = np.arange(-70, 71)
SCORES = np.arange(66)

# Ridge penalty on each log weight, toward the plain normal curve. It only
# matters for margins and scores that rarely happen.
PENALTY = 0.5

# The score shown is the most likely final at the line within this many
# points of the projected total. Without the limit, it would average 4 points
# under the total: scores skew high, so the likeliest final sits below the
# average one, and 20-17 would show for a quarter of games. 3 keeps it within
# 1.6 points of the total on average, on finals like 23-20 and 27-20.
TOTAL_WITHIN = 3.0

# Each final's |margin|, home score by away score.
GAP = np.abs(SCORES[:, None] - SCORES[None, :])


@dataclass
class Margins:
    """How final margins spread around a predicted one.

    log_weight  each |margin|'s weight, 0 to MARGINS.max(), as a log.
    sd          the normal curve's sd, in points.
    """
    log_weight: np.ndarray
    sd: float

    def probs(self, predicted):
        """Each game's chance of ending on each of MARGINS, one row per game."""
        predicted = np.asarray(predicted, float)
        lp = self.log_weight[np.abs(MARGINS)] - (MARGINS - predicted[:, None]) ** 2 / (2 * self.sd ** 2)
        p = np.exp(lp - lp.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)


@dataclass
class Scores:
    """How the two final scores spread around projected ones.

    log_weight      each score's weight, by SCORES, as a log.
    log_gap_weight  each |margin|'s weight on top of the scores', as a log.
    sd              each normal curve's sd, in points.
    """
    log_weight: np.ndarray
    log_gap_weight: np.ndarray
    sd: float

    def log_grid(self):
        """Every final's log weight, home score by away score."""
        return self.log_weight[:, None] + self.log_weight[None, :] + self.log_gap_weight[GAP]

    def log_curves(self, points):
        """Each game's normal curve around its projected points, one row per game, as logs peaking at 0."""
        lc = -(SCORES - np.asarray(points, float)[:, None]) ** 2 / (2 * self.sd ** 2)
        return lc - lc.max(axis=1, keepdims=True)


def fit_margins(predicted, margin, penalty=PENALTY):
    """Margins fitted on games' predicted and final home margins."""
    predicted, margin = np.asarray(predicted, float), np.asarray(margin, float)
    keep = np.abs(margin) <= MARGINS.max()
    predicted, margin = predicted[keep], margin[keep].astype(int)
    width = MARGINS.max() + 1
    seen = np.bincount(np.abs(margin), minlength=width)
    rows, at = np.arange(len(margin)), margin - MARGINS[0]

    # Minus the log-likelihood, penalised, and its gradient: for each log
    # weight, the games expected on that margin less those seen.
    def loss(x):
        m = Margins(x[:-1], np.exp(x[-1]))
        p = m.probs(predicted)
        ll = np.log(p[rows, at]).sum()
        expected = np.bincount(np.abs(MARGINS), weights=p.sum(axis=0), minlength=width)
        spread = ((MARGINS - predicted[:, None]) / m.sd) ** 2
        d_sd = (((margin - predicted) / m.sd) ** 2).sum() - (p * spread).sum()
        return -ll + penalty * (x[:-1] ** 2).sum(), np.r_[expected - seen + 2 * penalty * x[:-1], -d_sd]

    bounds = [(None, None)] * width + [(np.log(3.0), np.log(40.0))]
    x = optimize.minimize(loss, np.r_[np.zeros(width), np.log(13.0)], jac=True, method="L-BFGS-B",
                          bounds=bounds).x
    return Margins(x[:-1], float(np.exp(x[-1])))


def fit_scores(home_points, away_points, home_score, away_score, penalty=PENALTY):
    """Scores fitted on games' projected and final scores.

    The normaliser and expected counts are matrix products, the curves
    factor out of each final's weight.
    """
    hp, ap = np.asarray(home_points, float), np.asarray(away_points, float)
    h, a = np.asarray(home_score, float), np.asarray(away_score, float)
    keep = (h <= SCORES.max()) & (a <= SCORES.max())
    hp, ap, h, a = hp[keep], ap[keep], h[keep].astype(int), a[keep].astype(int)
    n, rows = len(SCORES), np.arange(len(h))
    seen = np.bincount(h, minlength=n) + np.bincount(a, minlength=n)
    seen_gap = np.bincount(np.abs(h - a), minlength=n)

    def loss(x):
        m = Scores(x[:n], x[n:2 * n], np.exp(x[-1]))
        lh, la, lw = m.log_curves(hp), m.log_curves(ap), m.log_grid()
        H, A, W = np.exp(lh), np.exp(la), np.exp(lw)
        z = ((H @ W) * A).sum(axis=1)
        ll = (lh[rows, h] + lw[h, a] + la[rows, a] - np.log(z)).sum()
        # Each final's expected count, and each game's chances for one side's score alone.
        cells = ((H / z[:, None]).T @ A) * W
        expected_gap = np.bincount(GAP.ravel(), weights=cells.ravel(), minlength=n)
        home_p, away_p = H * (A @ W.T) / z[:, None], A * (H @ W) / z[:, None]
        spread_h = ((SCORES - hp[:, None]) / m.sd) ** 2
        spread_a = ((SCORES - ap[:, None]) / m.sd) ** 2
        d_sd = ((((h - hp) / m.sd) ** 2 + ((a - ap) / m.sd) ** 2).sum()
                - (home_p * spread_h).sum() - (away_p * spread_a).sum())
        grad = np.r_[cells.sum(axis=1) + cells.sum(axis=0) - seen + 2 * penalty * x[:n],
                     expected_gap - seen_gap + 2 * penalty * x[n:2 * n], -d_sd]
        return -ll + penalty * (x[:2 * n] ** 2).sum(), grad

    bounds = [(None, None)] * (2 * n) + [(np.log(3.0), np.log(40.0))]
    x = optimize.minimize(loss, np.r_[np.zeros(2 * n), np.log(9.0)], jac=True, method="L-BFGS-B",
                          bounds=bounds).x
    return Scores(x[:n], x[n:2 * n], float(np.exp(x[-1])))


def lines(probs):
    """Each game's home line, to the half point: the one it is as likely to beat as to fall short of."""
    at_most = np.cumsum(probs, axis=1)
    # Whole-point lines first, then the half point above each: a push is neither side.
    under = np.concatenate([at_most - probs, at_most], axis=1)
    over = np.concatenate([1 - at_most, 1 - at_most], axis=1)
    return np.r_[MARGINS, MARGINS + 0.5][np.abs(over - under).argmin(axis=1)]


def at_spread(probs, spread):
    """Each game's chance the home side covers spread (nflverse's spread_line, the home margin), and of a push."""
    spread = np.asarray(spread, float)[:, None]
    covers = np.where(np.isnan(spread[:, 0]), np.nan, (probs * (MARGINS > spread)).sum(axis=1))
    push = np.where(np.isnan(spread[:, 0]), np.nan, (probs * (MARGINS == spread)).sum(axis=1))
    return covers, push


def score_at_line(scores, home_points, away_points, line, within=TOTAL_WITHIN):
    """Each game's most likely final with its margin at line: home scores, then away scores.

    The margin is the line itself or, at a half point, either side of it. It
    is never a tie: a line of 0 takes a one-point win either way. The total
    is within `within` points of the projected one, when a final at the line
    can be.
    """
    margin, total = SCORES[:, None] - SCORES[None, :], SCORES[:, None] + SCORES[None, :]
    grid = scores.log_grid()
    hp, ap = np.asarray(home_points, float), np.asarray(away_points, float)
    lh, la = scores.log_curves(hp), scores.log_curves(ap)
    home, away = [], []
    for ch, ca, at, projected in zip(lh, la, np.asarray(line, float), hp + ap):
        near = (np.abs(margin - at) <= 0.5) & (margin != 0) if at else np.abs(margin) == 1
        close = near & (np.abs(total - projected) <= within)
        lp = np.where(close if close.any() else near, ch[:, None] + grid + ca[None, :], -np.inf)
        i, j = np.unravel_index(lp.argmax(), lp.shape)
        home.append(int(SCORES[i]))
        away.append(int(SCORES[j]))
    return np.array(home), np.array(away)
