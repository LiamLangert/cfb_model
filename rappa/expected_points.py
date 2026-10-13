"""Expected points by down, distance and yards to goal, and every snap's value in them, for either league.

RAPPA rates each play by how many points it was worth. Rather than take that
from the data provider's own model, it is measured here, from the season's
own plays, so that a game's play values add up to its points:

    EP      a snap's expected points: the average next score in the half,
            for the side with the ball minus the other, as a smooth function
            of down, distance and yards to goal (features), fitted on every
            snap of the season (fit).
    value   of a snap to its offence: the points scored before the next node,
            plus that node's EP (negative when the other side has the ball),
            less this snap's EP (play_values). A node is a snap, a punt or a
            kickoff. A scoring play's next node is the kickoff after it, which
            counts zero: the kicking team's expected give-back of about 0.7
            points is left out, as predicting with it was no better. A half's
            last snap looks ahead to the end of the half, which counts zero
            too: its value is the points from it to the end of the half, less
            its EP.

Each league's loader turns its play-by-play into the frame these read, the
same frame rappa.special_teams takes, plus each snap's distance:

    game_id, period, down (1-4 on a snap, else missing), distance,
    yards_to_goal, offense, home, play_type (CFBD's names), h, a (the home and
    away score at the snap), h_after, a_after (and after the play)

The EP curve is fitted on the whole season, so a week's play values use a
curve that has seen later weeks. It is league-wide, not a team's: valuing
each college season's plays with the season before's curve instead predicted
the same (MAE -0.043 against -0.044 with this season's, on the version with
the give-back; values correlate 0.998).

Measured on college, walking forward over 2015-2026 with the chain re-rated
(2026-10-06): against CFBD's ppa as the play value, MAE -0.061 +/- 0.023
and log-loss -0.0014 +/- 0.0010, better in 10 of 12 seasons and in both
halves. CFBD's ppa correlates 0.85-0.93 with these values; it charges about
0.5 points more for a turnover, values third and fourth downs differently,
and leaves out the kickoff after a score. With the give-back counted, MAE
was -0.044 +/- 0.023.

Until 2026-10-09 a half's last snap had no value here, and each league's
loader fell back to the provider's (CFBD's ppa, nflverse's epa). Valuing it
against the end of the half instead was worth log-loss -0.0004 +/- 0.0003
in college's benchmark over 2015-2026 (better in 8 of 12 seasons; MAE
level) and -0.0004 +/- 0.0002 in the NFL's (9 of 12; MAE -0.005), and
-0.0010 +/- 0.0008 on the 2025-2026 college games Kalshi priced. Putting the time
left in the half into the EP curve, so that a drive reaching the opponent's
30 with seconds left is worth what it usually scores, did worse: log-loss
+0.0013 +/- 0.0003 (worse in 10 of 12 seasons), as late yards still say how
good an offence is and whether the drive scores in time says little more.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rappa.special_teams import KICKOFFS, NOT_SNAPS, PUNTS

log = logging.getLogger(__name__)


def next_scores(plays):
    """Each snap's possession's next score in its half, from its offence's side (0 if none before halftime).

    A possession is a run of snaps by one offence. Its points are the change
    in its score less the change in the other side's, up to the next
    possession's first snap (the end of the half, for the last one). A snap
    looks ahead to the first possession that scored, counting it for or
    against itself. Returns a Series on the snaps' index.
    """
    snaps = plays[plays.down.between(1, 4) & ~plays.play_type.isin(NOT_SNAPS) & plays.period.between(1, 4)]
    half = np.where(snaps.period <= 2, 1, 2)
    new = ((snaps.offense != snaps.offense.shift()) | (snaps.game_id != snaps.game_id.shift())).to_numpy()
    new = new | (half != np.roll(half, 1))
    poss = pd.Series(np.cumsum(new), index=snaps.index)
    first = snaps.groupby(poss).head(1)
    pos = pd.DataFrame({"game_id": first.game_id.to_numpy(), "half": half[new], "offense": first.offense.to_numpy(),
                        "home": first.home.to_numpy(), "h": first.h.to_numpy(), "a": first.a.to_numpy()})
    last = snaps.assign(half=half).groupby(["game_id", "half"])[["h", "a"]].max()
    same = (pos.game_id == pos.game_id.shift(-1)) & (pos.half == pos.half.shift(-1))
    key = pd.MultiIndex.from_frame(pos[["game_id", "half"]])
    end_h = np.where(same, pos.h.shift(-1), last.h.reindex(key).to_numpy())
    end_a = np.where(same, pos.a.shift(-1), last.a.reindex(key).to_numpy())
    sign = np.where(pos.offense == pos.home, 1.0, -1.0)
    pos["points"] = sign * ((end_h - pos.h) - (end_a - pos.a))
    pos["group"] = pos.game_id.astype(str) + "/" + pos.half.astype(str)
    nxt = pd.Series(np.where(pos.points.ne(0), pos.index, np.nan), index=pos.index).groupby(pos.group).bfill()
    has = nxt.notna()
    j = nxt[has].astype(int).to_numpy()
    same_side = pos.offense[has].to_numpy() == pos.offense.to_numpy()[j]
    pos["next_score"] = 0.0
    pos.loc[has, "next_score"] = np.where(same_side, 1.0, -1.0) * pos.points.to_numpy()[j]
    return poss.map(pd.Series(pos.next_score.to_numpy(), index=np.arange(1, len(pos) + 1)))


def features(down, distance, yards_to_goal):
    """The EP regression's columns: a cubic in yards to goal, log distance, goal to go, and each down's shift."""
    x = np.clip(np.asarray(yards_to_goal, float), 1, 99) / 100
    d = np.asarray(down, float)
    dist = np.asarray(distance, float)
    L = np.log(np.clip(dist, 1, 30))
    goal = (dist >= np.asarray(yards_to_goal, float)).astype(float)
    cols = [np.ones_like(x), x, x**2, x**3, L, goal]
    for k in (2, 3, 4):
        dk = (d == k).astype(float)
        cols += [dk, dk * L, dk * x]
    return np.column_stack(cols)


class ExpectedPoints:
    """A fitted EP curve: call it with down, distance and yards to goal (arrays) for each snap's EP."""

    def __init__(self, beta):
        self.beta = beta

    def __call__(self, down, distance, yards_to_goal):
        return features(down, distance, yards_to_goal) @ self.beta


def fit(plays):
    """The EP curve, fitted by least squares on every snap's next score."""
    ns = next_scores(plays)
    s = plays.loc[ns.index]
    ok = s.distance.notna() & s.yards_to_goal.notna()
    beta, *_ = np.linalg.lstsq(features(s.down[ok], s.distance[ok], s.yards_to_goal[ok]), ns[ok].to_numpy(),
                               rcond=None)
    ep = ExpectedPoints(beta)
    log.info("EP: 1st & 10 at own 25 %+.2f, midfield %+.2f, opponent 25 %+.2f; 3rd & 1 / 3rd & 10 at opponent 25 "
             "%+.2f / %+.2f (%d snaps)", *ep(np.array([1, 1, 1, 3, 3]), np.array([10, 10, 10, 1, 10]),
                                              np.array([75, 50, 25, 25, 25])), int(ok.sum()))
    return ep


def node_values(plays, ep=None):
    """Every node's value to its side in points, on plays' index; NaN off the nodes.

    A node is a snap of any kind (penalties and field goals too), a punt or a
    kickoff, and its side is the row's offence: for a kickoff, the kicking
    team, as CFBD has it and nfl.special_teams.plays_frame copies. A kickoff's
    own EP counts zero, as the scoring play before it sees it, so its value is
    the points before the next node plus that node's EP from the kicking side:
    what the kick and its return earned. In each half the nodes' values add up
    to the change in the score. plays must be in game order; ep is a fitted
    curve, fitted on plays when None.
    """
    ep = fit(plays) if ep is None else ep
    down = plays.down.where(plays.down.between(1, 4))
    snap = down.notna() & ~plays.play_type.isin(NOT_SNAPS) & plays.offense.notna()
    node = (snap | plays.play_type.isin(KICKOFFS) | plays.play_type.isin(PUNTS)) & plays.offense.notna()
    node &= plays.period.between(1, 5)
    half = pd.Series(np.where(plays.period <= 2, 1, np.where(plays.period <= 4, 2, 3)), index=plays.index)
    # The score when each half (overtime's first period the third) ends.
    timed = plays.period.between(1, 5)
    end = plays[timed].assign(half=half[timed]).groupby(["game_id", "half"])[["h_after", "a_after"]].max()
    n = plays[node].assign(down=down[node], half=half[node])
    kickoff = n.play_type.isin(KICKOFFS)
    n = n.assign(ep=np.where(kickoff, 0.0, ep(n.down.fillna(4), n.distance.fillna(10), n.yards_to_goal)))
    by = n.groupby(["game_id", "half"])
    nxt = {c: by[c].shift(-1) for c in ("ep", "offense", "h", "a")}
    # A half's last node looks ahead to the half's end: the score then, and nothing more to come.
    last = nxt["offense"].isna().to_numpy()
    at_end = end.reindex(pd.MultiIndex.from_arrays([n.game_id, n.half]))
    nxt["h"] = nxt["h"].where(~last, at_end.h_after.to_numpy())
    nxt["a"] = nxt["a"].where(~last, at_end.a_after.to_numpy())
    nxt["ep"] = nxt["ep"].where(~last, 0.0)
    nxt["offense"] = nxt["offense"].where(~last, n.offense)
    home_off = (n.offense == n.home).to_numpy()
    dh, da = (nxt["h"] - n.h).to_numpy(), (nxt["a"] - n.a).to_numpy()
    points = np.where(home_off, dh - da, da - dh)
    same = (nxt["offense"] == n.offense).to_numpy()
    then = np.where(same, nxt["ep"], -nxt["ep"])
    return pd.Series(points + then - n.ep.to_numpy(), index=n.index).reindex(plays.index)


def play_values(plays, ep=None):
    """Each snap's value to its offence in points (see the module docstring), on plays' index; NaN where it has none.

    node_values on the snaps: a scoring play's next node, the kickoff, counts zero. plays must be in game
    order. ep is a fitted curve, fitted on plays when None.
    """
    down = plays.down.where(plays.down.between(1, 4))
    return node_values(plays, ep).where(down.notna() & ~plays.play_type.isin(KICKOFFS))
