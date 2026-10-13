"""Special-teams ratings for either league: what each team's punt and kickoff units are worth, in points a game.

RAPPA's scrimmage ratings only see runs and passes. This rates the kicking
game: every punt and kickoff is valued in points, a ridge fit splits that
between the kicking and the returning unit, and the predictions add the gap
between the two teams straight onto the margin (WEIGHT).

Each kick is valued the same way in both leagues, from the play-by-play
alone, since CFBD gives kicks no PPA:

    EP      expected points of a first down, by yards to goal, fitted on the
            season's own plays: the average of the next score in the half,
            for the side with the ball minus the other (expected_points).
    value   of a kick to the kicking team: whatever was scored before the
            next snap, plus the expected points of that snap - negative
            when the receiving team has it. Clipped to +/-CLIP in the fit.
    over    the value against what a kick from that spot usually earns
            (a smooth fit on yards to goal, per kind), so the rating is
            the unit and not the field position it was handed.

Onside kicks are left out: the score calls them, not how good a team is.
So are field goals and extra points: in the NFL they added nothing on
2015-2022 and hurt on 2023-2026, since a kicker's accuracy does not hold up
well enough to predict, and in college made field goals over a make-rate
curve did not persist from game to game.

The two curves are league-wide and fitted on the whole season, so a week's
ratings use a curve that has seen later weeks. One game's share of tens of
thousands of first downs is too small to matter, and the team ratings
themselves are walked forward (each league's predict_games).

    units   four per team - punt (punter and coverage), punt_return, kickoff
            (kicker and coverage) and kick_return. Each kick is the kicking
            unit against the returning one, and punts and kickoffs get an
            intercept each, since their average value differs.
    value   each unit's rating times how many kicks a game a team's unit is
            on the field for, summed: points a game against an average team,
            so it goes onto the margin at face value.

Each league's module (research.cfb_special_teams, nfl.special_teams) turns its
play-by-play into the frame value_kicks reads, in CFBD's play types.

How it was measured, walking forward on game margins:

  - NFL, on 2015-2022 and checked on 2023-2026 with nflverse's epa as the
    value: punts carried almost all of it (-0.039 +/- 0.013 MAE on
    2015-2022, better in 10 of 11 seasons), kickoffs a little. ALPHA, CLIP
    and WEIGHT were chosen there. Fitting WEIGHT as a feature of the points
    conversion was worse: it swung between 0.3 and 2.4 from season to
    season. Shrinking toward last season's ratings changed MAE by under
    0.005, so each season stands alone.
  - College, valued as here: -0.066 +/- 0.027 MAE on 2023-2024 and on
    2025-2026, twice the NFL's gain, since college kicking units differ
    more (0.74 points a game, sd, in 2025). The NFL's constants held up.
  - NFL valued as here instead of by nflverse's epa (2026-10-06): level,
    MAE +0.002 +/- 0.005 and log-loss +0.0004 +/- 0.0003 over 2015-2026,
    worse in 5 of 12 seasons, while special teams as a whole are worth
    0.031 +/- 0.013. Team values correlate 0.91 with the epa ones but are
    less spread (sd 0.34 points a game against 0.48 in 2024); WEIGHT 1.2-1.4
    was no better. Both leagues now value kicks the one way.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

UNITS = ("punt", "punt_return", "kickoff", "kick_return")

# Ridge penalty. Ratings from a handful of punts are mostly noise, but
# 100-1000 shrank the signal away; 10 did better still on 2015-2022 in the
# NFL and lost it on 2023-2026.
ALPHA = 30.0

# Each kick's value is clipped to +/-CLIP. Return touchdowns and blocked
# punts reach +/-7; 2 predicted best, and a harder clip did better only on
# 2015-2022.
CLIP = 2.0

# Points of margin per point of special-teams value gap: at face value.
WEIGHT = 1.0

# Degree of the polynomial fits: EP on yards to goal, and a kick's usual
# value on the spot it is kicked from.
EP_DEGREE = 3
SPOT_DEGREE = 2

# Play types, in CFBD's names, which each league's loader maps its own onto.
PUNTS = {"Punt", "Punt Return", "Punt Return Touchdown", "Blocked Punt", "Blocked Punt Touchdown"}
KICKOFFS = {"Kickoff", "Kickoff Return (Offense)", "Kickoff Return Touchdown"}

# Not a snap: rows the next snap after a kick is looked for past.
NOT_SNAPS = PUNTS | KICKOFFS | {"Timeout", "End Period", "End of Half", "End of Game", "End of Regulation",
                                "Uncategorized", "placeholder", "Extra Point Good", "Extra Point Missed",
                                "Two Point Pass", "Two Point Rush", "Defensive 2pt Conversion"}


def expected_points(plays):
    """EP of a first down as a function of yards to goal, from the next score in the same half.

    A possession is a run of snaps by one offence. Its points are the change
    in its score less the change in the other side's, up to the next
    possession's first snap (the next score before halftime, for the last
    one). Every first down then looks ahead to the first possession that
    scored, counting it for or against itself; nothing before halftime
    counts zero.
    """
    snaps = plays[plays.down.between(1, 4) & ~plays.play_type.isin(NOT_SNAPS) & plays.period.between(1, 4)]
    half = np.where(snaps.period <= 2, 1, 2)
    new = ((snaps.offense != snaps.offense.shift()) | (snaps.game_id != snaps.game_id.shift())).to_numpy()
    new = new | (half != np.roll(half, 1))
    poss = pd.Series(np.cumsum(new), index=snaps.index)
    first = snaps.groupby(poss).head(1)
    pos = pd.DataFrame({"game_id": first.game_id.to_numpy(), "half": half[new], "offense": first.offense.to_numpy(),
                        "home": first.home.to_numpy(), "h": first.h.to_numpy(), "a": first.a.to_numpy()})
    # The score at the next possession's first snap, or at the end of the half.
    last = snaps.assign(half=half).groupby(["game_id", "half"])[["h", "a"]].max()
    same = (pos.game_id == pos.game_id.shift(-1)) & (pos.half == pos.half.shift(-1))
    end_h = np.where(same, pos.h.shift(-1), last.h.reindex(pd.MultiIndex.from_frame(pos[["game_id", "half"]])).to_numpy())
    end_a = np.where(same, pos.a.shift(-1), last.a.reindex(pd.MultiIndex.from_frame(pos[["game_id", "half"]])).to_numpy())
    sign = np.where(pos.offense == pos.home, 1.0, -1.0)
    pos["points"] = sign * ((end_h - pos.h) - (end_a - pos.a))

    # For each possession, the next one in its half that scored, and whose it was.
    pos["group"] = pos.game_id.astype(str) + "/" + pos.half.astype(str)
    scored = pos.points.ne(0)
    nxt = pd.Series(np.where(scored, pos.index, np.nan), index=pos.index).groupby(pos.group).bfill()
    has = nxt.notna()
    j = nxt[has].astype(int).to_numpy()
    same_side = pos.offense[has].to_numpy() == pos.offense.to_numpy()[j]
    pos["next_score"] = 0.0
    pos.loc[has, "next_score"] = np.where(same_side, 1.0, -1.0) * pos.points.to_numpy()[j]

    firsts = snaps[snaps.down == 1]
    ns = poss[firsts.index].map(pd.Series(pos.next_score.to_numpy(), index=np.arange(1, len(pos) + 1)))
    ytg = firsts.yards_to_goal.clip(1, 99)
    curve = np.polynomial.Polynomial.fit(ytg, ns, EP_DEGREE)
    log.info("EP of a first down: own 25 %+.2f, midfield %+.2f, opponent 25 %+.2f (%d first downs)",
             curve(75), curve(50), curve(25), len(firsts))
    return curve


def value_kicks(plays):
    """Every punt and kickoff from the kicking team's side, valued against its spot.

    plays is a season's play-by-play in game order (index 0..n-1), with:

        game_id, week
        period          1-4, then overtime
        down            1-4 on a snap, missing otherwise (kickoffs, timeouts)
        offense         the team with the ball; on a kickoff, the kicking team
        defense         the other team
        home            the home team
        yards_to_goal   from the offence's side; on a kickoff, the kicking team's
        play_type       in CFBD's names (PUNTS, KICKOFFS, NOT_SNAPS and the rest)
        play_text       for spotting onside kicks
        h, a            the home and away score at the snap

    Returns one row per kick: game_id, week, kind ("punt" or "kickoff"),
    kicking, receiving, epa (the value against the spot's usual, which fit
    rates), value and spot.
    """
    ep = expected_points(plays)

    # The next snap after every row, in the same game and half.
    half = np.where(plays.period <= 2, 1, np.where(plays.period <= 4, 2, 3))
    snap = plays.down.between(1, 4) & ~plays.play_type.isin(NOT_SNAPS)
    at = pd.Series(np.where(snap, plays.index, np.nan), index=plays.index)
    keys = [plays.game_id, half]
    nxt = at.groupby(keys).shift(-1).groupby(keys).bfill()

    punt = plays.play_type.isin(PUNTS)
    kickoff = plays.play_type.isin(KICKOFFS) & ~plays.play_text.str.contains("onside", case=False, na=False)
    kicks = plays[(punt | kickoff) & plays.offense.notna() & plays.defense.notna()]
    j = nxt[kicks.index]
    ok = j.notna().to_numpy()
    kicks, j = kicks[ok], j[ok].astype(int).to_numpy()
    after = plays.loc[j]
    kicker_home = (kicks.offense == kicks.home).to_numpy()
    # Points scored before the next snap, from the kicker's side, then that snap's EP.
    dh, da = after.h.to_numpy() - kicks.h.to_numpy(), after.a.to_numpy() - kicks.a.to_numpy()
    scored = np.where(kicker_home, dh - da, da - dh)
    snap_ep = ep(after.yards_to_goal.clip(1, 99).to_numpy())
    has_ball = after.offense.to_numpy() == kicks.offense.to_numpy()
    value = scored + np.where(has_ball, snap_ep, -snap_ep)

    kind = np.where(kicks.play_type.isin(PUNTS), "punt", "kickoff")
    spot = kicks.yards_to_goal.clip(1, 99).to_numpy()
    over = np.empty(len(kicks))
    for k in ("punt", "kickoff"):
        m = kind == k
        clipped = np.clip(value[m], -CLIP, CLIP)
        usual = np.polynomial.Polynomial.fit(spot[m], clipped, SPOT_DEGREE)
        over[m] = value[m] - usual(spot[m])
    return pd.DataFrame({
        "game_id": kicks.game_id.to_numpy(),
        "week": kicks.week.to_numpy(dtype=float),
        "kind": kind,
        "kicking": kicks.offense.to_numpy(),
        "receiving": kicks.defense.to_numpy(),
        "epa": over,
        "value": value,
        "spot": spot,
    })


def fit(kicks, alpha=ALPHA, clip=CLIP):
    """Each team's four unit ratings (points per kick) and its value (points per game).

    Ridge with an unpenalised intercept per kind of kick, solved in closed
    form: there are only 4 x teams + 1 columns.
    """
    teams = sorted(set(kicks.kicking) | set(kicks.receiving))
    col = {t: i for i, t in enumerate(teams)}
    n, p = len(kicks), 4 * len(teams) + 1
    punt = (kicks.kind == "punt").to_numpy()
    X = np.zeros((n, p))
    rows = np.arange(n)
    X[rows, kicks.kicking.map(col).to_numpy() * 4 + np.where(punt, 0, 2)] = 1.0
    X[rows, kicks.receiving.map(col).to_numpy() * 4 + np.where(punt, 1, 3)] = -1.0
    X[:, -1] = punt  # the punt intercept, on top of the kickoff one
    y = kicks.epa.clip(-clip, clip).to_numpy() if clip else kicks.epa.to_numpy()

    Xc, yc = X - X.mean(axis=0), y - y.mean()
    penalty = np.full(p, alpha)
    penalty[-1] = 0.0
    coef = np.linalg.solve(Xc.T @ Xc + np.diag(penalty), Xc.T @ yc)
    ratings = pd.DataFrame(coef[:-1].reshape(len(teams), 4), index=teams, columns=UNITS)

    # How often a team's unit is on the field, per team-game in these kicks.
    team_games = len(pd.concat([kicks[["game_id", "kicking"]].set_axis(["g", "t"], axis=1),
                                kicks[["game_id", "receiving"]].set_axis(["g", "t"], axis=1)]).drop_duplicates())
    per_game = np.array([punt.sum(), punt.sum(), (~punt).sum(), (~punt).sum()]) / team_games
    ratings["value"] = ratings[list(UNITS)].to_numpy() @ per_game
    ratings.index.name = "team"
    return ratings
