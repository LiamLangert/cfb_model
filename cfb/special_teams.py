"""Special-teams ratings for college: what each team's punt and kickoff units are worth, in points a game.

The college counterpart of nfl.special_teams, whose ridge fit (fit) it uses
unchanged. What differs is where each kick's value comes from: CFBD gives
punts and kickoffs no PPA, so it is worked out here.

    EP      expected points of a first down, by yards to goal, fitted on the
            season's own plays: the average of the next score in the half,
            for the side with the ball minus the other (expected_points).
    value   of a kick to the kicking team: whatever was scored before the
            next snap, plus the expected points of that snap - negative
            when the receiving team has it. Clipped to +/-CLIP.
    over    the value against what a kick from that spot usually earns
            (a smooth fit on yards to goal, per kind), so the rating is
            the unit and not the field position it was handed.

Onside kicks are left out, as in the NFL. So are field goals and extra
points (they were noise in the NFL).

The two curves are league-wide and fitted on the whole season, so a week's
ratings use a curve that has seen later weeks; one game's share of 50,000
first downs is too small to matter, and the team ratings themselves are
walked forward as in the NFL.

Walked forward over 2023-2024 and checked on 2025-2026 (see
cfb.predict_games), this cut MAE by 0.066 +/- 0.027 points a game in both,
about twice what it did in the NFL: college kicking units differ more, by
0.74 points a game (sd) in 2025. The NFL's ALPHA, CLIP and WEIGHT held up:
other values moved MAE by under 0.04.

`uv run -m cfb.special_teams <season>` prints a season's ratings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.ratings import FCS, ORDER, attach_game_context, pool_fcs
from nfl.special_teams import UNITS, fit as nfl_fit

log = logging.getLogger(__name__)

PUNTS = {"Punt", "Punt Return", "Punt Return Touchdown", "Blocked Punt", "Blocked Punt Touchdown"}
KICKOFFS = {"Kickoff", "Kickoff Return (Offense)", "Kickoff Return Touchdown"}

# Not a snap: rows the next snap after a kick is looked for past.
NOT_SNAPS = PUNTS | KICKOFFS | {"Timeout", "End Period", "End of Half", "End of Game", "End of Regulation",
                                "Uncategorized", "placeholder", "Extra Point Good", "Extra Point Missed",
                                "Two Point Pass", "Two Point Rush", "Defensive 2pt Conversion"}

# Ridge penalty, clip and weight, as in nfl.special_teams.
ALPHA = 30.0
CLIP = 2.0
WEIGHT = 1.0

# Degree of the polynomial fits: EP on yards to goal, and a kick's usual
# value on the spot it is kicked from.
EP_DEGREE = 3
SPOT_DEGREE = 2


def scores(plays):
    """Each play's home and away score at the snap (CFBD reports them after it; see cfb.ratings)."""
    home_off = plays.offense == plays.home
    home = pd.Series(np.where(home_off, plays.offense_score, plays.defense_score), index=plays.index)
    away = pd.Series(np.where(home_off, plays.defense_score, plays.offense_score), index=plays.index)
    home, away = home.groupby(plays.game_id).cummax(), away.groupby(plays.game_id).cummax()
    pre = pd.DataFrame({"h": home, "a": away}).groupby(plays.game_id).shift(1).fillna(0)
    return pre.h, pre.a


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


def load(season, out):
    """One season's punts and kickoffs, from the kicking team's side, valued against their spot."""
    plays_path, games_path = out / str(season) / "plays.csv", out / str(season) / "games.csv"
    if not plays_path.exists() or not games_path.exists():
        raise SystemExit(f"need {plays_path} and {games_path} - run `uv run -m cfb.pull {season}` first")
    plays, games = pool_fcs(pd.read_csv(plays_path, low_memory=False), pd.read_csv(games_path))
    plays = attach_game_context(plays.sort_values(ORDER).reset_index(drop=True), games)
    plays["h"], plays["a"] = scores(plays)
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
    """nfl.special_teams.fit, with the college constants as defaults."""
    return nfl_fit(kicks, alpha, clip)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season(s) to rate, fitted separately")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"ridge penalty (default: {ALPHA:g})")
    p.add_argument("--clip", type=float, default=CLIP, help=f"clip each kick's value (default: {CLIP:g}; 0 = off)")
    p.add_argument("--top", type=int, default=15, help="how many teams to print at each end (default: 15)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for season in args.seasons:
        kicks = load(season, args.out)
        ratings = fit(kicks, args.alpha, args.clip).drop(index=FCS, errors="ignore")
        ratings = ratings.sort_values("value", ascending=False)
        print(f"\n{season} special teams: {len(kicks)} punts and kickoffs. Units in points per kick, value in "
              f"points per game vs average ({', '.join(UNITS)}):")
        print(pd.concat([ratings.head(args.top), ratings.tail(args.top)]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
