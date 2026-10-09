"""What changes over an NFL offseason, as shifts of each team's prior: how much of last season's lineup returns.

A season's prior is last season's ratings (nfl.predict_games), which know
nothing of who left or arrived. Each team's units are shifted by how much of
last season's regular-season snaps were played by men on its week-1 roster:

    offence   OFF_RETURNING_SHIFT per SD of the offence's returning share,
              quarterbacks left out, since the quarterback term already
              follows the quarterback.
    defence   DEF_RETURNING_SHIFT per SD of the defence's.

Snap counts are nflverse's (snaps.csv, from 2012) and rosters its weekly
ones, the first week of the regular season (roster_week1.csv); nfl/pull.py
fetches both. A player is followed by his Pro Football Reference id, or,
where the roster has none (every offensive lineman), by his name and
position group when that is unique. Players listed as cut or retired are
gone. Each SD is over the season's 32 teams; a season without both files is
not shifted.

The shift fades: whole for week 2 and before, then less by a seventh each
week until it is gone from FADE_WEEK on (faded). At full strength all
season, it helped weeks 2-8 but cost 0.03-0.09 points a game from week 9
on, as continuity is a stand-in for what the season's own plays then show,
and it netted nothing.

Walked forward on 2015-2026 (2014 the warm-up) with the fade (2026-10-07):
MAE -0.024 +/- 0.009 and log-loss -0.0010 +/- 0.0005, better in 9 of 12
seasons, and in weeks 2-4 MAE -0.076 and log-loss -0.0047. Twice the size
cut MAE by 0.036 +/- 0.017 with twice the noise, and offence alone 0.023 +/-
0.011. Transferred snaps and a new head coach added nothing. The sizes and
the fade were picked among about ten variants on these seasons, the fade
after seeing the late-season harm, so the gain is likely smaller than
measured.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rappa.model import UNITS

log = logging.getLogger(__name__)

OFF_RETURNING_SHIFT = 0.01
DEF_RETURNING_SHIFT = 0.01

# The first week the shift is gone entirely; it is whole up to week 2.
FADE_WEEK = 9

OFFENCE = ["rush_off", "pass_off"]
DEFENCE = ["rush_def", "pass_def"]

# Snap counts and rosters keep a relocated team's old abbreviation, and some years spell a few differently.
TEAMS = {"STL": "LA", "SD": "LAC", "OAK": "LV", "LAR": "LA", "JAC": "JAX"}

POSITION_GROUP = {
    "C": "OL", "G": "OL", "T": "OL", "OL": "OL", "OT": "OL", "OG": "OL", "QB": "QB", "RB": "RB", "FB": "RB",
    "HB": "RB", "WR": "WR", "TE": "TE", "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "LB": "LB", "ILB": "LB",
    "OLB": "LB", "MLB": "LB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "SAF": "DB",
}

# Roster statuses of players no longer with the team.
GONE = {"CUT", "RET"}


def name_key(names):
    """Names for matching: lower case, letters only, without a Jr./Sr./II-V suffix."""
    n = names.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    n = n.str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True)
    return n.str.replace(r"\s+", " ", regex=True).str.strip()


def returning(season, out):
    """Each team's share of last season's regular-season snaps played by its week-1 roster, offence and defence.

    One row per team, with columns offence (quarterbacks left out) and defence, or None without the files.
    """
    snaps_path, roster_path = out / str(season - 1) / "snaps.csv", out / str(season) / "roster_week1.csv"
    if not snaps_path.exists() or not roster_path.exists():
        log.warning("%d: no %s or %s, no returning-snap shifts - run `uv run nfl/pull.py %d %d`",
                    season, snaps_path, roster_path, season - 1, season)
        return None
    snaps = pd.read_csv(snaps_path)
    snaps = snaps[snaps.game_type == "REG"].assign(team=lambda d: d.team.replace(TEAMS))
    roster = pd.read_csv(roster_path, low_memory=False)
    roster = roster[~roster.status.isin(GONE)]
    roster = roster.assign(team=roster.team.replace(TEAMS), group=roster.position.map(POSITION_GROUP),
                           key=name_key(roster.full_name.fillna("")))
    by_id = roster.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id").team
    no_id = roster[roster.pfr_id.isna()]
    by_name = no_id.groupby(["key", "group"]).team.agg(lambda t: t.iloc[0] if t.nunique() == 1 else None)

    players = snaps.groupby(["pfr_player_id", "player", "team", "position"])[["offense_snaps", "defense_snaps"]]
    players = players.sum().reset_index()
    players["now"] = players.pfr_player_id.map(by_id)
    miss = players.now.isna()
    fallback = pd.MultiIndex.from_arrays([name_key(players.player[miss]), players.position[miss].map(POSITION_GROUP)])
    players.loc[miss, "now"] = by_name.reindex(fallback).to_numpy()

    shares = {}
    for side, col in (("offence", "offense_snaps"), ("defence", "defense_snaps")):
        p = players[players.position != "QB"] if side == "offence" else players
        total = p.groupby("team")[col].sum()
        stayed = p[p.now == p.team].groupby("team")[col].sum().reindex(total.index).fillna(0.0)
        shares[side] = stayed / total
    return pd.DataFrame(shares).rename_axis("team")


def zscore(x):
    """x in SDs across teams; zero everywhere when there are too few teams or no spread."""
    if x.notna().sum() <= 10 or not x.std() > 0:
        return pd.Series(0.0, index=x.index)
    return ((x - x.mean()) / x.std()).fillna(0.0)


def prior_shift(season, out):
    """Each team's shift, in epa per play, of its four units' prior at full strength, or None without the data.

    One row per team, a column per unit; nfl.ratings.fit_qbs adds it (faded) to the weighted prior.
    """
    share = returning(season, out)
    if share is None:
        return None
    shift = pd.DataFrame(0.0, index=share.index, columns=list(UNITS))
    off, dfn = OFF_RETURNING_SHIFT * zscore(share.offence), DEF_RETURNING_SHIFT * zscore(share.defence)
    for unit in OFFENCE:
        shift[unit] += off
    for unit in DEFENCE:
        shift[unit] += dfn
    log.info("%d returning snaps: offence %.2f-%.2f, defence %.2f-%.2f of last season's (%d teams)", season,
             share.offence.min(), share.offence.max(), share.defence.min(), share.defence.max(), len(share))
    return shift


def fade(week):
    """How much of the shift is left for a week's predictions: 1 to week 2, then a step less a week, 0 at FADE_WEEK."""
    return float(np.clip((FADE_WEEK - week) / (FADE_WEEK - 2), 0.0, 1.0))


def faded(shift, week):
    """shift as it stands for week: scaled by fade, or None when nothing is left of it (or there is none)."""
    if shift is None or fade(week) == 0:
        return None
    return shift * fade(week)
