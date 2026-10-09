"""What changes over a college offseason, as shifts of each team's prior: recruiting, returning production, coaches,
and the defence that returns or transfers in.

A season's prior is last season's ratings (cfb.ratings.conference_prior),
which know nothing of who left, who arrived or who is coaching. Three things
CFBD publishes before a season starts move each team's own part of it, in
ppa per play (cfb.pull_offseason fetches them):

    defence   RECRUITING_SHIFT per SD of the team's 4-year recruiting average,
              247's class points for the classes of S-3 to S, the players on
              S's roster. Defences regress toward their recruiting.
    offence   RETURNING_SHIFT per SD of returning non-passing usage: the mean
              of the shares of last season's rushing and receiving plays by
              players still on the roster. Passing is left out, since the
              quarterback term already follows the quarterback.
    offence   NEW_COACH_SHIFT for a new head coach, against the season's share
              of new coaches (so a team that kept its coach is shifted up a
              little).
    defence   DEF_RETURNING_SHIFT per SD of returning defensive production:
              the share of last season's tackles, and of its havoc plays
              (tackles for loss, sacks, passes defended and hurries), made by
              players on this season's roster (defensive_turnover). CFBD
              publishes returning production for offences only, so this is
              built from its rosters and player season stats, which carry
              player ids that follow a transfer.
    defence   DEF_INCOMING_SHIFT per SD of the defensive production its
              transfers brought: the same stats at their old schools, in an
              average team's season.

Each SD is over the season's FBS teams, and a team with no data is not
shifted. They are shifts, not changes to how much carries over: scaling the
carry-over weight by returning production or a coaching change did nothing,
and at the team level none of them changed how strongly last season's gap
predicted this one.

Walked forward on FBS-vs-FBS game margins, 2015-2026 (2014 the warm-up), the
three together cut MAE by 0.14 +/- 0.03 on 2015-2021 and 0.09 +/- 0.03 on
2022-2026, and log-loss by 0.007 and 0.004; better in 11 of 12 seasons (not
2020). The gain is early: 0.23 points a game in weeks 2-4, 0.05 by weeks
13-16. Alone, recruiting was worth 0.05-0.07 in both periods; returning usage
0.06 before 2022 but 0.01 since, as the portal made last season's roster a
weaker guide. The talent composite, net transfer ratings (2021+ only) and the
change in talent added nothing. The sizes were picked on all of 2015-2026 and
moved MAE by under 0.02 over 0.015-0.035 (recruiting), 0.015-0.03 (returning)
and -0.02 to -0.04 (coach).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rappa.model import UNITS

log = logging.getLogger(__name__)

RECRUITING_SHIFT = 0.025
RETURNING_SHIFT = 0.02
NEW_COACH_SHIFT = -0.02

# Returning and incoming defence, measured from rosters and player stats (defensive_turnover), which CFBD has
# from 2016: seasons from 2017 are shifted. Walked forward on FBS-vs-FBS margins over 2015-2026 with the chain
# re-rated, on top of the shifts above and the play values of rappa.expected_points, they cut MAE by 0.042
# +/- 0.013 and log-loss by 0.0019 +/- 0.0006, better in 10 of 12 seasons (2026-10-07); 0.051 +/- 0.016 over
# 2017-2026. The gain runs through the season, the most in weeks 5-8: an experienced defence stays underrated
# well after its first games. Returning defence alone was worth 0.033 +/- 0.011; twice its size (0.04)
# overshot early. Returning offence is in RETURNING_SHIFT; transferred offence added nothing (quarterbacks are
# followed by name, cfb.ratings.qb_prior_means). The sizes were picked among five on these seasons.
DEF_RETURNING_SHIFT = 0.02
DEF_INCOMING_SHIFT = 0.015

# The stats a defender's production is counted in: tackles, and havoc plays.
HAVOC = ["TFL", "SACKS", "PD", "QB HUR"]

OFFENCE = ["rush_off", "pass_off"]
DEFENCE = ["rush_def", "pass_def"]


def zscore(x):
    return ((x - x.mean()) / x.std()).fillna(0.0)


def need(path, season):
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run -m cfb.pull_offseason {season}` first "
                         "(or pass --no-offseason)")
    return path


def recruiting_average(season, out):
    """Each team's mean class points over the classes of season-3 to season (the ones it has)."""
    classes = [pd.read_csv(need(out / str(year) / "recruiting.csv", season)).set_index("team").points.rename(year)
               for year in range(season - 3, season + 1)]
    return pd.concat(classes, axis=1).mean(axis=1)


def head_coaches(out, season):
    """Each (year, school)'s head coach; with an interim, the one who coached most games."""
    c = pd.read_csv(need(out / "coaches.csv", season))
    c = c.assign(coach=c.first_name + " " + c.last_name)
    return (c.sort_values("games", kind="stable").drop_duplicates(["year", "school"], keep="last")
            .set_index(["year", "school"]).coach)


def features(season, out):
    """The season's FBS teams (those CFBD has returning production for) and the three measures."""
    returning = pd.read_csv(need(out / str(season) / "returning.csv", season)).set_index("team")
    teams = returning.index
    coaches = head_coaches(out, season)
    if season - 1 not in coaches.index.get_level_values("year"):
        raise SystemExit(f"coaches.csv has no {season - 1} - run `uv run -m cfb.pull_offseason {season}`")
    now = coaches.xs(season, level="year").reindex(teams) if season in coaches.index.get_level_values("year") \
        else pd.Series(np.nan, index=teams)
    before = coaches.xs(season - 1, level="year").reindex(teams)
    new = (now != before).astype(float).where(now.notna() & before.notna())
    return pd.DataFrame({
        "recruiting": recruiting_average(season, out).reindex(teams),
        "returning": (returning.receiving_usage + returning.rushing_usage) / 2,
        "new_coach": new,
    }, index=teams)


def defensive_turnover(season, out):
    """Each team's returning and incoming defence, from season's roster and the season before's player stats.

    returning   the mean of two shares of last season's defence still on the roster: of its tackles, and
                of its havoc plays (HAVOC)
    incoming    the same two counts for players on the roster who played for another team last season, as
                a share of an average team's (every team in the stats, FCS too), averaged

    None when either file is missing (CFBD has player stats from 2016).
    """
    stats_path, roster_path = out / str(season - 1) / "player_defense.csv", out / str(season) / "roster.csv"
    if not stats_path.exists() or not roster_path.exists():
        return None
    st = pd.read_csv(stats_path)
    w = st.pivot_table(index=["player_id", "team"], columns="stat_type", values="stat", aggfunc="sum").fillna(0)
    w["tackles"] = w.get("TOT", 0)
    w["havoc"] = w[[c for c in HAVOC if c in w]].sum(axis=1)
    w = w[["tackles", "havoc"]].reset_index()
    roster = pd.read_csv(roster_path)[["id", "team"]].rename(columns={"id": "player_id", "team": "now"})
    m = w.merge(roster, on="player_id", how="left")
    team_total = w.groupby("team")[["tackles", "havoc"]].sum()
    stayed = m[m.now == m.team].groupby("team")[["tackles", "havoc"]].sum().reindex(team_total.index).fillna(0)
    arrived = m[m.now.notna() & (m.now != m.team)].groupby("now")[["tackles", "havoc"]].sum()
    return pd.DataFrame({"returning": (stayed / team_total).mean(axis=1),
                         "incoming": (arrived / team_total.mean()).mean(axis=1)}).rename_axis("team")


def fbs_teams(season, out):
    games = pd.read_csv(out / str(season) / "games.csv")
    return set(games.home_team[games.home_classification == "fbs"]) | set(games.away_team[games.away_classification == "fbs"])


def prior_shift(season, out):
    """Each team's shift, in ppa per play, of its four units' prior: one row per team, a column per unit.

    Added to the team's own part of the prior (cfb.ratings.fit_qbs's shift).
    A team whose coach is unknown counts as not new.
    """
    f = features(season, out)
    shift = pd.DataFrame(0.0, index=f.index, columns=list(UNITS))
    off = RETURNING_SHIFT * zscore(f.returning) + NEW_COACH_SHIFT * (f.new_coach.fillna(0.0) - f.new_coach.mean())
    dfn = RECRUITING_SHIFT * zscore(f.recruiting)
    for unit in OFFENCE:
        shift[unit] = off.fillna(0.0)
    for unit in DEFENCE:
        shift[unit] = dfn
    log.info("%d offseason shifts: %d teams, %d new coaches, sd %.3f offence and %.3f defence (ppa/play)",
             season, len(f), int(f.new_coach.sum()), off.std(), dfn.std())
    turnover = defensive_turnover(season, out)
    if turnover is None:
        return shift
    # Every FBS team on a roster, with returning production or with defensive stats is shifted; each SD is
    # over them. A team with no transfers in brought in nothing.
    roster = pd.read_csv(out / str(season) / "roster.csv")
    known = set(roster.team) | set(f.index) | set(turnover.index)
    teams = pd.Index(sorted(known & fbs_teams(season, out)))
    returning = turnover.returning.reindex(teams)
    incoming = turnover.incoming.reindex(teams).fillna(0.0)
    dfn_turnover = DEF_RETURNING_SHIFT * zscore(returning) + DEF_INCOMING_SHIFT * zscore(incoming)
    shift = shift.reindex(shift.index.union(teams)).fillna(0.0)
    for unit in DEFENCE:
        shift.loc[teams, unit] += dfn_turnover
    log.info("%d defensive turnover shifts: %d teams, sd %.3f (ppa/play)", season, len(teams), dfn_turnover.std())
    return shift
