"""College RAPPA ratings, fitted on CFBD play-by-play.

The model is in rappa/model.py - see there for how it works. This module gets
CFBD's plays into the shape it expects and holds the constants measured on
college data:

    target      CFBD's ppa
    plays       rushes and passes, read off CFBD's play types (see classify);
                kneels and spikes are dropped
    FCS         every non-FBS opponent is pooled into one team (see pool_fcs)
    home field  HFA below, held fixed - about five times the NFL's
    garbage     GARBAGE below
    clip        CLIP below
    conferences every conference gets a shared rating of its own
                (rappa.model.Groups, CONF_ constants below), so a team is its
                conference's part plus its own; see conferences
    prior       with conferences, CONF_PRIOR_WEIGHT of each conference's
                average and CONF_TEAM_WEIGHT of each team's gap from it;
                without, PRIOR_WEIGHT and PRIOR_WEIGHT_DEF of the whole rating
    passers     quarterbacks are rated on their own (fit_qbs and the QB_
                constants below), so a team's pass_off is its supporting cast.
                CFBD has no player ids, so passers are read off the play text
                (see passer_key) and a quarterback who transfers is followed
                by name (see qb_prior_means).

Run `uv run -m cfb.pull <season>` first. Writes <out>/<season>/rappa.csv
(default data/cfb/<season>/rappa.csv), one row per team, and qbs.csv next to
it, one row per passer. In rappa.csv, qb is the team's passers' rating
weighted by their dropbacks, and off and net count it.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rappa import model
from rappa.model import GARBAGE_RULES, UNITS, Groups, conference_home_edge, garbage_time, summarize

log = logging.getLogger(__name__)

# Chronological play order within a game.
ORDER = ["game_id", "drive_number", "play_number"]

# Play types that are unambiguously one or the other.
RUSH_TYPES = {"Rush", "Rushing Touchdown"}
PASS_TYPES = {
    "Pass Reception",
    "Pass Completion",  # CFBD's label for some 2025 completions
    "Pass Incompletion",
    "Passing Touchdown",
    "Sack",
    "Pass Interception Return",
    "Interception",
    "Interception Return Touchdown",
}
# Fumbles and safeties can be either, so they are read off the play text.
# Plain "Fumble" (2025 on) also covers the odd punt, which the text drops.
AMBIGUOUS = {"Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Safety", "Fumble Return Touchdown",
             "Fumble"}

# Clock management, not football. CFBD labels kneels as rushes and gives them a
# real PPA (mean -0.88), and teams kneel because they are winning - so leaving
# them in quietly penalises the best offences. Only play_text identifies them.
DEAD_BALL = r"kneel|knee down|takes a knee|spike"

# The one team every FCS (or lower-division) opponent is rated as.
FCS = "FCS"

# Home-field advantage in PPA per play, per side: the home offence is expected
# to be HFA above average and the away offence HFA below. Held fixed rather
# than fitted. A fitted column comes out three to four times too big: outside
# conference play the home team is usually the better one, and the column
# soaks up whatever part of that talent gap the ridge penalty shrinks off the
# team ratings. This is conference_home_edge(), where schedules are balanced:
# 0.0105, 0.0168 and 0.0210 in 2022-2024. Roughly 2 points a game. Walked
# forward on game margins (see below), anything from 0 to 0.035 predicted
# within 0.03 points of it: home field in points is learned by
# cfb.predict_games's stage 2 instead, at about 3 points (the NFL's is 2).
HFA = 0.016

# The constants below were re-tuned on game margins rather than holdout plays,
# as the NFL's were: cfb.predict_games walked forward over 2023-2024, choosing
# on MAE and log-loss of FBS-vs-FBS games from week 2 on, then checked on
# 2025-2026, with 2022 as the warm-up. With special teams and quarterbacks
# they cut MAE by 0.08 and log-loss by 0.008 against the play-tuned values
# (prior weight 0.57, penalty 300, NFL quarterback constants), and they
# changed less than 0.03 anywhere near their optimum.

# Plays by a team leading by at least this much are dropped
# (rappa.model.GARBAGE_RULES). Unlike in the NFL, no rule helped on game
# margins: flat21, the NFL's, cost 0.13-0.20 points of MAE, and the rest
# 0.03-0.11, bar q4_21, which was level.
GARBAGE = "none"

# Each play's PPA is clipped to +/-CLIP before fitting. Pick-sixes and fumble
# returns reach -8 and are mostly luck; clipping them predicted later weeks'
# unclipped PPA better in the 2022-2024 backtests, and 3 was also best on
# game margins (no clip, 2 and 4-5 were 0.02-0.06 worse).
CLIP = 3.0

# How much of last season's rating to carry forward when shrinking toward it,
# offence (PRIOR_WEIGHT) and defence (PRIOR_WEIGHT_DEF), in a fit without
# conferences (--no-conferences; see the CONF_ constants for the default). 0.57, the
# year-over-year OLS slope of the ratings on 2022-2024, left early-season
# predictions far too timid: in weeks 2-4 they spread 9.4 points against the
# market's 13.8. 0.85 for both (with the penalty at 200) did best on game
# margins, and anything in 0.7-1.0 was within 0.02 points. College
# defences carry over more than NFL ones (0.3 there). A second season of
# prior added nothing.
PRIOR_WEIGHT = 0.85
PRIOR_WEIGHT_DEF = 0.85

# Quarterbacks get a rating of their own (rappa.model.fit_passers), as in
# nfl.ratings: every pass play, sacks included, is the team's pass unit plus
# its passer, and a game uses the quarterback who starts it.
#
# It helps less than in the NFL: leaving it out cost 0.04-0.06 points of MAE
# and 0.003-0.008 of log-loss, against 0.11 and 0.010 there.
#
# QB_PRIOR_WEIGHT is how much of last season's rating a quarterback carries,
# wherever he plays. 0.6-0.9 were level; 1.0 was worse.
QB_PRIOR_WEIGHT = 0.9

# QB_REPLACEMENT is the rating, in PPA per dropback, of a quarterback we know
# nothing about, and what every passer is shrunk toward in a season's own fit.
# The NFL's -0.5 was a little worse than anything from 0 to -0.3 (-0.7 worse
# still); -0.2 is in the middle of that plateau.
QB_REPLACEMENT = -0.2

# PASS_PRIOR_WEIGHT is the team pass unit's carry-over once the quarterback is
# out of it. 0.55-0.85 were level; the NFL's 0.3 is below the other units
# there too.
PASS_PRIOR_WEIGHT = 0.7

# Conferences get a shared rating each (rappa.model.Groups): four units, like
# a team's, added to every member's. About two thirds of a team's plays are
# against its own conference, and those cannot tell a better conference from
# a worse one - lift every SEC team alike and every SEC game looks the same -
# so only the non-conference games can. With team ratings alone the penalty
# charges a conference-wide difference once per member, and the fit kept 43%
# of one planted on the 2024 schedule (62% for one team's) and took the rest
# from a prior that itself lost 15% a season. That left P4 teams underrated
# against G5 ones by 6.4 points a game in 2023-2026 (7.0 in weeks 2-4). With
# conference ratings, 85% of it is kept and the gap was +0.6 +/- 1.0.
#
# Walked forward over 2023-2026 on FBS-vs-FBS game margins (2022 the warm-up),
# conferences cut MAE by 0.17 on 2023-2024 and 0.22 on 2025-2026, and
# log-loss by 0.004-0.006. These values were chosen looking at all four
# seasons, so the gain is a little optimistic.
#
# CONF_ALPHA is the conference ratings' penalty (the teams' stays at their
# alpha). 500-1000 were level; 200 was 0.02 worse, 5-50 worse still, as the
# conference ratings then chase a few non-conference games.
CONF_ALPHA = 500.0

# CONF_PRIOR_WEIGHT is how much of each conference's average rating last
# season (over this season's members) carries forward: all of it. 0.9 and
# 1.1 were level. CONF_TEAM_WEIGHT is each team's gap from that average, and
# CONF_PASS_WEIGHT the pass unit's once the quarterback is out of it: 0.7
# and 0.55 beat the no-conference 0.85 and 0.7 by 0.02-0.03, and 0.55 for
# every unit was level. A team new to FBS starts at its conference's average.
CONF_PRIOR_WEIGHT = 1.0
CONF_TEAM_WEIGHT = 0.7
CONF_PASS_WEIGHT = 0.55

# The penalty for a season's own fit when quarterbacks are rated, instead of
# cross-validating (which the passers' offset throws off; see nfl.ratings).
# The same as cfb.predict_games.ALPHA: 300 made next season's prior worse
# (log-loss +0.001).
SEASON_ALPHA = 200.0

# Because every passer is shrunk toward QB_REPLACEMENT, the fitted ratings
# carry offsets: quarterbacks sit well below zero and some team units above
# it. They are common to every team and cancel in predictions, so the files
# keep them (they are next season's prior), and centered() takes them out
# for reading.

# CFBD's play text comes in two styles, sometimes for the same player in
# different games: "Jalen Milroe pass complete to ..." and "(13:44) Shotgun #4
# J.Milroe pass complete to ...". Touchdowns can read "Ryan Williams 45 Yd pass
# from Jalen Milroe (Will Reichard Kick)". Interception-return touchdowns do
# not name the passer, and those plays go without one.
CLOCK = re.compile(r"^\(\s*\d+:\d+\s*\)\s*")
FORMATION = re.compile(r"^(?:(?:No Huddle|Shotgun|Pistol|Under Center|Wildcat)[- ]*)+", re.I)
TD_PASS = re.compile(r"\d+ Yd pass from (.+?)(?:\s*\(|$)")
PASSER = re.compile(r"^(?:#\d+\s+)?([^#()]+?)\s+(?:pass|sacked)\b", re.I)
SUFFIX = re.compile(r"\s+(?:Jr\.?|Sr\.?|II|III|IV|V)$", re.I)
INITIALS = re.compile(r"^(?:[A-Z]\.){1,3}$|^[A-Z]{1,3}$")
ABBREVIATED = re.compile(r"^([A-Za-z]{1,2})\.([A-Za-z].*)$")


def continuous_week(games):
    """Week numbers that run in real time across the whole season.

    CFBD restarts the count at week 1 for the postseason, so bowls and playoff
    games otherwise sort as if they were played in August. Anything trained on
    "weeks <= N" then silently includes games from December and January.
    """
    weeks = games.week.astype(float).copy()
    regular = games.season_type == "regular"
    offset = games.loc[regular, "week"].max() if regular.any() else 0
    return weeks.where(regular, offset + weeks)


def pool_fcs(plays, games):
    """Rename every non-FBS team to FCS, in plays and games, so they share one rating.

    FCS teams play one or two FBS games a season, too few plays to rate them
    one by one, but pooled those games still say something about the FBS
    side. Keeping them this way beat dropping them on 2022-2024 holdout team
    PPA (corr 0.360 vs 0.346, mostly in the first few weeks) and tied on game
    margins. Games with no FBS team are dropped, or the pool would play itself.
    """
    games = games[games.home_classification.eq("fbs") | games.away_classification.eq("fbs")]
    lower = set(games.home_team[games.home_classification.ne("fbs")])
    lower |= set(games.away_team[games.away_classification.ne("fbs")])

    def pool(teams):
        return teams.where(~teams.isin(lower), FCS)

    plays = plays[plays.game_id.isin(games.id)]
    # home/away too, or home_field_column would misread a pooled home team.
    plays = plays.assign(**{c: pool(plays[c]) for c in ("offense", "defense", "home", "away")})
    games = games.assign(home_team=pool(games.home_team), away_team=pool(games.away_team))
    return plays, games


def attach_game_context(plays, games):
    """Bring each play's week, venue and conference-game flag across from its game."""
    info = games.assign(_w=continuous_week(games)).set_index("id")
    return plays.assign(
        week=plays.game_id.map(info._w),
        neutral_site=plays.game_id.map(info.neutral_site).fillna(False).astype(bool),
        conference_game=plays.game_id.map(info.conference_game).fillna(False).astype(bool),
    )


def add_pre_play_margin(df):
    """Score margin faced by the offense at the snap.

    CFBD reports the score *after* each play, and leaves it stale on clock and
    penalty rows, so take a running maximum (scores only go up) and shift it by
    one play. Must run on the full play set, before any filtering, or the
    running score will skip the plays that were dropped.
    """
    df = df.sort_values(ORDER).reset_index(drop=True)
    offense_is_home = df.offense == df.home
    home = np.where(offense_is_home, df.offense_score, df.defense_score)
    away = np.where(offense_is_home, df.defense_score, df.offense_score)
    home = pd.Series(home, index=df.index).groupby(df.game_id).cummax()
    away = pd.Series(away, index=df.index).groupby(df.game_id).cummax()

    by_game = pd.DataFrame({"h": home, "a": away}).groupby(df.game_id)
    pre_home = by_game.h.shift(1).fillna(0)
    pre_away = by_game.a.shift(1).fillna(0)
    df["margin"] = np.where(
        offense_is_home, pre_home - pre_away, pre_away - pre_home
    )
    return df


def classify(df):
    """Label each play rush or pass, dropping what is neither."""
    text_all = df.play_text.fillna("").str.lower()
    dead = text_all.str.contains(DEAD_BALL, regex=True)
    if dead.any():
        log.info("dropping %d kneels/spikes", dead.sum())
        df = df[~dead]

    kind = pd.Series(pd.NA, index=df.index, dtype=object)
    kind[df.play_type.isin(RUSH_TYPES)] = "rush"
    kind[df.play_type.isin(PASS_TYPES)] = "pass"

    amb = df.play_type.isin(AMBIGUOUS) & kind.isna()
    text = text_all.loc[df.index]
    # A sack is a pass play, and it is the strongest signal in the text.
    kind[amb & text.str.contains("pass|sack")] = "pass"
    kind[amb & kind.isna() & text.str.contains(" run| rush")] = "rush"

    df = df.assign(play_kind=kind)
    dropped = df.play_kind.isna().sum()
    log.info("classified %d plays (%d dropped as neither rush nor pass)", len(df) - dropped, dropped)
    return df[df.play_kind.notna()].copy()


def passer_name(text):
    """The passer named in one pass play's text, as written, or None."""
    if not isinstance(text, str):
        return None
    text = CLOCK.sub("", text)
    td = TD_PASS.search(text)
    if td:
        return td.group(1).strip()
    found = PASSER.match(FORMATION.sub("", text))
    return found.group(1).strip() if found else None


def passer_key(name):
    """One spelling for both styles of a name: first initial and last name, lower case.

    "Jalen Milroe", "J.Milroe" and "Jalen Milroe Jr." all come out j.milroe;
    "C.J. Stroud" and "CJ Stroud" c.stroud; "G.Van Buren" g.van buren.
    """
    name = SUFFIX.sub("", name.strip().rstrip(","))
    tokens = name.split()
    if not tokens:
        return None
    short = ABBREVIATED.match(tokens[0])
    if len(tokens) > 1 and INITIALS.match(tokens[0]):
        initial, last = tokens[0][0], " ".join(tokens[1:])
    elif short and not INITIALS.match(tokens[0]):
        initial, last = short.group(1)[0], " ".join([short.group(2), *tokens[1:]])
    elif len(tokens) > 1:
        initial, last = tokens[0][0], " ".join(tokens[1:])
    else:
        return tokens[0].lower()
    return f"{initial}.{last.strip('.')}".lower()


def add_passers(df):
    """A passer id for every pass play whose text names one: its key and team, "j.milroe|Alabama".

    The team is part of the id so that two quarterbacks with the same key on
    different teams stay apart; qb_prior_means follows one who transfers.
    passer_name keeps the text's spelling, for qb_table.
    """
    is_pass = df.play_kind == "pass"
    names = df.play_text.where(is_pass).map(passer_name)
    keys = names.map(lambda n: passer_key(n) if isinstance(n, str) else None)
    passer = (keys + "|" + df.offense).where(keys.notna())
    return df.assign(passer=passer, passer_name=names.where(keys.notna()))


def load(season, out):
    """One season's rush and pass plays, in the columns rappa.fit expects, and its games.

    FCS teams are pooled in both. Games include those not played yet, with no
    plays; margin is home minus away, NaN until a game is completed.
    """
    plays_path, games_path = out / str(season) / "plays.csv", out / str(season) / "games.csv"
    if not plays_path.exists() or not games_path.exists():
        raise SystemExit(f"need {plays_path} and {games_path} - run `uv run -m cfb.pull {season}` first")
    games = pd.read_csv(games_path)
    games["margin"] = (games.home_points - games.away_points).where(games.completed == True)  # noqa: E712
    plays, games = pool_fcs(pd.read_csv(plays_path, low_memory=False), games)
    games = games.assign(week=continuous_week(games))
    plays = attach_game_context(add_pre_play_margin(plays), games)
    plays = plays[plays.ppa.notna() & plays.offense.notna() & plays.defense.notna()]
    return add_passers(classify(plays).dropna(subset=["week"])), games


def conferences(games):
    """Each FBS team's conference this season, read off its games: {team: conference}.

    Independents and the FCS pool have none, so they get no conference rating.
    """
    sides = [games[[f"{side}_team", f"{side}_conference"]].set_axis(["team", "conference"], axis=1)
             for side in ("home", "away")]
    table = pd.concat(sides).dropna()
    table = table[table.team.ne(FCS) & table.conference.ne("FBS Independents")]
    return table.drop_duplicates("team").set_index("team").conference.to_dict()


def prior_weights(prior_weight=None, use_qbs=True, use_conferences=True):
    """What each unit of a team's prior is multiplied by.

    prior_weight is one weight for every unit, or an (offence, defence) pair;
    None is the defaults above for the kind of fit. With conferences they
    apply to each team's gap from its conference.
    """
    if prior_weight is not None:
        off, dfn = prior_weight if isinstance(prior_weight, tuple) else (prior_weight, prior_weight)
        return {"rush_off": off, "pass_off": off, "rush_def": dfn, "pass_def": dfn}
    if use_conferences:
        off, pass_off, dfn = CONF_TEAM_WEIGHT, CONF_PASS_WEIGHT, CONF_TEAM_WEIGHT
    else:
        off, pass_off, dfn = PRIOR_WEIGHT, PASS_PRIOR_WEIGHT, PRIOR_WEIGHT_DEF
    return {"rush_off": off, "pass_off": pass_off if use_qbs else off, "rush_def": dfn, "pass_def": dfn}


def conference_prior(prior, members):
    """Last season's ratings as this season's conference averages and each team's gap from its own.

    The averages are over this season's members that have a prior, so a
    conference that changed members is judged by the teams it has now. A team
    outside every conference keeps its whole rating as its gap. Returns (gaps,
    averages), unweighted.
    """
    units = prior[list(UNITS)]
    conference = units.index.map(members)
    averages = units.groupby(conference).mean()
    gaps = units - averages.reindex(conference).fillna(0.0).to_numpy()
    return gaps, averages


def _prior_and_groups(prior, weights, members):
    """The team prior to pass rappa.model, weighted, and the conferences' Groups (None without them)."""
    groups = None
    if members is not None:
        groups = Groups(members, CONF_ALPHA)
        if prior is not None:
            prior, averages = conference_prior(prior, members)
            groups.prior = averages * CONF_PRIOR_WEIGHT
    if prior is not None:
        # rappa.model takes one weight, so each unit's goes into the prior itself.
        prior = prior.assign(**{unit: prior[unit] * w for unit, w in weights.items()})
    return prior, groups


def fit(df, alpha=None, seed=0, prior=None, prior_weight=None, half_life=None, hfa=HFA, clip=CLIP,
        members=None):
    """rappa.model.fit, with the college constants as defaults.

    prior_weight is as prior_weights takes it. members is conferences(games),
    or None to fit without conference ratings; with them alpha must be given.
    Quarterbacks are not rated; see fit_qbs.
    """
    if prior is not None and "qb" in prior:
        # A quarterback-aware rappa.csv leaves the passers out of pass_off:
        # put them back for a fit that does not rate them, and take out its
        # offsets (see SEASON_ALPHA).
        prior = centered(prior.assign(pass_off=prior.pass_off + prior.qb))
    prior, groups = _prior_and_groups(prior, prior_weights(prior_weight, False, members is not None), members)
    return model.fit(df, alpha, hfa=hfa, clip=clip, prior=prior, prior_weight=1.0,
                     half_life=half_life, seed=seed, groups=groups)


def fit_qbs(df, alpha=None, seed=0, prior=None, qb_prior=None, prior_weight=None, half_life=None,
            hfa=HFA, clip=CLIP, members=None):
    """rappa.model.fit_passers with the college constants: team and quarterback ratings.

    prior is last season's rappa.csv, and qb_prior what each passer is shrunk
    toward (qb_prior_means); either may be None. prior_weight is as
    prior_weights takes it, and members as fit takes it. Returns the team
    ratings, the quarterback ratings (a Series by passer id), the penalty and
    the intercept.
    """
    prior, groups = _prior_and_groups(prior, prior_weights(prior_weight, True, members is not None), members)
    return model.fit_passers(df, alpha, hfa=hfa, clip=clip, prior=prior, prior_weight=1.0,
                             half_life=half_life, seed=seed, passer_prior=qb_prior,
                             passer_default=QB_REPLACEMENT, groups=groups)


def previous_ids(last_ids, ids):
    """Each of this season's passer ids that can be found last season, mapped to his id then.

    A passer on the same team matches by id. One on a new team matches by key
    alone, when exactly one of last season's passers had that key, nobody
    kept it on his old team, and nobody else took it to a new one.
    """
    ids = pd.Index(pd.unique(pd.Series(ids).dropna()))
    last_ids = pd.Index(last_ids)

    def key(idx):
        return pd.Series(idx.str.rsplit("|", n=1).str[0], index=idx)

    stayed = ids.intersection(last_ids)
    last_key = key(last_ids)
    movable = last_key.drop(stayed)
    movable = movable[~movable.duplicated(keep=False) & ~movable.isin(set(last_key[stayed]))]
    new_key = key(ids.difference(stayed))
    new_key = new_key[~new_key.duplicated(keep=False)]
    moved = new_key.map(pd.Series(movable.index, index=movable.to_numpy())).dropna()
    return pd.concat([pd.Series(stayed, index=stayed), moved])


def qb_prior_means(qb_prior, passers):
    """What each of `passers` is shrunk toward: QB_PRIOR_WEIGHT of his gap to replacement last season.

    qb_prior is last season's qbs.csv; a passer is found in it by
    previous_ids, which follows a quarterback who transferred. Anyone not
    found is left out, and so starts at QB_REPLACEMENT.
    """
    if qb_prior is None:
        return None
    before = previous_ids(qb_prior.index, passers)
    rating = pd.Series(qb_prior.rating.reindex(before.to_numpy()).to_numpy(), index=before.index)
    return QB_REPLACEMENT + QB_PRIOR_WEIGHT * (rating - QB_REPLACEMENT)


def qb_table(qbs, df):
    """One row per passer: name, team, dropbacks, rating.

    name is the fullest spelling the text used for him: "Jalen Milroe" over
    "J.Milroe".
    """
    thrown = df[df.passer.notna()]
    names = thrown.groupby("passer").passer_name.agg(lambda s: max(s.unique(), key=len))
    table = pd.DataFrame({
        "name": names,
        "team": thrown.groupby("passer").offense.first(),
        "dropbacks": thrown.passer.value_counts(),
        "rating": qbs,
    })
    table = table[table.dropbacks.notna()]
    table["dropbacks"] = table.dropbacks.astype(int)
    # For reading: against the average dropback this season.
    table["vs_average"] = table.rating - np.average(table.rating, weights=table.dropbacks)
    table.index.name = "passer_id"
    return table.sort_values("rating", ascending=False).round({"rating": 4, "vs_average": 4})


def centered(ratings):
    """rappa.csv's ratings against the average team, for reading (see SEASON_ALPHA's note)."""
    cols = [c for c in ("rush_off", "pass_off", "qb", "rush_def", "pass_def", "off", "def", "net") if c in ratings]
    return ratings.assign(**{c: ratings[c] - ratings[c].mean() for c in cols}).round(4)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season(s) to rate, fitted separately")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--alpha", type=float, help="fix the ridge penalty (default: cross-validated, or "
                                                f"{SEASON_ALPHA:g} when rating quarterbacks)")
    p.add_argument(
        "--garbage-time",
        choices=sorted(GARBAGE_RULES),
        default=GARBAGE,
        help=f"drop plays in garbage time (default: {GARBAGE})",
    )
    p.add_argument("--top", type=int, default=15, help="how many teams to print (default: 15)")
    p.add_argument("--hfa", type=float, default=HFA,
                   help=f"home-field edge in ppa/play per side, held fixed (default: {HFA}; 0 = none)")
    p.add_argument("--clip", type=float, default=CLIP,
                   help=f"clip each play's ppa to +/- this before fitting (default: {CLIP}; 0 = off)")
    p.add_argument("--half-life", type=float,
                   help="weeks after which a play counts half as much (default: no decay)")
    p.add_argument("--prior", type=Path, help="previous season's rappa.csv to shrink toward "
                                              "(its qbs.csv is picked up from the same folder)")
    p.add_argument("--prior-weight", type=float,
                   help="how much of the prior to carry forward, for every unit (default: "
                        f"{PRIOR_WEIGHT} for offence, {PRIOR_WEIGHT_DEF} for defence)")
    p.add_argument("--no-qbs", action="store_true", help="leave quarterbacks in their teams' pass_off")
    p.add_argument("--no-conferences", action="store_true",
                   help="rate teams without conference ratings (then rate every season before this one "
                        "the same way, as they are its prior)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for season in args.seasons:
        df, games = load(season, args.out)
        log.info("=== %d: %d plays from %d games (%d against %s)", season, len(df), df.game_id.nunique(),
                 (games.away_team.eq(FCS) | games.home_team.eq(FCS)).sum(), FCS)

        if args.garbage_time != "none":
            junk = garbage_time(df, args.garbage_time)
            log.info("dropping %d garbage-time plays (%.1f%%, rule=%s)",
                     junk.sum(), 100 * junk.mean(), args.garbage_time)
            df = df[~junk]

        # A sanity check on the fixed home field, not an input to the fit.
        log.info("conference-game home edge this season %+.4f (fitting with %+.4f)",
                 conference_home_edge(df, args.clip), args.hfa)
        prior = pd.read_csv(args.prior, index_col="team") if args.prior else None
        dest = args.out / str(season) / "rappa.csv"
        members = None if args.no_conferences else conferences(games)
        if args.no_qbs:
            # A fit with conferences cannot cross-validate its penalty.
            alpha = SEASON_ALPHA if args.alpha is None and members is not None else args.alpha
            ratings, _, _ = fit(
                df, alpha, args.seed, prior, args.prior_weight, args.half_life, args.hfa,
                args.clip, members,
            )
            ratings = summarize(ratings, df)
        else:
            qb_path = args.prior.parent / "qbs.csv" if args.prior else None
            qb_prior = pd.read_csv(qb_path, index_col="passer_id") if qb_path and qb_path.exists() else None
            alpha = SEASON_ALPHA if args.alpha is None else args.alpha
            ratings, qbs, _, _ = fit_qbs(df, alpha, args.seed, prior, qb_prior_means(qb_prior, df.passer),
                                         args.prior_weight, args.half_life, args.hfa, args.clip, members)
            ratings = summarize(ratings, df, qbs)
            table = qb_table(qbs, df)
            table.to_csv(dest.parent / "qbs.csv")
            log.info("wrote %s (%d passers)", dest.parent / "qbs.csv", len(table))
            print(f"\nQuarterbacks with 250+ dropbacks, {season} (vs_average: ppa per dropback against the "
                  "average dropback, on top of the team's pass_off):")
            print(table[table.dropbacks >= 250].drop(columns="rating").head(args.top * 2).to_string())
        ratings.to_csv(dest)
        log.info("wrote %s (%d teams)", dest, len(ratings))
        print(f"\nTop {args.top} by net RAPPA, {season} (ppa per play vs the average team):")
        print(centered(ratings).head(args.top).to_string())
        if members is not None:
            by_conference = centered(ratings).groupby(ratings.index.map(members)).net.agg(["mean", "size"])
            print(f"\nConferences by their teams' average net RAPPA, {season}:")
            print(by_conference.sort_values("mean", ascending=False).rename(columns={"size": "teams"})
                  .rename_axis("conference")
                  .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
