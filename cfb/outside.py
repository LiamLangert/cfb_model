"""College special teams and penalties: rappa.outside on CFBD's play-by-play.

rappa.outside has the method: every point in a game outside RAPPA's play
ratings (kicks, penalties, field goals, big plays past the cap and the
rest), rated by team and added to a predicted margin at its WEIGHT. This
module only reads a CFBD season into it: the fitted plays are cfb.ratings.load's,
FCS teams are pooled (and left out of the average the ratings are centred
on), and kneels and spikes are read off the play text, as cfb.ratings drops
them.

game_points caches each season's per-game totals as <out>/<season>/outside.csv,
rebuilt when the season's plays or games, or the code that values them, are
newer. `uv run -m cfb.outside <season>` prints a season's ratings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from cfb import ratings as college
from cfb.ratings import (CLIP, DEAD_BALL, FCS, add_pre_play_margin, attach_game_context, continuous_week, load,
                         pool_fcs, scores_after, snap_scores)
from rappa import expected_points, special_teams
from rappa import outside as shared
from rappa.special_teams import KICKOFFS, PUNTS

log = logging.getLogger(__name__)

CACHE = "outside.csv"
SOURCES = [Path(__file__), Path(shared.__file__), Path(college.__file__), Path(expected_points.__file__),
           Path(special_teams.__file__)]


def _season_points(season, out):
    """rappa.outside.game_points for one CFBD season."""
    fitted, games = load(season, out)
    plays, games_all = pool_fcs(pd.read_csv(out / str(season) / "plays.csv", low_memory=False),
                                pd.read_csv(out / str(season) / "games.csv"))
    plays = attach_game_context(add_pre_play_margin(plays), games_all.assign(week=continuous_week(games_all)))
    h, a = snap_scores(plays)
    h_after, a_after = scores_after(plays)
    frame = plays[["game_id", "period", "down", "distance", "yards_to_goal", "offense", "home", "play_type"]]
    kneel = (plays.play_text.fillna("").str.lower().str.contains(DEAD_BALL, regex=True)
             & ~plays.play_type.isin(PUNTS | KICKOFFS) & ~plays.index.isin(fitted.index))
    return shared.game_points(frame.assign(h=h, a=a, h_after=h_after, a_after=a_after), kneel, fitted,
                              games.rename(columns={"id": "game_id"}), CLIP)


def game_points(season, out=Path("data/cfb")):
    """The season's completed games with their points outside the play ratings, home minus away, from the cache.

    Columns game_id, week (cfb.ratings.continuous_week), home_team, away_team
    (FCS teams pooled), neutral_site and points. Games with no plays are left out.
    """
    inputs = [out / str(season) / f for f in ("plays.csv", "games.csv")] + SOURCES
    return shared.cached(out / str(season) / CACHE, inputs, lambda: _season_points(season, out))


def fit(games, prior=None):
    """rappa.outside.fit, the FCS pool rated but left out of the average."""
    return shared.fit(games, prior, pool=FCS)


def prior(season, out=Path("data/cfb")):
    """Last season's final ratings, the chain run from the first season with plays on disk; None for that one."""
    seasons = sorted(int(p.parent.name) for p in out.glob("*/plays.csv") if p.parent.name.isdigit())
    return shared.chain([s for s in seasons if s < season], lambda s: game_points(s, out), pool=FCS)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("season", type=int)
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--top", type=int, default=15, help="how many teams to print at each end (default: 15)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    games = game_points(args.season, args.out)
    table = fit(games, prior(args.season, args.out)).drop(index=FCS, errors="ignore").sort_values("rating")
    print(f"{args.season}: special teams and penalties, points a game against an average team "
          f"({len(games)} games; value = {shared.WEIGHT} x rating, what a predicted margin adds)")
    print(table.iloc[::-1].head(args.top).round(2).to_string())
    print("...")
    print(table.head(args.top).iloc[::-1].round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
