"""NFL special teams and penalties: rappa.outside on nflverse play-by-play.

rappa.outside has the method: every point in a game outside RAPPA's play
ratings (kicks, penalties, field goals, big plays past the cap and the
rest), rated by team and added to a predicted margin at its WEIGHT. This
module only reads an nflverse season into it: the fitted plays are
nfl.ratings.in_fit's, the play-by-play is nfl.special_teams.plays_frame's (as
nfl.ratings values plays), and kneels and spikes are nflverse's own play
types. Teams are named as nfl.predict_games.load_games names them.

game_points caches each season's per-game totals as <out>/<season>/outside.csv,
rebuilt when the season's plays or games, or the code that values them, are
newer. `uv run -m nfl.outside <season>` prints a season's ratings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nfl import ratings as league
from nfl import special_teams as kicks
from nfl.ratings import CLIP, COLUMNS, in_fit, load
from nfl.special_teams import plays_frame
from rappa import expected_points
from rappa import outside as shared

log = logging.getLogger(__name__)

CACHE = "outside.csv"
# predict_games.py's load_games names the teams; it is read by path, as it imports this module.
SOURCES = [Path(__file__), Path(__file__).with_name("predict_games.py"), Path(shared.__file__), Path(league.__file__),
           Path(kicks.__file__), Path(expected_points.__file__)]


def _season_points(season, out):
    """rappa.outside.game_points for one nflverse season."""
    from nfl.predict_games import load_games  # here, as nfl.predict_games imports this module
    raw = pd.read_csv(out / str(season) / "plays.csv", usecols=COLUMNS, low_memory=False)
    frame = plays_frame(raw, no_play_needs_penalty=True)
    kneel = pd.Series(np.isin(raw.play_type.reindex(frame.row.to_numpy()), ["qb_kneel", "qb_spike"]), index=frame.index)
    return shared.game_points(frame, kneel, in_fit(load(season, out)), load_games(season, out), CLIP)


def game_points(season, out=Path("data/nfl")):
    """The season's completed games with their points outside the play ratings, home minus away, from the cache.

    Columns game_id, week, home_team, away_team, neutral_site and points.
    Games with no plays are left out.
    """
    inputs = [out / str(season) / f for f in ("plays.csv", "games.csv")] + SOURCES
    return shared.cached(out / str(season) / CACHE, inputs, lambda: _season_points(season, out))


def fit(games, prior=None):
    """rappa.outside.fit."""
    return shared.fit(games, prior)


def prior(season, out=Path("data/nfl")):
    """Last season's final ratings, the chain run from the first season with plays on disk; None for that one."""
    seasons = sorted(int(p.parent.name) for p in out.glob("*/plays.csv") if p.parent.name.isdigit())
    return shared.chain([s for s in seasons if s < season], lambda s: game_points(s, out))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("season", type=int)
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    games = game_points(args.season, args.out)
    table = fit(games, prior(args.season, args.out)).sort_values("rating", ascending=False)
    print(f"{args.season}: special teams and penalties, points a game against an average team "
          f"({len(games)} games; value = {shared.WEIGHT} x rating, what a predicted margin adds)")
    print(table.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
