# /// script
# requires-python = ">=3.12"
# dependencies = ["nflreadpy>=0.1.5"]
# ///
"""Pull NFL play-by-play and schedules from nflverse for one or more seasons.

Run with `uv run nfl/pull.py 2023 2024 2025` - by path, not with -m.
nflreadpy needs pydantic 2 and cfbd needs pydantic 1, so the two cannot share
the project environment; the block above gives this script its own, and uv
only reads it when the script is run by path.

Writes CSVs under <out>/<season>/ (default data/nfl/<season>/) so the data only
has to be fetched once. Re-running skips anything already on disk unless
--force is passed - use it to refresh a season that is still in progress.

games.csv is the nflverse schedule, one row per game, with its closing line
already included. Unlike CFBD, spread_line is the home team's expected margin:
positive when the home team is favoured, so it lines up with -spread in the
college data. Weeks run straight through the playoffs instead of restarting.

plays.csv is nflverse play-by-play. epa is the counterpart of CFBD's ppa.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    import nflreadpy as nfl
except ImportError:
    raise SystemExit("nflreadpy not found - run this as `uv run nfl/pull.py`, not with -m or plain python")

log = logging.getLogger(__name__)

DATASETS = ("games", "plays")

# The first season nflverse has play-by-play for.
FIRST_SEASON = 1999


def fetch(dataset, season):
    if dataset == "games":
        return nfl.load_schedules([season])
    return nfl.load_pbp([season])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season year(s), e.g. 2023 2024 2025")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="output dir (default: data/nfl)")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="which datasets to pull (default: all)",
    )
    p.add_argument("--force", action="store_true", help="refetch even if CSVs exist")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    current = nfl.get_current_season()
    for season in args.seasons:
        if not FIRST_SEASON <= season <= current:
            raise SystemExit(f"{season}: nflverse covers {FIRST_SEASON}-{current}")

    for season in args.seasons:
        season_dir = args.out / str(season)
        for dataset in args.datasets:
            path = season_dir / f"{dataset}.csv"
            if path.exists() and not args.force:
                log.info("%d %s: %s already exists, skipping (--force to refetch)", season, dataset, path)
                continue
            df = fetch(dataset, season)
            if df.is_empty():
                log.warning("%d %s: no rows, skipping write", season, dataset)
                continue
            season_dir.mkdir(parents=True, exist_ok=True)
            df.write_csv(path)
            log.info("wrote %s (%d rows)", path, len(df))
        if season == current:
            log.info("%d is the current season - rerun with --force to pick up new weeks", season)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
