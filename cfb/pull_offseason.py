"""Pull what CFBD knows about each team before a season starts: recruiting, returning production, rosters, coaches.

Run with `uv run -m cfb.pull_offseason 2025 2026`. Needs CFBD_API_KEY in .env
or the environment. cfb.offseason turns these into shifts of each season's
prior; see there for what is used and why.

For each season S it writes, under <out> (default data/cfb):

    <year>/recruiting.csv   247's team recruiting rankings for the classes of
                            S-3 to S, the players on S's roster, one call each
    <S>/returning.csv       CFBD's returning production for S: the share of
                            last season's offence (PPA and usage, by passing,
                            rushing and receiving) still on the roster
    <S>/roster.csv          S's FBS rosters: CFBD player ids, positions and
                            class years, one call
    <S-1>/player_defense.csv
                            every player's defensive stats for S-1 (tackles,
                            tackles for loss, sacks, passes defended, hurries),
                            regular season and bowls, one call; CFBD has them
                            from 2016. With the roster, cfb.offseason measures
                            the defence that returns and that transferred in
    coaches.csv             every FBS head coach's seasons, one call for the
                            span; merged into what is already there

Anything already on disk is kept unless --force is passed. A season still to
be played should be pulled again once its recruiting class and roster are set
(returning production is published in the summer).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cfbd
import pandas as pd
from dotenv import load_dotenv

from cfb.pull import to_frame, with_retry

log = logging.getLogger(__name__)


def write(df, path):
    if df.empty:
        log.warning("%s: CFBD returned nothing, skipping write", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(df))


def coach_seasons(coaches):
    """One row per head coach per season: CFBD nests the seasons under each coach."""
    rows = []
    for coach in coaches:
        d = coach.dict()
        for season in d.get("seasons") or []:
            rows.append({"first_name": d["first_name"], "last_name": d["last_name"],
                         "hire_date": d.get("hire_date"), **season})
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="seasons to pull for, e.g. 2025 2026")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
    p.add_argument("--force", action="store_true", help="refetch even if the CSVs exist")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    api_key = os.getenv("CFBD_API_KEY")
    if not api_key:
        sys.exit("CFBD_API_KEY not found - add it to .env or the environment")

    seasons = sorted(set(args.seasons))
    classes = sorted({year for s in seasons for year in range(s - 3, s + 1)})
    with cfbd.ApiClient(cfbd.Configuration(access_token=api_key)) as client:
        recruiting, players = cfbd.RecruitingApi(client), cfbd.PlayersApi(client)
        for year in classes:
            path = args.out / str(year) / "recruiting.csv"
            if path.exists() and not args.force:
                log.info("%s already exists, skipping (--force to refetch)", path)
                continue
            write(to_frame(with_retry(recruiting.get_team_recruiting_rankings, year=year)), path)
        teams, stats = cfbd.TeamsApi(client), cfbd.StatsApi(client)
        for season in seasons:
            fetches = [
                (args.out / str(season) / "returning.csv",
                 lambda: with_retry(players.get_returning_production, year=season)),
                (args.out / str(season) / "roster.csv",
                 lambda: with_retry(teams.get_roster, year=season, classification=cfbd.DivisionClassification("fbs"))),
                (args.out / str(season - 1) / "player_defense.csv",
                 lambda: with_retry(stats.get_player_season_stats, year=season - 1, category="defensive",
                                    season_type=cfbd.SeasonType("both"))),
            ]
            for path, fetch in fetches:
                if path.exists() and not args.force:
                    log.info("%s already exists, skipping (--force to refetch)", path)
                    continue
                write(to_frame(fetch()), path)

        # A season's coaching change is against the season before, so fetch from one earlier.
        path = args.out / "coaches.csv"
        span = range(seasons[0] - 1, seasons[-1] + 1)
        have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["year"])
        if args.force or not set(span) <= set(have.year):
            fresh = coach_seasons(with_retry(cfbd.CoachesApi(client).get_coaches,
                                             min_year=span.start, max_year=span.stop - 1))
            kept = have[~have.year.isin(span)]
            write(pd.concat([kept, fresh], ignore_index=True).sort_values(["year", "school"]), path)
        else:
            log.info("%s already covers %d-%d, skipping (--force to refetch)", path, span.start, span.stop - 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
