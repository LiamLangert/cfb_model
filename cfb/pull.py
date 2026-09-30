"""Pull play-by-play, games, and betting lines from the CFBD API for a season.

Run with `uv run -m cfb.pull 2025`. Needs CFBD_API_KEY in .env or the
environment.

Writes CSVs under <out>/<season>/ (default data/cfb/<season>/) so the data only
has to be fetched once. Re-running skips anything already on disk unless
--force is passed.

games.csv carries the consensus betting line merged in, one row per game.
The per-sportsbook lines are kept alongside it in lines.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path

import cfbd
import pandas as pd
from dotenv import load_dotenv

log = logging.getLogger(__name__)

SEASON_TYPES = ("regular", "postseason", "both")
# "games" writes games.csv (with the consensus line merged in) and lines.csv.
DATASETS = ("games", "plays")
OUTPUTS = ("games", "lines", "plays")

# CFBD's own model outputs, dropped in favour of the betting lines.
MODEL_COLS = (
    "home_pregame_elo",
    "home_postgame_elo",
    "away_pregame_elo",
    "away_postgame_elo",
    "home_postgame_win_probability",
    "away_postgame_win_probability",
)

# Numbers medianed across sportsbooks to form the consensus.
LINE_COLS = (
    "spread",
    "spread_open",
    "over_under",
    "over_under_open",
    "home_moneyline",
    "away_moneyline",
)


def normalize(value):
    """Flatten a pydantic field value into something CSV can hold."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    return value


def to_frame(models, expand=()):
    """Turn a list of cfbd pydantic models into a DataFrame.

    Fields named in `expand` hold sub-objects (e.g. a play clock); their keys
    are lifted into `<field>_<key>` columns instead of being dumped as JSON.
    """
    rows = []
    for m in models:
        row = {}
        for key, value in m.dict().items():
            if key in expand:
                sub = value or {}
                for sub_key, sub_value in sub.items():
                    row[f"{key}_{sub_key}"] = normalize(sub_value)
            else:
                row[key] = normalize(value)
        rows.append(row)
    return pd.DataFrame(rows)


def with_retry(fn, *args, attempts=4, backoff=2.0, **kwargs):
    """Call a CFBD endpoint, retrying on transient failures."""
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced after the last attempt
            if attempt == attempts:
                raise
            delay = backoff ** attempt
            log.warning("%s failed (%s); retrying in %.0fs", fn.__name__, exc, delay)
            time.sleep(delay)


def week_plan(games_api, season, season_type):
    """(week, season_type, finished) for the weeks of this season that have started.

    Weeks yet to start are left out rather than fetched as empty, and a week
    still being played is marked unfinished so it is not cached half-done.
    """
    calendar = with_retry(games_api.get_calendar, year=season)
    now = datetime.now(timezone.utc)
    plan = []
    for wk in calendar:
        st = wk.season_type.value if isinstance(wk.season_type, Enum) else wk.season_type
        if season_type != "both" and st != season_type:
            continue
        if wk.start_date and wk.start_date > now:
            continue
        plan.append((wk.week, st, not wk.end_date or wk.end_date < now))
    return sorted(set(plan))


def fetch_games(games_api, season, season_type, classification):
    games = with_retry(
        games_api.get_games,
        year=season,
        season_type=season_type,
        classification=classification,
    )
    df = to_frame(games)
    df = df.drop(columns=[c for c in MODEL_COLS if c in df.columns])
    log.info("games: %d rows", len(df))
    return df


def fetch_lines(betting_api, season, season_type, game_ids, provider):
    """One row per game/provider; the API nests providers under each game."""
    betting_games = with_retry(
        betting_api.get_lines, year=season, season_type=season_type
    )
    rows = []
    for bg in betting_games:
        if game_ids is not None and bg.id not in game_ids:
            continue
        game_fields = {
            k: normalize(v) for k, v in bg.dict().items() if k != "lines"
        }
        game_fields["game_id"] = game_fields.pop("id")
        for line in bg.lines or []:
            if provider and line.provider != provider:
                continue
            rows.append({**game_fields, **{k: normalize(v) for k, v in line.dict().items()}})
    log.info("lines: %d rows (%d games)", len(rows), len(betting_games))
    return pd.DataFrame(rows)


def consensus_lines(lines_df):
    """Collapse the per-sportsbook lines to one consensus row per game.

    Uses the median so each figure stays a number a book actually posted -
    averaging American moneylines across the -100/+100 boundary would not.
    """
    if lines_df.empty:
        return pd.DataFrame()
    cols = [c for c in LINE_COLS if c in lines_df.columns]
    grouped = lines_df.groupby("game_id")
    consensus = grouped[cols].median().reset_index()
    consensus["n_books"] = grouped["provider"].nunique().values
    log.info("consensus: %d games with a line", len(consensus))
    return consensus


def fetch_plays(plays_api, season, plan, classification, parts_dir, force, pause):
    """Fetch plays week by week, caching each finished week so runs can resume."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for week, st, finished in plan:
        part = parts_dir / f"{st}_wk{week:02d}.csv"
        if part.exists() and not force:
            log.info("plays %s wk%02d: cached", st, week)
            frames.append(pd.read_csv(part))
            continue
        plays = with_retry(
            plays_api.get_plays,
            year=season,
            week=week,
            season_type=st,
            classification=classification,
        )
        df = to_frame(plays, expand=("clock",))
        if finished:
            df.to_csv(part, index=False)
        log.info("plays %s wk%02d: %d rows%s", st, week, len(df), "" if finished else " (in progress, not cached)")
        frames.append(df)
        time.sleep(pause)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write(df, path, label):
    if df.empty:
        log.warning("%s: no rows, skipping write", label)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(df))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("season", type=int, help="season year, e.g. 2024")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="output dir (default: data/cfb)")
    p.add_argument("--season-type", choices=SEASON_TYPES, default="both")
    p.add_argument(
        "--classification",
        default="fbs",
        help="division filter: fbs, fcs, ii, iii (default: fbs); 'all' for no filter",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="which datasets to pull (default: all); 'games' also writes lines.csv",
    )
    p.add_argument("--provider", help="only keep this betting provider, e.g. 'ESPN Bet'")
    p.add_argument("--force", action="store_true", help="refetch even if CSVs exist")
    p.add_argument("--pause", type=float, default=0.25, help="seconds between play requests")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    load_dotenv()
    api_key = os.getenv("CFBD_API_KEY")
    if not api_key:
        sys.exit("CFBD_API_KEY not found - add it to .env or the environment")

    classification = None if args.classification == "all" else args.classification
    season_dir = args.out / str(args.season)
    paths = {name: season_dir / f"{name}.csv" for name in OUTPUTS}

    want_games = "games" in args.datasets and (
        args.force or not paths["games"].exists()
    )
    want_plays = "plays" in args.datasets and (
        args.force or not paths["plays"].exists()
    )
    if "games" in args.datasets and not want_games:
        log.info("games: %s already exists, skipping (--force to refetch)", paths["games"])
    if "plays" in args.datasets and not want_plays:
        log.info("plays: %s already exists, skipping (--force to refetch)", paths["plays"])
    if not (want_games or want_plays):
        return 0

    config = cfbd.Configuration(access_token=api_key)
    with cfbd.ApiClient(config) as client:
        if want_games:
            games_df = fetch_games(
                cfbd.GamesApi(client), args.season, args.season_type, classification
            )
            # Lines cover every division, so restrict them to the games we pulled.
            game_ids = set(games_df["id"]) if not games_df.empty else None
            lines_df = fetch_lines(
                cfbd.BettingApi(client), args.season, args.season_type, game_ids, args.provider
            )
            write(lines_df, paths["lines"], "lines")

            consensus = consensus_lines(lines_df)
            if not consensus.empty:
                games_df = games_df.merge(
                    consensus, how="left", left_on="id", right_on="game_id"
                ).drop(columns=["game_id"])
                games_df["n_books"] = games_df["n_books"].fillna(0).astype(int)
            else:
                log.warning("no betting lines returned; games.csv has no line columns")
            write(games_df, paths["games"], "games")

        if want_plays:
            plan = week_plan(cfbd.GamesApi(client), args.season, args.season_type)
            log.info("plays: %d weeks to cover", len(plan))
            plays_df = fetch_plays(
                cfbd.PlaysApi(client),
                args.season,
                plan,
                classification,
                season_dir / "_plays_weekly",
                args.force,
                args.pause,
            )
            write(plays_df, paths["plays"], "plays")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
