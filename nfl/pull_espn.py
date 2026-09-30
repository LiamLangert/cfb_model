"""Pull ESPN's FPI for one or more NFL seasons: its projection for every game, and its weekly team ratings.

Run with `uv run -m nfl.pull_espn 2023 2024 2025`, after nfl/pull.py for the
same seasons: games are looked up by the ESPN event id in its games.csv. Both
come from ESPN's undocumented core API, so they can change without notice.

Writes CSVs under <out>/<season>/ (default data/nfl/<season>/). Re-running
skips anything already on disk unless --force is passed - use it to refresh
a season that is still in progress.

espn.csv is ESPN's matchup predictor, one row per game. espn_margin is the
home team's projected margin, the same convention as spread_line, and
espn_home_wp its win probability. For a game already played ESPN keeps the
pre-game number - it tracks the FPI snapshot from before the game much more
closely than the one after - but for a game not played yet it keeps moving
until kickoff.

fpi.csv is ESPN's team ratings, one row per team per weekly snapshot.
through_week is the last nflverse week whose games a snapshot includes, 0
for the preseason one, so a game in week W joins on through_week == W - 1.
fpi is expected margin against an average team on a neutral field, and is
exactly fpi_off + fpi_def + fpi_st (points; positive is good for the team).
Before 2023 the snapshots' dates are often out of step with the schedule
(most of 2019's are dated 9 September), so it is unclear what they saw.
in_window is True only for a snapshot dated after that week's last game and
before the next week's first.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

DATASETS = ("espn", "fpi")

# ESPN's postseason weeks, by nflverse game_type. Its week 4 is the Pro Bowl.
POSTSEASON = {1: "WC", 2: "DIV", 3: "CON", 5: "SB"}

# Where ESPN's abbreviation differs from nflverse's. The core API's teams are
# per season, so the old names (STL, SD, OAK) already come through right.
ALIASES = {"WSH": "WAS", "LAR": "LA"}

# A snapshot dated sooner than this after a week's last kickoff may not have
# seen that game finish. ESPN's 2025 week 2 and week 18 snapshots are dated
# 4 hours after a late kickoff, when both games were long over.
GAME_LENGTH = pd.Timedelta(hours=3.5)

# Parallel requests. A season is about 285 predictor calls.
WORKERS = 8


def get(url, tries=3):
    """One JSON document from ESPN, or None when there is nothing at that address."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except (urllib.error.URLError, TimeoutError) as e:
            err = e
        time.sleep(2**attempt)
    raise RuntimeError(f"{url}: {err}")


def ref_id(ref):
    """The id at the end of an ESPN $ref link: .../teams/33?lang=en -> '33'."""
    return ref.rsplit("/", 1)[1].split("?")[0]


def load_games(season, out):
    path = out / str(season) / "games.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run nfl/pull.py {season}` first")
    return pd.read_csv(path)


def pull_espn(season, games):
    """ESPN's projection for every game on the schedule."""
    games = games.dropna(subset=["espn"]).astype({"espn": int})

    def one(eid):
        doc = get(f"{CORE}/events/{eid}/competitions/{eid}/predictor")
        if doc is None:
            return None
        home = {s["name"]: s.get("value") for s in doc.get("homeTeam", {}).get("statistics", [])}
        wp = home.get("gameProjection")
        return {"espn": eid, "espn_margin": home.get("teamPredPtDiff"),
                "espn_home_wp": None if wp is None else wp / 100, "modified": doc.get("lastModified")}

    with ThreadPoolExecutor(WORKERS) as ex:
        found = [r for r in ex.map(one, games.espn) if r]
    df = games[["game_id", "espn"]].merge(pd.DataFrame(found), on="espn")
    missing = len(games) - df.espn_margin.notna().sum()
    if missing:
        log.warning("%d espn: no projection for %d of %d games", season, missing, len(games))
    return df


def team_map(season):
    """ESPN team id -> nflverse abbreviation, for that season's names."""
    refs = [t["$ref"] for t in get(f"{CORE}/seasons/{season}/teams?limit=40")["items"]]
    with ThreadPoolExecutor(WORKERS) as ex:
        teams = list(ex.map(get, refs))
    return {t["id"]: ALIASES.get(t["abbreviation"], t["abbreviation"]) for t in teams}


def snapshot(season, season_type, week):
    doc = get(f"{CORE}/seasons/{season}/types/{season_type}/weeks/{week}/powerindex?limit=50")
    return doc.get("items", []) if doc else []


def pull_fpi(season, games):
    """Every weekly FPI snapshot for the season that ESPN has."""
    teams = team_map(season)

    # (ESPN season type, ESPN week, nflverse through_week). Of the preseason
    # snapshots, the latest one ESPN has is the one closest to week 1.
    wanted = [(2, w, w) for w in range(1, int(games.week[games.game_type == "REG"].max()) + 1)]
    for espn_week, game_type in POSTSEASON.items():
        weeks = games.week[games.game_type == game_type]
        if not weeks.empty:
            wanted.append((3, espn_week, int(weeks.iloc[0])))
    with ThreadPoolExecutor(WORKERS) as ex:
        found = list(ex.map(lambda w: snapshot(season, w[0], w[1]), wanted))
    for week in (4, 3, 2, 1):
        pre = snapshot(season, 1, week)
        if pre:
            found.insert(0, pre)
            wanted.insert(0, (1, week, 0))
            break

    rows = []
    for (_, _, through_week), items in zip(wanted, found):
        for item in items:
            p = {q["name"]: q.get("value") for q in item.get("predictives", [])}
            rows.append({
                "season": season, "through_week": through_week, "team": teams[ref_id(item["team"]["$ref"])],
                "fpi": p.get("fpi"), "fpi_off": p.get("epaoffense"), "fpi_def": p.get("epadefense"),
                "fpi_st": p.get("epaspecialteams"), "updated": item.get("lastUpdated"),
            })
    df = pd.DataFrame(rows)
    unknown = set(df.team) - set(games.home_team) - set(games.away_team)
    if unknown:
        log.warning("%d fpi: ESPN teams with no nflverse match: %s", season, sorted(unknown))
    return df.assign(in_window=in_window(df, games))


def in_window(fpi, games):
    """Whether each snapshot is dated after its week's last game and before the next week's first."""
    kickoff = pd.to_datetime(games.gameday + " " + games.gametime, errors="coerce").dt.tz_localize("America/New_York")
    by_week = kickoff.groupby(games.week)
    after = (by_week.max() + GAME_LENGTH).reindex(fpi.through_week).to_numpy()
    before = by_week.min().reindex(fpi.through_week + 1).to_numpy()
    updated = pd.to_datetime(fpi.updated, utc=True).to_numpy()
    # Week 0 has no games to be after; the Super Bowl has no week to be before.
    return (pd.isna(after) | (updated > after)) & (pd.isna(before) | (updated < before))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season year(s), e.g. 2023 2024 2025")
    p.add_argument("--out", type=Path, default=Path("data/nfl"), help="data dir (default: data/nfl)")
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
    pull = {"espn": pull_espn, "fpi": pull_fpi}

    for season in args.seasons:
        games = load_games(season, args.out)
        for dataset in args.datasets:
            path = args.out / str(season) / f"{dataset}.csv"
            if path.exists() and not args.force:
                log.info("%d %s: %s already exists, skipping (--force to refetch)", season, dataset, path)
                continue
            df = pull[dataset](season, games)
            if df.empty:
                log.warning("%d %s: ESPN has nothing, skipping write", season, dataset)
                continue
            df.to_csv(path, index=False)
            log.info("wrote %s (%d rows)", path, len(df))
            if dataset == "fpi" and not df.in_window.all():
                stale = sorted(df.through_week[~df.in_window].unique().tolist())
                log.warning("%d fpi: snapshots dated outside their week for through_week %s", season, stale)
        # Upcoming, not just unplayed: 2022's cancelled BUF-CIN has no result either.
        if (games.result.isna() & (pd.to_datetime(games.gameday) >= pd.Timestamp.today().normalize())).any():
            log.info("%d has games still to play - their projections move until kickoff; "
                     "rerun with --force once they are played", season)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
