"""Pull ESPN's college FPI for one or more seasons: its projection for every game, and its weekly team ratings.

Run with `uv run -m cfb.pull_espn 2023 2024 2025`, after cfb.pull for the
same seasons: CFBD's game ids are ESPN's event ids and its team ids are
ESPN's, so games.csv is all the lookup this needs. Both come from ESPN's
undocumented core API, as in nfl.pull_espn, so they can change without notice.

Writes CSVs under <out>/<season>/ (default data/cfb/<season>/). Re-running
skips anything already on disk unless --force is passed - use it to refresh
a season that is still in progress.

espn.csv is ESPN's matchup predictor, one row per game. espn_margin is the
home team's projected margin, positive when ESPN favours the home team (the
opposite sign to CFBD's spread), and espn_home_wp its win probability. For
a game not played yet it keeps moving until kickoff.

fpi.csv is ESPN's FBS team ratings, one row per team per weekly regular-season
snapshot. through_week is the last CFBD week whose games a snapshot includes,
0 for the preseason one, so a game in week W joins on through_week == W - 1.
fpi is expected margin against an average team on a neutral field (points).
in_window is True only for a snapshot dated after that week's last game and
before the next week's first.

teams.csv is ESPN's name, abbreviation, colours and logo for every FBS team
that season, for cfb/site/build.py; team_id is CFBD's (and ESPN's) id.
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

from nfl.pull_espn import GAME_LENGTH, WORKERS, ref_id

log = logging.getLogger(__name__)

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"

DATASETS = ("espn", "fpi", "teams")


def get(url, tries=3):
    """One JSON document from ESPN, or None when there is nothing at that address.

    nfl.pull_espn.get, except that a 400 also means nothing there: ESPN
    answers that way for college games it never projected.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            err = e
        except (urllib.error.URLError, TimeoutError) as e:
            err = e
        time.sleep(2**attempt)
    raise RuntimeError(f"{url}: {err}")


def load_games(season, out):
    path = out / str(season) / "games.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `uv run -m cfb.pull {season}` first")
    return pd.read_csv(path)


def pull_espn(season, games):
    """ESPN's projection for every game on the schedule."""
    def one(eid):
        doc = get(f"{CORE}/events/{eid}/competitions/{eid}/predictor")
        if doc is None:
            return None
        home = {s["name"]: s.get("value") for s in doc.get("homeTeam", {}).get("statistics", [])}
        wp = home.get("gameProjection")
        return {"game_id": eid, "espn_margin": home.get("teamPredPtDiff"),
                "espn_home_wp": None if wp is None else wp / 100, "modified": doc.get("lastModified")}

    with ThreadPoolExecutor(WORKERS) as ex:
        found = [r for r in ex.map(one, games.id.astype(int)) if r]
    df = pd.DataFrame(found, columns=["game_id", "espn_margin", "espn_home_wp", "modified"])
    missing = len(games) - df.espn_margin.notna().sum()
    if missing:
        log.warning("%d espn: no projection for %d of %d games", season, missing, len(games))
    return df


def snapshot(season, season_type, week):
    doc = get(f"{CORE}/seasons/{season}/types/{season_type}/weeks/{week}/powerindex?limit=200")
    return doc.get("items", []) if doc else []


def pull_fpi(season, games):
    """Every weekly regular-season FPI snapshot for the season that ESPN has, and the latest preseason one."""
    names = dict(zip(games.home_id, games.home_team)) | dict(zip(games.away_id, games.away_team))
    regular = games[games.season_type == "regular"]
    wanted = [(2, w, w) for w in range(1, int(regular.week.max()) + 1)]
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
            team_id = int(ref_id(item["team"]["$ref"]))
            rows.append({
                "season": season, "through_week": through_week, "team_id": team_id,
                "team": names.get(team_id), "fpi": p.get("fpi"), "updated": item.get("lastUpdated"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    unknown = df.team.isna()
    if unknown.any():
        log.warning("%d fpi: %d ESPN team ids not in games.csv", season, df.team_id[unknown].nunique())
    return df.assign(in_window=in_window(df, regular))


def pull_teams(season, games):
    """ESPN's name, abbreviation, colours and logos for the season's FBS teams."""
    fbs = pd.concat([games.loc[games.home_classification.eq("fbs"), ["home_id", "home_team"]].set_axis(["id", "team"], axis=1),
                     games.loc[games.away_classification.eq("fbs"), ["away_id", "away_team"]].set_axis(["id", "team"], axis=1)])
    fbs = fbs.drop_duplicates("id")

    def one(team_id):
        doc = get(f"{CORE}/seasons/{season}/teams/{team_id}")
        if doc is None:
            return None
        logos = {tuple(logo.get("rel", [])): logo["href"] for logo in doc.get("logos", [])}
        return {"team_id": team_id, "espn_name": doc.get("displayName"), "abbreviation": doc.get("abbreviation"),
                "mascot": doc.get("name"), "color": doc.get("color"), "alternate_color": doc.get("alternateColor"),
                "logo": logos.get(("full", "default")), "logo_dark": logos.get(("full", "dark"))}

    with ThreadPoolExecutor(WORKERS) as ex:
        found = [r for r in ex.map(one, fbs.id.astype(int)) if r]
    df = fbs.rename(columns={"id": "team_id"}).merge(pd.DataFrame(found), on="team_id", how="left")
    if df.espn_name.isna().any():
        log.warning("%d teams: ESPN has nothing for %s", season, sorted(df.team[df.espn_name.isna()]))
    return df


def in_window(fpi, games):
    """Whether each snapshot is dated after its week's last game and before the next week's first."""
    kickoff = pd.to_datetime(games.start_date, utc=True)
    by_week = kickoff.groupby(games.week)
    after = (by_week.max() + GAME_LENGTH).reindex(fpi.through_week).to_numpy()
    before = by_week.min().reindex(fpi.through_week + 1).to_numpy()
    updated = pd.to_datetime(fpi.updated, utc=True).to_numpy()
    # Week 0 has no games to be after; the last week has no week to be before.
    return (pd.isna(after) | (updated > after)) & (pd.isna(before) | (updated < before))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seasons", type=int, nargs="+", help="season year(s), e.g. 2023 2024 2025")
    p.add_argument("--out", type=Path, default=Path("data/cfb"), help="data dir (default: data/cfb)")
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
    pull = {"espn": pull_espn, "fpi": pull_fpi, "teams": pull_teams}

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
        # Upcoming, not just unfinished: some games were cancelled and never played.
        ahead = (games.completed != True) & (pd.to_datetime(games.start_date, utc=True) >= pd.Timestamp.now("UTC"))  # noqa: E712
        if ahead.any():
            log.info("%d has games still to play - their projections move until kickoff; "
                     "rerun with --force once they are played", season)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
