"""RAPPA's week handed over to ../kalshi, which trades it (kalshi.cfb): the win probabilities and nothing else.

    uv run -m cfb.handoff                 # the season's next week, built first if missing or stale
    uv run -m cfb.handoff 2026 8          # another season and week
    uv run -m cfb.handoff --no-publish    # written to ../kalshi/data/probs/cfb.json, not pushed

./update.sh runs it after re-rating the season. It builds the week's predictions with cfb.week_parts when
data/cfb/<season>/parts_wk<NN>.csv is missing or older than the season's games or plays (--no-build only
warns), then hands over every game of the week not yet played, in kalshi.handoff's columns:

    game_id   CFBD's
    day       the gameday, US Eastern
    start     the kickoff in UTC, from the Eastern gameday and gametime; null while the time is TBD
    home, away, home_names, away_names   CFBD's names (kalshi.cfb.names matches Kalshi's to them)
    p_home    RAPPA's home win probability (rappa_wp)

with the season, the week, and its first and last day (counting the games already played, for kalshi.cfb's
week budget). kalshi.handoff.publish commits it to the kalshi repo's data branch and pushes it, where the
scheduled run (kalshi.run) reads it; the run refuses a handoff more than 8 days old.

On the Sunday and Monday before Tuesday's pull, a week's predictions built then would leave the week before's
games out of the ratings, so a missing week waits for the pull after them ("waiting": nothing is handed
over, and the last handoff stands, its games all kicked off).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from kalshi import handoff as kalshi_handoff

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
# A game is taken to be over this long after kickoff, for whether a pull has its plays.
GAME_LENGTH = timedelta(hours=4)


def pulled_through(season, week, out):
    """Whether the season's last pull (games.csv) came after every earlier week's games were over."""
    from cfb.ratings import continuous_week
    path = out / str(season) / "games.csv"
    games = pd.read_csv(path, low_memory=False)
    start = pd.to_datetime(games.start_date, utc=True)
    before = start[continuous_week(games) < week]
    pulled = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return before.empty or pulled >= before.max() + GAME_LENGTH


def stale(season, week, out):
    """Whether the week's predictions are missing or older than the season's games or plays."""
    path = out / str(season) / f"parts_wk{week:02d}.csv"
    sources = [out / str(season) / name for name in ("games.csv", "plays.csv")]
    return not path.exists() or any(s.exists() and s.stat().st_mtime > path.stat().st_mtime for s in sources)


def ensure_parts(season, week, out):
    """The week's predictions, built with cfb.week_parts when missing or older than games or plays: "ok",
    "built", or "waiting" when missing and the last pull predates the week before's games."""
    path = out / str(season) / f"parts_wk{week:02d}.csv"
    if not stale(season, week, out):
        return "ok"
    if not path.exists() and not pulled_through(season, week, out):
        return "waiting"
    log.info("%s is %s: building it with cfb.week_parts", path, "stale" if path.exists() else "missing")
    from cfb import week_parts
    week_parts.main([str(season), str(week), "--out", str(out)])
    return "built"


def kickoff_utc(game):
    """The game's kickoff in UTC, from its Eastern gameday and gametime; None if the time is TBD."""
    if pd.isna(game.get("gametime")) or pd.isna(game.get("gameday")):
        return None
    return datetime.fromisoformat(f"{game['gameday']}T{game['gametime']}").replace(tzinfo=ET).astimezone(timezone.utc)


def games_of(parts):
    """The week's games not yet played (no home score), as kalshi.handoff's rows."""
    rows = parts[parts.home_score.isna()].to_dict("records")
    return [{"game_id": str(g["game_id"]), "day": str(g["gameday"]), "start": kickoff_utc(g),
             "home": g["home_team"], "away": g["away_team"], "p_home": float(g["rappa_wp"]),
             "home_names": [g["home_team"]], "away_names": [g["away_team"]]} for g in rows]


def hand_over(season, week, out=Path("data/cfb"), publish=True, data=None, made_at=None):
    """Write the week's handoff from its parts file (and publish it): its path."""
    parts = pd.read_csv(out / str(season) / f"parts_wk{week:02d}.csv")
    days = pd.to_datetime(parts.gameday).dt.date
    path = kalshi_handoff.write("cfb", games_of(parts), model="RAPPA", season=season, week=week,
                                first_day=days.min().isoformat(), last_day=days.max().isoformat(),
                                made_at=made_at, data=data)
    if publish:
        kalshi_handoff.publish(path, f"hand over RAPPA's {season} week {week}")
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("season", type=int, nargs="?", help="default: the latest season with games")
    p.add_argument("week", type=int, nargs="?", help="default: cfb.week_parts' next week")
    p.add_argument("--out", type=Path, default=Path("data/cfb"))
    p.add_argument("--no-build", action="store_true", help="don't build missing or stale predictions")
    p.add_argument("--no-publish", action="store_true", help="write the handoff without pushing it")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from cfb.week_parts import upcoming_week
    season = args.season or max(int(p.name) for p in args.out.iterdir() if p.name.isdigit()
                                and (p / "games.csv").exists())
    week = args.week or upcoming_week(season, args.out)
    if week is None:
        print(f"no week ahead in {season}: nothing to hand over")
        return 0
    if args.no_build:
        if not (args.out / str(season) / f"parts_wk{week:02d}.csv").exists():
            raise SystemExit(f"no predictions for week {week} of {season}: run `uv run -m cfb.week_parts {season}`")
        if stale(season, week, args.out):
            log.warning("week %d's predictions are older than the season's games or plays: rerun cfb.week_parts",
                        week)
    elif ensure_parts(season, week, args.out) == "waiting":
        print(f"week {week}'s predictions wait for the pull after week {week - 1}'s games (./update.sh)")
        return 0
    path = hand_over(season, week, args.out, publish=not args.no_publish)
    print(f"handed over week {week} of {season}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
