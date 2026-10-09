"""Gather every data/cfb/<season>/rappa.csv and qbs.csv, and data/cfb/benchmark.csv, into cfb/site/data.json for index.html.

Run with `uv run cfb/site/build.py`, then serve the folder:
    python3 -m http.server 8001 -d cfb/site
Team names, colours and logos come from each season's teams.csv
(`uv run -m cfb.pull_espn <season> --datasets teams`), and conferences from
its games.csv. The pooled FCS team is left off the page.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from cfb.display import QB_SHARE, season_pass_level, with_pass_level  # noqa: E402
from cfb.ratings import FCS, previous_ids  # noqa: E402

DATA = ROOT.parents[1] / "data" / "cfb"
RATING_COLUMNS = ("rush_off", "pass_off", "qb", "rush_def", "pass_def", "off", "def", "net")

# The conferences the site groups as the power ones: the Pac-12 until it
# broke up after 2023, and Notre Dame, the one power independent.
POWER = {"SEC", "Big Ten", "Big 12", "ACC"}
POWER_UNTIL = {"Pac-12": 2023}
POWER_TEAMS = {"Notre Dame"}

# A quarterback is ranked in a season once he has this share of the season's
# most dropbacks - about 160 in a full season, and scaled down for one in progress.
QB_QUALIFY = 0.3


def conferences(games):
    """Each FBS team's conference that season: the one most of its games list it in."""
    sides = [games.loc[games[f"{s}_classification"].eq("fbs"), [f"{s}_team", f"{s}_conference"]]
             .set_axis(["team", "conference"], axis=1) for s in ("home", "away")]
    return pd.concat(sides).groupby("team").conference.agg(lambda c: c.mode().iat[0])


def progress(games):
    """Whether the season still has games to play, and the last regular-season week with a result."""
    ahead = (games.completed != True) & (pd.to_datetime(games.start_date, utc=True) >= pd.Timestamp.now("UTC"))  # noqa: E712
    done = games[(games.completed == True) & (games.season_type == "regular")]  # noqa: E712
    return bool(ahead.any()), int(done.week.max()) if len(done) else 0


def teams(season_dir):
    path = season_dir / "teams.csv"
    if not path.exists():
        print(f"no {path} - run `uv run -m cfb.pull_espn {season_dir.name} --datasets teams` for logos and colours")
        return pd.DataFrame(columns=["team"]).set_index("team")
    return pd.read_csv(path).drop_duplicates("team").set_index("team")


def ratings():
    seasons, confs = {}, {}
    for path in sorted(DATA.glob("*/rappa.csv")):
        season = int(path.parent.name)
        games = pd.read_csv(path.parent / "games.csv")
        conf = conferences(games)
        info = teams(path.parent)
        table = pd.read_csv(path)
        table = table[table.team != FCS]
        # Against the average FBS team: quarterback-aware files carry offsets
        # that cancel between teams (see cfb.ratings.centered).
        cols = [c for c in RATING_COLUMNS if c in table]
        table = table.assign(**{c: (table[c] - table[c].mean()).round(4) for c in cols})
        # A share of each conference's average passing game is shown on its quarterbacks (cfb.display).
        level = season_pass_level(path.parent)
        if level is not None and "qb" in table:
            table = with_pass_level(table.set_index("team"), level).round(4).reset_index()
        table["conference"] = table.team.map(conf).fillna("")
        table["power"] = (table.conference.isin(POWER) | table.team.isin(POWER_TEAMS)
                          | table.conference.map(POWER_UNTIL).ge(season))
        for col in ("espn_name", "abbreviation", "mascot", "color", "logo", "logo_dark"):
            table[col] = table.team.map(info[col]) if col in info else None
        table["color"] = "#" + table.color.fillna("888888")
        seasons[str(season)] = json.loads(table.to_json(orient="records"))
        order = table.groupby("conference").power.all().sort_index()
        confs[str(season)] = sorted(order.index, key=lambda c: (not order[c], c == "FBS Independents", c))
    return seasons, confs


def quarterbacks():
    """Each season's qualified quarterbacks, ranked by rating against the average dropback.

    The rating includes a share of his conference's average passing game
    (cfb.display), so it compares across conferences.

    passer_id includes the team, so a quarterback who transfers gets a new
    one; career follows him across seasons the way his prior does
    (cfb.ratings.previous_ids).
    """
    seasons, career, last = {}, {}, None
    for path in sorted(DATA.glob("*/qbs.csv")):
        qbs = pd.read_csv(path, index_col="passer_id")
        before = previous_ids(last.index, qbs.index) if last is not None else pd.Series(dtype=object)
        season_career = {q: career.get(before.get(q), q) for q in qbs.index}
        games = pd.read_csv(path.parent / "games.csv")
        partial, through = progress(games)
        fbs = qbs[qbs.team != FCS]
        level = season_pass_level(path.parent)
        fbs = fbs.assign(shown=fbs.vs_average + QB_SHARE * fbs.team.map(level).fillna(0.0))
        ranked = fbs[fbs.dropbacks >= QB_QUALIFY * fbs.dropbacks.max()].sort_values("shown", ascending=False)
        seasons[path.parent.name] = {
            "partial": partial,
            "through_week": through,
            "rows": [{"id": season_career[q], "name": r["name"], "team": r.team, "dropbacks": int(r.dropbacks),
                      "rating": round(float(r.shown), 4), "rank": k}
                     for k, (q, r) in enumerate(ranked.iterrows(), start=1)],
        }
        career, last = season_career, qbs
    return seasons


def main():
    seasons, confs = ratings()
    if not seasons:
        raise SystemExit(f"no rappa.csv under {DATA} - run `uv run -m cfb.ratings <season>` first")
    out = {"seasons": seasons, "conferences": confs, "qbs": quarterbacks()}

    # The accuracy table, when cfb.benchmark has been run.
    bench = DATA / "benchmark.csv"
    if bench.exists():
        out["benchmark"] = json.loads(pd.read_csv(bench, dtype={"season": str}).to_json(orient="records"))
    else:
        print(f"no {bench} - run `uv run -m cfb.benchmark <seasons>` to add the accuracy table")

    (ROOT / "data.json").write_text(json.dumps(out))
    print(f"wrote {ROOT / 'data.json'} ({', '.join(seasons)}{', benchmark' if bench.exists() else ''}, "
          f"quarterbacks for {len(out['qbs'])} seasons)")


if __name__ == "__main__":
    main()
