# /// script
# requires-python = ">=3.12"
# dependencies = ["nflreadpy>=0.1.5", "numpy>=2", "pandas>=3", "scipy>=1.14", "scikit-learn>=1.5"]
# ///
"""Gather both leagues' ratings and predictions, week by week, into site/data.json and site/seasons/ for index.html.

Run with `uv run site/build.py`, then serve the folder:
    python3 -m http.server 8000 -d site

For each league (data/nfl and data/cfb) and each season with a rappa.csv:

  ratings      the season's team ratings through each of its weeks
               (nfl.history, cfb.history.ratings_by_week): the fit that
               reproduces rappa.csv, on the plays up to the week, special
               teams included. A week's ratings know nothing of later games.
               Kept in <season>/rappa_weekly.csv and refitted when older than
               the season's plays, rappa.csv or the code; a season that no fit
               reproduces shows rappa.csv alone.
  rating       each team's rating in points, which the page ranks by: net at
               stage 2's points a game for each unit a play, fitted on the
               games through the week, plus special teams. Before stage 2 has
               enough games (2014, the start of 2015) teams rank by net.
  predictions  every week's games since 2015 (history.predictions): win
               probability and the score at RAPPA's line, each week from
               ratings and fits on the weeks before it, and the final score
               once played. One walk forward (nfl.predict_week.walk,
               cfb.week_parts.walk, a few minutes) serves every week; the
               result is kept in data/<league>/history.csv and
               history_ppp.csv, and a rerun recomputes only the weeks not
               yet final, unless older data or the code changed.

Each season goes to site/seasons/<league>/<season>.json, which the page
fetches when the season is picked. data.json holds, per league: the
seasons, the seasons with predictions and the week the Predictions page
opens on, filters (the conference and division choices), progress
(whether a season is still being played), teams (names, colours and logos)
and facts (the latest figures the page's explanation quotes).

NFL team names, colours and logos come from nflverse; college ones from each
season's teams.csv (`uv run -m cfb.pull_espn <season> --datasets teams`),
and conferences from its games.csv. College's pooled FCS team is left off.

Run by path, not with -m: nflreadpy needs pydantic 2 and cfbd needs
pydantic 1, so this declares its own dependencies (PEP 723).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from cfb import history as cfb_history  # noqa: E402
from cfb import previous_week as cfb_previous_week  # noqa: E402
from cfb import special_teams as cfb_special_teams  # noqa: E402
from cfb import week_parts as cfb_week_parts  # noqa: E402
from cfb.display import season_pass_level, with_pass_level  # noqa: E402
from cfb.ratings import FCS  # noqa: E402
from nfl import history as nfl_history  # noqa: E402
from nfl import predict_week as nfl_predict_week  # noqa: E402
from nfl import previous_week as nfl_previous_week  # noqa: E402
from nfl import special_teams as nfl_special_teams  # noqa: E402
from rappa import special_teams as shared_special_teams  # noqa: E402
from rappa import weeks as rappa_weeks  # noqa: E402

DATA = ROOT.parent / "data"
# One file a season, which the page loads when that season is picked.
SEASONS = ROOT / "seasons"
RATING_COLUMNS = ("rush_off", "pass_off", "qb", "rush_def", "pass_def", "off", "def", "net")

# The college conferences grouped as the power ones: the Pac-12 until it
# broke up after 2023, and Notre Dame, the one power independent.
POWER = {"SEC", "Big Ten", "Big 12", "ACC"}
POWER_UNTIL = {"Pac-12": 2023}
POWER_TEAMS = {"Notre Dame"}


def cached(path, sources):
    """Whether path exists and is newer than every one of sources."""
    return path.exists() and all(path.stat().st_mtime >= p.stat().st_mtime for p in sources)


def special_teams(season_dir, module, inputs):
    """A season's special-teams ratings (points a game), refitted when older than its inputs or the code."""
    cache = season_dir / "special_teams.csv"
    sources = [season_dir / f for f in inputs] + [Path(module.__file__), Path(shared_special_teams.__file__)]
    if cached(cache, sources):
        return pd.read_csv(cache, index_col="team")
    print(f"rating special teams for {season_dir.parent.name} {season_dir.name}")
    ratings = module.fit(module.load(int(season_dir.name), season_dir.parent))
    ratings.round(4).to_csv(cache)
    return ratings


def strength(table, slope):
    """Each team's rating in points, what the page ranks by: net at slope points a game for each unit a play, plus
    special teams, a team with no kicks counting as average. The margin RAPPA expects against an average team on a
    neutral field, as the predictions add them. Net alone when slope is None."""
    if slope is None:
        return table.net
    return slope * table.net + table.st.fillna(0.0)


def centered(table):
    """Ratings against the average team shown: quarterback-aware files carry offsets that cancel between teams."""
    cols = [c for c in (*RATING_COLUMNS, "st") if c in table]
    return table.assign(**{c: (table[c] - table[c].mean()).round(4) for c in cols})


def ratings(r):
    """One team's ratings, with a missing one (no kicks, say) as null."""
    return {k: None if pd.isna(v) else float(v) for k, v in r.items()}


def num(v, digits=3):
    """A number for data.json, missing as null."""
    return None if v is None or pd.isna(v) else round(float(v), digits)


# ---- Week by week ----------------------------------------------------------

def weekly_ratings(season_dir, history, previous_week):
    """A season's ratings through each of its weeks (history.ratings_by_week), one row per team and week, or None
    when no fit reproduces its rappa.csv. Kept in rappa_weekly.csv and refitted when older than its inputs or code."""
    cache = season_dir / "rappa_weekly.csv"
    sources = previous_week.sources(int(season_dir.name), season_dir.parent) + [Path(history.__file__)]
    if cached(cache, sources):
        frame = pd.read_csv(cache)
        return None if frame.empty else frame
    print(f"rating {season_dir.parent.name} {season_dir.name} week by week")
    weeks = history.ratings_by_week(int(season_dir.name), season_dir.parent)
    if weeks is None:
        pd.DataFrame(columns=["week"]).to_csv(cache, index=False)
        return None
    frames = []
    for week, label, table, *level in weeks:
        table = table.assign(week=week, label=label)
        if level:
            table["level"] = table.index.map(level[0])
        frames.append(table.rename_axis("team").reset_index())
    pd.concat(frames, ignore_index=True).round(4).to_csv(cache, index=False)
    # Read back, so a fresh fit shows exactly as a kept one does.
    return pd.read_csv(cache)


def season_weeks(path, history, previous_week, through_week):
    """[(week, label, table)]: a season's weekly ratings, or rappa.csv alone, as its latest week, without them."""
    frame = weekly_ratings(path.parent, history, previous_week)
    if frame is None:
        return [(through_week, "Latest", pd.read_csv(path, index_col="team"))]
    return [(int(week), group.label.iloc[0], group.drop(columns=["week", "label"]).set_index("team"))
            for week, group in frame.groupby("week", sort=True)]


def predicted_weeks(out, module, history):
    """(games, ppp): every week's predictions since module.FIRST_SEASON (history.predictions), and stage 2's points
    per play through each week (history.points_per_play), by season * 100 + week.

    Both come from one walk-forward run (module.walk, a few minutes), kept in
    history.csv and history_ppp.csv under out until older than its inputs or
    code. A rerun redoes only the weeks the cache can't vouch for: a week
    every game of which was final is kept, unless something other than the
    latest season's own files changed.
    """
    season = max(int(p.parent.name) for p in out.glob("*/games.csv"))
    sources = module.sources(season, out) + [Path(history.__file__), Path(rappa_weeks.__file__)]
    games_path, ppp_path = out / "history.csv", out / "history_ppp.csv"
    if cached(games_path, sources) and cached(ppp_path, sources):
        return pd.read_csv(games_path, low_memory=False), pd.read_csv(ppp_path, index_col="slot").ppp
    print(f"predicting every {out.name} week since {module.FIRST_SEASON} (a walk forward: a few minutes)")
    rows, preds = module.walk(season, out)
    kept = pd.DataFrame()
    current = out / str(season)
    if games_path.exists() and cached(games_path, [p for p in sources if current not in p.parents]):
        old = pd.read_csv(games_path, low_memory=False)
        slot = old.season * 100 + old.week
        final = (old.home_score.notna() & old.away_score.notna()).groupby(slot).all()
        kept = old[slot.isin(final.index[final])]
    new = history.predictions(rows, preds, skip=set(kept.season * 100 + kept.week) if len(kept) else ())
    games = pd.concat([kept, new], ignore_index=True).sort_values(["season", "week"], kind="stable")
    games.to_csv(games_path, index=False)
    slots = sorted(set(rows.season * 100 + rows.week))
    ppp = pd.Series([history.points_per_play(rows, s // 100, s % 100) for s in slots],
                    index=pd.Index(slots, name="slot"), name="ppp", dtype=float)
    ppp.round(4).to_csv(ppp_path)
    # Read back, so a fresh run shows exactly as a kept one does.
    return pd.read_csv(games_path, low_memory=False), pd.read_csv(ppp_path, index_col="slot").ppp


def ppp_through(ppp, season, week):
    """Stage 2's points per play through a week: the latest fitted on games no later than it, None before any."""
    known = ppp[(ppp.index <= season * 100 + week) & ppp.notna()]
    return None if known.empty else float(known.iloc[-1])


def game_entry(r):
    """One predicted game, as the page shows it: the score at RAPPA's line, win chances and the final."""
    def side(s):
        return {"team": r[f"{s}_team"], "points": int(r[f"{s}_points_at_line"]),
                "score": None if pd.isna(r[f"{s}_score"]) else int(r[f"{s}_score"])}
    return {"id": r["game_id"], "gameday": r["gameday"],
            "gametime": r["gametime"] if pd.notna(r["gametime"]) else None, "neutral": bool(r["neutral_site"]),
            "home_wp": num(r["rappa_wp"], 4), "away": side("away"), "home": side("home")}


def games_by_season(games):
    """{season: [week entries]} of the predictions with a win probability: stage 2 and sigma wait for enough games."""
    games = games[games.rappa_wp.notna()]
    out = {}
    for (season, week), g in games.groupby(["season", "week"], sort=True):
        out.setdefault(str(season), []).append({"week": int(week), "label": g.label.iloc[0],
                                                "games": [game_entry(r) for r in g.to_dict("records")]})
    return out


def default_week(games):
    """The week the Predictions page opens on: the latest season's first with a game not yet played, else its
    last."""
    season = max(games, key=int)
    weeks = games[season]
    ahead = [w for w in weeks if any(g["home"]["score"] is None for g in w["games"])]
    return {"season": season, "week": (ahead[0] if ahead else weeks[-1])["week"]}


def facts(games):
    """The figures the explanation quotes that move week to week: the latest week's home field and sigma."""
    last = games[games.rappa_wp.notna()].iloc[-1]
    return {"hfa": num(last.hfa, 2), "sigma": num(last.sigma, 2)}


def write_season(league, season, data):
    """One season's weeks, ratings and predictions, as site/seasons/<league>/<season>.json, which the page loads
    when it is picked."""
    path = SEASONS / league / f"{season}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")))


# ---- NFL -------------------------------------------------------------------

def nfl_league():
    out = DATA / "nfl"
    info = {t["team_abbr"]: t for t in nfl.load_teams().to_dicts()}
    league = {"name": "NFL", "seasons": [], "filters": {}, "progress": {}, "teams": {}}
    games, ppp = predicted_weeks(out, nfl_predict_week, nfl_history)
    by_season = games_by_season(games)
    for path in sorted(out.glob("*/rappa.csv")):
        season = path.parent.name
        played = pd.read_csv(path.parent / "games.csv", low_memory=False)
        played = played[played.result.notna()]
        # Fewer than a full regular season's games played: still in progress.
        through = int(played.week.max()) if len(played) else 0
        league["progress"][season] = {"partial": len(played) < 256, "through_week": through}
        st_full = special_teams(path.parent, nfl_special_teams, ["plays.csv"]).value
        weeks = []
        for week, label, table in season_weeks(path, nfl_history, nfl_previous_week, through):
            table = table.drop(columns=["plays_off", "plays_def"], errors="ignore")
            if "st" not in table:
                table["st"] = st_full
            table = centered(table)
            slope = ppp_through(ppp, int(season), week)
            if slope is not None:
                table["rating"] = strength(table, slope).round(2)
            rows = [{"team": team, "conference": info.get(team, {}).get("team_conf"),
                     "division": info.get(team, {}).get("team_division"), **ratings(r)}
                    for team, r in table.iterrows()]
            weeks.append({"week": week, "label": label, "points_per_play": num(slope), "teams": rows})
        write_season("nfl", season, {"ratings": weeks, "games": by_season.get(season, [])})
        league["seasons"].append(season)
        divisions = sorted({info[t]["team_division"] for t in info if info[t].get("team_division")})
        league["filters"][season] = [["", [["all", "All teams"], ["conference:AFC", "AFC"], ["conference:NFC", "NFC"]]],
                                     ["Division", [[f"division:{d}", d] for d in divisions]]]

    for abbr, t in info.items():
        league["teams"][abbr] = {"name": t["team_name"], "abbr": abbr, "color": t["team_color"],
                                 "logo": t["team_logo_espn"], "logo_dark": None}
    league["game_seasons"] = sorted(by_season, key=int)
    league["games_default"] = default_week(by_season)
    league["facts"] = {"points_per_play": num(ppp.dropna().iloc[-1]), **facts(games)}
    return league


# ---- College ---------------------------------------------------------------

def conferences(games):
    """Each FBS team's conference that season: the one most of its games list it in."""
    sides = [games.loc[games[f"{s}_classification"].eq("fbs"), [f"{s}_team", f"{s}_conference"]]
             .set_axis(["team", "conference"], axis=1) for s in ("home", "away")]
    return pd.concat(sides).groupby("team").conference.agg(lambda c: c.mode().iat[0])


def cfb_progress(games):
    """Whether the season still has games to play, and the last regular-season week with a result."""
    ahead = (games.completed != True) & (pd.to_datetime(games.start_date, utc=True) >= pd.Timestamp.now("UTC"))  # noqa: E712
    done = games[(games.completed == True) & (games.season_type == "regular")]  # noqa: E712
    return {"partial": bool(ahead.any()), "through_week": int(done.week.max()) if len(done) else 0}


def cfb_teams(out):
    """Names, colours and logos by team, from every season's teams.csv; a later season's wins."""
    frames = [pd.read_csv(p) for p in sorted(out.glob("*/teams.csv"))]
    if not frames:
        print(f"no teams.csv under {out} - run `uv run -m cfb.pull_espn <season> --datasets teams` for logos")
        return {}
    teams = pd.concat(frames).drop_duplicates("team", keep="last").set_index("team")
    return {team: {"name": team, "abbr": r.abbreviation if pd.notna(r.abbreviation) else "",
                   "color": "#" + (r.color if pd.notna(r.color) else "888888"),
                   "logo": r.logo if pd.notna(r.logo) else None,
                   "logo_dark": r.logo_dark if pd.notna(r.logo_dark) else None}
            for team, r in teams.iterrows()}


def cfb_league():
    out = DATA / "cfb"
    info = cfb_teams(out)
    league = {"name": "College", "seasons": [], "filters": {}, "progress": {}, "teams": info}
    games, ppp = predicted_weeks(out, cfb_week_parts, cfb_history)
    by_season = games_by_season(games)
    for path in sorted(out.glob("*/rappa.csv")):
        season = path.parent.name
        season_games = pd.read_csv(path.parent / "games.csv", low_memory=False)
        league["progress"][season] = cfb_progress(season_games)
        conf = conferences(season_games)
        st_full = special_teams(path.parent, cfb_special_teams, ["plays.csv", "games.csv"]).value
        full_level = season_pass_level(path.parent)
        weeks, power = [], {}
        for week, label, table in season_weeks(path, cfb_history, cfb_previous_week,
                                               league["progress"][season]["through_week"]):
            table = table.drop(columns=["plays_off", "plays_def"], errors="ignore")
            if "st" not in table:
                table["st"] = st_full
            level = table.pop("level") if "level" in table else full_level
            table = centered(table.drop(index=FCS, errors="ignore"))
            # A share of each conference's average passing game is shown on its quarterbacks (cfb.display).
            if level is not None and "qb" in table:
                moved = with_pass_level(table, level.dropna())
                table = table.assign(pass_off=moved.pass_off.round(4), qb=moved.qb.round(4))
            slope = ppp_through(ppp, int(season), week)
            if slope is not None:
                table["rating"] = strength(table, slope).round(2)
            rows = []
            for team, r in table.iterrows():
                c = conf.get(team, "")
                power[c] = c in POWER or POWER_UNTIL.get(c, 0) >= int(season)
                rows.append({"team": team, "conference": c, "power": power[c] or team in POWER_TEAMS,
                             **ratings(r)})
            weeks.append({"week": week, "label": label, "points_per_play": num(slope), "teams": rows})
        write_season("cfb", season, {"ratings": weeks, "games": by_season.get(season, [])})
        league["seasons"].append(season)
        # Power conferences first, then the rest, independents last.
        confs = sorted((c for c in power if c), key=lambda c: (not power[c], c == "FBS Independents", c))
        league["filters"][season] = [["", [["all", "All FBS"], ["power", "Power conferences"], ["g5", "Group of Five"]]],
                                     ["Conference", [[f"conference:{c}", c] for c in confs]]]
    league["game_seasons"] = sorted(by_season, key=int)
    league["games_default"] = default_week(by_season)
    league["facts"] = {"points_per_play": num(ppp.dropna().iloc[-1]), **facts(games)}
    return league


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    leagues = {}
    for key, build in (("nfl", nfl_league), ("cfb", cfb_league)):
        if not any((DATA / key).glob("*/rappa.csv")):
            print(f"no rappa.csv under {DATA / key} - run `uv run -m {key}.ratings <season>` to add the {key} ratings")
            continue
        leagues[key] = build()
    if not leagues:
        raise SystemExit(f"no ratings under {DATA}")
    (ROOT / "data.json").write_text(json.dumps(leagues, allow_nan=False))
    for key, league in leagues.items():
        seasons = league["seasons"]
        print(f"{league['name']}: ratings for {seasons[0]}-{seasons[-1]}, predictions for "
              f"{league['game_seasons'][0]}-{league['game_seasons'][-1]}")
    print(f"wrote {ROOT / 'data.json'} and {SEASONS}")


if __name__ == "__main__":
    main()
