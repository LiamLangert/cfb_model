# /// script
# requires-python = ">=3.12"
# dependencies = ["nflreadpy>=0.1.5"]
# ///
"""Gather every data/nfl/<season>/rappa.csv and qbs.csv, and data/nfl/benchmark.csv, into nfl/site/data.json for index.html.

Run with `uv run nfl/site/build.py`, then serve the folder:
    python3 -m http.server 8000 -d nfl/site
Team names, colours and logos come from nflverse.
"""

from __future__ import annotations

import json
from pathlib import Path

import nflreadpy as nfl
import polars as pl

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "nfl"
RATING_COLUMNS = ("rush_off", "pass_off", "qb", "rush_def", "pass_def", "off", "def", "net")


# A quarterback is ranked in a season once he has this share of the season's
# most dropbacks - about 225 in a full season, and scaled down for one in progress.
QB_QUALIFY = 0.3


def quarterbacks():
    """Each season's qualified quarterbacks, ranked by rating against the average dropback."""
    seasons = {}
    for path in sorted(DATA.glob("*/qbs.csv")):
        games = pl.read_csv(path.parent / "games.csv", infer_schema_length=0)
        # Full names come from the schedule's starters; nflverse's passer names are abbreviated.
        names = pl.concat([
            games.select(pl.col(f"{side}_qb_id").alias("passer_id"), pl.col(f"{side}_qb_name").alias("full"))
            for side in ("home", "away")
        ]).drop_nulls().unique("passer_id", keep="first")
        played = games.filter(pl.col("result").is_not_null())
        qbs = pl.read_csv(path)
        ranked = (qbs.filter(pl.col("dropbacks") >= QB_QUALIFY * qbs["dropbacks"].max())
                  .join(names, on="passer_id", how="left")
                  .with_columns(pl.col("full").fill_null(pl.col("name")))
                  .sort("vs_average", descending=True)
                  .with_row_index("rank", offset=1))
        seasons[path.parent.name] = {
            # Fewer than a full regular season's games played: still in progress.
            "partial": played.height < 256,
            "through_week": int(played["week"].cast(pl.Int32).max()) if played.height else 0,
            "rows": ranked.select(
                pl.col("passer_id").alias("id"), pl.col("full").alias("name"), "team", "dropbacks",
                pl.col("vs_average").alias("rating"), pl.col("rank").cast(pl.Int32),
            ).to_dicts(),
        }
    return seasons


def main():
    teams = nfl.load_teams().select(
        pl.col("team_abbr").alias("team"), "team_name", "team_nick", "team_color",
        "team_color2", "team_logo_espn", "team_conf", "team_division",
    )
    info = {row["team"]: row for row in teams.to_dicts()}

    seasons = {}
    for path in sorted(DATA.glob("*/rappa.csv")):
        ratings = pl.read_csv(path)
        # Against the average team: quarterback-aware files carry offsets that
        # cancel between teams (see nfl.ratings.centered).
        cols = [c for c in RATING_COLUMNS if c in ratings.columns]
        rows = ratings.with_columns([(pl.col(c) - pl.col(c).mean()).round(4) for c in cols]).to_dicts()
        for row in rows:
            row.update({k: v for k, v in info.get(row["team"], {}).items() if k != "team"})
        seasons[path.parent.name] = rows

    if not seasons:
        raise SystemExit(f"no rappa.csv under {DATA} - run `uv run -m nfl.ratings <season>` first")
    out = {"seasons": seasons}

    out["qbs"] = quarterbacks()

    # The accuracy table, when nfl.benchmark has been run.
    bench = DATA / "benchmark.csv"
    if bench.exists():
        out["benchmark"] = pl.read_csv(bench, schema_overrides={"season": pl.String}).to_dicts()
    else:
        print(f"no {bench} - run `uv run -m nfl.benchmark <seasons>` to add the accuracy table")

    (ROOT / "data.json").write_text(json.dumps(out))
    print(f"wrote {ROOT / 'data.json'} ({', '.join(seasons)}{', benchmark' if bench.exists() else ''}, "
          f"quarterbacks for {len(out['qbs'])} seasons)")


if __name__ == "__main__":
    main()
