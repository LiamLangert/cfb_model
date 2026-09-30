# RAPPA

Regularized Adjusted Predicted Points Added: team ratings from play-by-play.
A RAPM-style ridge regression gives every team four ratings (rush offence,
pass offence, rush defence, pass defence) in points per play against an
average opponent. The same model runs on college football (CFBD) and the NFL
(nflverse). Walk-forward game predictions are graded against the closing
spread.

## Layout

```
rappa/              shared by both leagues
  model.py          the ridge fit: plays in, team (and optionally passer and conference) ratings out
  validation.py     scoring ratings on holdout plays, one team-game at a time
  games.py          ratings -> game margins, graded against the spread
cfb/                college football, from CFBD and ESPN
  pull.py           plays, games and betting lines from the CFBD API
  pull_espn.py      ESPN FPI projections, weekly ratings, and team logos and colours
  ratings.py        CFBD play prep, passers from the play text, conferences, college constants, writes rappa.csv and qbs.csv
  special_teams.py  punt and kickoff ratings, valued by expected points, added to the predicted margin
  validate.py       backtests of the rating choices
  predict_games.py  walk-forward predictions across seasons, vs the market
  predict_week.py   one week's predictions, laid out to read
  backtest_week.py  those predictions graded ATS, week by week
  compare_espn.py   RAPPA and ESPN FPI graded ATS on the same games
  benchmark.py      RAPPA, ESPN FPI and the market: MAE and log-loss by season
  site/             static page: team ratings, QB rankings by season, the benchmark table
nfl/                NFL, from nflverse and ESPN
  pull.py           plays and schedules (with closing lines) from nflverse
  pull_espn.py      ESPN FPI projections and weekly ratings
  ratings.py        nflverse play prep, NFL constants, writes rappa.csv and qbs.csv
  special_teams.py  punt and kickoff ratings, added to the predicted margin
  validate.py       backtests of the rating choices
  predict_games.py  walk-forward predictions across seasons, vs the market
  backtest_week.py  those predictions graded ATS, week by week
  compare_espn.py   RAPPA and ESPN FPI graded ATS on the same games
  benchmark.py      RAPPA, ESPN FPI and the market: MAE and log-loss by season
  site/             static page: team ratings, QB rankings by season, the benchmark table
data/               not in git
  cfb/<season>/     plays, games, lines, espn, fpi, teams, rappa, qbs, predictions_wkNN
  nfl/<season>/     plays, games, espn, fpi, rappa, qbs
```

Both leagues run the same pipeline: quarterbacks rated on their own, special
teams added at face value, and a points conversion trained on every earlier
game. Each league's constants (home field, the per-play clip, how much of last
season carries over, the quarterback replacement level) are set in its
`ratings.py`, with how each was measured. College also gives every conference
a shared rating, so a team is its conference's part plus its own. Most of a
team's games are against its own conference, and those cannot tell one
conference from another, so without it the fit left most of the gap between
conferences out and underrated P4 teams against G5 ones by about 6 points a
game. College carries far more of last season forward than the NFL (all of
each conference's average and 0.7 of each team's gap from it, against
0.3-0.5), and its home field is bigger (0.016 ppa a play against 0.003,
about 3 points a game against 2).
Every script takes `--help`, and its docstring explains its method and its
defaults.

## Setup

```
uv sync
echo "CFBD_API_KEY=..." > .env    # only needed for cfb.pull
```

## College

Run from the project root:

```
uv run -m cfb.pull 2025
uv run -m cfb.pull_espn 2022 2023 2024 2025 2026
uv run -m cfb.ratings 2025 --prior data/cfb/2024/rappa.csv
uv run -m cfb.validate 2024 --prior data/cfb/2023/rappa.csv
uv run -m cfb.predict_games 2022 2023 2024 2025 2026
uv run -m cfb.predict_week 2026 5
uv run -m cfb.backtest_week 2023 2024 2025 2026
uv run -m cfb.compare_espn 2023 2024 2025 2026
uv run -m cfb.benchmark 2023 2024 2025 2026
uv run cfb/site/build.py && python3 -m http.server 8001 -d cfb/site
```

The seasons chain: each season's rappa.csv and qbs.csv are the next one's
prior, so after changing a constant, re-rate them in order from the first
(2022, which has no prior). `cfb.predict_games` and everything built on it
pick the prior up from `data/cfb/<season - 1>/` by themselves, and the first
season given is a warm-up for the points conversion. `--no-conferences` (on
cfb.ratings, cfb.predict_games and cfb.benchmark) runs the model without
conference ratings; it needs a chain re-rated with
`cfb.ratings --no-conferences`. `cfb.benchmark` grades
FBS-vs-FBS games only, since every FCS opponent is rated as one pooled team.
It writes `data/cfb/benchmark.csv`, which `cfb/site/build.py` puts on the
site's "Accuracy vs FPI" view.

## NFL

```
uv run nfl/pull.py 2023 2024 2025 2026
uv run -m nfl.pull_espn 2023 2024 2025 2026
uv run -m nfl.ratings 2025 --prior data/nfl/2024/rappa.csv
uv run -m nfl.validate 2023 2024 2025
uv run -m nfl.predict_games 2023 2024 2025 2026
uv run -m nfl.backtest_week 2023 2024 2025 2026
uv run -m nfl.compare_espn 2023 2024 2025 2026
uv run -m nfl.benchmark 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
uv run nfl/site/build.py && python3 -m http.server 8000 -d nfl/site
```

`nfl/pull.py` and `nfl/site/build.py` are run by path, not with `-m`. They
need nflreadpy, which needs pydantic 2, while cfbd needs pydantic 1, so they
cannot share the project environment. Each declares its own dependencies
inline (PEP 723), and uv only reads those when a script is run by path.

`nfl.benchmark` needs plays, games and rappa.csv from 2014, and espn.csv
from 2015, when ESPN's projections start. 2014 is only a warm-up for the
points conversion and 2015's prior. It writes `data/nfl/benchmark.csv`, which `nfl/site/build.py` puts on
the site's "Accuracy vs FPI" view.
