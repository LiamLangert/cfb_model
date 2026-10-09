# RAPPA

Regularized Adjusted Predicted Points Added: team ratings from play-by-play.
A RAPM-style ridge regression gives every team four ratings (rush offence,
pass offence, rush defence, pass defence) in points per play against an
average opponent. Each play is worth its value in expected points, measured
from each season's own plays by down, distance and yards to goal
(rappa.expected_points) rather than taken from CFBD's ppa or nflverse's epa,
so that a game's play values add up to its points. The same model runs on college football (CFBD) and the NFL
(nflverse). Walk-forward game predictions are graded against the closing
spread.

## Layout

```
rappa/              shared by both leagues
  model.py          the ridge fit: plays in, team (and optionally passer and conference) ratings out
  validation.py     scoring ratings on holdout plays, one team-game at a time
  games.py          ratings -> game margins, graded against the spread
  special_teams.py  punt and kickoff ratings: each kick valued by expected points, units rated by ridge
  expected_points.py  expected points by down, distance and yards to goal, fitted on each season; each play's value in them
  weeks.py          running a function over many weeks in parallel processes, one thread each
cfb/                college football, from CFBD and ESPN
  pull.py           plays, games and betting lines from the CFBD API
  pull_espn.py      ESPN FPI projections, weekly ratings, and team logos and colours
  pull_offseason.py recruiting classes, returning production, rosters, defensive player stats and head coaches from the CFBD API
  offseason.py      those turned into shifts of each team's prior
  ratings.py        CFBD play prep, passers from the play text, conferences, college constants, writes rappa.csv and qbs.csv
  special_teams.py  CFBD punts and kickoffs for rappa.special_teams, added to the predicted margin
  validate.py       backtests of the rating choices
  predict_games.py  walk-forward predictions across seasons, vs the market
  predict_week.py   one week's predictions, laid out to read
  stage2.py         stage 2's edges scaled by each game's expected plays, and win probability's fitted sigma
  week_parts.py     one week's predictions taken apart for the site: win probability, line, projected score, and the points from each factor
  pace.py           each team's pace and scoring so far, for week_parts' projected total
  previous_week.py  the season rated without its latest week; which prior reproduces rappa.csv
  history.py        the site's weeks: ratings through each week, and every week's predictions
  backtest_week.py  those predictions graded ATS, week by week
  compare_espn.py   RAPPA and ESPN FPI graded ATS on the same games
  benchmark.py      RAPPA, ESPN FPI and the market: MAE and log-loss by season
nfl/                NFL, from nflverse and ESPN
  pull.py           plays, schedules (with closing lines), snap counts and week-1 rosters from nflverse
  pull_espn.py      ESPN FPI projections and weekly ratings
  offseason.py      how much of last season's snaps each team's week-1 roster played, as shifts of its prior
  ratings.py        nflverse play prep, NFL constants, writes rappa.csv and qbs.csv
  special_teams.py  nflverse punts and kickoffs for rappa.special_teams, added to the predicted margin
  validate.py       backtests of the rating choices
  predict_games.py  walk-forward predictions across seasons, vs the market
  key_numbers.py    margins and scores that bunch on 3 and 7: a line set as the market sets one, and a score at it
  pace.py           each team's pace and points in its games so far, for the projected total
  predict_week.py   one week's predictions taken apart: win probability, line, projected score, and the points from each factor
  previous_week.py  the season rated without its latest week; which prior reproduces rappa.csv
  history.py        the site's weeks: ratings through each week, and every week's predictions
  backtest_week.py  those predictions graded ATS, week by week
  compare_espn.py   RAPPA and ESPN FPI graded ATS on the same games
  benchmark.py      RAPPA, ESPN FPI and the market: MAE and log-loss by season
site/               the ratings page for both leagues: team ratings and predictions for any week since 2015
  build.py          gathers both leagues' weekly ratings and predictions into data.json and seasons/
  index.html        the page, with a switch between the NFL and college
  deploy.sh         rebuilds data.json and publishes the page to fb.liamlangert.com
  weekly.sh         the Tuesday update: pulls, re-rates, then deploy.sh
  vercel.json       Vercel's settings: a static site, and redirects from the old site's pages
data/               not in git
  cfb/<season>/     plays, games, lines, espn, fpi, teams, recruiting, returning, roster, player_defense, rappa, qbs, predictions_wkNN
  cfb/coaches.csv   every FBS head coach's seasons
  nfl/<season>/     plays, games, snaps, roster_week1, espn, fpi, rappa, qbs
```

Both leagues run the same pipeline: quarterbacks rated on their own, special
teams added at face value, and a points conversion trained on every earlier
game. Both keep every play, garbage time included, and clip each one's value
to +/-3 points. Each league's other constants (home field, how much of last
season carries over, the quarterback replacement level) are set in its
`ratings.py`, with how each was measured. College also gives every conference
a shared rating, so a team is its conference's part plus its own, and
scales each game's edges by how many plays it should have, the league's
recent plays per game and the two teams' tempo, before the points
conversion (`cfb/stage2.py`), with win probability's sigma fitted on
earlier games' winners. Most of a
team's games are against its own conference, and those cannot tell one
conference from another, so without it the fit left most of the gap between
conferences out and underrated P4 teams against G5 ones by about 6 points a
game. College carries far more of last season forward than the NFL (all of
each conference's average and 0.7 of each team's gap from it, against
0.3-0.5), and moves each team's prior for what changed over the offseason:
its defence by its recruiting and by how much of last season's defence
returns or transferred in (from CFBD's rosters and player stats, whose player
ids follow a transfer), its offence by how much of last season's rushing and
receiving returns and by a coaching change. Its home field is
bigger (0.016 ppa a play against 0.003, about 3 points a game against 2).
The NFL also puts each quarterback's designed runs on his rating, leaves
penalty no-plays out of the fit as college does, and adds the rest gap
between the two teams at the closing line's price of 0.15 points a day. Its
prior moves each team by how much of last season's snaps, offence (bar the
quarterbacks) and defence, its week-1 roster played. Unlike college's shifts,
that one fades out by week 9: kept all season, it hurt the second half as
much as it helped the first.
College counts every carry by a quarterback as one of his dropbacks instead,
since CFBD logs scrambles as runs; rest was worth nothing there.
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
uv run -m cfb.pull_offseason 2025
uv run -m cfb.ratings 2025 --prior data/cfb/2024/rappa.csv
uv run -m cfb.validate 2024 --prior data/cfb/2023/rappa.csv
uv run -m cfb.predict_games 2022 2023 2024 2025 2026
uv run -m cfb.predict_week 2026 5
uv run -m cfb.week_parts 2026
uv run -m cfb.backtest_week 2023 2024 2025 2026
uv run -m cfb.compare_espn 2023 2024 2025 2026
uv run -m cfb.benchmark 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

The seasons chain: each season's rappa.csv and qbs.csv are the next one's
prior, so after changing a constant, re-rate them in order from the first
(2014, which has no prior). `cfb.predict_games` and everything built on it
pick the prior up from `data/cfb/<season - 1>/` by themselves, and the first
season given is a warm-up for the points conversion. `--no-conferences` and
`--no-offseason` (on cfb.ratings, cfb.predict_games and cfb.benchmark) run the
model without conference ratings or offseason shifts; each needs a chain
re-rated with the same flag. `cfb.benchmark` grades FBS-vs-FBS games only,
since every FCS opponent is rated as one pooled team. As with the NFL, it
needs plays, games and rappa.csv from 2014, a warm-up for the points
conversion and 2015's prior, and espn.csv from 2015. CFBD has moneylines only
from 2021, so before that the market's win probability comes from its
spread. It writes `data/cfb/benchmark.csv`.

## NFL

```
uv run nfl/pull.py 2023 2024 2025 2026
uv run -m nfl.pull_espn 2023 2024 2025 2026
uv run -m nfl.ratings 2025 --prior data/nfl/2024/rappa.csv
uv run -m nfl.validate 2023 2024 2025
uv run -m nfl.predict_games 2023 2024 2025 2026
uv run -m nfl.predict_week 2026
uv run -m nfl.backtest_week 2023 2024 2025 2026
uv run -m nfl.compare_espn 2023 2024 2025 2026
uv run -m nfl.benchmark 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

`nfl/pull.py` and `site/build.py` are run by path, not with `-m`. They
need nflreadpy, which needs pydantic 2, while cfbd needs pydantic 1, so they
cannot share the project environment. Each declares its own dependencies
inline (PEP 723), and uv only reads those when a script is run by path.

`nfl.benchmark` needs plays, games and rappa.csv from 2014, and espn.csv
from 2015, when ESPN's projections start. The returning-snap shifts need
snaps.csv from 2014 and roster_week1.csv from 2015 (`--no-offseason` on
nfl.predict_games and nfl.benchmark leaves them out). 2014 is only a warm-up
for the points conversion and 2015's prior. It writes `data/nfl/benchmark.csv`.

## Site

```
uv run site/build.py && python3 -m http.server 8000 -d site
```

One page for both leagues, with a switch between them, a theme switch, and
a "How does it work?" page (#how) that explains the model for a general
reader. Both lists browse any week of any season, and every week shows
only what was known then.

Teams lists every team's ratings through the week picked, with its special
teams in points a game. The build rates each season week by week
(`<league>.history.ratings_by_week`): the fit that reproduces the season's
rappa.csv (with or without last season's prior, as `<league>.previous_week`
checks), on the plays up to each week, with special teams on the same
games' kicks. The last week's is rappa.csv. It keeps the result in
rappa_weekly.csv next to rappa.csv until the season's plays, rappa.csv or
the code change; a season that no fit reproduces shows rappa.csv alone. Run
it after re-rating a season.

Teams are ranked by Rating, in points: the margin RAPPA expects against an
average team on a neutral field, added up as the predictions add it. That is
net at stage 2's points a game for each unit a play, fitted on the games
through the week (about 57 in college, at the week's average plays, and 62
in the NFL), plus special
teams. Before stage 2 has enough games (2014 and the start of 2015) teams
rank by net. Move is each team's change in that rank since the week before.

Predictions lists any week's games since 2015 (college's FBS-vs-FBS ones),
by season and week, opening on the latest season's next week. Each game
shows RAPPA's score at its line (the middle outcome, which bunches on 3 and
7 as the market's does) and win probability, and once it is played the
final score, green when RAPPA's favourite won and red when it lost, with
the week's record above. Every week comes from `<league>.history.predictions`
on one walk forward (`nfl.predict_week.walk`, `cfb.week_parts.walk`, a few
minutes each): ratings, stage 2, sigma and the key-number fits all trained
on the weeks before. The build keeps them in data/<league>/history.csv and
redoes only the weeks not yet final, unless older data or the code changed.
The weeks' score fits run in parallel, a thread each (`rappa.weeks`). The
conference filter narrows both lists. Pull the latest schedule and plays
(`uv run nfl/pull.py <season>`, `uv run -m cfb.pull <season>`) and rebuild
to move a league on a week.

The build writes data.json (the leagues, their seasons and filters) and
seasons/<league>/<season>.json, one file a season, which the page fetches
when the season is picked.

### Publishing

```
site/deploy.sh
```

The page is published at https://fb.liamlangert.com (and
https://liams-rankings.vercel.app), the Vercel project liams-rankings, which
site/ is linked to (site/.vercel). fb is a CNAME in liamlangert.com's
Cloudflare DNS, set to DNS only, as Vercel needs. The script
rebuilds data.json and the season files and runs `vercel deploy --prod`. It needs the Vercel CLI
(`npm install -g vercel`) and a one-time `vercel login`. Only index.html,
data.json and seasons/ are uploaded (site/.vercelignore). vercel.json tells Vercel to
serve the folder as it is, with no build, and redirects the old Next.js
site's pages: /cfb and /cfb/games to College's Teams and Predictions, and
/cbb to the home page. Every earlier deployment stays in the Vercel
dashboard, and `vercel rollback <deployment url>` puts one back.

### Every Tuesday

```
site/weekly.sh               # pull, re-rate, rebuild and publish
site/weekly.sh --no-deploy   # the same, without publishing
```

`site/weekly.sh` is the whole week's update: both leagues' pulls (`--force`,
for a season in progress), the season re-rated (college on last season's
prior, the NFL standalone, as the repo's seasons are), then `site/deploy.sh`.
A league whose pull or re-rating fails is left as it was and the other still
publishes; the script then exits 1. From March through August it does
nothing. Each run's output goes to data/logs/weekly-<date>.log.

Windows Task Scheduler runs it every Tuesday at 10am, as the task "RAPPA
weekly update". A run missed while the laptop was off happens when it is
next on. It needs the Windows user to be logged in, and it can wake the
laptop from sleep. `vercel login`'s session refreshes itself while it is
used. For a token that does not depend on that, make one under Vercel's
Account Settings > Tokens and add it to .env as `VERCEL_TOKEN=...`; the
script passes it to the Vercel CLI.
