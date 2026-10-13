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

## Using it

Two scripts, run from the project root whenever you like:

```
./update.sh                # pull the latest games, re-rate both leagues, rebuild the site's data
~/my_website/deploy.sh     # then publish it to fb.liamlangert.com
./bets.sh                  # Kalshi: the orders so far, this week's plan, and place it if you say so
```

`./update.sh` is the whole update: both leagues' pulls (`--force`, for a
season in progress), the season re-rated (college on last season's prior,
the NFL standalone, as the repo's seasons are), then `site/build.py`. It
defaults to the season in progress (from August; before that, last year's)
and takes another as `./update.sh 2025`. It publishes nothing: deploy the
website to put the update live. A league whose pull or re-rating fails is
left as it was and the other is still rebuilt; the script then exits 1.
Each run's output goes to data/logs/update-<date>.log.

`./bets.sh` first saves and prints what became of every Kalshi order so far
(fills, settlements, profit: `betting.orders --track`), then plans the rest
of the week's college games at this moment's prices (`betting.daily`). Only
games kicking off in the next 30 hours are placed; it shows those, reviews
any resting orders (keep, cancel, or replace), and asks for a typed PLACE
before sending anything. Later games are listed at today's prices with the
week's budget held for them: run it again on a later day to place those.
It builds the week's predictions first if they are missing or older than
the latest pull. Extra arguments go to `betting.daily` (`--bankroll 50`,
`--max-week-share`, ...; `uv run --with cryptography -m betting.daily --help`
lists them). Each run's output goes to data/logs/bets-<date>.log. Nothing
runs on a schedule.

## Layout

```
update.sh           pull, re-rate and rebuild the site's data, for both leagues
bets.sh             Kalshi: track the orders so far, plan the week, place after a typed PLACE
rappa/              the model, shared by both leagues
  model.py          the ridge fit: plays in, team (and optionally passer and conference) ratings out
  games.py          ratings -> game margins and win probabilities, graded against the spread
  special_teams.py  punt and kickoff ratings: each kick valued by expected points, units rated by ridge
  outside.py        special teams and penalties: every point outside the play ratings, per game, rated by team
  expected_points.py  expected points by down, distance and yards to goal, fitted on each season; each play's value in them
  weeks.py          running a function over many weeks in parallel processes, one thread each
cfb/                college football, from CFBD and ESPN
  pull.py           plays, games and betting lines from the CFBD API
  pull_espn.py      ESPN FPI projections, weekly ratings, and team logos and colours
  pull_offseason.py recruiting classes, returning production, rosters, defensive player stats and head coaches from the CFBD API
  offseason.py      those turned into shifts of each team's prior
  ratings.py        CFBD play prep, passers from the play text, conferences, college constants, writes rappa.csv and qbs.csv
  outside.py        CFBD games for rappa.outside: special teams and penalties, added to the predicted margin
  predict_games.py  walk-forward predictions across seasons, vs the market
  stage2.py         stage 2's edges scaled by each game's expected plays, and win probability's fitted sigma
  week_parts.py     one week's predictions taken apart: win probability, line, projected score, and the points from each factor
  pace.py           each team's pace and scoring so far, for week_parts' projected total
  display.py        how the site shows a quarterback's share of the passing rating
  previous_week.py  the season rated without its latest week; which prior reproduces rappa.csv
  history.py        the site's weeks: ratings through each week, and every week's predictions
nfl/                NFL, from nflverse and ESPN
  pull.py           plays, schedules (with closing lines), snap counts and week-1 rosters from nflverse
  pull_espn.py      ESPN FPI projections and weekly ratings
  offseason.py      how much of last season's snaps each team's week-1 roster played, as shifts of its prior
  ratings.py        nflverse play prep, NFL constants, writes rappa.csv and qbs.csv
  outside.py        nflverse games for rappa.outside: special teams and penalties, added to the predicted margin
  special_teams.py  nflverse plays in the frame rappa.special_teams and rappa.expected_points read
  predict_games.py  walk-forward predictions across seasons, vs the market
  key_numbers.py    margins and scores that bunch on 3 and 7: a line set as the market sets one, and a score at it
  pace.py           each team's pace and points in its games so far, for the projected total
  predict_week.py   one week's predictions taken apart: win probability, line, projected score, and the points from each factor
  previous_week.py  the season rated without its latest week; which prior reproduces rappa.csv
  history.py        the site's weeks: ratings through each week, and every week's predictions
betting/            Kalshi, college games only
  kalshi.py         the API client: signed requests, read-only unless asked, fees, order books, team names
  orders.py         fair value, guards and Kelly sizing; placing, cancelling, and --track
  daily.py          the week's plan, placed 2-30 hours before each kickoff, and the review of resting orders
  display.py        the terminal tables
research/           backtests and benchmarks; nothing above runs them
  cfb_validate.py, nfl_validate.py          the rating choices, scored on holdout plays (validation.py)
  cfb_benchmark.py, nfl_benchmark.py        RAPPA, ESPN FPI and the market: MAE and log-loss by season
  cfb_compare_espn.py, nfl_compare_espn.py  RAPPA and ESPN FPI graded ATS on the same games
  cfb_backtest_week.py, nfl_backtest_week.py  the walk-forward predictions graded ATS, week by week
  cfb_predict_week.py                       one week's college predictions, laid out to read
  cfb_special_teams.py                      CFBD punts and kickoffs for rappa.special_teams (not in predictions)
site/build.py       gathers both leagues' weekly ratings and predictions into ~/my_website/fb/data/
tests/              the Kalshi code's tests, with fake transports
data/               not in git
  cfb/<season>/     plays, games, lines, espn, fpi, teams, recruiting, returning, roster, player_defense, rappa, qbs, outside (cache), parts_wkNN
  cfb/coaches.csv   every FBS head coach's seasons
  nfl/<season>/     plays, games, snaps, roster_week1, espn, fpi, rappa, qbs, outside (cache)
  kalshi/orders/    every order placed, its fills and settlements, each run's plan and review
  logs/             each update.sh and bets.sh run's output
```

Both leagues run the same pipeline: quarterbacks rated on their own, special
teams and penalties added to the margin - every point outside the play
ratings, rated game by game (`rappa/outside.py`), at 0.7 - and a points
conversion trained on every earlier game. Both keep every play, garbage time included, and clip each one's value
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
echo "CFBD_API_KEY=..." > .env    # for cfb.pull
# and KALSHI_ID, KALSHI_PRIVATE_KEY in .env for bets.sh
```

The Kalshi code needs `cryptography`, which is not in the project's
dependencies: run it with `uv run --with cryptography`, as bets.sh does. Its
tests: `PYTHONPATH=. uv run --with cryptography --with pytest pytest tests/`.

## Rating seasons from scratch, and the research

update.sh re-rates only the season in progress. The rest is by hand, from
the project root. College:

```
uv run -m cfb.pull 2025
uv run -m cfb.pull_espn 2022 2023 2024 2025 2026
uv run -m cfb.pull_offseason 2025
uv run -m cfb.ratings 2025 --prior data/cfb/2024/rappa.csv
uv run -m cfb.predict_games 2022 2023 2024 2025 2026
uv run -m cfb.week_parts 2026
uv run -m research.cfb_validate 2024 --prior data/cfb/2023/rappa.csv
uv run -m research.cfb_predict_week 2026 5
uv run -m research.cfb_backtest_week 2023 2024 2025 2026
uv run -m research.cfb_compare_espn 2023 2024 2025 2026
uv run -m research.cfb_benchmark 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

The seasons chain: each season's rappa.csv and qbs.csv are the next one's
prior, so after changing a constant, re-rate them in order from the first
(2014, which has no prior). `cfb.predict_games` and everything built on it
pick the prior up from `data/cfb/<season - 1>/` by themselves, and the first
season given is a warm-up for the points conversion. `--no-conferences` and
`--no-offseason` (on cfb.ratings, cfb.predict_games and research.cfb_benchmark) run the
model without conference ratings or offseason shifts; each needs a chain
re-rated with the same flag. `research.cfb_benchmark` grades FBS-vs-FBS games only,
since every FCS opponent is rated as one pooled team. As with the NFL, it
needs plays, games and rappa.csv from 2014, a warm-up for the points
conversion and 2015's prior, and espn.csv from 2015. CFBD has moneylines only
from 2021, so before that the market's win probability comes from its
spread. It writes `data/cfb/benchmark.csv`.

NFL:

```
uv run nfl/pull.py 2023 2024 2025 2026
uv run -m nfl.pull_espn 2023 2024 2025 2026
uv run -m nfl.ratings 2025 --prior data/nfl/2024/rappa.csv
uv run -m nfl.predict_games 2023 2024 2025 2026
uv run -m nfl.predict_week 2026
uv run -m research.nfl_validate 2023 2024 2025
uv run -m research.nfl_backtest_week 2023 2024 2025 2026
uv run -m research.nfl_compare_espn 2023 2024 2025 2026
uv run -m research.nfl_benchmark 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

`nfl/pull.py` and `site/build.py` are run by path, not with `-m`. They
need nflreadpy, which needs pydantic 2, while cfbd needs pydantic 1, so they
cannot share the project environment. Each declares its own dependencies
inline (PEP 723), and uv only reads those when a script is run by path.

`research.nfl_benchmark` needs plays, games and rappa.csv from 2014, and espn.csv
from 2015, when ESPN's projections start. The returning-snap shifts need
snaps.csv from 2014 and roster_week1.csv from 2015 (`--no-offseason` on
nfl.predict_games and research.nfl_benchmark leaves them out). 2014 is only a warm-up
for the points conversion and 2015's prior. It writes `data/nfl/benchmark.csv`.

## Site

```
uv run site/build.py && python3 -m http.server 8000 -d ~/my_website   # then open localhost:8000/fb/
```

The page itself is part of the website, ~/my_website/fb/index.html, styled
with the site's shared look (~/my_website/shared/). This repo only makes its
data. One page for both leagues, with a switch between them, a theme switch, and
a "How does it work?" page (#how) that explains the model for a general
reader. Both lists browse any week of any season, and every week shows
only what was known then.

Teams lists every team's ratings through the week picked, with its special
teams and penalties in points a game. The build rates each season week by week
(`<league>.history.ratings_by_week`): the fit that reproduces the season's
rappa.csv (with or without last season's prior, as `<league>.previous_week`
checks), on the plays up to each week, with special teams on the same
games. The last week's is rappa.csv. It keeps the result in
rappa_weekly.csv next to rappa.csv until the season's plays, rappa.csv or
the code change; a season that no fit reproduces shows rappa.csv alone. Run
it after re-rating a season.

Teams are ranked by Rating, in points: the margin RAPPA expects against an
average team on a neutral field, added up as the predictions add it. That is
net at stage 2's points a game for each unit a play, fitted on the games
through the week (about 57 in college, at the week's average plays, and 62
in the NFL), plus special
teams and penalties. Before stage 2 has enough games (2014 and the start of 2015) teams
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
conference filter narrows both lists. `./update.sh` pulls the latest
schedule and plays and rebuilds, to move both leagues on a week.

The build writes data.json (the leagues, their seasons and filters) and
seasons/<league>/<season>.json, one file a season, which the page fetches
when the season is picked, into ~/my_website/fb/data/
(`RAPPA_SITE_DATA=<folder>` writes them elsewhere). The website keeps them
out of git, but deploys them.

### Publishing

```
~/my_website/deploy.sh
```

The page is published with the rest of the website, the Vercel project
liamlangert: at https://liamlangert.com/fb/ and https://fb.liamlangert.com,
which the website's vercel.json routes to the page, along with the old
site's redirects (/cfb and /cfb/games to College's Teams and Predictions,
/cbb to the home page). fb is a CNAME in liamlangert.com's Cloudflare DNS,
set to DNS only, as Vercel needs. Nothing here deploys: a rebuilt week goes
live the next time the website is deployed. Every earlier deployment stays
in the Vercel dashboard, and `vercel rollback <deployment url>` puts one
back.
