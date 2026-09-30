"""NFL: RAPPA on nflverse play-by-play, benchmarked against ESPN's FPI.

Run from the project root, in order:

    uv run nfl/pull.py       fetch plays and schedules from nflverse (by path - see its docstring)
    pull_espn                fetch ESPN's FPI game projections and weekly team ratings
    ratings                  fit and write each season's team ratings
    special_teams            punt and kickoff ratings (predict_games adds them to the margin)
    validate                 backtest the rating choices (home field, clip, prior, ...)
    predict_games            walk-forward game predictions across seasons, vs the market
    backtest_week            those predictions graded against the spread, week by week
    compare_espn             RAPPA and ESPN graded against the spread on the same games
    benchmark                RAPPA, ESPN and the market: MAE and log-loss, season by season
    uv run nfl/site/build.py build the ratings page in nfl/site

The rest run as `uv run -m nfl.<module>`.
"""
