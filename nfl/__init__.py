"""NFL: RAPPA on nflverse play-by-play, benchmarked against ESPN's FPI.

Run from the project root, in order:

    uv run nfl/pull.py       fetch plays, schedules, snaps and week-1 rosters (by path - see its docstring)
    pull_espn                fetch ESPN's FPI game projections and weekly team ratings
    ratings                  fit and write each season's team ratings
    outside                  special teams and penalties: points outside the play ratings (predict_games adds them)
    special_teams            nflverse plays in the frame rappa.special_teams and rappa.expected_points read
    offseason                how much of each lineup returns, as early-season shifts of the prior
    validate                 backtest the rating choices (home field, clip, prior, ...)
    predict_games            walk-forward game predictions across seasons, vs the market
    backtest_week            those predictions graded against the spread, week by week
    compare_espn             RAPPA and ESPN graded against the spread on the same games
    benchmark                RAPPA, ESPN and the market: MAE and log-loss, season by season
    uv run site/build.py     build the ratings page's data, for both leagues, in ~/my_website/fb/data/

The rest run as `uv run -m nfl.<module>`.
"""
