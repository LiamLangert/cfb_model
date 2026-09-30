"""College football: RAPPA on CFBD play-by-play, from the FBS point of view, benchmarked against ESPN's FPI.

The same pipeline as nfl, with college constants (see cfb.ratings). Run from
the project root, in order:

    pull                     fetch plays, games and betting lines from the CFBD API
    pull_espn                fetch ESPN's FPI game projections, weekly team ratings, and team logos
    ratings                  fit and write each season's team and quarterback ratings
    special_teams            punt and kickoff ratings (predict_games adds them to the margin)
    validate                 backtest the rating choices on holdout plays (garbage time, prior, decay)
    predict_games            walk-forward game predictions across seasons, vs the market
    predict_week             one week's predictions from predict_games, laid out to read
    backtest_week            those predictions graded against the spread, week by week
    compare_espn             RAPPA and ESPN graded against the spread on the same games
    benchmark                RAPPA, ESPN and the market: MAE and log-loss, season by season
    uv run cfb/site/build.py build the ratings page in cfb/site

The rest run as `uv run -m cfb.<module>`.
"""
