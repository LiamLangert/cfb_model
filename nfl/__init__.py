"""NFL: RAPPA on nflverse play-by-play.

./update.sh runs pull and ratings for the season in progress; the rest are
what those and site/build.py use. Run from the project root:

    uv run nfl/pull.py       fetch plays, schedules, snaps and week-1 rosters (by path - see its docstring)
    pull_espn                fetch ESPN's FPI game projections and weekly team ratings
    ratings                  fit and write each season's team ratings
    outside                  special teams and penalties: points outside the play ratings (predict_games adds them)
    special_teams            nflverse plays in the frame rappa.special_teams and rappa.expected_points read
    offseason                how much of each lineup returns, as early-season shifts of the prior
    predict_games            walk-forward game predictions across seasons, vs the market
    key_numbers              margins and scores that bunch on 3 and 7: a line set as the market sets one
    pace                     each team's pace and points so far, for the projected total
    predict_week             one week's predictions taken apart: win probability, line, score, each factor's points
    previous_week            the season rated without its latest week; which prior reproduces rappa.csv
    history                  the site's weeks: ratings through each week, and every week's predictions

The rest run as `uv run -m nfl.<module>`. The backtests and benchmarks are in research/.
"""
