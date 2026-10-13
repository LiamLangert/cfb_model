"""College football: RAPPA on CFBD play-by-play, from the FBS point of view.

The same pipeline as nfl, with college constants (see cfb.ratings).
./update.sh runs pull and ratings for the season in progress; the rest are
what those and site/build.py use. Run from the project root:

    pull                     fetch plays, games and betting lines from the CFBD API
    pull_espn                fetch ESPN's FPI game projections, weekly team ratings, and team logos
    pull_offseason           fetch recruiting classes, returning production and head coaches from CFBD
    offseason                turn those into shifts of each team's prior (ratings and predict_games use it)
    ratings                  fit and write each season's team and quarterback ratings
    outside                  special teams and penalties: points outside the play ratings (predict_games adds them)
    predict_games            walk-forward game predictions across seasons, vs the market
    stage2                   each game's edges scaled by its expected plays, and win probability's sigma
    pace                     each team's pace and scoring so far, for the projected total
    week_parts               one week's predictions taken apart: win probability, line, score, each factor's points
    display                  how the site shows a quarterback's share of the passing rating
    previous_week            the season rated without its latest week; which prior reproduces rappa.csv
    history                  the site's weeks: ratings through each week, and every week's predictions

Each runs as `uv run -m cfb.<module>`. The backtests and benchmarks are in research/.
"""
