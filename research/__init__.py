"""Backtests and benchmarks: how each modelling choice in cfb, nfl and rappa was tested.

Nothing in the update or the betting plan runs these. Run from the project
root as `uv run -m research.<module>`, each with --help:

    cfb_validate, nfl_validate          the rating choices, scored on holdout plays
    validation                          the scoring both of those share
    cfb_benchmark, nfl_benchmark        RAPPA, ESPN's FPI and the market: MAE and log-loss, season by season
    cfb_compare_espn, nfl_compare_espn  RAPPA and ESPN graded against the spread on the same games
    cfb_backtest_week, nfl_backtest_week  the walk-forward predictions graded against the spread, week by week
    cfb_predict_week                    one week's college predictions from cfb.predict_games, laid out to read
    cfb_special_teams                   CFBD punts and kickoffs rated by rappa.special_teams (not in predictions)
"""
