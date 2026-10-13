"""Kalshi: college football maker orders priced off RAPPA, on the shared ../kalshi package. ./bets.sh runs it.

    orders      the League, WEIGHT, the week's games, matching Kalshi's names to them; the planner's command
                line (plan, place, cancel, --track, --smoke)
    daily       the run ./bets.sh uses: the week's plan, placed 2-30 hours before each kickoff, to a week's
                budget, and the review of resting orders
    names       Kalshi's team names made comparable with CFBD's
    display     kalshi.display's tables, with RAPPA and kickoffs

Each runs as `uv run -m betting.<module>`, with --help.
"""
