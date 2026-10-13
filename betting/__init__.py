"""Kalshi: college football maker orders priced off RAPPA. ./bets.sh runs it.

    kalshi      the API client: signed requests, read-only unless asked, fees, order books, team names
    orders      fair value, guards and Kelly sizing; placing, cancelling, and --track (what became of every order)
    daily       the week's plan, placed 2-30 hours before each kickoff, and the review of resting orders
    display     the terminal tables

Each runs as `uv run --with cryptography -m betting.<module>`, with --help.
"""
