#!/usr/bin/env bash
# The betting plan, any day: what became of the Kalshi orders so far (fills,
# settlements, profit), then the plan for the rest of the week's college games
# at this moment's prices (betting.daily).
#
# Games kicking off in the next 30 hours can be placed now: it asks for a
# typed PLACE before sending anything, and any other answer sends nothing.
# Later games are shown at today's prices, with the budget held for them;
# run it again on a later day to place those. It also reviews resting orders,
# cancelling or replacing any the market has moved past, after the same PLACE.
# If the week's predictions are missing or older than the latest pull, it
# builds them first (a few minutes).
#
#   ./bets.sh                   track, plan, and ask before placing
#   ./bets.sh --bankroll 50     extra arguments go to betting.daily (--help lists them)
#
# Needs KALSHI_ID and KALSHI_PRIVATE_KEY in .env. Each run's output is kept
# in data/logs/bets-<date>.log.
set -uo pipefail
cd "$(dirname "$0")"

# uv on the path, for a shell that has not read the profile. Unbuffered, so the log keeps up with the run.
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

mkdir -p data/logs
exec > >(tee -a "data/logs/bets-$(date +%F).log") 2>&1

echo "=== $(date '+%F %T') orders so far"
uv run --quiet -m betting.orders --track
echo "=== $(date '+%F %T') this week's plan"
uv run --quiet -m betting.daily --execute "$@"
status=$?
echo "=== $(date '+%F %T') done"
exit $status
