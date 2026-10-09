#!/usr/bin/env bash
# The daily Kalshi run: a --track snapshot of every order placed so far (cfb.kalshi_orders --track), then
# cfb.kalshi_daily's plan for the week, with a Windows notification when there is something to place or
# cancel. Both only read from Kalshi.
#
# Windows Task Scheduler runs it Tuesday to Saturday at 4pm, as the task "RAPPA Kalshi daily"; Tuesday's
# comes after the site's 10am pull, so it builds the new week's predictions first. To act on the
# notification, run `cfb/kalshi_daily.sh --execute` in a terminal: it replans at that moment's prices and
# asks for a typed PLACE (add --yes where there is no keyboard, as at Claude Code's ! prompt). Each run's
# output is kept in data/logs/kalshi-daily-<date>.log. From March through August there are no games, and it
# does nothing.
set -uo pipefail
cd "$(dirname "$0")/.."

# Task Scheduler starts a bare shell: put uv on the path. Unbuffered, so the log keeps up with the run.
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

month=$(date +%-m)
if [ "$month" -ge 3 ] && [ "$month" -le 8 ]; then
    echo "$(date '+%F %T') no games until September - nothing to do"
    exit 0
fi

mkdir -p data/logs
exec > >(tee -a "data/logs/kalshi-daily-$(date +%F).log") 2>&1

if [ "${1:-}" = "--execute" ]; then
    echo "=== $(date '+%F %T') Kalshi daily: placing"
    uv run --quiet --with cryptography -m cfb.kalshi_daily "$@"
    status=$?
    echo "=== $(date '+%F %T') done (exit $status)"
    exit $status
fi

echo "=== $(date '+%F %T') Kalshi order tracking"
uv run --quiet --with cryptography -m cfb.kalshi_orders --track
tracked=$?
echo "=== $(date '+%F %T') Kalshi daily plan"
uv run --quiet --with cryptography -m cfb.kalshi_daily --notify "$@"
planned=$?
echo "=== $(date '+%F %T') done (track exit $tracked, plan exit $planned)"
[ "$tracked" -eq 0 ] && [ "$planned" -eq 0 ]
