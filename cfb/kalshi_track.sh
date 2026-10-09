#!/usr/bin/env bash
# The daily Kalshi check: what became of every order cfb.kalshi_orders placed - fills, queue positions,
# cancellations, settlements and profit - saved under data/kalshi/orders/ (see cfb.kalshi_orders --track).
# It only reads from Kalshi; it never places or cancels anything.
#
# Windows Task Scheduler runs it every day at 10am, as the task "RAPPA Kalshi track". Run it by hand with
# `cfb/kalshi_track.sh`. Each run's output is kept in data/logs/kalshi-track-<date>.log. From March through
# August there are no games, and it does nothing.
set -uo pipefail
cd "$(dirname "$0")/.."

# Task Scheduler starts a bare shell: put uv on the path.
export PATH="$HOME/.local/bin:$PATH"

month=$(date +%-m)
if [ "$month" -ge 3 ] && [ "$month" -le 8 ]; then
    echo "$(date '+%F %T') no games until September - nothing to track"
    exit 0
fi

mkdir -p data/logs
exec > >(tee -a "data/logs/kalshi-track-$(date +%F).log") 2>&1
echo "=== $(date '+%F %T') Kalshi order tracking"
uv run --with cryptography -m cfb.kalshi_orders --track
status=$?
echo "=== $(date '+%F %T') done (exit $status)"
exit $status
