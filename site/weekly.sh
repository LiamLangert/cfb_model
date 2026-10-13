#!/usr/bin/env bash
# The Tuesday update: pull the week just played for both leagues, re-rate the
# season, and rebuild the ratings page's data (next week's predictions
# included) in the website, ~/my_website/fb/data. It publishes nothing: the
# new week goes live with the next deploy of the website (~/my_website/deploy.sh).
#
# Windows Task Scheduler runs it each Tuesday (README, Every Tuesday). Run it
# by hand with `site/weekly.sh`. Each run's output is kept in
# data/logs/weekly-<date>.log.
#
# Each league's pull and re-rating runs on its own, so a failure in one still
# rebuilds the other; the script then exits 1 so the task shows it failed.
# From March through August there are no games, and it does nothing.
set -uo pipefail
cd "$(dirname "$0")/.."

# Task Scheduler starts a bare shell: put uv on the path.
export PATH="$HOME/.local/bin:$PATH"

# A season runs from August, so January's bowls and playoffs count with the year before.
month=$(date +%-m)
season=$(date +%Y)
[ "$month" -lt 8 ] && season=$((season - 1))
if [ "$month" -ge 3 ] && [ "$month" -le 8 ]; then
    echo "$(date '+%F %T') no games until September - nothing to update"
    exit 0
fi

mkdir -p data/logs
exec > >(tee -a "data/logs/weekly-$(date +%F).log") 2>&1
echo "=== $(date '+%F %T') weekly update, $season season"

failed=()
step() {
    echo "--- $*"
    "$@" || { echo "!!! failed: $*"; failed+=("$*"); return 1; }
}

step uv run -m cfb.pull "$season" --force &&
    step uv run -m cfb.ratings "$season" --prior "data/cfb/$((season - 1))/rappa.csv"
# The repo's NFL seasons are rated without a prior (nfl.previous_week).
step uv run nfl/pull.py "$season" --force &&
    step uv run -m nfl.ratings "$season"

step uv run site/build.py

if [ ${#failed[@]} -gt 0 ]; then
    echo "=== $(date '+%F %T') done, with ${#failed[@]} failed: ${failed[*]}"
    exit 1
fi
echo "=== $(date '+%F %T') done"
