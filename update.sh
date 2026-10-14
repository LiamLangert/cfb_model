#!/usr/bin/env bash
# Update the ratings, any day: pull the latest plays and games for both
# leagues, re-rate the season, and rebuild the ratings page's data (next
# week's predictions included) in the website, ~/my_website/fb/data. The
# site goes live with the next deploy of the website (~/my_website/deploy.sh).
# Once college is re-rated, its next week's win probabilities are handed over
# to ../kalshi, which trades them (cfb.handoff: pushed to the kalshi repo's
# data branch).
#
#   ./update.sh          the season in progress (from August; before then, last year's)
#   ./update.sh 2025     another season
#
# Each league's pull and re-rating runs on its own, so a failure in one still
# rebuilds the other; the script then exits 1. Each run's output is kept in
# data/logs/update-<date>.log.
set -uo pipefail
cd "$(dirname "$0")"

# uv on the path, for a shell that has not read the profile.
export PATH="$HOME/.local/bin:$PATH"

# A season runs from August, so January's bowls and playoffs count with the year before.
season=${1:-$(date +%Y)}
[ $# -eq 0 ] && [ "$(date +%-m)" -lt 8 ] && season=$((season - 1))

mkdir -p data/logs
exec > >(tee -a "data/logs/update-$(date +%F).log") 2>&1
echo "=== $(date '+%F %T') update, $season season"

failed=()
step() {
    echo "--- $*"
    "$@" || { echo "!!! failed: $*"; failed+=("$*"); return 1; }
}

step uv run -m cfb.pull "$season" --force &&
    step uv run -m cfb.ratings "$season" --prior "data/cfb/$((season - 1))/rappa.csv" &&
    step uv run -m cfb.handoff "$season"
# The repo's NFL seasons are rated without a prior (nfl.previous_week).
step uv run nfl/pull.py "$season" --force &&
    step uv run -m nfl.ratings "$season"

step uv run site/build.py

if [ ${#failed[@]} -gt 0 ]; then
    echo "=== $(date '+%F %T') done, with ${#failed[@]} failed: ${failed[*]}"
    exit 1
fi
echo "=== $(date '+%F %T') done: publish with ~/my_website/deploy.sh"
