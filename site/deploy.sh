#!/usr/bin/env bash
# Rebuild data.json and publish the site to Vercel, at fb.liamlangert.com (and liams-rankings.vercel.app).
#
# Run from anywhere: site/deploy.sh. It needs the Vercel CLI (npm install -g vercel) and a one-time `vercel login`;
# the folder's link to the Vercel project is kept in site/.vercel.
# Re-rate the seasons first (README), as site/build.py reads their rappa.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run site/build.py
vercel deploy --prod --cwd site
