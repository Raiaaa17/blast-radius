#!/usr/bin/env bash
# Score a selector image on a labeled corpus under the scorer's exact docker
# flags. Usage: eval/score_image.sh <image> <corpus-dir> [--batch N] [--runs 3]
#   [--limit 180] [--json out.json] [--keep-output dir]
# See eval/scorer/score_image.py for what it reports.
set -euo pipefail
exec python3 "$(dirname "$(readlink -f "$0")")/scorer/score_image.py" "$@"
