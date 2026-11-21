#!/usr/bin/env bash
# Label an upstream corpus with br-eval's reference labeler (eval/labeler/label.py),
# in the corpus's priority order, a chunk at a time, so a partial run still covers
# the most important patches. Resumable: already-labeled ids are skipped.
#
#   eval/upstream/label.sh <corpus-dir> [chunk=24]
#
# Env:
#   BR_LABELER   path to label.py         (default: ../labeler/label.py next to this dir)
#   BR_SQLGLOT   sqlglot clone holding the pinned commit
#   BR_EVAL_DATA scratch dir for pristine export, trees, run logs (NOT br-eval's own)
#   BR_JOBS      parallel suite runs      (default 2)
#   BR_CPUSET    host CPUs, one per job   (default 6,7)
#   BR_MEMORY    docker memory per run    (default 2g)
set -euo pipefail
corpus=$(realpath "$1")
chunk=${2:-24}
here=$(dirname "$(realpath "$0")")
labeler=${BR_LABELER:-$here/../labeler/label.py}
sqlglot=${BR_SQLGLOT:-/home/rayana/firstmate/data/br-upstream/sqlglot}
export BR_EVAL_DATA=${BR_EVAL_DATA:-/home/rayana/firstmate/data/br-upstream/labelwork}
jobs=${BR_JOBS:-2}
cpuset=${BR_CPUSET:-6,7}
mkdir -p "$BR_EVAL_DATA/work" "$corpus/labels"

mapfile -t ids < "$corpus/priority.txt"
todo=()
for id in "${ids[@]}"; do
  [ -n "$id" ] && [ ! -e "$corpus/labels/$id.json" ] && todo+=("$id")
done
echo "label.sh: ${#todo[@]} of ${#ids[@]} to label, chunk $chunk, jobs $jobs on cpus $cpuset" >&2
for ((i = 0; i < ${#todo[@]}; i += chunk)); do
  python3 "$labeler" --corpus "$corpus" --tests "$corpus/tests.json" --sqlglot "$sqlglot" \
    --jobs "$jobs" --cpuset "$cpuset" --memory "${BR_MEMORY:-2g}" --only "${todo[@]:i:chunk}"
done
