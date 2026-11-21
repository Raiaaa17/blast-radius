#!/usr/bin/env bash
# Label an upstream corpus with br-eval's reference labeler (eval/labeler/label.py),
# one patch per suite run, in the corpus's priority order, so a partial run still
# covers the most important patches. Resumable: already-labeled ids are skipped.
#
# The host is shared, so each new run starts only when
#   - fewer than BR_JOBS of our runs are active (1 while br-eval itself runs >= 2), and
#   - MemAvailable is above BR_MIN_FREE_MB;
# every run is capped by docker --memory/--memory-swap BR_MEMORY.
#
#   eval/upstream/label.sh <corpus-dir>
#
# Env:
#   BR_LABELER      path to label.py         (default: ../labeler/label.py next to this dir)
#   BR_SQLGLOT      sqlglot clone holding the pinned commit
#   BR_EVAL_DATA    scratch dir for pristine export, trees, run logs (NOT br-eval's own)
#   BR_JOBS         max parallel suite runs  (default 2)
#   BR_CPUSET       host CPUs, one per run   (default 6,7)
#   BR_MEMORY       docker memory per run    (default 2500m)
#   BR_MIN_FREE_MB  start a run only above this MemAvailable (default 3072)
#   BR_ID_PATTERN   regex matching our patch ids in container names (default -(fwd|rev)-)
set -euo pipefail
corpus=$(realpath "$1")
here=$(dirname "$(realpath "$0")")
labeler=${BR_LABELER:-$here/../labeler/label.py}
sqlglot=${BR_SQLGLOT:-/home/rayana/firstmate/data/br-upstream/sqlglot}
export BR_EVAL_DATA=${BR_EVAL_DATA:-/home/rayana/firstmate/data/br-upstream/labelwork}
max_jobs=${BR_JOBS:-2}
IFS=, read -r -a cpus <<< "${BR_CPUSET:-6,7}"
memory=${BR_MEMORY:-2500m}
min_free=${BR_MIN_FREE_MB:-3072}
ours=${BR_ID_PATTERN:--(fwd|rev)-}
mkdir -p "$BR_EVAL_DATA/work" "$corpus/labels" "$corpus/logs"

mapfile -t ids < "$corpus/priority.txt"
todo=()
for id in "${ids[@]}"; do
  [ -n "$id" ] && [ ! -e "$corpus/labels/$id.json" ] && todo+=("$id")
done
echo "label.sh: ${#todo[@]} of ${#ids[@]} to label, <= $max_jobs runs on cpus ${cpus[*]}, $memory each, start above ${min_free} MB free" >&2

declare -A slot_pid=()
free_mb() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
others() { docker ps --format '{{.Names}}' | grep '^br-eval-label-' | grep -Evc -- "$ours" || true; }
active() {
  local n=0 c
  for c in "${!slot_pid[@]}"; do
    if kill -0 "${slot_pid[$c]}" 2>/dev/null; then n=$((n + 1)); else unset "slot_pid[$c]"; fi
  done
  echo "$n"
}
# wait for our own containers left over from an interrupted driver
while docker ps --format '{{.Names}}' | grep '^br-eval-label-' | grep -Eq -- "$ours"; do sleep 20; done

i=0
while [ "$i" -lt "${#todo[@]}" ]; do
  allowed=$max_jobs
  [ "$(others)" -ge 2 ] && allowed=1
  if [ "$(active)" -lt "$allowed" ] && [ "$(free_mb)" -gt "$min_free" ]; then
    for c in "${cpus[@]}"; do
      [ -z "${slot_pid[$c]:-}" ] && break
    done
    id=${todo[$i]}
    python3 "$labeler" --corpus "$corpus" --tests "$corpus/tests.json" --sqlglot "$sqlglot" \
      --jobs 1 --cpuset "$c" --memory "$memory" --only "$id" >> "$corpus/logs/label.log" 2>&1 &
    slot_pid[$c]=$!
    i=$((i + 1))
    sleep 45  # let the new run's memory show up before the next check
  else
    sleep 20
  fi
done
wait
python3 - "$labeler" "$corpus" <<'EOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("label", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.write_answers(Path(sys.argv[2]))
EOF
