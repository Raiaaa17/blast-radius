#!/bin/sh
# Build-time setup for staticeq (run from the Dockerfile after sqlglot is cloned
# and requirements.lock is installed):
#
#   staticeq/build.sh /path/to/pristine/sqlglot
#
# Writes staticeq/data/segments/ (statement index of every pristine .py file,
# ~4 s) next to the committed staticeq/data/filemap.json, and makes both
# readable by uid nobody. Set STATICEQ_TRACE=1 to also re-run the per-test
# file-access trace of the whole suite (~3 min on 3 cores) and rewrite
# filemap.json; the committed map was produced exactly that way at the pinned
# commit, so the default build does not need to.
set -eu
ROOT=${1:?usage: staticeq/build.sh SQLGLOT_ROOT}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP=$(dirname -- "$HERE")
PY=${PYTHON:-python}
EXTRA=""
if [ "${STATICEQ_TRACE:-0}" = "1" ]; then
    EXTRA="--trace --jobs ${STATICEQ_JOBS:-3}"
    if [ -f "$APP/sample/tests.json" ]; then EXTRA="$EXTRA --tests $APP/sample/tests.json"; fi
fi
# shellcheck disable=SC2086
PYTHONPATH="$APP" "$PY" -m staticeq.build_map --root "$ROOT" --out "$HERE/data" $EXTRA
chmod -R a+rX "$HERE"
