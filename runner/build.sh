#!/bin/sh
# Build-time setup for br-runner. Usage: runner/build.sh [SQLGLOT_ROOT]   (default /opt/sqlglot)
#
# The run disk is read-only, so Python can never cache bytecode at run time. Precompile:
#  - plain bytecode for sqlglot/ and tests/ (import sqlglot: 0.45 s -> 0.14 s CPU), and
#  - pytest's assertion-rewritten bytecode for the test modules (pytest writes it into
#    tests/**/__pycache__/*-pytest-<ver>.pyc while collecting), so no run-time rewrite.
# Run it after `pip install -r requirements.lock`, as the user that owns the checkout.
set -eu
ROOT="${1:-/opt/sqlglot}"
PY="${PYTHON:-python}"
"$PY" -m compileall -q -j 0 "$ROOT/sqlglot" "$ROOT/tests" >/dev/null
( cd "$ROOT" && SKIP_INTEGRATION=0 "$PY" -m pytest -q --collect-only -p no:cacheprovider tests >/dev/null )
chmod -R a+rX "$ROOT"
echo "runner/build.sh: precompiled $ROOT" >&2
