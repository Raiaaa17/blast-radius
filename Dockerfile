# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image the organizers labeled every commit in: its Python and
# SQLite decide which sqlglot tests run, so keep it.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1

# The exact labeling environment.
COPY requirements.lock /build/requirements.lock
RUN pip install -r /build/requirements.lock

# sqlglot at the pinned commit, from its own repository (allowed source).
ARG SQLGLOT_SHA=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
COPY selector/fetch_sqlglot.py /build/fetch_sqlglot.py
RUN python /build/fetch_sqlglot.py "$SQLGLOT_SHA" /opt/sqlglot

# Build-time maps: per-test coverage of the pristine tree. Only the build-time
# scripts are copied first, so runtime-only changes keep this layer cached.
COPY selector/build_maps.py selector/merge_maps.py selector/inline_pool.py selector/static_index.py \
     selector/build_timing.py selector/verify.py selector/runner_child.py /app/selector/
# Recorded test seconds, only to balance the build's shards.
COPY sample/tests.json /app/selector/sample_tests.json
RUN SKIP_INTEGRATION=0 python /app/selector/build_maps.py /opt/sqlglot /app/maps --jobs 4 \
 && python /app/selector/static_index.py /opt/sqlglot /app/maps \
 && python -m compileall -q -j 4 /opt/sqlglot /app/selector \
 && SKIP_INTEGRATION=0 python /app/selector/build_timing.py /opt/sqlglot /app/maps --jobs 4 \
 && chmod -R a+rX /opt/sqlglot /app

COPY selector/ /app/selector/
RUN python -m compileall -q /app/selector && chmod -R a+rX /app/selector

ENV SKIP_INTEGRATION=0
CMD ["python", "/app/selector/main.py"]
