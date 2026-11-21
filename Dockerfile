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

# Precompiled bytecode (plain and pytest-rewritten) in the pristine tree: the
# run disk is read-only, so nothing can be cached at run time.
COPY runner /app/runner
RUN sh /app/runner/build.sh /opt/sqlglot

# Build-time maps of the pristine tree. Only the build-time scripts are copied
# first, so runtime-only changes keep these layers cached.
#  maps.pkl     per-test line coverage, class-attribute reads, file opens (traced run)
#  static.pkl   code objects, import graph, class holders
#  zygote.pkl   verdict and CPU of every test through the runtime's warm verifier
#               (with every function's code swapped for an identical copy)
#  profiles.pkl comparison-site value profiles of every test (instrumented run)
#  state.pkl    the pristine import state as the run time digests it (state_keys.pkl: every
#               dict slot's entries), and which slots depend on the hash seed / the process
COPY selector/build_maps.py selector/merge_maps.py selector/inline_pool.py selector/static_index.py \
     selector/cachereset.py selector/zygote.py selector/runner_child.py selector/build_zygote.py \
     selector/profiles.py selector/build_profiles.py selector/statediff.py selector/verify.py \
     selector/sentinel.py selector/setters.py /app/selector/
# Recorded test seconds, only to balance the build's shards.
COPY sample/tests.json /app/selector/sample_tests.json
RUN SKIP_INTEGRATION=0 python /app/selector/build_maps.py /opt/sqlglot /app/maps --jobs 4 \
 && python /app/selector/static_index.py /opt/sqlglot /app/maps \
 && python /app/selector/statediff.py /opt/sqlglot /app/maps/state.pkl \
 && SKIP_INTEGRATION=0 python /app/selector/build_zygote.py /opt/sqlglot /app/maps --jobs 4 \
 && SKIP_INTEGRATION=0 python /app/selector/build_profiles.py /opt/sqlglot /app/maps --jobs 4 --min-seconds 0 \
 && chmod -R a+rX /opt/sqlglot /app

# Static patch analysis (neutrality proofs, data-file readers).
COPY staticeq /app/staticeq
RUN sh /app/staticeq/build.sh /opt/sqlglot

COPY selector/ /app/selector/
RUN python -m compileall -q /app/selector /app/staticeq && chmod -R a+rX /app

ENV SKIP_INTEGRATION=0
CMD ["python", "/app/selector/main.py"]
