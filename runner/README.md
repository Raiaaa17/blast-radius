# runner/ - runtime test verifier (br-runner)

Applies one patch to the read-only pristine sqlglot checkout inside the run container and
reports, for exactly the tests you name, whether each one **fails** under the grader's
semantics, **passed** (cannot fail), or is **unknown** (not decided: never treat it as passed).

```python
from runner import Verifier

v = Verifier()                         # starts one warm zygote (~1.0-1.4 s CPU, once per batch)
r = v.verify(patch_text_or_path, [1377, 1376, 1573], budget_s=20)
r.failing   # {1377}       confirmed failing: output it
r.passed    # {1376, 1573} passed or skipped: safe to drop
r.unknown   # set()        crash budget, verification timeout, file deleted...: include it
r.why       # {1377: "call subtest: AssertionError ..."}
r.cpu_s     # CPU of the verification child(ren), pool workers included
results = v.verify_many([(patch_a, tests_a), (patch_b, tests_b)])   # spread over lanes
v.close()
```

Tests are tests.json numbers (ints) or node ids (strs); results use the same type.
Patches are text, bytes or a path. `verify(patch_path, numbers, tests_json, budget_s)` is a
one-shot wrapper returning `{"failing", "passed", "unknown", "cpu_s"}`.

Constructor options (all optional): `pristine` (default `$BR_SQLGLOT_ROOT`, else `/opt/sqlglot`,
`/app/sqlglot`, `/sqlglot`), `tests_json` (`/data/tests.json`), `lanes` (parallel zygotes, 1),
`workers` (pool size inside tests, 1), `timeout` (global pytest-timeout seconds, None),
`cap_factor`/`cap_base_s` (per-test verification cap = `2 + 10 x recorded seconds`),
`hard_cap_s` (120), `default_budget_s` (None), `as_limit_mb` (None), `tmp_root` (`/tmp/br-runner`).
Per call: `budget_s`, `max_failures` (early exit), `max_timeouts` (3), `cap_factor`, `cap_base_s`,
`workers`, `as_limit_mb`.

## Build

`runner/build.sh [SQLGLOT_ROOT]` - run in the Dockerfile after `pip install -r requirements.lock`
and cloning sqlglot (default `/opt/sqlglot`). It precompiles plain bytecode and pytest's
assertion-rewritten bytecode into the pristine tree (the run disk is read-only, so nothing can be
cached at run time) and makes it world-readable. Import of sqlglot drops from 0.45 s to 0.14 s.

```dockerfile
COPY runner /app/runner
RUN /app/runner/build.sh /opt/sqlglot
```
Callers must have `/app` (the directory containing `runner/`) importable; the zygote is started as
`python -m runner.zygote` with `PYTHONPATH` set to that directory only.

## How it works

- **Zygote** (`runner/zygote.py`): one long-lived process per lane. It imports pytest, numpy,
  pandas, duckdb once, runs pytest once on a dummy test, prepares a pytest `Config` (arg parsing,
  plugin loading, entry points), starts the multiprocessing resource tracker, then `gc.freeze()`s
  its heap. Per request it applies the patch to its overlay and forks a child.
- **Overlay** (`runner/overlay.py`): `/tmp/br-runner/laneN/root` is one symlink per top-level
  entry of the pristine checkout. A patch materializes only the directories on the path to each
  changed file (real dirs of symlinks) and writes changed files as real files. The `.pyc` of a
  changed module is hidden. Every read the tests do (fixtures, `inspect`, `__file__`-relative
  paths) sees the patched tree. Test files that `resolve()` their own path (test_sqlglotc) are
  real copies. Cost: a few hundred symlinks; /tmp never holds a copy of the tree.
- **Child** (`runner/driver.py`): runs `pytest_cmdline_main` on the prepared config (or plain
  `pytest.main` when the patch touches conftest/ini files) with
  `-q -p no:cacheprovider --continue-on-collection-errors` over the files of the requested
  tests, deselecting everything else, in tests.json order. A plugin streams per-test verdicts to
  the zygote, so a crash keeps every finished verdict.
- **Spawn pools**: `TestOptimizer` and `TestExecutor` run work in
  `ProcessPoolExecutor(mp_context=get_context("spawn"))` (12 fresh interpreters importing
  sqlglot+duckdb+pandas on a 12-core host: ~1 s CPU each). The child maps the "spawn" context to
  a forkserver whose server is forked from the child right after collection, before any test ran,
  so workers start from the same post-import state as a spawn worker. Results are still pickled
  across a real process boundary (patches to `serde.py` break these tests exactly as in the
  grader). Pool size is `workers` (1).

## Verdict semantics (grader's)

- failing: any failed setup/call/teardown report, **any failed subtest** (pytest 9 prints PASSED
  beside SUBFAILED), an unexpected success, a collection/import error of the test's file or class,
  a crash of the interpreter while the test ran (segfault, `os._exit`, OOM kill), a pytest
  internal error while it ran.
- passed: reported and nothing failed (skips and expected failures cannot fail).
- unknown: the test hit its verification cap (`2 s + 10 x recorded`, pytest-timeout on the call
  phase) or the hard wall cap (`1.5 x cap + 15 s`, for code stuck where signals cannot land), the
  budget ran out, the test's file was deleted/renamed by the patch, it was not collected (renamed
  test), the patch did not apply, or the run was stopped after 3 verification timeouts / 2
  crash-on-first-test runs (a patch that hangs or crashes everything).

## Robustness

Children run in their own process group (killed as a group), with `oom_score_adj=1000` (if memory
runs out the kernel kills the child, not the zygote or selector), no core dumps, optional
`RLIMIT_AS`/`RLIMIT_CPU`. Hangs in Python are cut by pytest-timeout at the cap; hangs in C by the
zygote's hard cap. Crashes rerun the remaining tests in a fresh child. Tested on synthetic patches:
Python infinite loop, catastrophic regex, segfault, `os._exit`, syntax error in a core module,
ImportError in a new module, 6 GB allocation, infinite recursion, test-file edit, test-file syntax
error, setUpClass error, new conftest, setup.cfg pytest section, deleted fixture, deleted/renamed
test file, binary file (GIT binary patch), mode-only change.

## Measured costs (scorer limits: 2 CPUs, python:3.13-slim + requirements.lock)

| what | CPU |
| --- | --- |
| zygote warmup (once per lane) | 1.0-1.4 s |
| one request, trivial test, any patch (fork + pytest session + import sqlglot) | 0.11-0.13 s |
| + importing a dialect test module (e.g. test_snowflake) | +0.07 s |
| + test_optimizer module + TestOptimizer.setUpClass (duckdb) | +0.10 s |
| + first spawn-pool test (template + worker fork) | +0.3 s |
| + TestExecutor.setUpClass (loads TPC-H/TPC-DS CSVs) | +1.6-1.8 s |
| per test | ~1.5-2x its recorded seconds (+~5 ms pytest overhead) |
| all 1620 tests, empty patch, one request | 171 s (recorded total 103 s) |

`python -m runner sample` on the 30 sample commits (labeled failing tests + 40 random cheap
tests each): every labeled failing test reports failing or unknown, every sampled
non-failing test passes; see findings for numbers.

## CLI

```
python -m runner [--pristine DIR] [--tests-json F] verify PATCH 12,845 tests/x.py::C::t
python -m runner sample --commits /data/commits --answers sample/answers.json --extra 40
python -m runner.selftest 400     # patch applier: hand cases + differential test vs git
```
