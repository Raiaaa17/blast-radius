# eval/ - ground truth, stress corpus, local scorer

Evaluation only. Nothing here is read by the selector image, and no selection
rule may be derived from a corpus (the track's rules forbid any table from a
patch or kind of edit to the tests it breaks).

| Path | What |
| --- | --- |
| `ref/Dockerfile` | Reference labeling image `br-eval-ref`: the kit's base image digest + `requirements.lock`. |
| `labeler/label.py` | Labels every patch of a corpus by running the full sqlglot suite on the patched tree. |
| `labeler/brreport.py` | pytest plugin the labeler loads: per-test outcome, subtests, durations, collection errors. |
| `score_image.sh` | Scores a selector image on a labeled corpus under the scorer's exact docker flags. |
| `corpus/gen_mutations.py` | Generates stress patches (60 kinds of edit, plus multi-file mixes). |
| `corpus/subset/` | Every exactly labeled stress patch of at most 20 KB (165 of 173, 85 kinds), kept in git; the full corpus lives outside the repo. |

Grader semantics, timings and gotchas live in the cross-team notes at
`/home/rayana/firstmate/data/br-eval/findings.md`.

## Reference environment

```sh
docker build -t br-eval-ref -f eval/ref/Dockerfile .
```

sqlglot is not baked in: the labeler mounts a `git archive` export of the pinned
commit, with the patch applied, read-only at `/src` and runs, as uid 65534 with
`PYTHONPATH=/src:/plug` and `SKIP_INTEGRATION=0`:

```sh
python -m pytest -p no:cacheprovider -p brreport -q --continue-on-collection-errors --timeout=300 tests
```

A numbered test fails when any of its reports failed (setup/call/teardown error
or failure, any failing subtest, a pytest-timeout timeout) or its file could not
be collected. A numbered test that no longer exists after the patch (renamed or
deleted test, renamed or deleted test file) is listed as `vanished` in the
per-patch label, not as failing.

## Labeling a corpus

A corpus directory holds `commits/<id>.patch` and optionally `kinds.json`
(`{id: kind}`).

```sh
eval/labeler/label.py --corpus DIR [--jobs 2] [--cpuset 10,11] [--memory 2500m] [--min-free-gb 3] [--timeout 300] [--kinds perf] [--skip-kinds perf]
```

It writes `DIR/labels/<id>.json` (failing numbers with reasons, vanished tests,
per-test real seconds, collection errors, wall time) and `DIR/answers.json` in
the kit's format. It resumes where it stopped; `--force` relabels. Each job runs
one serial full suite pinned to one host CPU (`--cpuset-cpus`): under a CPU
*quota* the optimizer/executor tests spawn one process per host CPU and thrash.
Each run is hard-capped at `--memory` (swap disabled) and starts only when the
host has `--min-free-gb` available; the label records the run's peak memory,
and a run the kernel OOM-kills is marked `oom` and left out of answers.json.
`--timeout-scale S --timeout-floor F` replaces the uniform `--timeout` with a
per-test `max(F, S x recorded seconds)` (pytest-timeout markers added by
`brreport` from `BR_TIMEOUTS`): it bounds runs of non-terminating patches, whose
failing set does not depend on the timeout; a slowdown's label does, so those
keep a uniform timeout. A stop file is re-checked after the memory gate.
About 140-230 s per patch per core on a 12-core host.

## Generating a stress corpus

```sh
eval/corpus/gen_mutations.py --sqlglot <clean clone at eb11aec> --out DIR --per-kind 6 --seed 1
```

Each patch is a real `git diff` (sample format; renames as `rename from/to`,
mode changes, added and deleted files) of one generated edit in the clone's
working tree, which is reset after every patch. `DIR/meta.json` describes each
edit. Kinds (`kinds.json`) are namespaced: `code:*` (function bodies: operator
flips, constants, deleted statements, early return/raise, negated conditions,
swapped arguments), `sig:*` (defaults, added/removed parameters), `table:*`
(class and module tables: dialect FUNCTIONS/TRANSFORMS/KEYWORDS, type maps,
class attributes, reordered entries), `import:*`, `init:edit`, `syntax:*`,
`file:*` (renames, deletes, additions, mode-only), `fixture:*` (edits, added and
deleted lines, `--` comments, `#` meta lines), `test:*` (assertion edits, added,
renamed, deleted, skipped tests, helper edits, test-file syntax errors),
`comment:*`, `docstring`, `noop:*` (whitespace, local renames, `pass`,
reordered functions, files outside the package), `perf:*` (busy loops,
infinite loops, unbounded recursion) and `multi`.

Round 3 adds what a patch parser or a coverage map most easily gets wrong:
`format:*` (whole file to CRLF, final newline dropped, UTF-8 byte order mark),
`file:binary` (`GIT binary patch` of a TPC-H/TPC-DS `.csv.gz` table: a value, a
row, a truncated gzip; an added image), `file:quoted-path` (non-ASCII names git
quotes, names with spaces), `file:submodule` (gitlink bump), `outside:executed`
(`sqlglotc/setup.py`, which `tests/test_sqlglotc.py` runs with `runpy`),
`dialect:new` (a new dialect with or without its registries: `DIALECTS`, the
`Dialects` enum, parsers/generators modules; also a registry entry whose module
is missing), `refactor:rename-method` (a method and its in-file references,
which breaks name-based dispatch such as `<key>_sql`), `expr:arg-types`
(required/optional flips, added and dropped args), `test:conftest` (a new
`tests/conftest.py`: autouse fixtures, `collect_ignore`), `fixture:blank-line`,
`fixture:trailing-space`, `sweep:comment` / `sweep:mixed` (30-60 files, with or
without one real mutation among them), `noop:large` (~180 KB docs diff) and
`noop:empty` (an empty patch file). `Tree.diff` adds `--binary` only when a
binary file changed, so text patches keep the sample's abbreviated `index` lines.

## Scoring an image

```sh
eval/score_image.sh <image> <corpus-dir> [--batch 40] [--runs 3] [--limit 180] [--json out.json]
eval/score_image.sh blast-radius-entry sample      # the kit's sample set
```

It mounts each batch under the Makefile's `RUN_FLAGS`, scores the output with
the kit's `score.py` (imported unchanged, so simulated seconds and misses are
exact), and adds the container's CPU seconds (cgroup v2 `cpu.stat` of a private
systemd slice, `--cgroup-parent`), the wall time against the limit, every miss
with its kind and missed tests, cost per kind, and the middle of `--runs` totals.
`--batch N` runs the corpus as consecutive batches of N commits, like the
hidden sets.

## State when the work stopped (2026-11-21 13:20 UTC) and how to resume

- Stress corpus: 733 patches, 92 kinds, at `/home/rayana/firstmate/data/br-eval/corpora/stress`
  (`commits/`, `kinds.json`, `meta.json`, `labels/`, `answers.json`). 173 have exact labels (every
  `answers.json` entry is exact: partial, killed and unappliable runs are never listed). Labeled first: the
  shared-state pollution and dispatch-addition shapes, and one or more of each round-3 kind. Still
  unlabeled: most second and later patches of the round-1 kinds, and all `perf:*` patches.
- `corpus/subset/` = `make_subset.py <stress> --per-kind 1000 --max-bytes 20000`. The 8 labeled patches left
  out are the ones over 20 KB: 4 `file:binary`, 2 `noop:large`, 1 `format:crlf`, 1 `file:delete-test`.
- Resume labeling from where it stopped (labels already written are skipped). Use at most 2 jobs, each with a
  hard 2500m cap, starting only above 3 GB free:

  ```sh
  S=/home/rayana/firstmate/data/br-eval/corpora/stress
  eval/labeler/label.py --corpus $S --jobs 2 --cpuset 10,11 --memory 2500m --min-free-gb 3 \
      --reuse-reports --skip-kinds perf
  eval/labeler/label.py --corpus $S --kinds perf:infinite-loop --timeout-floor 20 --timeout-scale 6 --wall 5400
  eval/labeler/label.py --corpus $S --kinds perf --timeout 60 --wall 3600
  ```
- Scoring a selector on the labeled patches: `eval/score_image.sh <image> <stress-dir> --batch 40` skips
  unlabeled patches. Selector v3 (main c99a412) was being scored when the work stopped: its first 40 labeled
  stress patches gave 0 misses, 285.5 s simulated against a 154.5 s oracle floor, 25.3 s CPU, 26 s wall and a
  360 MB peak. The other 133 labeled patches were not scored.

