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
| `corpus/subset/` | A small labeled subset of the stress corpus, kept in git (the full corpus lives outside the repo). |

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
eval/labeler/label.py --corpus DIR [--jobs 3] [--cpuset 9,10,11] [--timeout 300] [--kinds perf] [--skip-kinds perf]
```

It writes `DIR/labels/<id>.json` (failing numbers with reasons, vanished tests,
per-test real seconds, collection errors, wall time) and `DIR/answers.json` in
the kit's format. It resumes where it stopped; `--force` relabels. Each job runs
one serial full suite pinned to one host CPU (`--cpuset-cpus`): under a CPU
*quota* the optimizer/executor tests spawn one process per host CPU and thrash.
About 160 s per patch per core on a 12-core host.

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
