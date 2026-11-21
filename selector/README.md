# selector

The submission's entry point (`python /app/selector/main.py`) and its build-time maps.

## Build (Dockerfile)

| step | script | output in `/app/maps` |
| --- | --- | --- |
| fetch sqlglot @ eb11aec (GitHub archive) | `fetch_sqlglot.py` | `/opt/sqlglot` (pristine, read-only at run time) |
| per-test tracing of the whole suite, 4 processes | `build_maps.py` + `merge_maps.py` | `maps.pkl` |
| static code / import index | `static_index.py` | `static.pkl` |
| untraced timing + self-check through the runtime runner | `build_timing.py` | `timing.pkl` |

`maps.pkl` (test sets are int bitmasks over `test_ids`, interned in `sets`):

- `lines[file][line]` - tests that executed the line (function bodies at run time; test-module import
  and class fixtures are attributed to the tests they serve);
- `attrs[(class_key, attr)][owner_key]`, `owners[owner_key]`, `owner_mro` - class-attribute reads,
  keyed by defining class and by the class they were read through;
- `opens[path]`, `lists[dir]` - files opened / directories listed under the tree;
- `import_codes` - code objects executed while importing sqlglot (module/class bodies, factories);
- `import_sensitive` - tests that imported sqlglot/tests modules themselves or spawned a process;
- `unreliable` - tests that did not pass under the tracer (always candidates).

## Run

`main.py` reads `/data/tests.json` and `/data/commits/*.patch`, prints one answer line per commit.
For each patch: `patch.py` parses/applies the diff, `analyze.py` classifies each changed Python unit by
comparing code objects (neutral / function / class / module), `main.assess` maps units to candidate
tests, `verify.py` runs the cheap candidates against a symlink-farm copy of the patched tree in forked
children of a warm parent (`runner_child.run`), and `pack.py` splits the kept tests over workers.

Anything not understood (renames, deletions, syntax errors, config files, binary diffs, import-time
functions, opaque module-level statements) selects every test; anything that cannot be verified
(expensive, timed out, crashed, past the deadline) is kept.

Environment overrides for local runs: `BR_DATA`, `BR_PRISTINE`, `BR_MAPS`, `BR_TMP`.
