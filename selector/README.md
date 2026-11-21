# selector

The submission's entry point (`python /app/selector/main.py`) and its build-time maps.

## Build (Dockerfile)

| step | script | output in `/app/maps` |
| --- | --- | --- |
| fetch sqlglot @ eb11aec (GitHub archive) | `fetch_sqlglot.py` | `/opt/sqlglot` (pristine, read-only at run time) |
| precompiled bytecode | `runner/build.sh` | `/opt/sqlglot/**/__pycache__` |
| state setters (test files leaving sqlglot state changed), then per-test tracing of the whole suite, 4 processes | `setters.py`, `build_maps.py` + `merge_maps.py` | `maps.pkl` |
| static code / import index (run-time first lines, attribute names) | `static_index.py` | `static.pkl` |
| pristine import state as the run time digests it, its hash-seed / process dependence | `statediff.py` | `state.pkl`, `state_keys.pkl` |
| every test through the warm verifier, every function code swapped for an identical copy; what the pristine suite changes in sqlglot state (`volatile`) | `build_zygote.py`, `sentinel.py` | `zygote.pkl` |
| comparison-site value profiles of every test (instrumented sqlglot, setters replayed) | `build_profiles.py` + `profiles.py` | `profiles.pkl` |
| statement segments for staticeq | `staticeq/build.sh` | `staticeq/data/segments/` |

`maps.pkl` (test sets are int bitmasks over `test_ids`, interned in `sets`):

- `lines[file][line]` - tests that executed the line (function bodies at run time; test-module import
  and class fixtures are attributed to the tests they serve);
- `attrs[(class_key, attr)][owner_key]`, `owners[owner_key]`, `owner_mro` - class-attribute reads,
  keyed by defining class and by the class they were read through;
- `opens[path]`, `lists[dir]` - files opened / directories listed under the tree (`lists` also holds the
  import system's own directory scans, so file adds/deletes use staticeq's cleaner listers instead);
- `dict_slots[slot]`, `key_reads[dict][key]`, `dict_leaky` - entry-level reads of every class- and
  module-level dict: during the trace each dict is replaced by a `TrackedDict` copy that records the keys a
  test looks up (`b"*"` = the whole dict: iteration, views, length, equality, copy, repr, pickling); a dict
  still reachable another way after the swap (`gc.get_referrers`) is leaky and keeps attribute-level mapping;
- `enums[("C", class) | ("M", module)]` - tests that listed a namespace with `dir()` / `vars()` (the
  generator's dispatch table is built from `dir(cls)`), for added or removed names;
- `import_codes` - code objects executed while importing sqlglot (module/class bodies, factories);
- `import_sensitive` - tests that imported sqlglot/tests modules themselves or spawned a process;
- `attrs_unknown` - tests that fail under the recorders but pass a line-only second pass: line coverage
  known, attribute/entry reads not (candidates for every import-state change);
- `unreliable` - tests that pass under neither (always candidates).

`state.pkl` is the pristine import state digested the way the run time does it (a child forked from a
selector-like process, the tree a symlink farm), with `callables` (code spans of function-valued class
attributes) and, in `state_keys.pkl`, every dict slot's per-entry encodings.

Process-wide caches of sqlglot (`cachereset.py`: module-level `*cache*` containers, lru caches; in
practice the generator dispatch table) are emptied before every traced test, so every test records the
code and class attributes it depends on, not only the first one of its shard.

## Run

`main.py` reads `/data/tests.json` and `/data/commits/*.patch` and prints one answer line per commit.

1. **Neutral**: `staticeq.analyze` proves comments, docstrings, annotations, unused files and fixture
   comment lines neutral -> `"workers": []`.
2. **Units**: `patch.py` applies the diff, `analyze.py` compares old/new code objects (line tables
   ignored; the pristine side from the build's bytecode) -> function bodies, class bodies, module level.
3. **Candidates**:
   - function body -> tests that executed a *changed statement* (statement-level diff; inserted
     statements are anchored on their neighbours / the enclosing header; whole function when the
     signature changed);
   - class body, module level, or a function that runs at import -> **import-state diff**
     (`statediff.py`): a child imports the patched tree and digests every module global, class
     attribute and MRO; a pristine child forked from the same process is the reference. Changed slots
     map to the tests that read that class attribute (or, for a new attribute, the base attribute it
     shadows), to every user of a class whose MRO or dunder changed, and to the code referencing a
     changed module global. A changed dict maps to the tests that read a changed entry (or the whole
     dict); a changed function-valued attribute to its readers that also ran the old value's code; a new
     or removed name also to the tests that listed that namespace. The reference is the build's pristine
     snapshot: each child diffs against it and returns only what changed. A tree that no longer imports
     selects every test;
   - data file -> tests that opened it (staticeq's map attributes setUpClass reads to the class);
   - script outside the package that tests run or read (`sqlglotc/setup.py`) -> those tests;
   - test code -> the test itself / its file and importers;
   - added / deleted / renamed / copied files -> the old side's readers (a deleted test module: its tests
     and its importers' tests) and the new side's listers and import attempts (staticeq); a new test
     module with import-time side effects, a new top-level module shadowing an installed one, a removed
     or renamed sqlglot module, a new package -> every test;
   - config, binary, symlink, syntax error -> every test.
4. **Proofs**: when a changed function differs only in single comparison operators, a test whose
   profile shows that every operand pair it produced at those sites compares the same under the old
   and the new operator (or never evaluated them) cannot see the change and is dropped
   (`profiles.py`; all tests except test_max_nodes are profiled).
5. **Verify or keep**: per candidate, its packed seconds vs its CPU to verify (build timing, class
   fixture cost) and its prior odds of failing, every CPU second weighted 2x (`CPU_WEIGHT`: the scorer
   charges all container CPU and measures about twice what this machine does); anything unverifiable
   (expensive, not trusted) is kept. numpy/pandas/duckdb are imported once, before the children fork,
   only when a child's test modules need them.
6. **Verification** (`zygote.py`): function-body-only patches whose functions did not run at import
   are verified *hot*: the selector imports pristine sqlglot and the candidate test modules once,
   forks a child per commit that swaps the changed functions' `__code__` in place (changed modules
   the warm process never imported load from the patched source); everything else runs *fresh*: a
   child of the third-party-only process imports from a symlink farm of the patched tree. Children
   run tests with unittest in tests.json order (`runner_child.py`), the process pool replaced by an
   in-process pickle round trip (`inline_pool.py`); watchdog per test; a commit whose tests hang or
   crash twice keeps the rest unverified. One child at a time (`VERIFY_PARALLEL`). Every test module
   is imported before the first test (pytest's collection does that); new test modules are imported too.
   - **State setters** (`setters.py`): test files that leave sqlglot state changed for every later file
     (test_dialect_entry_points empties the dialect registry; test_lineage / test_optimizer rebind
     `sqlglot.schema`) are found at build time (static scan, then a fixture-replay probe comparing
     structural digests). Their effect is re-created before the first file that follows them in suite
     order, here, in the map build and in the profile build; a registry they empty is emptied again
     before every later test, so each runs the refilling code itself whatever its neighbours.
   - **Pollution sentinel** (`sentinel.py`): for changed run-time code, a shallow fingerprint of every
     sqlglot global and class attribute (identity, length, entry identities) before and after the
     candidates run. What changed, beyond what the pristine suite changes by itself (`volatile` in
     zygote.pkl; for a registry it only empties and refills, entries holding a different value still
     count), maps to the tests that read it, which are kept: in the grader's single process every later
     test sees that state. A static check (a new mutation of `self.UPPER_TABLE`, `cls.*`, a global)
     forces verification so the sentinel runs.
7. **Packing** (`pack.py`): LPT + exact search for small sets, k in 1..8.

Environment overrides for local runs: `BR_DATA`, `BR_PRISTINE`, `BR_MAPS`, `BR_TMP`.
