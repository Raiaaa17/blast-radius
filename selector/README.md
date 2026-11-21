# selector

The submission's entry point (`python /app/selector/main.py`) and its build-time maps.

## Build (Dockerfile)

| step | script | output in `/app/maps` |
| --- | --- | --- |
| fetch sqlglot @ eb11aec (GitHub archive) | `fetch_sqlglot.py` | `/opt/sqlglot` (pristine, read-only at run time) |
| precompiled bytecode | `runner/build.sh` | `/opt/sqlglot/**/__pycache__` |
| per-test tracing of the whole suite, 4 processes | `build_maps.py` + `merge_maps.py` | `maps.pkl` |
| static code / import index | `static_index.py` | `static.pkl` |
| every test through the warm verifier, every function code swapped for an identical copy | `build_zygote.py` | `zygote.pkl` |
| comparison-site value profiles of every test (instrumented sqlglot) | `build_profiles.py` + `profiles.py` | `profiles.pkl` |
| hash-seed / process dependence of the import state | `statediff.py` | `state.pkl` |
| statement segments for staticeq | `staticeq/build.sh` | `staticeq/data/segments/` |

`maps.pkl` (test sets are int bitmasks over `test_ids`, interned in `sets`):

- `lines[file][line]` - tests that executed the line (function bodies at run time; test-module import
  and class fixtures are attributed to the tests they serve);
- `attrs[(class_key, attr)][owner_key]`, `owners[owner_key]`, `owner_mro` - class-attribute reads,
  keyed by defining class and by the class they were read through;
- `opens[path]`, `lists[dir]` - files opened / directories listed under the tree;
- `import_codes` - code objects executed while importing sqlglot (module/class bodies, factories);
- `import_sensitive` - tests that imported sqlglot/tests modules themselves or spawned a process;
- `unreliable` - tests that did not pass under the tracer (always candidates).

Process-wide caches of sqlglot (`cachereset.py`: module-level `*cache*` containers, lru caches; in
practice the generator dispatch table) are emptied before every traced test, so every test records the
code and class attributes it depends on, not only the first one of its shard.

## Run

`main.py` reads `/data/tests.json` and `/data/commits/*.patch` and prints one answer line per commit.

1. **Neutral**: `staticeq.analyze` proves comments, docstrings, annotations, unused files and fixture
   comment lines neutral -> `"workers": []`.
2. **Units**: `patch.py` applies the diff, `analyze.py` compares old/new code objects (line tables
   ignored) -> function bodies, class bodies, module level.
3. **Candidates**:
   - function body -> tests that executed a *changed statement* (statement-level diff; inserted
     statements are anchored on their neighbours / the enclosing header; whole function when the
     signature changed);
   - class body, module level, or a function that runs at import -> **import-state diff**
     (`statediff.py`): a child imports the patched tree and digests every module global, class
     attribute and MRO; a pristine child forked from the same process is the reference. Changed slots
     map to the tests that read that class attribute (or, for a new attribute, the base attribute it
     shadows), to every user of a class whose MRO or dunder changed, and to the code referencing a
     changed module global. A tree that no longer imports selects every test;
   - data file -> tests that opened it (staticeq's map attributes setUpClass reads to the class);
   - test code -> the test itself / its file and importers;
   - config, rename, delete, binary, syntax error -> every test.
4. **Proofs**: when a changed function differs only in single comparison operators, a test whose
   profile shows that every operand pair it produced at those sites compares the same under the old
   and the new operator (or never evaluated them) cannot see the change and is dropped
   (`profiles.py`; all tests except test_max_nodes are profiled).
5. **Verify or keep**: per candidate, its packed seconds vs its CPU to verify (build timing, class
   fixture cost) and its prior odds of failing; anything unverifiable (expensive, not trusted) is kept.
6. **Verification** (`zygote.py`): function-body-only patches whose functions did not run at import
   are verified *hot*: the selector imports pristine sqlglot and the candidate test modules once,
   forks a child per commit that swaps the changed functions' `__code__` in place (changed modules
   the warm process never imported load from the patched source); everything else runs *fresh*: a
   child of the third-party-only process imports from a symlink farm of the patched tree. Children
   run tests with unittest in tests.json order (`runner_child.py`), the process pool replaced by an
   in-process pickle round trip (`inline_pool.py`); watchdog per test; a commit whose tests hang or
   crash twice keeps the rest unverified.
7. **Packing** (`pack.py`): LPT + exact search for small sets, k in 1..8.

Environment overrides for local runs: `BR_DATA`, `BR_PRISTINE`, `BR_MAPS`, `BR_TMP`.
