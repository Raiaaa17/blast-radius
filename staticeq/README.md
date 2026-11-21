# staticeq: static patch analysis

Decides, without running any test, whether a patch to the pinned sqlglot tree
is **behavior-neutral** (the selector may then run nothing), and otherwise
lists exactly **what changed**: the Python units (function bodies, headers,
class bodies, module-level code) and the data files, with the tests already
known to read each data file. Standard library only, so it runs in the slim
run image. Typical cost: about 1-5 ms of CPU per patch in chunk mode (see
"Cost").

```python
from staticeq import analyze

verdict = analyze(patch_text, "/opt/sqlglot")   # patch_text: str or bytes
if verdict["neutral"]:
    workers = []                                # provably nothing can fail
```

## Result

```jsonc
{
  "neutral": false,                 // true only if every file is proven neutral
  "reasons": ["sqlglot/x.py: code changed", ...],   // why not neutral (human-readable)
  "files": [                        // one entry per file in the patch
    {"path": "sqlglot/x.py", "old_path": "...", "new_path": "...",
     "kind": "python",              // python | python-unused | data | data-unused | config |
                                    // structural-{added,deleted,rename,copy,binary,mode} |
                                    // unparsed | unapplied | missing | no-op | error
     "level": "changed",            // python: code-equal | ast-equal | docstring-only | changed | error
     "mode": "chunk",               // python: chunk (segment-level) or file (whole file)
     "neutral": false, "reason": "..."}
  ],
  "units": [                        // changed Python units (non-neutral Python files only)
    {"file": "sqlglot/parsers/dremio.py", "module": "sqlglot.parsers.dremio",
     "qualname": "DremioParser.<lambda>",   // co_qualname style; "" = module level
     "scope": "function",           // function | class | module
     "where": "body",               // see table below
     "old_lines": [110, 115],       // span of the unit in the PRISTINE file (decorators included)
     "def_line": 110,               // pristine def/class/lambda line (co_firstlineno of a def is its first decorator line, use old_lines[0] for that)
     "names": [],                   // class-body / module-level: names bound by the changed statements
     "detail": "lambda body changed",
     "package_init": false,         // unit lives in an __init__.py
     "test_ids": []}                // the unit is itself a test method -> its test id
  ],
  "data_files": [                   // non-Python files (and .py read as text) that matter
    {"path": "tests/fixtures/optimizer/simplify.sql", "status": "modified",
     // modified | added | deleted | rename | copy | binary | mode | mentioned | source-read | modified-script
     "tests": ["tests/test_optimizer.py::TestOptimizer::test_simplify"],  // tests seen opening it
     "readers": ["tests/helpers.py:load_sql_fixture_pairs"],
     "dir_listers": [],             // added/deleted files: tests that list an ancestor directory
     "known": true}                 // false = no file map was available (be broad)
  ]
}
```

`where` and what it implies for test selection:

| where | meaning | tests that can observe it |
| --- | --- | --- |
| `body` | statements inside a def or lambda (nested defs report their own qualname, e.g. `f.<locals>.g`) | tests that executed that function in the pristine run (coverage of `old_lines`) |
| `header` | decorators, defaults, annotations, parameters of a def at module/class level; evaluated at import | importers of the module (the def statement runs at import) |
| `class-body` | a class-level statement changed, or a method/attribute was added/removed/renamed (`names` lists them) | importers of the class's module (metaclass/registration effects), plus users of the names |
| `class-header` | bases, decorators, metaclass keywords of a class | importers |
| `module-level` | module statements (imports, globals, top-level calls), or the module no longer compiles (`detail` starts with `import fails`) | every test importing the module, transitively |
| `docstring` | only reported when `docstrings_neutral=False` | none at this commit (see evidence) |
| `file-added` / `file-deleted` / ... | structural change of a .py file | importers; for added modules also `data_files[].tests` = tests that tried to import that name |

Changes confined to a lambda body inside a module- or class-level statement
(for example one entry of a parser's `FUNCTIONS` table) are reported as that
lambda's `body`, not as a class-body change: the lambda's code runs only when
it is called.

## When a patch is neutral

A file is neutral when one of these holds:

1. **Python, line-only.** Old and new compile (as the import system does it:
   bytes, so encoding cookies apply; `dont_inherit`; `optimize=0`) to code
   objects that are equal recursively except for line positions:
   `co_firstlineno`, line/position tables, and the class-body constant stored
   into `__firstlineno__` (Python 3.13). Bytecode, constants, names,
   docstrings, defaults, annotations and exception tables are all compared.
   Or, failing that, the ASTs are equal with positions dropped. Comments,
   blank lines and reformatting pass. Any new compile warning on the patched
   side (say, `invalid escape sequence`) makes the file not neutral.
2. **Python, docstring-only** (`docstrings_neutral=True`, the default): the
   only AST difference is in docstrings. The evidence:
   - `__doc__` is read nowhere in sqlglot or its tests. The one use is
     `expressions/core.py` writing `cls.__doc__ = cls.__doc__ or ""` for pdoc.
   - `tests/test_docs.py` and `tests/test_integration_loader.py` use the
     unittest `load_tests` protocol, which pytest ignores, so no doctest is in
     the graded suite (br-eval confirmed this against the labels).
   - `tests/test_typing.py` compares annotator code objects (`co_consts`
     included, so docstrings and nested line tables count). The build-time
     `typing_guard` checks that no child/parent annotator pair is equal
     modulo docstrings and line tables (100 pairs at the pinned commit, none
     equal), so such an edit cannot make that test find a redundant entry.
3. **Python file nothing uses.** The file is never imported and never read by
   any test during the traced run, for example `setup.py`, `benchmarks/*`,
   `tests/gen_fixtures.py`, `tests/leakcheck.py` or `sqlglot/__main__.py`.
   This needs the file map.
4. **Data file nothing reads.** No test opened it during the traced run and
   no test source names it literally (examples: README.md, CHANGELOG.md,
   posts/, pdoc/). Needs the file map.
5. **SQL fixture, comment or blank lines only.** Every test that reads the
   file does so through `tests/helpers.py` `load_sql_fixtures` or
   `load_sql_fixture_pairs`, both of which run `_filter_comments`. This rule
   checks that `_filter_comments(old) == _filter_comments(new)`, so whole
   `--` comment lines and empty lines pass. Inline `-- x` comments,
   whitespace-only lines and `#` meta lines do not.
6. **Root `pyproject.toml` / `setup.cfg`**: only when neither version has a
   pytest section, the new one still parses, and no test reads it.

The following are never neutral: `conftest.py`, `pytest.ini`, `tox.ini`,
`sitecustomize.py`, `.pth` files, new, deleted, renamed, copied or binary
files, mode changes, patches that do not parse or apply exactly, and empty
patches. A Python edit to a file whose source text tests read, other than the
code-level readers below, is also never neutral.

**Source text readers at this commit**, found by the trace:
- `test_typing` reads `sqlglot/typing/*.py`, parses them with `ast` and looks
  only at `ImportFrom` nodes.
- `test_sqlglotc` runs `sqlglotc/setup.py` with `runpy`.

Both depend only on the compiled program. Nothing in sqlglot or its tests
reads line numbers: there is no `inspect.getsource`, `linecache`, `_getframe`
or `f_lineno`, and the only `inspect` uses are signatures and argspecs.

## Chunk mode (why it is cheap and still exact)

`build.sh` writes `data/segments/`, one JSON per pristine `.py` file. Each
splits the file into top-level statement segments (with the comment gap
before each one) and, for classes, into segments of the class body.

At run time only the segments that contain touched lines are compiled and
parsed. Class-body segments are compiled inside their real enclosing class
headers, which keeps qualnames, name mangling and `super()` intact. Segment
boundaries are points where the tokenizer is at depth 0, so if the new chunk
compiles on its own, the text before and after it parses exactly as before.

Anything that doesn't fit falls back to whole-file comparison:
- the patch touches lines 1-2 (an encoding cookie could appear)
- `__future__` text, CR, form-feed or NUL characters in the chunk
- a touched class header
- a chunk that doesn't compile on its own

`python -m staticeq.validate` mutates the real tree at random (comments,
inline comments, blank lines, docstrings, operators and constants) and
checks that chunk mode and whole-file mode agree. On 1500 mutations there
were 0 disagreements and 0 wrong verdicts. It cost 2.7 ms per patch in chunk
mode against 19 ms whole-file.

## File map (`data/filemap.json`, committed, about 170 KB)

The map comes from running the pristine suite once with
`staticeq/trace_plugin.py`, using the labeling environment:
`SKIP_INTEGRATION=0`, the base image and `requirements.lock`. The plugin
works like this:

- An audit hook records every `open` of a repo path, including failed opens
  of files that don't exist yet, every `os.listdir`/`scandir`/`glob` outside
  the import system, and every `.py` file read as text. Each record carries
  a reader signature such as `tests/helpers.py:load_sql_fixture_pairs`.
- A `meta_path` finder records failed imports of `sqlglot.*` and `tests.*`.
- Each test's **call phase runs in a forked child**, while setup and teardown
  stay in the parent. A cache filled by one test therefore can't hide a read
  from another. Reads made during `setUpClass` are attributed to the whole
  class, and import-time reads by a shared module to every test.
- `ProcessPoolExecutor` is swapped for an inline executor, so work that
  test_optimizer and test_executor send to spawned workers runs, and is
  traced, in-process. DuckDB is pinned to one thread so it survives fork.

Regenerate with `STATICEQ_TRACE=1 staticeq/build.sh /path/to/sqlglot`
(about 3 minutes on 3 cores). The pinned tree never changes, so the committed
map is what the build would produce.

## Build

```sh
staticeq/build.sh /opt/sqlglot      # after requirements.lock is installed; ~4 s
```

This writes `staticeq/data/segments/` and makes `staticeq/` readable by
nobody. Without it, analysis still works but always takes whole-file mode,
which is slower but gives the same verdicts. Override locations with
`STATICEQ_SEGMENTS=dir` and `STATICEQ_MAP=file`.

## CLI and tests

```sh
python -m staticeq --root /opt/sqlglot sample/commits/*.patch --brief   # JSON lines
python -m unittest discover -s staticeq/tests -t .                      # synthetic diffs
python -m staticeq.validate --root /opt/sqlglot --n 1500                # mutation check
```

On the 30 sample commits, all 3 `comment` commits come out neutral. Every
commit that breaks a test comes out not neutral, and so do all the other
`code` commits (their edits change bytecode). In the 3 `multi` commits the
comment-only Python files come out neutral, and the code and fixture files
are reported. Analysing all 30 takes about 70-100 ms of CPU in total, plus
about 25 ms to import the module.

## Cost

- Importing the module costs about 25 ms, once. The segment files load lazily,
  only for touched files.
- A comment edit costs about 1-3 ms, and a code edit with units about 2-10 ms.
- Whole-file fallback on the largest files (`parser.py`, `generator.py`) costs
  about 80-250 ms.
- Call `gc.freeze()` after loading large maps in the selector process. It cut
  analysis CPU by about 25% here, because the cyclic GC otherwise keeps
  rescanning long-lived map objects.
