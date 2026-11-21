# eval/upstream: real sqlglot changes as evaluation patches

Evaluation data only. Nothing in this directory, and no label produced from it, may
feed a selection rule (the track forbids a table from a patch or a kind of edit to the
tests it breaks). Use it to measure a selector on change kinds the synthetic sample
does not contain.

## What it makes

Patches shaped exactly like the kit's `sample/commits/*.patch` (plain `git diff`
against the pinned commit `eb11aec`, `--no-renames`, 7-character `index` hashes,
3 lines of context), from two real sources:

| direction | source | meaning |
| --- | --- | --- |
| `fwd` | commits after the pin on upstream `main` | the change sqlglot actually made next |
| `rev` | commits before the pin, reverse-applied | "undo that commit" against the pin |

Each origin commit is split into portions; every distinct portion that applies cleanly
to the pin becomes one patch, id `<sha7>-<fwd|rev>-<portion>`:

| portion | content |
| --- | --- |
| `whole` | every in-scope file of the commit (only when all apply) |
| `src` | `sqlglot/`, `sqlglotc/` files only |
| `tests` | `tests/**/*.py` only |
| `fixtures` | `tests/` non-`.py` files (fixtures) only |
| `partial` | the in-scope files that apply, when `whole` does not |
| `squash` | `fwd` only, when `whole` does not apply: pin -> that commit for the files it touches (also lands the intermediate history of those files) |

Out of scope and dropped (listed per patch in `dropped_paths`): docs, CI, packaging,
the integration-tests submodule pointer, binary files. Identical patch texts are emitted
once (the others are listed as `duplicates`).

The coarse `kind` comes from the patch alone, never from a test outcome: `code`;
`comment` (every touched source file has an unchanged AST, or only docstrings changed -
`change: docstring`); `test`; `fixture`; `multi` when more than one of those appears.

## Files

- `mine.py` - mines a pool: `mine.py --repo <sqlglot clone> --out <pool> [--rev-limit N] --tests-json sample/tests.json`.
  Needs a clone with history around the pin (a `--shallow-since` clone is enough). Works
  entirely in git's object store with a private index; never checks anything out.
- `select_corpus.py` - ranks the pool (all `fwd` first, then `rev` newest-first, greedily
  covering new (area, portion) pairs) and writes a corpus dir: `commits/`, `manifest.json`
  (origin, direction, portion, kind, files with area and change class, size, rank),
  `kinds.json`, `priority.txt`, `tests.json`.
- `label.sh` - labels a corpus in priority order with br-eval's reference labeler
  (`eval/labeler/label.py`: pristine pinned tree + patch, full suite in the reference
  image, the grader's failure semantics) and writes `answers.json` in the kit's format.
- `sample/` - a small representative subset of the corpus, with its labels, for quick checks.

## Reproduce

```sh
git clone --shallow-since=2026-06-01 --no-checkout https://github.com/tobymao/sqlglot.git /path/sqlglot
eval/upstream/mine.py --repo /path/sqlglot --out /path/pool --rev-limit 1300 --tests-json sample/tests.json
eval/upstream/select_corpus.py --pool /path/pool --out /path/corpus --limit 500
BR_JOBS=2 BR_CPUSET=6,7 eval/upstream/label.sh /path/corpus      # ~3 min of one core per patch, memory-gated
eval/score_image.sh <image> /path/corpus                          # br-eval's scorer
```

The full corpus and its labels live outside the repo (see the findings note for the path).
