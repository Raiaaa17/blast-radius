"""Build-time map of which tests touch which files (see trace_plugin.py).

The map is produced by ``staticeq/build.sh`` by running the pristine sqlglot
suite once with a file-access tracer, each test in a forked child so that
per-test attribution is not hidden by caches or class-level setup. The JSON
shape (all paths repo-relative, tests as indexes into ``tests``)::

    {
      "version": 2,
      "tests": ["tests/test_x.py::T::test_a", ...],
      "reads": {path: {"tests": [i, ...], "readers": ["tests/helpers.py:load_sql_fixtures", ...]}},
      "source_reads": {path.py: {"tests": [...], "readers": [...]}},   # .py read as text
      "listdirs": {dir: {"tests": [...], "readers": [...]}},
      "missing_imports": {module: [i, ...]},
      "imported": [path.py, ...],          # every repo .py imported anywhere in the run
      "all_tests_reads": [path, ...],      # read at import of a shared module: every test
      "outcomes": {"run": n, "failed": [ids...]}
    }
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

DEFAULT_MAP_PATHS = (
    os.environ.get("STATICEQ_MAP", ""),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "filemap.json"),
)


class DataMap:
    def __init__(self, raw: dict):
        self.raw = raw
        self.tests: list[str] = raw.get("tests", [])
        self.reads: dict = raw.get("reads", {})
        self.source_reads: dict = raw.get("source_reads", {})
        self.listdirs: dict = raw.get("listdirs", {})
        self.missing_imports: dict = raw.get("missing_imports", {})
        self.imported: set[str] = set(raw.get("imported", []))
        self.all_tests_reads: set[str] = set(raw.get("all_tests_reads", []))
        self.mentions: dict[str, list[str]] = raw.get("mentions", {})
        guard = raw.get("typing_guard")
        # Unknown guard (map built without it) -> treat sqlglot/typing as sensitive.
        self.typing_guard_files: set[str] = (
            set(guard.get("sensitive_files", [])) if isinstance(guard, dict) and guard.get("ok")
            else {"<all sqlglot/typing>"})
        self._by_file: dict[str, list[str]] | None = None

    def ids(self, idxs) -> list[str]:
        return [self.tests[i] for i in idxs]

    def readers_of(self, path: str) -> tuple[list[str], list[str]]:
        """(test ids, reader signatures) of tests that opened this path."""
        if path in self.all_tests_reads:
            return list(self.tests), ["<import-time read: every test>"]
        entry = self.reads.get(path) or self.source_reads.get(path)
        if not entry:
            return [], []
        return self.ids(entry["tests"]), list(entry.get("readers", []))

    def source_readers_of(self, path: str) -> tuple[list[str], list[str]]:
        entry = self.source_reads.get(path)
        if not entry:
            return [], []
        return self.ids(entry["tests"]), list(entry.get("readers", []))

    def listers_of_parents(self, path: str) -> tuple[list[str], list[str]]:
        """Tests that listed any ancestor directory of path (a new or removed
        entry there changes what they see)."""
        ids: set[str] = set()
        readers: set[str] = set()
        parts = path.split("/")
        for k in range(len(parts) - 1, -1, -1):
            d = "/".join(parts[:k]) if k else "."
            entry = self.listdirs.get(d)
            if entry:
                ids.update(self.ids(entry["tests"]))
                readers.update(entry.get("readers", []))
        return sorted(ids), sorted(readers)

    def import_attempts(self, module: str) -> list[str]:
        out: set[str] = set()
        for name, idxs in self.missing_imports.items():
            if name == module or name.startswith(module + ".") or module.startswith(name + "."):
                out.update(self.ids(idxs))
        return sorted(out)

    def tests_in_file(self, test_file: str) -> list[str]:
        if self._by_file is None:
            self._by_file = {}
            for t in self.tests:
                self._by_file.setdefault(t.split("::", 1)[0], []).append(t)
        return self._by_file.get(test_file, [])

    def mention_tests(self, path: str) -> tuple[list[str], list[str]]:
        files = self.mentions.get(path, [])
        ids: list[str] = []
        for f in files:
            ids.extend(self.tests_in_file(f))
        return sorted(set(ids)), files

    def is_relevant(self, path: str) -> bool:
        """Could any test observe this existing file at all?"""
        if path in self.imported or path in self.all_tests_reads:
            return True
        if path in self.reads or path in self.source_reads:
            return True
        return False


@lru_cache(maxsize=4)
def load_map(path: str | None = None) -> DataMap | None:
    candidates = [path] if path else [p for p in DEFAULT_MAP_PATHS if p]
    for p in candidates:
        try:
            with open(p, encoding="utf-8") as f:
                return DataMap(json.load(f))
        except (OSError, ValueError):
            continue
    return None
