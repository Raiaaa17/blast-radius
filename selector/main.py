"""Blast Radius selector: for each patch, print the tests that fail, packed.

Per commit:
1. staticeq proves the patch behavior-neutral (comments, docstrings,
   annotations, unused files, fixture comment lines): run nothing;
2. otherwise classify every changed unit (analyze.py) and map it to candidate
   tests through the build-time maps (coverage of function bodies, class
   attribute reads, fixture readers, the static name/import index), falling
   back to every test whenever a change is not understood;
3. decide per candidate whether to keep it as is or verify it by running it
   (its packed seconds against its CPU seconds and how likely it is to fail);
4. run the candidates to verify against the patched tree in warm forked
   children (zygote.py) and keep the ones that do not pass;
5. pack the result onto workers (pack.py).
"""

from __future__ import annotations

import json
import os
import pickle
import resource
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))  # staticeq lives next to selector/

import analyze  # noqa: E402
import pack  # noqa: E402
import patch as patchlib  # noqa: E402
import profiles  # noqa: E402
import sentinel  # noqa: E402
import statediff  # noqa: E402
import verify  # noqa: E402
import zygote  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", "/data"))
PRISTINE = Path(os.environ.get("BR_PRISTINE", "/opt/sqlglot"))
MAPS = Path(os.environ.get("BR_MAPS", "/app/maps"))
TMP = Path(os.environ.get("BR_TMP", "/tmp/br"))

VERIFY_MAX_S = 6.0  # never verify a test that costs more CPU than this
COMMIT_BUDGET_S = 15.0  # most verification CPU spent on one commit
# The scorer charges every CPU second the container uses, and its machine
# measures about twice the CPU this one does: a local CPU second costs about
# two simulated seconds.
CPU_WEIGHT = 2.0
# One child at a time: two on the scorer's two cores contend (SMT siblings), and the
# scorer charges CPU, not wall time; the batch limit leaves ample wall time.
VERIFY_PARALLEL = 1
HOT_FIXED_S = 0.03 + 0.1  # fork, code swap, pipe; a share of the warm process's import
FRESH_FIXED_S = 0.15  # fork, symlink farm, sqlglot import from disk, compile of the changed files
THIRD_PARTY_S = 0.85  # numpy + pandas + duckdb, imported once if any fresh child needs them
P_FAIL_CODE = 0.15  # prior that a test covering a changed unit fails
P_FAIL_DATA = 0.9  # prior that a direct reader of a changed data file fails
THIRD_PARTY = ("numpy", "pandas", "duckdb")
CONFIG_FILES = {"pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini", "conftest.py"}

START = time.monotonic()


def log(message: str) -> None:
    print(f"[selector {time.monotonic() - START:6.1f}s] {message}", file=sys.stderr, flush=True)


def module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def code_lines(code: types.CodeType) -> tuple[int, int]:
    lo, hi = code.co_firstlineno, code.co_firstlineno
    stack = [code]
    while stack:
        c = stack.pop()
        for _, _, line in c.co_lines():
            if line is not None:
                lo = min(lo, line)
                hi = max(hi, line)
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return lo, hi


class Index:
    def __init__(self) -> None:
        with open(MAPS / "maps.pkl", "rb") as handle:
            m = pickle.load(handle)
        with open(MAPS / "static.pkl", "rb") as handle:
            s = pickle.load(handle)
        self.ids: list[str] = m["test_ids"]
        self.bit = {tid: i for i, tid in enumerate(self.ids)}
        self.measured: list[float] = m["seconds"]
        self.sets: list[int] = m["sets"]
        self.lines: dict[str, dict[int, int]] = m["lines"]
        self.owners: dict[str, int] = m["owners"]
        self.sub_owners: dict[str, list[str]] = defaultdict(list)
        for owner, mro in m["owner_mro"].items():
            for key in mro:
                self.sub_owners[key].append(owner)
        self.opens: dict[str, int] = m["opens"]
        self.lists: dict[str, int] = m["lists"]
        self.import_codes = m["import_codes"]
        self.import_sensitive: int = m["import_sensitive"]
        self.unreliable: int = m["unreliable"]
        # Tests whose line coverage is known but not their class-attribute,
        # dict-entry or namespace reads: candidates for every import-state change.
        self.attrs_unknown: int = m.get("attrs_unknown", 0)
        self.setters: list[str] = m.get("setters", [])
        self.dict_slots: dict[tuple, int] = m.get("dict_slots", {})
        self.dict_leaky: set[int] = m.get("dict_leaky", set())
        self.key_reads: dict[int, dict[bytes, int]] = m.get("key_reads", {})
        # A dict's entries copied into a derived table (the generator's dispatch from TRANSFORMS):
        # lookups there are reads of the source's entries.
        self.derived: dict[int, set[int]] = m.get("derived", {})
        self.enums: dict[tuple, int] = m.get("enums", {})
        self.by_file: dict[str, int] = m["by_file"]
        self.all = (1 << len(self.ids)) - 1
        self.codes = s["codes"]
        # Per code entry: names it uses as attributes or imports (None: unknown, an older map).
        self.code_attr_names: dict[int, frozenset] | None = (
            {id(entry): names for entry, names in zip(self.codes, s["attr_names"])} if "attr_names" in s else None
        )
        self.codes_by_file: dict[str, list] = defaultdict(list)
        for entry in self.codes:
            self.codes_by_file[entry[0]].append(entry)
        self.importers: dict[str, set[str]] = s["importers"]
        self.modules: dict[str, str] = s["modules"]
        self.holders: dict[str, set[str]] = s["holders"]
        self._closure: dict[str, set[str]] = {}
        self.codes_by_name: dict[str, list] | None = None
        z: dict = {}
        if (MAPS / "zygote.pkl").exists():
            with open(MAPS / "zygote.pkl", "rb") as handle:
                z = pickle.load(handle)
        self.timing: dict[str, tuple[str, float]] = z.get("timing", {})
        self.class_cpu: dict[str, float] = z.get("class_cpu", {})
        # Slots the pristine suite itself changes while it runs (sentinel.py); None: unknown.
        self.volatile: set | None = z.get("volatile")
        self.warm_import_keys: set = z.get("import_keys", set())
        self.fresh_import: dict[str, float] = z.get("fresh_import", {})
        self.attrs: dict = m["attrs"]
        self.owner_mro: dict[str, list[str]] = m["owner_mro"]
        self.state: dict = {}
        self.state_unordered: set = set()
        self.state_unstable: set = set()
        self.state_mro: dict[str, list[str]] = {}
        self.state_dicts: set = set()
        self.state_callables: dict = {}
        self.state_untraced: set | None = None
        self.state_enums: set = set()
        self.state_eager: set | None = None  # modules `import sqlglot` imports itself
        if (MAPS / "state.pkl").exists():
            with open(MAPS / "state.pkl", "rb") as handle:
                st = pickle.load(handle)
            self.state = st["slots"]
            self.state_unordered = st["unordered"]
            self.state_unstable = st["unstable"]
            self.state_mro = st["mro"]
            self.state_dicts = st.get("dict_slots", set())
            self.state_callables = st.get("callables", {})
            self.state_untraced = st.get("untraced", None)
            self.state_enums = st.get("enums", set())
            self.state_eager = st.get("eager", None)
        self.profile_sites: dict = {}
        self.profiled = 0
        self.uninstrumented: set[str] = set()
        if (MAPS / "profiles.pkl").exists():
            with open(MAPS / "profiles.pkl", "rb") as handle:
                p = pickle.load(handle)
            self.profile_sites = p["sites"]
            self.profiled = p["profiled"]
            self.uninstrumented = p["uninstrumented"]

    # ------------------------------------------------------------ masks

    def lines_mask(self, rel: str, first: int, last: int) -> int:
        table = self.lines.get(rel)
        if not table:
            return 0
        mask = 0
        sets = self.sets
        if last - first + 1 <= len(table):
            for line in range(first, last + 1):
                sid = table.get(line)
                if sid is not None:
                    mask |= sets[sid]
        else:
            for line, sid in table.items():
                if first <= line <= last:
                    mask |= sets[sid]
        return mask

    def class_use(self, key: str) -> int:
        mask = 0
        for owner in self.sub_owners.get(key, ()):
            mask |= self.sets[self.owners[owner]]
        return mask

    def importer_closure(self, mod: str) -> set[str]:
        if mod in self._closure:
            return self._closure[mod]
        seen: set[str] = set()
        stack = [mod]
        while stack:
            for nxt in self.importers.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        self._closure[mod] = seen
        return seen


class Impact:
    def __init__(self) -> None:
        self.mask = 0
        self.data_mask = 0  # direct readers of a changed data file (likely to fail)
        self.broad: list[str] = []
        self.notes: list[str] = []
        self.hot = True  # every change can be swapped into a warm process
        self.hot_why = ""
        self.py_files: list[str] = []
        self.data_files: list[str] = []
        self.units: list[tuple[int, list | None]] = []  # (mask, comparison-operator edits or None)
        # Class-body / module-level / import-time edits of sqlglot modules: resolved
        # by diffing the patched import state; (fallback mask or None = broad, note).
        self.state_units: list[tuple[int | None, str]] = []
        self.state_modules: set[str] = set()
        # New test modules: pytest imports them at collection, before any test runs.
        self.preimport: list[str] = []
        # Changed run-time code (function bodies): it may mutate shared state while tests run.
        self.functions = False
        self.may_pollute = False  # a changed function adds a mutation of class/module state
        self.state_slots: set = set()  # import-state slots the patch itself changes

    def add(self, mask: int, sites: list | None = None) -> None:
        self.mask |= mask
        self.units.append((mask, sites))

    def not_hot(self, why: str) -> None:
        if self.hot:
            self.hot = False
            self.hot_why = why


def class_impact(ix: Index, rel: str, qualname: str, names: set[str] | None) -> int:
    mod = module_name(rel)
    key = f"{mod}:{qualname}"
    mask = ix.class_use(key)
    parts = qualname.split(".")
    for i in range(1, len(parts)):  # enclosing classes (a dialect around its Tokenizer)
        mask |= ix.class_use(f"{mod}:{'.'.join(parts[:i])}")
    for holder in ix.holders.get(key, ()):  # classes holding this one (tokenizer_class, ...)
        mask |= ix.class_use(holder)
    simple = parts[-1]
    files = {rel} | {ix.modules[m] for m in ix.importer_closure(mod) if m in ix.modules}
    for path in files:
        for _, kind, qn, first, last, cnames in ix.codes_by_file.get(path, ()):
            if kind != "class" or simple not in cnames:
                continue
            if names is not None and not (cnames & names):
                continue
            mask |= ix.class_use(f"{module_name(path)}:{qn}")
    return mask | ix.import_sensitive | ix.unreliable | ix.attrs_unknown


def name_impact(ix: Index, rel: str, names: set[str], runtime: bool = False) -> int:
    """Tests that run code referencing the names (and the users of what they name).

    runtime=True: only run-time code counts. With the import state diffed, what a
    class body built from a name at import is itself in the diff (as that class's
    own changed attributes), so class bodies that mention the name add nothing.
    """
    mod = module_name(rel)
    mask = ix.import_sensitive | ix.unreliable | ix.attrs_unknown
    files = {rel} | {ix.modules[m] for m in ix.importer_closure(mod) if m in ix.modules}
    if ix.codes_by_name is None:
        index: dict[str, list] = defaultdict(list)
        for entry in ix.codes:
            for name in entry[5]:
                index[name].append(entry)
        ix.codes_by_name = index
    seen = set()
    for name in names:
        for entry in ix.codes_by_name.get(name, ()):
            path, kind, qn, first, last, _ = entry
            if path not in files or id(entry) in seen:
                continue
            if runtime and path != rel and ix.code_attr_names is not None:
                # Another module reaches this module's global only as an attribute
                # (module.NAME) or an import; a global of the same name there is its
                # own (a from-import rebinding shows up as that module's changed slot).
                if name not in ix.code_attr_names.get(id(entry), frozenset((name,))):
                    continue
            seen.add(id(entry))
            if kind == "function":
                mask |= ix.lines_mask(path, first, last)
            elif kind == "class" and not runtime:
                mask |= ix.class_use(f"{module_name(path)}:{qn}")
    # The rebound objects themselves: a function's callers, a class's users.
    for _, kind, qn, first, last, _ in ix.codes_by_file.get(rel, ()):
        if qn in names:
            if kind == "function":
                mask |= ix.lines_mask(rel, first, last)
            elif kind == "class":
                mask |= class_impact(ix, rel, qn, None)
    return mask


def _top_level_importable(name: str) -> bool:
    """Whether a top-level module of this name exists outside the tree (stdlib, site-packages)."""
    if not name.isidentifier():
        return False
    if name in sys.modules or name in sys.builtin_module_names or name in ("sqlglot", "tests"):
        return True
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return True


def _quiet_test_module(source: bytes) -> bool:
    """A new test module whose import (at pytest collection) only defines things.

    pytest imports every test module before it runs any test, so a new module's
    import-time side effects reach every test. Imports, definitions, constant
    assignments and an `if __name__ == "__main__"` block are quiet; anything
    else (calls, attribute or item assignments, loops, ...) is not.
    """
    import ast

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return True  # it cannot be collected: only its own (unnumbered) tests fail

    def quiet(stmts: list, in_class: bool) -> bool:
        for stmt in stmts:
            if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.Pass)):
                continue
            if isinstance(stmt, ast.ClassDef):
                if not quiet(stmt.body, True):
                    return False
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                if all(isinstance(t, (ast.Name, ast.Tuple, ast.List)) for t in targets):
                    continue
                return False
            if (
                not in_class
                and isinstance(stmt, ast.If)
                and isinstance(stmt.test, ast.Compare)
                and isinstance(stmt.test.left, ast.Name)
                and stmt.test.left.id == "__name__"
            ):
                continue
            return False
        return True

    return quiet(tree.body, False)


def assess(
    ix: Index,
    diffs: list[patchlib.FileDiff],
    files: dict[str, bytes | None],
    neutral_files: set[str],
    data_tests: dict[str, set[str]],
    dir_listers: dict[str, set[str]] | None = None,
    filemap_ok: bool = True,
) -> Impact:
    """filemap_ok=False: staticeq's file map is missing, so readers and listers are unknown."""
    impact = Impact()
    dir_listers = dir_listers or {}

    def ids_mask(ids) -> int:
        mask = 0
        for tid in ids:
            b = ix.bit.get(tid)
            if b is not None:
                mask |= 1 << b
        return mask

    def readers(rel: str) -> int:
        """Tests that opened the file (or, per staticeq, read or tried to import it)."""
        if not filemap_ok and "no file map" not in impact.broad:
            impact.broad.append("no file map")
        mask = ix.sets[ix.opens[rel]] if rel in ix.opens else 0
        return mask | ids_mask(data_tests.get(rel, ()))

    def listers(rel: str) -> int:
        """Tests that list a directory above the file (a new or removed entry changes what they see).

        staticeq's file map records only listings made by test and library
        code; the coverage map's listings also hold the import system's own
        directory scans, which do not depend on unrelated entries.
        """
        return ids_mask(dir_listers.get(rel, ()))

    def test_module_users(rel: str) -> int:
        """A test module's own tests and the tests of every module importing it."""
        mod = module_name(rel)
        mask = ix.by_file.get(rel, 0)
        for m in ix.importer_closure(mod):
            other = ix.modules.get(m)
            if other:
                mask |= ix.by_file.get(other, 0)
        return mask

    def removed(rel: str) -> bool:
        """A path that disappears (deleted, or the old side of a rename). False: broad."""
        if not rel.endswith(".py") or not rel.startswith(("sqlglot/", "tests/")):
            # Data files, and scripts outside the package (no test imports those:
            # staticeq's map lists every imported module, all under sqlglot/ and tests/).
            direct = readers(rel)
            impact.add(direct | listers(rel))
            impact.data_mask |= direct
            impact.notes.append(f"removed {rel}: {bin(direct).count('1')} readers")
            return True
        if rel.startswith("tests/") and os.path.basename(rel) != "__init__.py":
            # Its own tests vanish (kept as cheap insurance) and its importers fail.
            mask = test_module_users(rel) | readers(rel) | listers(rel) | ix.import_sensitive | ix.unreliable
            impact.add(mask)
            impact.notes.append(f"removed test module {rel}: {bin(mask).count('1')}")
            return True
        return False

    def added(rel: str, content: bytes | None) -> bool:
        """A path that appears (new file, or the new side of a rename). False: broad."""
        top = rel.split("/", 1)[0]
        if rel.endswith(".py") and not rel.startswith(("sqlglot/", "tests/")):
            # The repository root is on sys.path: a new top-level module or
            # package shadows any installed module of that name.
            name = top[:-3] if "/" not in rel else top
            if "/" not in rel or rel == f"{top}/__init__.py":
                if _top_level_importable(name):
                    impact.broad.append(f"new top-level module {rel} shadows {name}")
                    return False
        if rel.endswith(".py") and os.path.basename(rel) == "__init__.py":
            if rel.startswith(("sqlglot/", "tests/")):
                impact.broad.append(f"new package {rel}")
                return False
        mask = readers(rel) | listers(rel) | ix.import_sensitive | ix.unreliable
        if rel.endswith(".py") and rel.startswith("tests/"):
            if not _quiet_test_module(content or b""):
                impact.broad.append(f"new test module {rel} has import-time side effects")
                return False
            impact.preimport.append(module_name(rel))
        impact.add(mask)
        impact.notes.append(f"added {rel}: {bin(mask).count('1')}")
        return True

    for diff in diffs:
        path = diff.path
        base = os.path.basename(path)
        in_tree = path.startswith(("sqlglot/", "tests/"))
        if path in neutral_files and not (diff.old_path is None or diff.new_path is None):
            impact.notes.append(f"neutral {path}")
            continue
        if diff.binary:
            if in_tree or readers(path) or listers(path):
                impact.broad.append(f"binary {path}")
            impact.not_hot("binary")
            continue
        if base in CONFIG_FILES and (in_tree or "/" not in path):
            impact.broad.append(f"config {path}")
            continue
        if diff.mode_change and not diff.hunks and not diff.binary and diff.old_path == diff.new_path:
            impact.notes.append(f"mode only {path}")
            continue
        if diff.new_mode == "120000" or (diff.mode_change and diff.new_mode not in ("", "100644", "100755")):
            impact.broad.append(f"symlink or special mode {diff.new_mode} {path}")
            continue
        renamed = bool(diff.old_path and diff.new_path and diff.old_path != diff.new_path)
        gone = None if diff.copy else (diff.old_path if (renamed or diff.new_path is None) else None)
        if renamed or diff.old_path is None or diff.new_path is None:
            impact.not_hot(f"structural {path}")
            if gone and gone.startswith("sqlglot/") and gone.endswith(".py"):
                impact.broad.append(f"removed {gone}")
                continue
            if gone:
                if not removed(gone):
                    if not impact.broad:
                        impact.broad.append(f"removed {gone}")
                    continue
            if diff.new_path and (renamed or diff.old_path is None):
                if not added(diff.new_path, files.get(diff.new_path)):
                    continue
                if diff.new_path.endswith(".py"):
                    impact.py_files.append(diff.new_path)
                else:
                    impact.data_files.append(diff.new_path)
            continue
        if not path.endswith(".py"):
            # Data and fixture files: the tests that read them.
            impact.not_hot(f"data file {path}")
            impact.data_files.append(path)
            direct = readers(path)
            impact.add(direct)
            impact.data_mask |= direct
            impact.notes.append(f"data {path}: {bin(direct).count('1')}")
            continue
        if not in_tree:
            # Scripts outside the package (sqlglotc/setup.py, ...): staticeq proved
            # the unused ones neutral; the rest are run or read by known tests.
            got = readers(path)
            if not got:
                impact.broad.append(f"changed script {path} used by unknown tests")
                continue
            impact.not_hot(f"script {path}")
            impact.add(got | ix.unreliable)
            impact.data_mask |= got
            impact.notes.append(f"script {path}: {bin(got).count('1')}")
            continue
        impact.py_files.append(path)
        if path in ix.opens:  # a test reads this source file as text
            impact.not_hot(f"source read {path}")
            impact.add(ix.sets[ix.opens[path]])
        old = (PRISTINE / path).read_bytes()
        new = files[path]
        try:
            old_text, new_text = old.decode("utf-8"), new.decode("utf-8")
        except UnicodeDecodeError:
            impact.broad.append(f"undecodable {path}")
            continue
        result = analyze.analyze_python(old_text, new_text, str(PRISTINE / path))
        if result.syntax_error:
            impact.broad.append(f"syntax error {path}")
            continue
        if result.neutral:
            impact.notes.append(f"neutral {path}")
            continue
        is_test = path.startswith("tests/")
        mod = module_name(path)
        for unit in result.units:
            if unit.kind == "function":
                code = unit.old_code
                if not is_test and (path, code.co_qualname, code.co_firstlineno) in ix.import_codes:
                    # It ran while sqlglot was imported: what it built is in the import state.
                    impact.state_units.append((None, f"import-time function {path}:{unit.qualname}"))
                    impact.state_modules.add(mod)
                if (path, code.co_qualname, code.co_firstlineno) in ix.warm_import_keys:
                    impact.not_hot(f"ran at import {path}:{unit.qualname}")
                    if is_test:
                        # A test helper run while test modules import (helpers.SKIP_INTEGRATION
                        # = string_to_bool(...)): module state every importer's tests see.
                        users = test_module_users(path)
                        impact.add(users | ix.unreliable)
                        impact.notes.append(f"test helper run at import {path}:{unit.qualname}: "
                                            f"{bin(users).count('1')}")
                if unit.ranges:
                    mask = 0
                    for first, last in unit.ranges:
                        mask |= ix.lines_mask(path, first, last)
                    where = ",".join(f"{a}-{b}" if a != b else str(a) for a, b in unit.ranges)
                else:
                    first, last = code_lines(code)
                    mask = ix.lines_mask(path, first, last)
                    where = "all"
                sites = None
                if unit.old_node is not None and unit.new_node is not None:
                    try:
                        sites = profiles.compare_only_diff(unit.old_node, unit.new_node, path) or None
                    except Exception:  # noqa: BLE001
                        sites = None
                if sites:
                    where += " cmp " + ",".join(f"{o}->{n}" for _, o, n in sites)
                # Tests the coverage tracer could not follow are always candidates;
                # their comparison profiles are complete, so they can be proven too.
                impact.add(mask | ix.unreliable, sites)
                impact.functions = True
                if unit.may_pollute and not is_test:
                    impact.may_pollute = True
                    impact.notes.append(f"may mutate shared state: {unit.qualname}")
                impact.notes.append(f"function {path}:{unit.qualname} [{where}]: {bin(mask).count('1')}")
                continue
            impact.not_hot(f"{unit.kind} body {path}:{unit.qualname}")
            if is_test:
                mask = ix.by_file.get(path, 0) | ix.unreliable
                for m in ix.importer_closure(mod):
                    rel = ix.modules.get(m)
                    if rel:
                        mask |= ix.by_file.get(rel, 0)
                impact.add(mask)
                impact.notes.append(f"test {unit.kind} {path}:{unit.qualname}: {bin(mask).count('1')}")
            elif unit.kind == "class":
                impact.state_units.append((None, f"class {path}:{unit.qualname} {unit.names}"))
                impact.state_modules.add(mod)
            else:
                impact.state_units.append(
                    (None, f"module {path} {sorted(unit.names) if unit.names is not None else '?'}")
                )
                impact.state_modules.add(mod)
    return impact


def prove(ix: Index, impact: Impact) -> int:
    """Tests (map bits) whose comparison-site profiles show every edit they can see invisible.

    A unit made only of comparison-operator edits is invisible to a profiled
    test when, at each edited site, the test either never evaluated it or every
    operand pair it produced gives the same result under both operators.
    """
    removable = ix.profiled
    for mask, sites in impact.units:
        if not mask:
            continue
        if sites is None:
            removable &= ~mask
            continue
        proven = ix.profiled
        for key, old_op, new_op in sites:
            # Only sqlglot's own modules were instrumented: a site anywhere else
            # (tests/helpers.py, ...) has no profile, which proves nothing.
            if key[0] in ix.uninstrumented or not str(key[0]).startswith("sqlglot/"):
                proven = 0
                break
            for sigs, tests in ix.profile_sites.get(key, ()):
                if not profiles.equivalent(sigs, old_op, new_op):
                    proven &= ~tests
        removable &= ~mask | proven
    return removable


def entry_readers(ix: Index, slot: tuple, entries: dict) -> int | None:
    """Tests that read a changed entry of a dict slot, or the whole dict; None: not known per entry."""
    changed = entries.get(slot)
    gid = ix.dict_slots.get(slot)
    if changed is None or gid is None or gid in ix.dict_leaky:
        return None
    mask = 0
    for g in [gid, *ix.derived.get(gid, ())]:
        if g in ix.dict_leaky:
            return None
        reads = ix.key_reads.get(g, {})
        for code in list(changed) + [b"*"]:
            sid = reads.get(code)
            if sid is not None:
                mask |= ix.sets[sid]
    return mask


def namespace_listers(ix: Index, kind: str, name: str) -> int:
    """Tests that listed a class's (or a subclass's) or a module's namespace (dir, vars)."""
    sid = ix.enums.get((kind, name))
    return ix.sets[sid] if sid is not None else 0


def slots_mask(ix: Index, slots: list, modules: set[str], entries: dict | None = None,
               removed: set | None = None) -> tuple[int | None, list[str]]:
    """Tests that read any of the changed import-state slots (None = every test).

    entries: for a changed dict slot, the keys of the entries that differ; the
    slot then maps to the tests that read those entries (or the whole dict)
    instead of every test that read the attribute.
    """
    entries = entries or {}
    removed = removed or set()
    mask = ix.import_sensitive | ix.unreliable | ix.attrs_unknown
    notes: list[str] = []
    check = [s for s in slots if s not in ix.state_unstable]
    # Slots whose value differs from process to process (a class-body loop
    # variable left over from iterating a set, ...) cannot be compared: count
    # them as changed when the patch edits their own module.
    for slot in ix.state_unstable:
        mod = slot[1].split(":")[0] if slot[0] in ("C", "MRO") else slot[1]
        if mod in modules:
            check.append(slot)
    failed = [slot[1] for slot in check if slot[0] == "M" and slot[2] == "<import failed>"]
    if failed:
        # Modules that no longer import. If `import sqlglot` needs one, every test fails;
        # a lazily loaded one (a dialect, its parser) fails only the tests that use it.
        if ix.state_eager is None or any(m in ix.state_eager or m not in ix.modules for m in failed):
            return None, [f"{failed[:3]} no longer import"]
        for m in failed:
            rel = ix.modules[m]
            got = ix.lines_mask(rel, 0, 1 << 20)
            for _, kind_, qn, _, _, _ in ix.codes_by_file.get(rel, ()):
                if kind_ == "class":
                    got |= ix.class_use(f"{m}:{qn}")
            for importer in ix.importer_closure(m) | {m}:
                other = ix.modules.get(importer)
                if other and other.startswith("tests/"):
                    got |= ix.by_file.get(other, 0)
            mask |= got
            notes.append(f"{m} no longer imports (lazily loaded): {bin(got).count('1')}")
        check = [s for s in check if not (s[0] == "M" and s[2] == "<import failed>")]
    for slot in check:
        kind = slot[0]
        if kind == "C":
            _, key, attr = slot
            mod_name, _, qual = key.partition(":")
            rel = ix.modules.get(mod_name)
            untraced = ix.state_untraced is None or key in ix.state_untraced
            if untraced and key in ix.state_enums and attr.startswith("_"):
                # An enum's derived tables (_member_map_, _value2member_map_, ...): what they
                # change is in its member slots (added, removed), mapped by name below.
                notes.append(f"{key}.{attr} (enum internals)")
                continue
            if untraced and key in ix.state_enums:
                # A member: the code that names it, and every lookup that missed it.
                got = name_impact(ix, rel, {attr}, runtime=True) if rel is not None else ix.all
                sid = ix.enums.get(("miss", key, attr))
                if sid is not None:
                    got |= ix.sets[sid]
                mask |= got
                notes.append(f"{key}.{attr} (enum member): {bin(got).count('1')}")
                continue
            if (attr.startswith("__") and attr.endswith("__")) or untraced:
                # Dunders (__init__ of a dataclass, ...) are looked up by the interpreter, and the
                # recorders skip enum, tuple and exception classes: neither leaves attribute reads.
                # The code that names the class or the attribute (constructors, Enum.MEMBER) is
                # what can observe the change.
                got = ix.class_use(key)
                if rel is not None:
                    got |= name_impact(ix, rel, {qual.rsplit(".", 1)[-1], attr}, runtime=True)
                mask |= got
                notes.append(f"{key}.{attr} (untraced or dunder): {bin(got).count('1')}")
                if attr.startswith("__") and attr.endswith("__"):
                    continue
            got = entry_readers(ix, slot, entries)
            if got is not None:
                mask |= got
                notes.append(f"{key.rsplit('.', 1)[-1]}.{attr}[{len(entries[slot])} entries]: {bin(got).count('1')}")
                continue
            owners = ix.attrs.get((key, attr))
            readers = 0
            if owners:
                for sid in owners.values():
                    readers |= ix.sets[sid]
            spans = ix.state_callables.get(slot)
            if spans and slot in ix.state and slot not in removed:
                # A function-valued attribute that changed (decorator, defaults, a
                # rebinding): only a call of the old value can observe it, and every
                # call runs that value's own code first.
                ran = 0
                for rel, lo, hi in spans:
                    ran |= ix.lines_mask(rel, lo, hi)
                notes.append(f"{key.rsplit('.', 1)[-1]}.{attr}: {bin(readers).count('1')} readers, "
                             f"{bin(readers & ran).count('1')} ran it")
                readers &= ran
            mask |= readers
            if slot not in ix.state or slot in removed:
                # A new (or removed) name: what lists the class's namespace sees it, and every
                # lookup of that name that found nothing (getattr(self, key + "_sql", None), ...).
                mask |= namespace_listers(ix, "C", key)
                sid = ix.enums.get(("miss", key, attr))
                if sid is not None:
                    mask |= ix.sets[sid]
            if slot not in ix.state:  # a new attribute shadows the bases' for this class
                for base in ix.state_mro.get(key, [key])[1:]:
                    for owner, sid in (ix.attrs.get((base, attr)) or {}).items():
                        if key in ix.owner_mro.get(owner, ()):
                            mask |= ix.sets[sid]
            notes.append(f"{key.rsplit('.', 1)[-1]}.{attr}")
        elif kind == "MRO":
            mask |= ix.class_use(slot[1])
            notes.append(f"mro {slot[1]}")
        elif kind == "M":
            _, mod, name = slot
            if name == "<import failed>":
                return None, [f"{mod} no longer imports"]
            rel = ix.modules.get(mod)
            if rel is None:
                continue  # a new module: its importers are handled as a new file
            got = entry_readers(ix, slot, entries)
            if got is not None:
                mask |= got
                notes.append(f"{mod}.{name}[{len(entries[slot])} entries]: {bin(got).count('1')}")
                continue
            got = name_impact(ix, rel, {name}, runtime=True)
            if slot not in ix.state or slot in removed:
                got |= namespace_listers(ix, "M", mod)
                sid = ix.enums.get(("missM", mod, name))
                if sid is not None:
                    got |= ix.sets[sid]
            mask |= got
            notes.append(f"{mod}.{name}: {bin(got).count('1')}")
    return mask, notes


def pollution_readers(ix: Index, polluted: dict) -> int:
    """Tests that read state the candidates left changed: {slot: changed dict entries or None}."""
    mask = ix.unreliable | ix.attrs_unknown
    for slot, entries in polluted.items():
        kind, owner, name = slot
        if entries is not None:
            got = entry_readers(ix, slot, {slot: [k for k, _ in entries]})
            if got is not None:
                mask |= got
                continue
        if kind == "C":
            for sid in (ix.attrs.get((owner, name)) or {}).values():
                mask |= ix.sets[sid]
            mask |= ix.class_use(owner) if name.startswith("__") or ix.state_untraced is None \
                or owner in ix.state_untraced else 0
        else:
            rel = ix.modules.get(owner)
            if rel is None:
                return ix.all
            mask |= name_impact(ix, rel, {name}, runtime=True)
    return mask


_THIRD_PARTY_USE: dict[str, bool] = {}


def needs_third_party(module: str) -> bool:
    """Whether a test module imports numpy, pandas or duckdb (a look at its source)."""
    got = _THIRD_PARTY_USE.get(module)
    if got is None:
        try:
            text = (PRISTINE / (module.replace(".", "/") + ".py")).read_text()
        except OSError:
            text = ""
        got = any(f"import {name}" in text or f"from {name}" in text for name in THIRD_PARTY) or not text
        _THIRD_PARTY_USE[module] = got
    return got


def staticeq_verdict(data: bytes) -> dict | None:
    try:
        from staticeq import analyze as se_analyze

        return se_analyze(data, str(PRISTINE))
    except Exception as error:  # noqa: BLE001 - fall back to our own analysis
        log(f"staticeq failed: {error!r}")
        return None


class Plan:
    def __init__(self, commit: str):
        self.commit = commit
        self.files: dict[str, bytes | None] = {}
        self.diffs: list = []
        self.candidates: set[int] = set()
        self.impact = Impact()
        self.keep: set[int] = set()
        self.verify: list[int] = []
        self.mode = ""


def main() -> None:
    # Before anything can import numpy (here, in the warm process or a child):
    # single-threaded BLAS keeps every process safe to fork.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    recorded_ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
    id_of = {t["number"]: t["id"] for t in tests}
    patches = sorted((DATA / "commits").glob("*.patch"))
    n = len(patches)
    # The batch limit is 3 minutes (4.5 in the final, at 1.5x the commits);
    # assume the tighter one unless the batch is clearly a final-sized one.
    limit_s = 270.0 if n > 75 else 180.0
    deadline = START + limit_s - 45.0

    ix = Index()
    import gc

    gc.collect()
    gc.freeze()
    log(f"loaded maps for {len(ix.ids)} tests; {n} commits")
    bit_number: dict[int, int] = {}
    unknown: set[int] = set()
    for t in tests:
        b = ix.bit.get(t["id"])
        if b is None:
            unknown.add(t["number"])
        else:
            bit_number[b] = t["number"]

    def numbers(mask: int) -> set[int]:
        out = set()
        while mask:
            low = mask & -mask
            b = low.bit_length() - 1
            if b in bit_number:
                out.add(bit_number[b])
            mask ^= low
        return out

    plans: list[Plan] = []
    for patch_path in patches:
        plan = Plan(patch_path.stem)
        plans.append(plan)
        data = patch_path.read_bytes()
        verdict = staticeq_verdict(data)
        if verdict is not None and verdict.get("neutral"):
            plan.impact.notes.append("staticeq: neutral")
            log(f"{plan.commit}: neutral ({len(verdict.get('files', []))} files)")
            continue
        neutral_files: set[str] = set()
        data_tests: dict[str, set[str]] = {}
        dir_listers: dict[str, set[str]] = {}
        unknown_files = False
        if verdict is not None:
            for f in verdict.get("files", []):
                if f.get("neutral") and f.get("path"):
                    neutral_files.add(f["path"])
            for d in verdict.get("data_files", []):
                data_tests.setdefault(d["path"], set()).update(d.get("tests", ()))
                dir_listers.setdefault(d["path"], set()).update(d.get("dir_listers", ()))
                if not d.get("known", True):
                    unknown_files = True
        try:
            plan.diffs = patchlib.parse(data)
            for diff in plan.diffs:
                old = None
                if diff.old_path:
                    src = PRISTINE / diff.old_path
                    old = src.read_bytes() if src.is_file() else None
                if diff.binary:
                    continue
                plan.files[diff.new_path or diff.old_path] = patchlib.apply(old, diff)
                if diff.old_path and diff.new_path and diff.old_path != diff.new_path and not diff.copy:
                    plan.files[diff.old_path] = None
            plan.impact = assess(ix, plan.diffs, plan.files, neutral_files, data_tests, dir_listers,
                                 filemap_ok=verdict is not None and not unknown_files)
        except Exception as error:  # anything unexpected: run everything we can
            log(f"{plan.commit}: analysis failed ({error!r}); broad")
            plan.diffs, plan.files, plan.impact = [], {}, Impact()
            plan.impact.broad.append("analysis failed")

    # Class-body, module-level and import-time edits: diff the patched import state.
    state_jobs = []
    for plan in plans:
        impact = plan.impact
        if impact.broad or not impact.state_units:
            continue
        if not ix.state:
            for fallback, note in impact.state_units:
                if fallback is None:
                    impact.broad.append(note)
                else:
                    impact.add(fallback)
                    impact.notes.append(note)
            continue
        root = TMP / "state" / plan.commit
        try:
            verify.build_tree(PRISTINE, root, plan.files)
            state_jobs.append((plan.commit, root))
        except Exception as error:  # noqa: BLE001
            log(f"{plan.commit}: cannot build tree for the state diff ({error!r})")
    diffs: dict[str, dict | None] = {}
    if state_jobs:
        # The reference is the build's pristine snapshot, taken the way these
        # children take theirs; each child reports only what differs from it.
        base = {"slots": ix.state, "dict_slots": ix.state_dicts}
        diffs = statediff.run_snapshots(state_jobs, ix.state_unordered, parallel=VERIFY_PARALLEL, log=log, base=base,
                                        keys_file=MAPS / "state_keys.pkl")
        log(f"state diffs for {len(state_jobs)} commits")
    for plan in plans:
        impact = plan.impact
        if impact.broad or not impact.state_units:
            continue
        got = diffs.get(plan.commit)
        slots = None if got is None else got.get("changed")
        if slots:
            impact.state_slots = set(slots)
        mask, notes = (None, ["state diff failed"]) if slots is None else slots_mask(
            ix, slots, impact.state_modules, got.get("entries"), set(got.get("removed", ())))
        if mask is None:
            # The patched tree no longer imports (or the diff failed): every test.
            impact.broad.append(f"state: {notes}")
        else:
            impact.add(mask)
            shown = notes if os.environ.get("BR_DEBUG") else (
                notes[:12] + ([f"... {len(notes) - 12} more"] if len(notes) > 12 else []))
            impact.notes.append(f"state {[n for _, n in impact.state_units]}: {len(slots)} slots {shown}: "
                                f"{bin(mask).count('1')}")
    import shutil

    shutil.rmtree(TMP / "state", ignore_errors=True)

    for plan in plans:
        impact = plan.impact
        if impact.broad:
            plan.candidates = set(recorded_ms)
        else:
            plan.candidates = numbers(impact.mask)
            if impact.mask or any(not x.startswith("neutral") for x in impact.notes):
                plan.candidates |= unknown
        proved = [] if impact.broad or not ix.profiled else sorted(numbers(prove(ix, impact)) & plan.candidates)
        plan.candidates -= set(proved)
        log(
            f"{plan.commit}: {len(plan.candidates)} candidates; broad={impact.broad} {impact.notes}"
            + (f"; proved unaffected: {[id_of[c].split('::', 1)[1] for c in proved]}" if proved else "")
        )

    # ------------------------------------------------------------ decide

    cost_of: dict[int, float] = {}  # CPU seconds to verify, from the build's warm timing pass
    trusted: set[int] = set()  # passes in the warm verifier on the pristine tree
    class_of: dict[int, str] = {}
    for t in tests:
        entry = ix.timing.get(t["id"])
        file_part, _, rest = t["id"].partition("::")
        class_of[t["number"]] = file_part[:-3].replace("/", ".") + "." + rest.split("::")[0]
        if entry is not None:
            cost_of[t["number"]] = entry[1]
            if entry[0] == "passed":
                trusted.add(t["number"])
        else:
            cost_of[t["number"]] = 10.0

    def split(plan: Plan) -> None:
        """Keep or verify each candidate: its packed seconds against its CPU and its odds of failing."""
        impact = plan.impact
        hot = impact.hot and not impact.broad
        data_numbers = numbers(impact.data_mask)
        keep: set[int] = set()
        maybe: list[int] = []
        for c in plan.candidates:
            if plan.files and c in trusted and cost_of[c] <= VERIFY_MAX_S and not impact.broad:
                maybe.append(c)
            else:
                keep.add(c)
        plan.keep = keep
        if not maybe:
            return
        bins = pack.pack([(c, recorded_ms[c]) for c in keep])
        loads = [sum(recorded_ms[c] for c in b) for b in bins]
        verify_list: list[int] = []
        classes: set[str] = set()
        for c in sorted(maybe, key=lambda c: -recorded_ms[c]):
            r = recorded_ms[c]
            top = max(loads) if loads else 0
            options = []
            if loads:
                i = loads.index(min(loads))
                options.append((max(0, loads[i] + r - top), i))
            if len(loads) < pack.MAX_WORKERS:
                options.append((pack.WORKER_MS + max(0, r - top), -1))
            extra_ms, where = min(options)
            p_fail = P_FAIL_DATA if c in data_numbers else P_FAIL_CODE
            cpu_ms = 1000 * cost_of[c]
            if class_of[c] not in classes:
                cpu_ms += 1000 * ix.class_cpu.get(class_of[c], 0.0)
            cpu_ms *= CPU_WEIGHT
            # A change that may mutate shared state must run, so the sentinel can see what it leaves.
            if (1 - p_fail) * extra_ms <= cpu_ms and not impact.may_pollute:
                keep.add(c)
                if where == -1:
                    loads.append(r)
                else:
                    loads[where] += r
            else:
                verify_list.append(c)
                classes.add(class_of[c])
        verify_list.sort(key=lambda c: cost_of[c])
        spent = 0.0
        for i, c in enumerate(verify_list):
            spent += cost_of[c]
            if spent > COMMIT_BUDGET_S:
                keep.update(verify_list[i:])
                verify_list = verify_list[:i]
                break
        if verify_list:
            # Verifying has a fixed cost: not worth it to save less than that.
            fixed = HOT_FIXED_S if hot else FRESH_FIXED_S
            mods = {class_of[c].rsplit(".", 1)[0] for c in verify_list}
            if not hot:
                fixed += sum(ix.fresh_import.get(m, 0.05) for m in mods)
            if any(needs_third_party(m) for m in mods):
                fixed += THIRD_PARTY_S / 2  # imported once, shared with the batch's other children
            cpu = fixed + sum(cost_of[c] for c in verify_list)
            cpu += sum(ix.class_cpu.get(k, 0.0) for k in {class_of[c] for c in verify_list})
            all_ms = pack.cost_ms([(c, recorded_ms[c]) for c in keep | set(verify_list)])
            keep_ms = pack.cost_ms([(c, recorded_ms[c]) for c in keep])
            p_fail = P_FAIL_DATA if all(c in data_numbers for c in verify_list) else P_FAIL_CODE
            if (all_ms - keep_ms) * (1 - p_fail) <= CPU_WEIGHT * cpu * 1000 and not impact.may_pollute:
                keep.update(verify_list)
                verify_list = []
        plan.verify = sorted(verify_list)  # tests.json order, like the grader's run
        plan.mode = "hot" if hot else "fresh"

    for plan in plans:
        split(plan)

    # ------------------------------------------------------------ verify

    def use_sentinel(plan: Plan) -> bool:
        return plan.impact.functions and ix.volatile is not None

    fresh_jobs: list[zygote.Job] = []
    hot_plans: list[Plan] = []
    for plan in plans:
        if not plan.verify:
            continue
        ids = [id_of[c] for c in plan.verify]
        allowance = [max(1.0, 3.0 * cost_of[c] + 1.0, 2.0 * recorded_ms[c] / 1000 + 1.0) for c in plan.verify]
        if plan.mode == "hot":
            hot_plans.append(plan)
            continue
        root = TMP / plan.commit
        try:
            verify.build_tree(PRISTINE, root, plan.files)
        except Exception as error:
            log(f"{plan.commit}: cannot build tree ({error!r}); keeping candidates")
            plan.keep.update(plan.verify)
            plan.verify = []
            continue
        fresh_jobs.append(zygote.Job(plan.commit, ids, allowance, "fresh", root,
                                     preimport=plan.impact.preimport, sentinel=use_sentinel(plan)))

    # The pollution sentinel: what the pristine suite changes by itself does not count.
    volatile = ix.volatile if isinstance(ix.volatile, dict) else {s: "full" for s in (ix.volatile or ())}
    zygote.SENTINEL_VOLATILE = {**volatile, **{s: "full" for s in ix.state_unstable}}
    zygote.SENTINEL_KEYS = MAPS / "state_keys.pkl"
    zygote.SENTINEL_UNORDERED = set(ix.state_unordered)
    # Test files that leave sqlglot state changed for every later file: replayed in each child.
    zygote.SETTERS = list(ix.setters)
    order_of: dict[str, int] = {}
    for t in tests:
        order_of.setdefault(t["id"].split("::", 1)[0], len(order_of))
    zygote.SUITE_ORDER = order_of
    jobs_by_key: dict[str, zygote.Job] = {}
    if fresh_jobs:
        # Third-party modules a patch cannot change, imported once here so the
        # children share them (only when a child's test modules need them).
        fresh_mods = {tid.split("::", 1)[0][:-3].replace("/", ".") for j in fresh_jobs for tid in j.ids}
        zygote.warm_third_party(heavy=any(needs_third_party(m) for m in fresh_mods))
        log(f"fresh: verifying {sum(len(j.ids) for j in fresh_jobs)} tests over {len(fresh_jobs)} commits")
        zygote.run_jobs(fresh_jobs, parallel=VERIFY_PARALLEL, deadline=deadline, log=log)
        for job in fresh_jobs:
            jobs_by_key[job.key] = job
    if hot_plans and time.monotonic() < deadline:
        modules = sorted({id_of[c].split("::", 1)[0][:-3].replace("/", ".") for p in hot_plans for c in p.verify})
        warm = zygote.Warm(PRISTINE)
        # With the sentinel, every sqlglot module first: a lazy import inside a test must not
        # hide the state it then changes.
        watch = any(use_sentinel(p) for p in hot_plans)
        warm.load(modules, all_sqlglot=watch)
        log(f"warm: {len(modules)} test modules, {len(warm.functions)} function codes, "
            f"{len(warm.import_keys)} ran at import")
        if watch:
            zygote.SENTINEL_BASE = sentinel.fingerprint()
        hot_jobs = []
        for plan in hot_plans:
            ids = [id_of[c] for c in plan.verify]
            allowance = [max(1.0, 3.0 * cost_of[c] + 1.0, 2.0 * recorded_ms[c] / 1000 + 1.0) for c in plan.verify]
            patches = []
            sources: dict[str, Path] = {}
            packages: set[str] = set()
            try:
                for rel in plan.impact.py_files:
                    mod = module_name(rel)
                    if mod in sys.modules:
                        patches += warm.plan(rel, (PRISTINE / rel).read_bytes(), plan.files[rel])
                    else:
                        src = TMP / "src" / plan.commit / rel
                        src.parent.mkdir(parents=True, exist_ok=True)
                        src.write_bytes(plan.files[rel])
                        sources[mod] = src
                        if rel.endswith("__init__.py"):
                            packages.add(mod)
            except zygote.NotHot as error:
                log(f"{plan.commit}: not hot after all ({error}); keeping candidates")
                plan.keep.update(plan.verify)
                plan.verify = []
                continue
            hot_jobs.append(zygote.Job(plan.commit, ids, allowance, "hot", PRISTINE, patches=patches,
                                       sources=sources, packages=packages, sentinel=use_sentinel(plan)))
        if hot_jobs:
            log(f"hot: verifying {sum(len(j.ids) for j in hot_jobs)} tests over {len(hot_jobs)} commits")
            zygote.run_jobs(hot_jobs, parallel=VERIFY_PARALLEL, deadline=deadline, log=log)
            for job in hot_jobs:
                jobs_by_key[job.key] = job
    else:
        for plan in hot_plans:
            plan.keep.update(plan.verify)
            plan.verify = []

    import shutil

    shutil.rmtree(TMP, ignore_errors=True)
    for plan in plans:
        job = jobs_by_key.get(plan.commit)
        if job is None:
            plan.keep.update(plan.verify)
            continue
        failed = 0
        for c in plan.verify:
            status = job.verdicts.get(id_of[c])
            if status not in ("passed", "skipped"):
                plan.keep.add(c)
                failed += status is not None
        log(
            f"{plan.commit}: {job.mode} ran {len(job.verdicts)}/{len(job.ids)} in {job.child_cpu:.2f} cpu-s, "
            f"{failed} failing; selected {len(plan.keep)}"
        )
        polluted = {s: e for s, e in (job.polluted or {}).items() if s not in plan.impact.state_slots}
        if polluted:
            # The candidates left shared state changed: every later test in the grader's
            # process sees it, so the tests that read that state are kept.
            if ("?", "?", "?") in polluted:
                victims = set(recorded_ms)
            else:
                victims = numbers(pollution_readers(ix, polluted))
            plan.keep |= victims
            log(f"{plan.commit}: candidates left shared state changed {sorted(polluted)[:6]}: "
                f"keeping {len(victims)} readers; selected {len(plan.keep)}")

    for plan in plans:
        workers = pack.pack([(c, recorded_ms[c]) for c in sorted(plan.keep)])
        print(json.dumps({"commit": plan.commit, "workers": workers}, separators=(",", ":")))
    sys.stdout.flush()
    usage_self = resource.getrusage(resource.RUSAGE_SELF)
    usage_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = usage_self.ru_utime + usage_self.ru_stime + usage_kids.ru_utime + usage_kids.ru_stime
    cgroup = ""
    try:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            if line.startswith("usage_usec"):
                cgroup = f", container cgroup {int(line.split()[1]) / 1e6:.2f}s"
    except OSError:
        pass
    log(f"done: cpu {cpu:.2f}s (self {usage_self.ru_utime + usage_self.ru_stime:.2f}s){cgroup}")


if __name__ == "__main__":
    main()
