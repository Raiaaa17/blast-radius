"""Build-time maps of the pristine sqlglot tree, keyed by test id.

Runs every sqlglot test in-process under a tracer and records, per test:

* line coverage of every file under ``sqlglot/`` and ``tests/`` (sys.monitoring
  LINE events, each location disabled after its first hit and re-armed at every
  test boundary, so the cost is one callback per distinct line per test);
* class use: every read of a class attribute of a sqlglot class (data tables
  and methods alike) through recording descriptors, keyed by the class the
  attribute was defined on, the attribute name and the class it was read
  through (``type(self)``), so a class-body edit maps to exactly the tests that
  read the edited table through that class or one of its subclasses;
* the verdict and the real duration of the test in this environment.

All sqlglot modules are imported before tracing starts, so code that runs at
import time (module bodies, class bodies, metaclass work) is never attributed
to whichever test happened to import a module first; class-body and
module-level edits are mapped through class use and name use instead.

Usage: python build_maps.py <sqlglot-root> <out-dir> [--jobs N] [--files a,b]
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inline_pool import install_inline_pool  # noqa: E402


# --------------------------------------------------------------------------
# Worker: runs a shard of test files under the tracer.
# --------------------------------------------------------------------------


def _iter_sqlglot_modules(src: Path) -> list[str]:
    names = []
    for path in sorted((src / "sqlglot").rglob("*.py")):
        rel = path.relative_to(src).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__main__":
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def _collect_classes(modules) -> list[type]:
    """Every class defined in a sqlglot module, nested classes included."""
    seen: dict[int, type] = {}
    stack = []
    for module in modules:
        for value in list(vars(module).values()):
            if isinstance(value, type):
                stack.append(value)
    while stack:
        cls = stack.pop()
        if id(cls) in seen:
            continue
        if not getattr(cls, "__module__", "").startswith("sqlglot"):
            continue
        seen[id(cls)] = cls
        for value in list(vars(cls).values()):
            if isinstance(value, type):
                stack.append(value)
    return list(seen.values())


def class_key(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


class _State:
    """Per-test recording buffers, swapped at every test boundary."""

    lines: set = set()
    attrs: set = set()
    events: set = set()
    keys: set = set()  # (dict id, key) entries read from tracked dicts; (dict id, ALL) whole-dict reads
    enums: set = set()  # classes and modules whose namespace was listed (dir / vars)
    copying: set | None = None  # while a derived table is built: the dicts it copies whole


class _AllKeys:
    """Marks a read of a whole tracked dict (iteration, length, copy, comparison)."""

    def __repr__(self) -> str:
        return "<ALL>"


ALL = _AllKeys()


def _note(d, key) -> None:
    dk = getattr(d, "_dk", None)
    if dk is not None:
        if _State.copying is not None and dk in _State.copying:
            return  # dict(d) reads keys() then every entry: part of the copy
        try:
            _State.keys.add((dk, key))
        except TypeError:  # unhashable key: the lookup fails anyway; count it as a whole read
            _State.keys.add((dk, ALL))


def _note_all(d) -> None:
    dk = getattr(d, "_dk", None)
    if dk is not None:
        if _State.copying is not None:
            _State.copying.add(dk)  # copied into a derived table, whose reads are recorded instead
            return
        _State.keys.add((dk, ALL))


def _install_dispatch_tracker(dict_table: list, derived: dict) -> None:
    """The generator's dispatch table: a per-class copy of TRANSFORMS (plus the *_sql methods
    found with dir), looked up by expression class for every node generated.

    Copying TRANSFORMS would count as reading all of it; instead the table itself is
    tracked (as ("D", generator class)) and linked to the dicts it copied, so an edited
    TRANSFORMS entry maps to the tests that looked that expression type up.
    """
    import sqlglot.generator as generator_module

    real = getattr(generator_module, "_build_dispatch", None)
    if real is None:  # no such table here: TRANSFORMS stays a whole-dict read
        return
    ids: dict[str, int] = {}

    def build_dispatch(cls):
        outer, _State.copying = _State.copying, set()
        try:
            table = real(cls)
            copied = _State.copying
        finally:
            _State.copying = outer
        ck = class_key(cls)
        dk = ids.get(ck)
        if dk is None:
            dk = ids[ck] = len(dict_table)
            dict_table.append([("D", ck)])
        derived.setdefault(dk, set()).update(copied)
        tracked = TrackedDict(table)
        tracked._dk = dk
        return tracked

    generator_module._build_dispatch = build_dispatch


class TrackedDict(dict):
    """A copy of a class- or module-level dict that records which entries each test reads.

    Single-key operations record the key; anything that sees the whole dict
    (iteration, views, length, equality, copies, repr, pickling) records ALL.
    Overriding __iter__ also takes C code (dict(d), {**d}, f(**d)) off the
    fast path that would read the entries without calling a method.
    """

    __slots__ = ("_dk",)

    def __getitem__(self, key):
        _note(self, key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        _note(self, key)
        return dict.get(self, key, default)

    def __contains__(self, key):
        _note(self, key)
        return dict.__contains__(self, key)

    def setdefault(self, key, default=None):
        _note(self, key)
        return dict.setdefault(self, key, default)

    def pop(self, key, *default):
        _note(self, key)
        return dict.pop(self, key, *default)

    def __setitem__(self, key, value):
        _note(self, key)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        _note(self, key)
        dict.__delitem__(self, key)

    def __iter__(self):
        _note_all(self)
        return dict.__iter__(self)

    def keys(self):
        _note_all(self)
        return dict.keys(self)

    def values(self):
        _note_all(self)
        return dict.values(self)

    def items(self):
        _note_all(self)
        return dict.items(self)

    def __len__(self):
        _note_all(self)
        return dict.__len__(self)

    def __eq__(self, other):
        _note_all(self)
        return dict.__eq__(self, other)

    def __ne__(self, other):
        _note_all(self)
        return dict.__ne__(self, other)

    def __reversed__(self):
        _note_all(self)
        return dict.__reversed__(self)

    def copy(self):
        _note_all(self)
        return dict(dict.items(self))

    def __or__(self, other):
        _note_all(self)
        return dict.__or__(self, other)

    def __ror__(self, other):
        _note_all(self)
        return dict.__ror__(self, other)

    def __ior__(self, other):
        _note_all(self)
        return dict.__ior__(self, other)

    def update(self, *args, **kwargs):
        _note_all(self)
        dict.update(self, *args, **kwargs)

    def popitem(self):
        _note_all(self)
        return dict.popitem(self)

    def clear(self):
        _note_all(self)
        dict.clear(self)

    def __repr__(self):
        _note_all(self)
        return dict.__repr__(self)

    def __reduce_ex__(self, protocol):
        _note_all(self)
        return (dict, (dict(dict.items(self)),))

    __hash__ = None  # type: ignore[assignment]


def _install_dict_trackers(modules, attr_wrappers: dict, memo: dict) -> tuple[list, list]:
    """Replace class- and module-level dicts of sqlglot with TrackedDicts.

    memo: id(original) -> tracked copy, shared by every place the same dict
    object is bound, so they stay one object. Returns (dict_table, originals):
    dict_table[dk] lists the slots bound to tracked dict dk. A dict that is
    still reachable from anywhere else after the replacement (a container,
    a default argument, a closure, an instance) is marked leaky by
    check_dict_leaks: its reads through that path would not be recorded.
    """
    import types

    table: list[list] = []
    originals: list = []

    def tracked(value, slot):
        got = memo.get(id(value))
        if got is None:
            got = TrackedDict(value)
            got._dk = len(table)
            table.append([])
            originals.append(value)
            memo[id(value)] = got
        table[got._dk].append(slot)
        return got

    # Class attributes: their recording descriptors hand out the tracked copy.
    for (ck, name), wrapper in attr_wrappers.items():
        value = wrapper.v
        if type(value) is dict:
            wrapper.v = tracked(value, ("C", ck, name))
    # Module globals (process-wide caches stay untouched: the trace empties them).
    for module in modules:
        mod_name = module.__name__
        for name, value in list(vars(module).items()):
            if type(value) is not dict or name.startswith("__") or "cache" in name.lower():
                continue
            if isinstance(value, types.ModuleType):
                continue
            setattr(module, name, tracked(value, ("M", mod_name, name)))
    return table, originals


def check_dict_leaks(originals: list) -> set[int]:
    import gc
    import types

    gc.collect()
    leaky = set()
    for dk in range(len(originals)):
        refs = [
            r for r in gc.get_referrers(originals[dk])
            if r is not originals and not isinstance(r, types.FrameType)
        ]
        if refs:
            leaky.add(dk)
    return leaky


def _install_namespace_recorders() -> None:
    """dir() and vars() calls on sqlglot classes, instances and modules: what lists a namespace."""
    import builtins
    import types

    real_dir, real_vars = builtins.dir, builtins.vars

    def note(obj) -> None:
        try:
            if isinstance(obj, types.ModuleType):
                if obj.__name__.startswith("sqlglot"):
                    _State.enums.add(("M", obj.__name__))
                return
            cls = obj if isinstance(obj, type) else type(obj)
            for c in cls.__mro__:
                if str(real_getattr(c, "__module__", "")).startswith("sqlglot"):
                    _State.enums.add(("C", class_key(c)))
        except Exception:  # noqa: BLE001
            pass

    real_getattr = builtins.getattr

    def note_miss(obj, name) -> None:
        # A lookup that found nothing: a new attribute of that name would change it.
        try:
            if not isinstance(name, str):
                return
            if isinstance(obj, types.ModuleType):
                if obj.__name__.startswith("sqlglot"):
                    _State.enums.add(("missM", obj.__name__, name))
                return
            cls = obj if isinstance(obj, type) else type(obj)
            for c in cls.__mro__:
                if str(real_getattr(c, "__module__", "")).startswith("sqlglot"):
                    _State.enums.add(("miss", class_key(c), name))
        except Exception:  # noqa: BLE001
            pass

    def br_getattr(obj, name, *default):
        if not default:
            try:
                return real_getattr(obj, name)
            except AttributeError:
                note_miss(obj, name)
                raise
        try:
            return real_getattr(obj, name)
        except AttributeError:
            note_miss(obj, name)
            return default[0]

    def br_hasattr(obj, name):
        try:
            real_getattr(obj, name)
        except AttributeError:
            note_miss(obj, name)
            return False
        return True

    def br_dir(*args):
        if not args:
            return sorted(sys._getframe(1).f_locals)
        note(args[0])
        return real_dir(*args)

    def br_vars(*args):
        if not args:
            return dict(sys._getframe(1).f_locals)
        note(args[0])
        return real_vars(*args)

    builtins.dir = br_dir
    builtins.vars = br_vars
    builtins.getattr = br_getattr
    builtins.hasattr = br_hasattr


def _install_attr_recorders(classes: list[type]) -> tuple[list[tuple[str, str]], dict]:
    """Replace class attributes with descriptors recording (attr id, owner)."""
    import enum

    attr_table: list[tuple[str, str]] = []
    state = _State

    class _Value:
        __slots__ = ("v", "k")

        def __init__(self, v, k):
            self.v = v
            self.k = k

        def __get__(self, obj, owner=None):
            state.attrs.add((self.k, type(obj) if obj is not None else owner))
            return self.v

    class _Bound:
        __slots__ = ("v", "k")

        def __init__(self, v, k):
            self.v = v
            self.k = k

        def __get__(self, obj, owner=None):
            state.attrs.add((self.k, type(obj) if obj is not None else owner))
            return self.v.__get__(obj, owner)

    class _Prop:
        __slots__ = ("v", "k", "__doc__")

        def __init__(self, v, k):
            self.v = v
            self.k = k
            self.__doc__ = getattr(v, "__doc__", None)

        def __get__(self, obj, owner=None):
            state.attrs.add((self.k, type(obj) if obj is not None else owner))
            return self.v.__get__(obj, owner)

        def __set__(self, obj, value):
            state.attrs.add((self.k, type(obj)))
            self.v.__set__(obj, value)

        def __delete__(self, obj):
            self.v.__delete__(obj)

    wrapped = 0
    wrappers: dict = {}
    for cls in classes:
        if issubclass(cls, (enum.Enum, tuple, BaseException)):
            continue
        ck = class_key(cls)
        for name, value in list(vars(cls).items()):
            if name.startswith("__") and name.endswith("__"):
                continue
            vtype = type(value)
            k = len(attr_table)
            if isinstance(value, property):
                wrapper = _Prop(value, k)
            elif hasattr(vtype, "__set__") or hasattr(vtype, "__delete__"):
                continue  # slots and other data descriptors: per-instance storage
            elif hasattr(vtype, "__get__") and not isinstance(value, type):
                wrapper = _Bound(value, k)
            else:
                wrapper = _Value(value, k)
            try:
                setattr(cls, name, wrapper)
            except (AttributeError, TypeError):
                continue
            attr_table.append((ck, name))
            if isinstance(wrapper, _Value):
                wrappers[(ck, name)] = wrapper
            wrapped += 1
    print(f"[maps] wrapped {wrapped} class attributes in {len(classes)} classes", file=sys.stderr)
    _unwrap_for_mock((_Value, _Bound, _Prop))
    return attr_table, wrappers


_MOCK_RESTORE: list = []


def _unwrap_for_mock(wrapper_types: tuple) -> None:
    """mock.patch(autospec=True) on a wrapped class attribute must see the real attribute.

    patch reads the original from the class __dict__ (our recording wrapper)
    and would autospec the wrapper; hand it the wrapped value instead and put
    the wrapper back after the test (restore_mock_wrappers), since patch
    restores the value it was given.
    """
    from unittest import mock

    real = mock._patch.get_original

    def get_original(self):
        original, local = real(self)
        if isinstance(original, wrapper_types):
            _MOCK_RESTORE.append((self.target, self.attribute, original))
            original = original.v
        return original, local

    mock._patch.get_original = get_original


def restore_mock_wrappers() -> None:
    while _MOCK_RESTORE:
        target, attribute, wrapper = _MOCK_RESTORE.pop()
        try:
            if vars(target).get(attribute) is not wrapper:
                setattr(target, attribute, wrapper)
        except (AttributeError, TypeError):
            pass


def worker(src: Path, files: list[str], out_path: Path, lines_only: bool = False, only: set | None = None,
           setter_files: list[str] | None = None, order: dict[str, int] | None = None) -> None:
    """lines_only: no class-attribute, dict or namespace recorders (a second pass for the
    tests that do not pass under them); only: run just these test ids. setter_files: test
    files that leave sqlglot state changed (setters.py), replayed untraced before the first
    file that follows them in suite order (`order`), as the grader's single process has them."""
    import importlib
    import unittest

    os.chdir(src)
    sys.path.insert(0, str(src))
    os.environ.setdefault("SKIP_INTEGRATION", "0")

    install_inline_pool()
    started = time.perf_counter()

    # Import-time lines: everything the import of sqlglot executes (module and
    # class bodies, metaclasses, factories called from class bodies).
    import_lines: set = set()
    src_str = str(src) + os.sep
    early_tool = sys.monitoring.PROFILER_ID
    sys.monitoring.use_tool_id(early_tool, "br-maps-import")

    def on_import_line(code, line):
        # Which code objects run at import time (function granularity).
        if code.co_filename.startswith(src_str):
            import_lines.add((os.path.relpath(code.co_filename, src), code.co_qualname, code.co_firstlineno))
        return sys.monitoring.DISABLE

    # LINE events never fire for code on a function's first line (one-line defs and
    # lambdas: their only line is the RESUME's): PY_START covers those calls.
    sys.monitoring.register_callback(early_tool, sys.monitoring.events.LINE, on_import_line)
    sys.monitoring.register_callback(early_tool, sys.monitoring.events.PY_START, on_import_line)
    sys.monitoring.set_events(early_tool, sys.monitoring.events.LINE | sys.monitoring.events.PY_START)
    modules = []
    for name in _iter_sqlglot_modules(src):
        try:
            modules.append(importlib.import_module(name))
        except Exception as error:  # pragma: no cover - reported, not fatal
            print(f"[maps] import {name} failed: {error!r}", file=sys.stderr)
    sys.monitoring.set_events(early_tool, 0)
    sys.monitoring.register_callback(early_tool, sys.monitoring.events.LINE, None)
    sys.monitoring.register_callback(early_tool, sys.monitoring.events.PY_START, None)
    sys.monitoring.free_tool_id(early_tool)

    classes = _collect_classes(modules)
    dict_table: list = []
    leaky: set = set()
    derived: dict[int, set[int]] = {}  # derived table -> the tracked dicts it copies
    if lines_only:
        attr_table: list = []
    else:
        attr_table, wrappers = _install_attr_recorders(classes)
        dict_table, originals = _install_dict_trackers(modules, wrappers, {})
        leaky = check_dict_leaks(originals)
        del originals
        _install_namespace_recorders()
        _install_dispatch_tracker(dict_table, derived)
        print(f"[maps] {len(dict_table)} tracked dicts, {len(leaky)} also reachable elsewhere", file=sys.stderr)
    from cachereset import find_caches, reset as reset_caches

    caches = find_caches()
    print(f"[maps] {len(caches)} process-wide caches reset before every test", file=sys.stderr)

    # File ids for line coverage: every .py under sqlglot/ and tests/.
    file_ids: dict[str, int] = {}
    file_names: list[str] = []
    for base in ("sqlglot", "tests"):
        for path in sorted((src / base).rglob("*.py")):
            file_ids[str(path)] = len(file_names)
            file_names.append(str(path.relative_to(src)))

    src_prefix = str(src) + os.sep
    watched = {"open", "os.listdir", "os.scandir", "glob.glob", "subprocess.Popen", "import", "os.system", "os.exec", "os.posix_spawn"}

    def audit(event, args):
        if event not in watched:
            return
        if event == "open":
            path = args[0]
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            if isinstance(path, str):
                full = os.path.abspath(path)
                if full.startswith(src_prefix):
                    _State.events.add(("open", os.path.relpath(full, src)))
        elif event in ("os.listdir", "os.scandir", "glob.glob"):
            path = args[0] if args else "."
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            if isinstance(path, str):
                full = os.path.abspath(path)
                if full.startswith(src_prefix) or full == str(src):
                    _State.events.add(("list", os.path.relpath(full, src)))
        elif event == "import":
            name = str(args[0])
            if name.startswith(("sqlglot", "tests")):
                _State.events.add(("import", name))
        else:
            _State.events.add(("spawn", event))

    sys.addaudithook(audit)

    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, "br-maps")
    disable = mon.DISABLE
    state = _State
    fid_get = file_ids.get

    def on_line(code, line):
        fid = fid_get(code.co_filename)
        if fid is not None:
            state.lines.add((fid << 20) | line)
        return disable

    def on_start(code, offset):
        # A call: its first line, which LINE events skip (a one-line def or lambda).
        fid = fid_get(code.co_filename)
        if fid is not None:
            state.lines.add((fid << 20) | code.co_firstlineno)
        return disable

    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.register_callback(tool, mon.events.PY_START, on_start)
    traced_events = mon.events.LINE | mon.events.PY_START

    # Test modules are imported under the tracer: what their import runs is
    # shared by every test of the module.
    mon.set_events(tool, traced_events)
    test_modules = {}
    module_buckets: dict[str, tuple[set, set, set]] = {}
    for rel in files:
        modname = rel[:-3].replace("/", ".")
        state.lines, state.attrs, state.events, state.keys, state.enums = set(), set(), set(), set(), set()
        mon.restart_events()
        try:
            test_modules[rel] = importlib.import_module(modname)
        except Exception as error:
            print(f"[maps] import {modname} failed: {error!r}", file=sys.stderr)
            test_modules[rel] = None
        module_buckets[rel] = (state.lines, state.attrs, state.events, state.keys, state.enums)
    state.lines, state.attrs, state.events, state.keys, state.enums = set(), set(), set(), set(), set()
    mon.set_events(tool, 0)
    import_s = time.perf_counter() - started

    results: dict[str, dict] = {}
    gaps: list[tuple[str | None, str | None, set, set]] = []
    from setters import clears_before, reclear

    setter_list = list(setter_files or [])
    setter_of = {s[0]: s for s in setter_list}
    order = order or {}
    applied: set[str] = set()
    clears_of: dict[str, list] = {}

    class Result(unittest.TestResult):
        current: str | None = None
        last: str | None = None

        def _id(self, test) -> str:
            cls = type(test)
            module = sys.modules[cls.__module__]
            rel = os.path.relpath(module.__file__, src)
            return f"{rel}::{cls.__name__}::{test._testMethodName}"

        def startTest(self, test):
            super().startTest(test)
            tid = self._id(test)
            # Registries a preceding setter empties are emptied again before each later test
            # (setters.clears_before); it lands in the gap, not in this test's reads.
            rel_of_test = tid.split("::", 1)[0]
            clears = clears_of.get(rel_of_test)
            if clears is None:
                clears = clears_of[rel_of_test] = clears_before(setter_list, order, rel_of_test)
            if clears:
                saved = state.keys
                state.keys = set()
                reclear(clears)
                state.keys = saved
            # Whatever ran since the last test (class/module fixtures) is a gap.
            gaps.append((self.last, tid, state.lines, state.attrs, state.events, state.keys, state.enums))
            reset_caches(caches)
            state.lines, state.attrs, state.events, state.keys, state.enums = set(), set(), set(), set(), set()
            mon.restart_events()
            self.current = tid
            results[tid] = {"status": "passed", "t0": time.perf_counter(), "c0": time.process_time()}

        def stopTest(self, test):
            tid = self.current
            entry = results[tid]
            entry["seconds"] = time.perf_counter() - entry.pop("t0")
            entry["cpu"] = time.process_time() - entry.pop("c0")
            entry["lines"] = state.lines
            entry["attrs"] = state.attrs
            entry["events"] = state.events
            entry["keys"] = state.keys
            entry["enums"] = state.enums
            restore_mock_wrappers()
            state.lines, state.attrs, state.events, state.keys, state.enums = set(), set(), set(), set(), set()
            mon.restart_events()
            self.last, self.current = tid, None
            super().stopTest(test)

        def _mark(self, test, status):
            tid = self.current
            if tid is None or tid not in results:
                return
            if status != "passed" and results[tid]["status"] in ("passed", "skipped"):
                results[tid]["status"] = status

        def addError(self, test, err):
            super().addError(test, err)
            if isinstance(test, unittest.TestCase):
                self._mark(test, "error")
            else:  # class or module fixture failed
                gaps.append((self.last, None, set(), set(), set(), set(), set()))
                print(f"[maps] fixture error {test}: {err[1]!r}", file=sys.stderr)

        def addFailure(self, test, err):
            super().addFailure(test, err)
            self._mark(test, "failed")

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err is not None:
                self._mark(test, "failed")

        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self._mark(test, "skipped")

        def addUnexpectedSuccess(self, test):
            super().addUnexpectedSuccess(test)
            self._mark(test, "failed")

    loader = unittest.TestLoader()
    mon.set_events(tool, traced_events)
    run_started = time.perf_counter()
    from setters import apply as apply_setter

    for rel, module in test_modules.items():
        due = [f for f in setter_of if f not in applied and order.get(f, 1 << 30) < order.get(rel, 1 << 30)]
        if due:
            mon.set_events(tool, 0)
            keep = (state.lines, state.attrs, state.events, state.keys, state.enums)
            state.lines, state.attrs, state.events, state.keys, state.enums = set(), set(), set(), set(), set()
            for f in due:
                apply_setter(setter_of[f])
                applied.add(f)
            state.lines, state.attrs, state.events, state.keys, state.enums = keep
            restore_mock_wrappers()
            mon.set_events(tool, traced_events)
            mon.restart_events()
        if rel in setter_of:
            applied.add(rel)
        if module is None:
            continue
        suite = unittest.TestSuite()
        for name, value in vars(module).items():
            if (
                isinstance(value, type)
                and issubclass(value, unittest.TestCase)
                and value.__module__ == module.__name__
            ):
                for method in loader.getTestCaseNames(value):
                    if only is not None and f"{rel}::{value.__name__}::{method}" not in only:
                        continue
                    suite.addTest(value(method))
        result = Result()
        suite.run(result)
        # Module teardown runs after the last stopTest: close the gap.
        gaps.append((result.last, None, state.lines, state.attrs, state.events, state.keys, state.enums))
        state.lines, state.attrs, state.events, state.keys, state.enums = set(), set(), set(), set(), set()
        mon.restart_events()
    mon.set_events(tool, 0)
    run_s = time.perf_counter() - run_started

    # Owners are classes: turn them into stable keys.
    owner_mro: dict[str, list[str]] = {}

    def norm_attrs(attrs):
        out = set()
        for k, owner in attrs:
            try:
                key = class_key(owner)
            except Exception:
                key = "?"
            if key not in owner_mro and key != "?":
                try:
                    owner_mro[key] = [class_key(c) for c in owner.__mro__]
                except Exception:
                    owner_mro[key] = [key]
            out.add((attr_table[k], key))
        return out

    # Dict entries read: keys as statediff encodes them (ALL as b"*"), dicts by id.
    sys.path.insert(0, str(HERE))
    from statediff import Digester

    enc = Digester().fp
    key_memo: dict = {}

    def norm_keys(keys):
        out = set()
        for dk, key in keys:
            if key is ALL:
                out.add((dk, b"*"))
                continue
            try:
                code = key_memo.get((type(key), key))
            except TypeError:
                code = None
            if code is None:
                try:
                    code = enc(key)
                except Exception:  # noqa: BLE001
                    code = b"*"
                try:
                    key_memo[(type(key), key)] = code
                except TypeError:
                    pass
            out.add((dk, code))
        return out

    for entry in results.values():
        entry["attrs"] = norm_attrs(entry["attrs"])
        entry["keys"] = norm_keys(entry["keys"])
    gaps = [(a, b, lines, norm_attrs(attrs), events, norm_keys(keys), enums)
            for a, b, lines, attrs, events, keys, enums in gaps]
    module_buckets = {
        rel: (lines, norm_attrs(attrs), events, norm_keys(keys), enums)
        for rel, (lines, attrs, events, keys, enums) in module_buckets.items()
    }

    with open(out_path, "wb") as handle:
        pickle.dump(
            {
                "files": file_names,
                "attr_table": attr_table,
                "results": results,
                "gaps": gaps,
                "module_buckets": module_buckets,
                "import_lines": import_lines,
                "owner_mro": owner_mro,
                "dict_table": dict_table,
                "leaky": leaky,
                "derived": derived,
                "lines_only": lines_only,
                "import_s": import_s,
                "run_s": run_s,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(
        f"[maps] shard of {len(files)} files: {len(results)} tests, import {import_s:.1f}s, run {run_s:.1f}s",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------
# Driver: shards the test files over processes and merges the shards.
# --------------------------------------------------------------------------


def _test_files(src: Path) -> list[str]:
    files = []
    for path in sorted((src / "tests").rglob("test_*.py")):
        files.append(str(path.relative_to(src)))
    return files


def _shard(files: list[str], weights: dict[str, float], jobs: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(jobs)]
    loads = [0.0] * jobs
    for rel in sorted(files, key=lambda f: -weights.get(f, 1.0)):
        i = loads.index(min(loads))
        shards[i].append(rel)
        loads[i] += weights.get(rel, 1.0)
    return [s for s in shards if s]


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--worker":
        only = None
        if "--only" in args:
            with open(args[args.index("--only") + 1]) as handle:
                only = set(json.load(handle))
        setter_files, order = None, None
        if "--setters" in args:
            with open(args[args.index("--setters") + 1]) as handle:
                got = json.load(handle)
            setter_files, order = got["setters"], got["order"]
        worker(Path(args[1]), args[3].split(","), Path(args[2]), lines_only="--lines-only" in args, only=only,
               setter_files=setter_files, order=order)
        return
    src, out = Path(args[0]), Path(args[1])
    jobs = 4
    only = None
    if "--jobs" in args:
        jobs = int(args[args.index("--jobs") + 1])
    if "--files" in args:
        only = args[args.index("--files") + 1].split(",")
    out.mkdir(parents=True, exist_ok=True)
    files = only or _test_files(src)
    weights: dict[str, float] = {}
    sample = HERE / "sample_tests.json"
    if sample.exists():
        for test in json.loads(sample.read_text())["tests"]:
            weights[test["file"]] = weights.get(test["file"], 0.0) + test["seconds"] + 0.01
    # Files that leave sqlglot state changed for every later file (setters.py).
    import setters

    order = setters.suite_order(sample) if sample.exists() else {f: i for i, f in enumerate(sorted(files))}
    files = sorted(files, key=lambda f: (order.get(f, 1 << 30), f))
    setter_files = setters.detect(src, files, log=lambda m: print(m, file=sys.stderr))
    setters_file = out / "setters.json"
    setters_file.write_text(json.dumps({"setters": setter_files, "order": order}))
    print(f"[maps] state setters: {setter_files}", file=sys.stderr)
    # The two giants get their own process each so the shards finish together.
    shards = [sorted(shard, key=lambda f: (order.get(f, 1 << 30), f)) for shard in _shard(files, weights, jobs)]
    started = time.perf_counter()
    procs = []
    for i, shard in enumerate(shards):
        shard_out = out / f"shard{i}.pkl"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(src), str(shard_out), ",".join(shard),
               "--setters", str(setters_file)]
        procs.append((subprocess.Popen(cmd), shard_out))
    failed = False
    for proc, _ in procs:
        if proc.wait() != 0:
            failed = True
    if failed:
        raise SystemExit("a coverage shard failed")
    print(f"[maps] all shards done in {time.perf_counter() - started:.1f}s", file=sys.stderr)
    # Tests that do not pass under the recorders (mock.patch with autospec on a
    # wrapped attribute, ...): a second pass with line coverage only.
    bad: list[str] = []
    for _, shard_out in procs:
        with open(shard_out, "rb") as handle:
            results = pickle.load(handle)["results"]
        bad += [tid for tid, entry in results.items() if entry["status"] != "passed"]
    second = None
    if bad:
        only_file = out / "second_only.json"
        only_file.write_text(json.dumps(bad))
        second = out / "shard_lines.pkl"
        bad_files = sorted({tid.split("::", 1)[0] for tid in bad})
        bad_files = sorted(bad_files, key=lambda f: (order.get(f, 1 << 30), f))
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(src), str(second), ",".join(bad_files),
               "--lines-only", "--only", str(only_file), "--setters", str(setters_file)]
        subprocess.run(cmd, check=True)
        only_file.unlink()
    from merge_maps import merge

    merge(src, [p for _, p in procs], out, second=second, setter_files=setter_files)
    setters_file.unlink()
    for _, shard_out in procs:
        shard_out.unlink()
    if second is not None:
        second.unlink()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
