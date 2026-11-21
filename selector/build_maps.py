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
            wrapped += 1
    print(f"[maps] wrapped {wrapped} class attributes in {len(classes)} classes", file=sys.stderr)
    return attr_table, {}


def worker(src: Path, files: list[str], out_path: Path) -> None:
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

    sys.monitoring.register_callback(early_tool, sys.monitoring.events.LINE, on_import_line)
    sys.monitoring.set_events(early_tool, sys.monitoring.events.LINE)
    modules = []
    for name in _iter_sqlglot_modules(src):
        try:
            modules.append(importlib.import_module(name))
        except Exception as error:  # pragma: no cover - reported, not fatal
            print(f"[maps] import {name} failed: {error!r}", file=sys.stderr)
    sys.monitoring.set_events(early_tool, 0)
    sys.monitoring.register_callback(early_tool, sys.monitoring.events.LINE, None)
    sys.monitoring.free_tool_id(early_tool)

    classes = _collect_classes(modules)
    attr_table, _ = _install_attr_recorders(classes)
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

    mon.register_callback(tool, mon.events.LINE, on_line)

    # Test modules are imported under the tracer: what their import runs is
    # shared by every test of the module.
    mon.set_events(tool, mon.events.LINE)
    test_modules = {}
    module_buckets: dict[str, tuple[set, set, set]] = {}
    for rel in files:
        modname = rel[:-3].replace("/", ".")
        state.lines, state.attrs, state.events = set(), set(), set()
        mon.restart_events()
        try:
            test_modules[rel] = importlib.import_module(modname)
        except Exception as error:
            print(f"[maps] import {modname} failed: {error!r}", file=sys.stderr)
            test_modules[rel] = None
        module_buckets[rel] = (state.lines, state.attrs, state.events)
    state.lines, state.attrs, state.events = set(), set(), set()
    mon.set_events(tool, 0)
    import_s = time.perf_counter() - started

    results: dict[str, dict] = {}
    gaps: list[tuple[str | None, str | None, set, set]] = []

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
            # Whatever ran since the last test (class/module fixtures) is a gap.
            gaps.append((self.last, tid, state.lines, state.attrs, state.events))
            reset_caches(caches)
            state.lines, state.attrs, state.events = set(), set(), set()
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
            state.lines, state.attrs, state.events = set(), set(), set()
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
                gaps.append((self.last, None, set(), set(), set()))
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
    mon.set_events(tool, mon.events.LINE)
    run_started = time.perf_counter()
    for rel, module in test_modules.items():
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
                    suite.addTest(value(method))
        result = Result()
        suite.run(result)
        # Module teardown runs after the last stopTest: close the gap.
        gaps.append((result.last, None, state.lines, state.attrs, state.events))
        state.lines, state.attrs, state.events = set(), set(), set()
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

    for entry in results.values():
        entry["attrs"] = norm_attrs(entry["attrs"])
    gaps = [(a, b, lines, norm_attrs(attrs), events) for a, b, lines, attrs, events in gaps]
    module_buckets = {
        rel: (lines, norm_attrs(attrs), events) for rel, (lines, attrs, events) in module_buckets.items()
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
        worker(Path(args[1]), args[3].split(","), Path(args[2]))
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
    # The two giants get their own process each so the shards finish together.
    shards = _shard(files, weights, jobs)
    started = time.perf_counter()
    procs = []
    for i, shard in enumerate(shards):
        shard_out = out / f"shard{i}.pkl"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(src), str(shard_out), ",".join(shard)]
        procs.append((subprocess.Popen(cmd), shard_out))
    failed = False
    for proc, _ in procs:
        if proc.wait() != 0:
            failed = True
    if failed:
        raise SystemExit("a coverage shard failed")
    print(f"[maps] all shards done in {time.perf_counter() - started:.1f}s", file=sys.stderr)
    from merge_maps import merge

    merge(src, [p for _, p in procs], out)
    for _, shard_out in procs:
        shard_out.unlink()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
