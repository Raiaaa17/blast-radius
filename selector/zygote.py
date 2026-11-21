"""Warm verification: run candidate tests against a patched sqlglot cheaply.

Two ways to get a process that behaves like the patched tree:

* "hot": every change is a function body (no class body, module level,
  default argument or data file changed) and none of the changed functions
  ran while sqlglot and the test modules were imported. The selector process
  imports the pristine sqlglot and the test modules once; each commit gets a
  forked child that swaps the changed functions' code objects in place
  (``fn.__code__ = new``) and runs the tests. Changed modules the warm process
  never imported are loaded from the patched source when a test imports them.
* "fresh": anything else. The child is forked from a process holding only
  third-party modules and imports sqlglot and the tests from a symlink farm of
  the patched tree (runner_child.run), exactly like an interpreter started on
  the patched checkout.

Both kinds of child report "S i" / "E i status cpu" / "C class cpu" lines on a
pipe; the parent multiplexes up to ``parallel`` children with select() and
kills a child whose test overruns its allowance (that test is "timeout" and
the rest continue in a new child).
"""

from __future__ import annotations

import gc
import importlib
import importlib.abc
import importlib.util
import os
import select
import signal
import sys
import time
import types
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
CO_OPTIMIZED = 0x0001
BETWEEN_TESTS_S = 20.0  # module import / class fixture allowance between tests
MAX_STOPS = 2  # timeouts/crashes after which a commit's remaining tests stay unverified


def code_key(code: types.CodeType) -> tuple[str, str, int]:
    return (code.co_filename, code.co_qualname, code.co_firstlineno)


# --------------------------------------------------------------------------
# Pairing old and new code objects of one file
# --------------------------------------------------------------------------


class NotHot(Exception):
    pass


def _const_eq(a, b) -> bool:
    import math

    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (a == b and math.copysign(1, a) == math.copysign(1, b)) or (a != a and b != b)
    if isinstance(a, complex):
        return _const_eq(a.real, b.real) and _const_eq(a.imag, b.imag)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_const_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, frozenset):
        return a == b and sorted(map(repr, a)) == sorted(map(repr, b))
    return a == b


def own_equal(a: types.CodeType, b: types.CodeType) -> bool:
    """Equal bytecode, names and non-code constants (line tables ignored)."""
    if not (
        a.co_code == b.co_code
        and a.co_names == b.co_names
        and a.co_varnames == b.co_varnames
        and a.co_freevars == b.co_freevars
        and a.co_cellvars == b.co_cellvars
        and a.co_flags == b.co_flags
        and a.co_argcount == b.co_argcount
        and a.co_posonlyargcount == b.co_posonlyargcount
        and a.co_kwonlyargcount == b.co_kwonlyargcount
        and a.co_name == b.co_name
        and a.co_qualname == b.co_qualname
        and a.co_exceptiontable == b.co_exceptiontable
        and a.co_stacksize == b.co_stacksize
        and len(a.co_consts) == len(b.co_consts)
    ):
        return False
    is_class = not (a.co_flags & CO_OPTIMIZED)
    for x, y in zip(a.co_consts, b.co_consts):
        if isinstance(x, types.CodeType) or isinstance(y, types.CodeType):
            if not (isinstance(x, types.CodeType) and isinstance(y, types.CodeType)):
                return False
            continue
        if not _const_eq(x, y):
            # Python 3.13 stores a class's first line number in its body.
            if (
                is_class
                and isinstance(x, int)
                and isinstance(y, int)
                and x == a.co_firstlineno
                and y == b.co_firstlineno
                and "__firstlineno__" in a.co_names
            ):
                continue
            return False
    return True


def _walk_pairs(old: types.CodeType, new: types.CodeType, out: list, top: bool) -> bool:
    """Collect (old, new, own_changed) for every function whose code differs.

    Raises NotHot when a class or module body changed (it ran at import) or the
    nesting structure differs. Returns whether anything below differed.
    """
    if not own_equal(old, new):
        if top or not (old.co_flags & CO_OPTIMIZED):
            raise NotHot(f"body of {old.co_qualname or '<module>'} changed")
        if old.co_freevars != new.co_freevars:
            raise NotHot(f"free variables of {old.co_qualname} changed")
        # Nested code is replaced wholesale with the function; still pair it
        # so closures that already exist get the new code too.
        _pair_nested(old, new, out, force=True)
        out.append((old, new, True))
        return True
    changed = _pair_nested(old, new, out, force=False)
    if changed and (old.co_flags & CO_OPTIMIZED) and not top:
        out.append((old, new, False))  # new nested code for closures made from now on
    return changed


def _pair_nested(old: types.CodeType, new: types.CodeType, out: list, force: bool) -> bool:
    olds = [c for c in old.co_consts if isinstance(c, types.CodeType)]
    news = [c for c in new.co_consts if isinstance(c, types.CodeType)]
    if len(olds) != len(news):
        if force:
            # The function's own code changed and added/removed a nested
            # function: closures made from now on use the new code; existing
            # ones (made at import) cannot be paired.
            for a in olds:
                if a.co_flags & CO_OPTIMIZED:
                    out.append((a, None, True))
            return True
        raise NotHot(f"nested code of {old.co_qualname or '<module>'} differs")
    changed = False
    for a, b in zip(olds, news):
        if a.co_qualname != b.co_qualname:
            if force:
                out.append((a, None, True))
                changed = True
                continue
            raise NotHot(f"nested code renamed in {old.co_qualname}")
        changed |= _walk_pairs(a, b, out, top=False)
    return changed


def hot_pairs(old_src: bytes, new_src: bytes, filename: str) -> list[tuple]:
    """(old_code, new_code_or_None, own_changed) for a function-body-only change."""
    try:
        old = compile(old_src, filename, "exec", dont_inherit=True)
        new = compile(new_src, filename, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        raise NotHot(f"compile: {error}") from None
    out: list = []
    _walk_pairs(old, new, out, top=True)
    return out


# --------------------------------------------------------------------------
# The warm process
# --------------------------------------------------------------------------


class ImportTrace:
    """Code objects executed while importing (sys.monitoring PY_START)."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.keys: set[tuple[str, str, int]] = set()
        self.tool = None

    def __enter__(self):
        mon = sys.monitoring
        for tool in (mon.PROFILER_ID, 3, 4, mon.OPTIMIZER_ID):
            try:
                mon.use_tool_id(tool, "br-import-trace")
                self.tool = tool
                break
            except ValueError:
                continue
        if self.tool is None:
            raise RuntimeError("no free sys.monitoring tool id")
        prefix, keys, disable = self.prefix, self.keys, mon.DISABLE

        def on_start(code, offset):
            if code.co_filename.startswith(prefix):
                keys.add((code.co_filename, code.co_qualname, code.co_firstlineno))
            return disable

        mon.register_callback(self.tool, mon.events.PY_START, on_start)
        mon.set_events(self.tool, mon.events.PY_START)
        return self

    def __exit__(self, *exc):
        mon = sys.monitoring
        mon.set_events(self.tool, 0)
        mon.register_callback(self.tool, mon.events.PY_START, None)
        mon.free_tool_id(self.tool)
        return False


def warm_third_party() -> None:
    """Import the third-party modules the tests use, before any fork.

    BLAS thread pools are disabled first so the process stays single-threaded
    and safe to fork.
    """
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    for name in (
        "unittest",
        "unittest.mock",
        "concurrent.futures",
        "concurrent.futures.process",
        "multiprocessing",
        "json",
        "decimal",
        "datetime",
        "csv",
        "gzip",
        "numpy",
        "pandas",
        "pandas.testing",
        "duckdb",
    ):
        try:
            importlib.import_module(name)
        except Exception:
            pass


class Warm:
    """Pristine sqlglot and test modules imported in this process."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.prefix = str(root) + os.sep
        self.import_keys: set[tuple[str, str, int]] = set()
        self.functions: dict[tuple[str, str, int], list[types.FunctionType]] = {}
        self.failed_imports: set[str] = set()

    def load(self, test_modules: list[str]) -> None:
        sys.path.insert(0, str(self.root))
        os.chdir(self.root)
        sys.path.insert(0, str(HERE))
        from inline_pool import install_inline_pool

        install_inline_pool()
        with ImportTrace(self.prefix) as trace:
            importlib.import_module("sqlglot")
            for name in test_modules:
                try:
                    importlib.import_module(name)
                except BaseException:  # noqa: BLE001 - such tests are run fresh
                    self.failed_imports.add(name)
        self.import_keys = trace.keys
        prefix = self.prefix
        index: dict[tuple[str, str, int], list[types.FunctionType]] = {}
        for obj in gc.get_objects():
            if type(obj) is types.FunctionType:
                code = obj.__code__
                if code.co_filename.startswith(prefix):
                    index.setdefault((code.co_filename, code.co_qualname, code.co_firstlineno), []).append(obj)
        self.functions = index
        gc.collect()
        gc.freeze()

    def module_loaded(self, rel: str) -> bool:
        return module_name(rel) in sys.modules

    def plan(self, rel: str, old_src: bytes, new_src: bytes) -> list[tuple]:
        """Patches [(function objects, new code)] for one changed file, or NotHot."""
        filename = str(self.root / rel)
        pairs = hot_pairs(old_src, new_src, filename)
        patches = []
        covered_by_parent = set()
        for old, new, own in pairs:
            key = code_key(old)
            if own and key in self.import_keys:
                raise NotHot(f"{old.co_qualname} ran at import")
            funcs = self.functions.get(key, [])
            for fn in funcs:
                if not own_equal(fn.__code__, old):
                    raise NotHot(f"live code of {old.co_qualname} differs from the pristine source")
            if new is None:
                if funcs:
                    raise NotHot(f"{old.co_qualname} exists and has no counterpart")
                continue
            if funcs:
                patches.append((funcs, new))
                covered_by_parent.add(key)
        # Every changed function must be reachable: live objects of its own, or
        # an enclosing function whose new code will create it.
        for old, new, own in pairs:
            if not own:
                continue
            key = code_key(old)
            if key in covered_by_parent:
                continue
            if not _has_patched_ancestor(old.co_qualname, filename, pairs, covered_by_parent):
                raise NotHot(f"no live function for {old.co_qualname}")
        return patches


def _has_patched_ancestor(qualname: str, filename: str, pairs, covered) -> bool:
    for old, new, own in pairs:
        if new is None or code_key(old) not in covered:
            continue
        q = old.co_qualname
        if qualname.startswith(q + ".<locals>.") and old.co_filename == filename:
            return True
    return False


def module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class PatchedSourceFinder(importlib.abc.MetaPathFinder):
    """Loads changed modules that the warm process never imported from their new source."""

    def __init__(self, sources: dict[str, Path], packages: set[str]) -> None:
        self.sources = sources
        self.packages = packages

    def find_spec(self, fullname, path=None, target=None):
        src = self.sources.get(fullname)
        if src is None:
            return None
        if fullname in self.packages:
            return importlib.util.spec_from_file_location(
                fullname, src, submodule_search_locations=[str(src.parent)]
            )
        return importlib.util.spec_from_file_location(fullname, src)


# --------------------------------------------------------------------------
# Jobs and children
# --------------------------------------------------------------------------


class Job:
    def __init__(self, key: str, ids: list[str], allowance: list[float], mode: str, root: Path,
                 patches=None, sources=None, packages=None):
        self.key = key
        self.ids = ids
        self.allowance = allowance
        self.mode = mode  # "hot" | "fresh"
        self.root = root  # symlink farm of the patched tree (fresh) or cwd (hot)
        self.patches = patches or []
        self.sources = sources or {}
        self.packages = packages or set()
        self.verdicts: dict[str, str] = {}
        self.cpu: dict[str, float] = {}
        self.class_cpu: dict[str, float] = {}
        self.gave_up = False
        self.stops = 0
        self.child_cpu = 0.0


def _child_main(job: Job, ids: list[str], w: int) -> None:
    try:  # if memory runs out, the kernel kills a child, not the selector
        with open("/proc/self/oom_score_adj", "w") as handle:
            handle.write("1000")
    except OSError:
        pass
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    out = os.fdopen(w, "w", buffering=1)

    def emit(line: str) -> None:
        out.write(line + "\n")
        out.flush()

    os.environ["SKIP_INTEGRATION"] = "0"
    sys.path.insert(0, str(HERE))
    from runner_child import run

    if job.mode == "hot":
        for funcs, new in job.patches:
            for fn in funcs:
                fn.__code__ = new
        if job.sources:
            sys.meta_path.insert(0, PatchedSourceFinder(job.sources, job.packages))
            importlib.invalidate_caches()
        os.chdir(job.root)
        run(str(job.root), ids, emit, fresh=False)
    else:
        run(str(job.root), ids, emit, fresh=True)


def _fork(job: Job, ids: list[str]) -> tuple[int, int]:
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            _child_main(job, ids, w)
        except BaseException:  # noqa: BLE001
            pass
        finally:
            os._exit(0)
    os.close(w)
    return pid, r


def run_jobs(jobs: list[Job], parallel: int, deadline: float, log=None) -> None:
    queue = deque(jobs)
    active: dict[int, dict] = {}

    def finish(fd: int, timed_out: bool) -> None:
        st = active.pop(fd)
        try:
            os.close(fd)
        except OSError:
            pass
        if timed_out:
            try:
                os.kill(st["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            _, status, usage = os.wait4(st["pid"], 0)
            st["job"].child_cpu += usage.ru_utime + usage.ru_stime
        except ChildProcessError:
            pass
        job: Job = st["job"]
        if st["finished"]:
            return
        if st["started"] is not None:
            tid = job.ids[st["batch"][st["started"]]]
            job.verdicts[tid] = "timeout" if timed_out else "crash"
            job.stops += 1
            if log:
                log(f"{job.key}: child stopped on {tid}: {job.verdicts[tid]}")
            if job.stops >= MAX_STOPS:
                # A patch that hangs or kills test after test: keep the rest unverified.
                job.gave_up = True
                if log:
                    log(f"{job.key}: {job.stops} tests stopped; rest unverified")
            else:
                queue.appendleft(job)  # continue with the rest in a fresh child
        else:
            job.gave_up = True
            if log:
                log(f"{job.key}: child stopped outside a test; rest unverified")

    while queue or active:
        now = time.monotonic()
        if now >= deadline:
            for fd in list(active):
                finish(fd, True)
            for job in queue:
                job.gave_up = True
            if log:
                log("verification deadline reached")
            break
        while queue and len(active) < parallel:
            job = queue.popleft()
            if job.gave_up:
                continue
            batch = [i for i, tid in enumerate(job.ids) if tid not in job.verdicts]
            if not batch:
                continue
            pid, fd = _fork(job, [job.ids[i] for i in batch])
            active[fd] = {
                "job": job,
                "pid": pid,
                "batch": batch,
                "started": None,
                "limit": now + BETWEEN_TESTS_S,
                "buf": b"",
                "finished": False,
            }
        if not active:
            continue
        wait = min(min(st["limit"] for st in active.values()), deadline) - time.monotonic()
        ready, _, _ = select.select(list(active), [], [], max(0.0, wait))
        for fd in ready:
            st = active[fd]
            chunk = os.read(fd, 65536)
            if not chunk:
                finish(fd, False)
                continue
            st["buf"] += chunk
            job = st["job"]
            while b"\n" in st["buf"]:
                line, st["buf"] = st["buf"].split(b"\n", 1)
                parts = line.decode().split()
                if not parts:
                    continue
                if parts[0] == "DONE":
                    st["finished"] = True
                elif parts[0] == "S":
                    st["started"] = int(parts[1])
                    st["limit"] = time.monotonic() + job.allowance[st["batch"][st["started"]]]
                elif parts[0] == "E":
                    tid = job.ids[st["batch"][int(parts[1])]]
                    job.verdicts[tid] = parts[2]
                    if len(parts) > 3:
                        job.cpu[tid] = float(parts[3])
                    st["started"] = None
                    st["limit"] = time.monotonic() + BETWEEN_TESTS_S
                elif parts[0] == "C" and len(parts) > 2:
                    job.class_cpu[parts[1]] = job.class_cpu.get(parts[1], 0.0) + float(parts[2])
        now = time.monotonic()
        for fd in [fd for fd, st in active.items() if st["limit"] <= now]:
            finish(fd, True)
