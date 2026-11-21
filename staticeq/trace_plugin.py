"""pytest plugin (build time only): record which files every test touches.

Load with ``-p staticeq.trace_plugin --staticeq-out DIR [--staticeq-shard i/N
--staticeq-tests tests.json]``. Mechanics:

* A ``sys.addaudithook`` hook sees every ``open`` (including failed opens of
  missing files), ``os.listdir``/``os.scandir`` and ``glob.glob`` under the
  repository root, with a compact reader signature taken from the stack.
  Import-system reads are separated from code that reads source text.
* A finder appended to ``sys.meta_path`` records failed imports of
  repository modules (a new file with that name would be picked up).
* Each test's *call* phase runs in a forked child (setup and teardown stay in
  the parent), so a lazily-filled cache in one test cannot hide a read from a
  later test; reads in setup/teardown (setUpClass, fixtures) are attributed
  to every test of the class, reads during collection to every test of the
  module.
* ProcessPoolExecutor is replaced by an inline executor, so work that
  sqlglot's tests ship to spawned workers runs (and is traced) in-process,
  and DuckDB connections are forced to one thread so they survive fork.

Output: one JSON per process in DIR, merged by build_map.py.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import pytest

_WATCHED = frozenset({"open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2"})
_STDLIB_TAGS = ("linecache", "inspect", "traceback", "runpy", "pkgutil", "doctest", "tokenize",
                "zipimport")

_root = ""
_root_sep = ""
_ctx: tuple[str, str] = ("init", "")
_events: dict[tuple, None] = {}
_missing: dict[tuple, None] = {}
_outcomes: dict[str, str] = {}
_collect_stack: list[str] = []
_child_timeout = 900


def _rel(path, dir_fd_possible: bool = False) -> str | None:
    try:
        if isinstance(path, int):
            return None
        path = os.fspath(path)
        if isinstance(path, bytes):
            path = os.fsdecode(path)
    except TypeError:
        return None
    if not os.path.isabs(path):
        if dir_fd_possible:
            return None  # os.open(name, dir_fd=...) as in shutil.rmtree: not cwd-relative
        path = os.path.join(os.getcwd(), path)
    path = os.path.normpath(path)
    if path == _root:
        return "."
    if not path.startswith(_root_sep):
        return None
    return path[len(_root_sep):]


def _stack_info():
    """(is_import, reader signature, stdlib tags) for the current open."""
    f = sys._getframe(2)
    is_import = False
    reader = None
    tags: list[str] = []
    depth = 0
    while f is not None and depth < 60:
        fn = f.f_code.co_filename
        if fn.startswith("<frozen importlib"):
            is_import = True
        elif reader is None and fn.startswith(_root_sep):
            rel = fn[len(_root_sep):]
            if not rel.startswith("staticeq/"):
                reader = f"{rel}:{f.f_code.co_qualname}"
        else:
            base = os.path.basename(fn)[:-3]
            if base in _STDLIB_TAGS and base not in tags:
                tags.append(base)
        f = f.f_back
        depth += 1
    return is_import, reader or "<no repo frame>", tags


def _audit(event, args):
    if event not in _WATCHED or not _root:
        return
    try:
        os_open = event == "open" and len(args) > 1 and args[1] is None
        rel = _rel(args[0], dir_fd_possible=os_open) if args else None
        if rel is None and event in ("os.listdir", "os.scandir") and args and args[0] is None:
            rel = _rel(os.getcwd())
        if rel is None:
            return
        if event == "open":
            if "__pycache__" in rel or os.path.isdir(os.path.join(_root, rel)):
                return
            mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
            is_import, reader, tags = _stack_info()
            if rel.endswith(".py") and is_import:
                return  # module import: covered by the import map
            kind = "write" if any(c in mode for c in "wax+") else "read"
            if rel.endswith(".py"):
                kind = "source-" + kind
            _events[(_ctx[0], _ctx[1], kind, rel, reader, ",".join(tags))] = None
        else:
            is_import, reader, tags = _stack_info()
            if is_import:
                return  # FileFinder directory cache; failed imports are tracked separately
            _events[(_ctx[0], _ctx[1], "listdir", rel, reader, ",".join(tags))] = None
    except Exception:  # never disturb the traced test
        return


class _MissingFinder:
    """Last meta_path entry: sees only imports every real finder failed."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if fullname.split(".")[0] in ("sqlglot", "tests"):
            try:
                reader = _stack_info()[1]
            except Exception:
                reader = "<no repo frame>"
            _missing[(_ctx[0], _ctx[1], fullname, reader)] = None
        return None


class _InlineExecutor:
    """Drop-in for ProcessPoolExecutor that runs work immediately in-process."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(),
                 **kwargs):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, /, *args, **kwargs):
        from concurrent.futures import Future

        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001 - mirror executor semantics
            fut.set_exception(e)
        return fut

    def map(self, fn, *iterables, timeout=None, chunksize=1, **kwargs):
        return map(fn, *iterables)

    def shutdown(self, wait=True, *, cancel_futures=False):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def _patch_runtime() -> None:
    import concurrent.futures
    import concurrent.futures.process

    concurrent.futures.ProcessPoolExecutor = _InlineExecutor
    concurrent.futures.process.ProcessPoolExecutor = _InlineExecutor
    try:
        import duckdb

        real_connect = duckdb.connect

        def connect(*args, **kwargs):
            config = dict(kwargs.pop("config", None) or {})
            config.setdefault("threads", 1)
            return real_connect(*args, config=config, **kwargs)

        duckdb.connect = connect
    except Exception:
        pass


def _imported_files() -> list[str]:
    out = set()
    for m in list(sys.modules.values()):
        fn = getattr(m, "__file__", None)
        if isinstance(fn, str):
            rel = _rel(fn)
            if rel and rel.endswith(".py"):
                out.add(rel)
    return sorted(out)


# ---------------------------------------------------------------------------
# pytest hooks


def pytest_addoption(parser):
    g = parser.getgroup("staticeq")
    g.addoption("--staticeq-out", default=None)
    g.addoption("--staticeq-shard", default="0/1")
    g.addoption("--staticeq-tests", default=None, help="tests.json for duration-balanced shards")
    g.addoption("--staticeq-child-timeout", type=int, default=900)


def pytest_configure(config):
    global _root, _root_sep, _child_timeout
    if not config.getoption("--staticeq-out"):
        return
    _root = os.path.normpath(str(config.rootpath))
    _root_sep = _root + os.sep
    _child_timeout = config.getoption("--staticeq-child-timeout")
    _patch_runtime()
    sys.addaudithook(_audit)
    sys.meta_path.append(_MissingFinder())


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    if not _root:
        return (yield)
    _collect_stack.append(collector.nodeid)
    global _ctx
    prev = _ctx
    _ctx = ("collect", collector.nodeid)
    try:
        return (yield)
    finally:
        _collect_stack.pop()
        _ctx = prev


def pytest_collection_modifyitems(session, config, items):
    if not _root:
        return
    spec = config.getoption("--staticeq-shard")
    k, n = (int(x) for x in spec.split("/"))
    if n <= 1:
        return
    seconds: dict[str, float] = {}
    path = config.getoption("--staticeq-tests")
    if path:
        with open(path, encoding="utf-8") as f:
            for t in json.load(f)["tests"]:
                seconds[t["id"]] = t["seconds"]
    loads = [0.0] * n
    owner: dict[str, int] = {}
    order = sorted(items, key=lambda it: (-seconds.get(it.nodeid, 0.01), it.nodeid))
    for it in order:
        w = loads.index(min(loads))
        owner[it.nodeid] = w
        loads[w] += seconds.get(it.nodeid, 0.01) + 0.003
    keep = [it for it in items if owner[it.nodeid] == k]
    drop = [it for it in items if owner[it.nodeid] != k]
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if not _root:
        return None
    from _pytest.runner import call_and_report

    global _ctx
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    _ctx = ("setup", item.nodeid)
    rep = call_and_report(item, "setup", log=True)
    outcome = rep.outcome if not rep.passed else "?"
    if rep.passed:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:  # child: run the call phase, ship events, exit hard
            os.close(r)
            code = 0
            try:
                _events.clear()
                _missing.clear()
                _ctx = ("call", item.nodeid)

                def on_alarm(signum, frame):
                    raise TimeoutError("staticeq trace child timeout")

                signal.signal(signal.SIGALRM, on_alarm)
                signal.alarm(_child_timeout)
                t0 = time.perf_counter()
                crep = call_and_report(item, "call", log=False)
                signal.alarm(0)
                result = crep.outcome
                elapsed = time.perf_counter() - t0
            except BaseException as e:  # noqa: BLE001
                result = f"crash:{type(e).__name__}"
                elapsed = -1.0
                code = 1
            try:
                payload = json.dumps({
                    "events": list(_events),
                    "missing": list(_missing),
                    "outcome": result,
                    "seconds": elapsed,
                    "imported": _imported_files(),
                }).encode()
                view = memoryview(payload)
                while view:
                    n = os.write(w, view)
                    view = view[n:]
            finally:
                os._exit(code)
        os.close(w)
        chunks = []
        while True:
            b = os.read(r, 1 << 20)
            if not b:
                break
            chunks.append(b)
        os.close(r)
        os.waitpid(pid, 0)
        try:
            data = json.loads(b"".join(chunks) or b"{}")
        except ValueError:
            data = {}
        for ev in data.get("events", []):
            _events[tuple(ev)] = None
        for ev in data.get("missing", []):
            _missing[tuple(ev)] = None
        for f in data.get("imported", []):
            _imported.add(f)
        outcome = data.get("outcome", "lost")
    _outcomes[item.nodeid] = outcome
    _ctx = ("teardown", item.nodeid)
    call_and_report(item, "teardown", log=True, nextitem=nextitem)
    _ctx = ("idle", "")
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


_imported: set[str] = set()


def pytest_collection_finish(session):
    if not _root:
        return
    _imported.update(_imported_files())


def pytest_sessionfinish(session, exitstatus):
    if not _root:
        return
    out = session.config.getoption("--staticeq-out")
    os.makedirs(out, exist_ok=True)
    shard = session.config.getoption("--staticeq-shard").replace("/", "of")
    _imported.update(_imported_files())
    items = [it.nodeid for it in session.items]
    payload = {
        "root": _root,
        "items": items,
        "events": list(_events),
        "missing": list(_missing),
        "outcomes": _outcomes,
        "imported": sorted(_imported),
        "python": sys.version,
    }
    with open(os.path.join(out, f"trace-{shard}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)
