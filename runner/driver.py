"""Code that runs inside one forked verification child.

The child is forked from a warm zygote (pytest, numpy, pandas, duckdb and - optionally -
pristine sqlglot plus every test module already imported). It drops the modules a patch
can affect, then runs the requested tests through `pytest.main` in-process so pytest's
own unittest integration, subtests, pytest-timeout and assertion rewriting decide every
verdict exactly as in the organizers' labeling run.

Verdict per requested test id (grader semantics):
  failing  any setup/call/teardown report failed, any subtest report failed (pytest 9
           prints PASSED next to SUBFAILED: that is a failure), pytest-timeout fired, or
           the test's file/class failed to collect (import error);
  passed   reported and nothing failed (a skip or an expected failure cannot fail);
  unknown  never reported (not collected, run aborted, crash) - the caller must not
           treat this as passed.

Events are streamed to the zygote as JSON lines on `result_fd` so that a crash or kill
keeps every verdict finished before it.

spawn-context process pools (tests/test_optimizer.py and tests/test_executor.py use
`ProcessPoolExecutor(mp_context=get_context("spawn"))`, which would start ~os.cpu_count()
fresh interpreters that each import sqlglot, duckdb and pandas) are redirected to a
forkserver whose server process is forked from this child right after collection,
before any test ran: workers start from the same post-import state a spawn worker
reaches, but cost a fork instead of ~1 s of imports. Results still cross a real process
boundary and are pickled exactly as with spawn.
"""

from __future__ import annotations

import json
import os
import socket
import sys

PURGE_PREFIXES = {
    "none": (),
    "tests": ("tests",),
    "all": ("tests", "sqlglot"),
}


def purge_modules(level: str) -> int:
    prefixes = PURGE_PREFIXES[level]
    if not prefixes:
        return 0
    doomed = [
        name
        for name in sys.modules
        if any(name == p or name.startswith(p + ".") for p in prefixes)
    ]
    for name in doomed:
        del sys.modules[name]
    return len(doomed)


# --------------------------------------------------------------------- spawn -> template


class _Template:
    """A forkserver whose server process is a fork of this (pristine) child."""

    pid: int | None = None
    started_pristine = False


def _start_template() -> None:
    import multiprocessing.connection as mpconn
    import multiprocessing.forkserver as mfs

    fs = mfs._forkserver
    if _Template.pid is not None:
        try:
            pid, _ = os.waitpid(_Template.pid, os.WNOHANG)
        except ChildProcessError:
            pid = _Template.pid
        if not pid:
            return
        _Template.pid = None
    with socket.socket(socket.AF_UNIX) as listener:
        address = mpconn.arbitrary_address("AF_UNIX")
        listener.bind(address)
        os.chmod(address, 0o600)
        listener.listen()
        alive_r, alive_w = os.pipe()
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:  # the template server
            code = 0
            try:
                os.close(alive_w)
                rfd = _CHILD_STATE.get("result_fd")
                if rfd is not None:
                    try:
                        os.close(rfd)
                    except OSError:
                        pass
                mfs.main(listener.fileno(), alive_r, [])
            except SystemExit:
                pass
            except BaseException:
                code = 1
            finally:
                os._exit(code)
        os.close(alive_r)
    fs._forkserver_address = address
    fs._forkserver_alive_fd = alive_w
    fs._forkserver_pid = pid
    _Template.pid = pid


def _ensure_running() -> None:
    import multiprocessing.resource_tracker as rt

    rt.ensure_running()
    _start_template()


def install_spawn_redirect(workers: int | None) -> None:
    import multiprocessing.context as mpc
    import multiprocessing.forkserver as mfs

    mfs._forkserver.ensure_running = _ensure_running
    mpc._concrete_contexts["spawn"] = mpc._concrete_contexts["forkserver"]
    if workers:
        os.process_cpu_count = lambda: workers  # ProcessPoolExecutor's default max_workers


def stop_template() -> None:
    import multiprocessing.forkserver as mfs

    fs = mfs._forkserver
    if fs._forkserver_alive_fd is not None:
        try:
            os.close(fs._forkserver_alive_fd)
        except OSError:
            pass
        fs._forkserver_alive_fd = None
    if _Template.pid is not None:
        try:
            os.waitpid(_Template.pid, 0)
        except ChildProcessError:
            pass
        _Template.pid = None


def _module_uses_mp(mod) -> bool:
    import concurrent.futures.process as cfp
    import multiprocessing

    for v in vars(mod).values():
        if v is multiprocessing or v is cfp.ProcessPoolExecutor:
            return True
        if getattr(v, "__module__", None) in ("multiprocessing.context",):
            return True
        if getattr(v, "__name__", None) == "get_context" and "multiprocessing" in str(
            getattr(v, "__module__", "") or getattr(getattr(v, "__self__", None), "__module__", "")
        ):
            return True
    return False


# --------------------------------------------------------------------- pytest plugin

_CHILD_STATE: dict = {}


class Collector:
    def __init__(self, out, wanted: list[str], max_failures: int | None = None,
                 caps: dict | None = None):
        self.out = out
        self.caps = caps or {}
        self.timed_out: set[str] = set()
        self.max_timeouts = None
        self.max_failures = max_failures
        self.session = None
        self.wanted = set(wanted)
        self.failed: dict[str, str] = {}
        self.skipped: set[str] = set()
        self.seen: set[str] = set()
        self.collect_errors: dict[str, str] = {}

    def emit(self, **ev):
        try:
            self.out.write(json.dumps(ev) + "\n")
            self.out.flush()
        except Exception:
            pass

    # collection --------------------------------------------------------------
    def pytest_collectreport(self, report):
        if report.failed:
            msg = _short(report.longreprtext) if hasattr(report, "longreprtext") else "collect"
            self.collect_errors[report.nodeid] = msg
            self.emit(ev="collect_error", node=report.nodeid, why=msg)

    def pytest_collection_modifyitems(self, session, config, items):
        self.session = session
        self.emit(ev="t", what="collected", cpu=_cpu())
        keep = [it for it in items if it.nodeid in self.wanted]
        drop = [it for it in items if it.nodeid not in self.wanted]
        items[:] = keep
        if self.caps:
            import pytest

            for it in keep:
                cap = self.caps.get(it.nodeid)
                if cap:
                    # call phase only: class setup (TestExecutor loads TPC data ~2 s) is
                    # bounded by the zygote's hard cap instead
                    it.add_marker(pytest.mark.timeout(cap, func_only=True))
        if drop:
            config.hook.pytest_deselected(items=drop)
        self.emit(ev="collected", n=len(keep))

    def pytest_sessionstart(self, session):
        self.emit(ev="t", what="sessionstart", cpu=_cpu())

    def pytest_runtestloop(self, session):
        # Fork the pristine template before any test runs when a selected test module
        # uses multiprocessing (spawn pools); otherwise it is started lazily on demand.
        mods = {id(getattr(it, "module", None)): getattr(it, "module", None) for it in session.items}
        if any(m is not None and _module_uses_mp(m) for m in mods.values()):
            try:
                import multiprocessing.resource_tracker as rt

                rt.ensure_running()
                _start_template()
                _Template.started_pristine = True
            except Exception as e:  # never fatal: lazy start remains available
                self.emit(ev="log", msg=f"template start failed: {e!r}")

    # running -----------------------------------------------------------------
    def pytest_runtest_logstart(self, nodeid, location):
        self.emit(ev="start", id=nodeid)

    def pytest_runtest_logreport(self, report):
        nid = report.nodeid
        if report.failed:
            text = getattr(report, "longreprtext", "") or ""
            if "Timeout (>" in text and "pytest-timeout" in text:
                self.timed_out.add(nid)
            ctx = getattr(report, "context", None)
            where = report.when + (" subtest" if ctx is not None else "")
            if nid not in self.failed:
                self.failed[nid] = f"{where}: {_short(getattr(report, 'longreprtext', ''))}"
        elif report.skipped and report.when in ("setup", "call") and getattr(report, "context", None) is None:
            self.skipped.add(nid)

    def pytest_runtest_logfinish(self, nodeid, location):
        self.seen.add(nodeid)
        if nodeid in self.timed_out:
            self.emit(ev="done", id=nodeid, v="timeout", why=self.failed.get(nodeid, "timeout"))
            if self.max_timeouts and len(self.timed_out) >= self.max_timeouts and self.session:
                # a patch that hangs everything would otherwise cost a full cap per test
                self.session.shouldstop = "too many verification timeouts"
                self.emit(ev="early_exit", why="timeouts")
        elif nodeid in self.failed:
            self.emit(ev="done", id=nodeid, v="failing", why=self.failed[nodeid])
            if self.max_failures and len(self.failed) >= self.max_failures and self.session:
                self.session.shouldstop = "max failures reached"
                self.emit(ev="early_exit")
        elif nodeid in self.skipped:
            self.emit(ev="done", id=nodeid, v="skipped")
        else:
            self.emit(ev="done", id=nodeid, v="passed")


PREPARED: dict = {}  # {"config": Config, "timeout": float|None} set by the zygote before forking


def base_pytest_args(timeout) -> list[str]:
    # --continue-on-collection-errors: one unimportable file must not stop the others
    # (the grader marks that file's tests failing and still runs everything else)
    args = ["-q", "-p", "no:cacheprovider", "--continue-on-collection-errors"]
    if timeout:
        args.append(f"--timeout={timeout}")
    return args


def prepare_config(timeout) -> None:
    """Parse args and load plugins once (in the zygote); children only run the session."""
    from _pytest.config import _prepareconfig

    PREPARED["config"] = _prepareconfig(base_pytest_args(timeout) + ["tests"], [])
    PREPARED["timeout"] = timeout


# patches touching these make the zygote-prepared config stale: use plain pytest.main
CONFIG_FILES = ("conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml")


def _vmpeak() -> int | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _cpu() -> float:
    import resource

    s = resource.getrusage(resource.RUSAGE_SELF)
    return round(s.ru_utime + s.ru_stime, 4)


def _short(text: str, limit: int = 300) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    tail = "\n".join(lines[-4:])
    return tail[-limit:]


def run_child(spec: dict, result_fd: int) -> None:
    """Entry point in the forked child; never returns."""
    code = 0
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    try:
        try:  # if memory runs out, the kernel kills this child, never the zygote/selector
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")
        except OSError:
            pass
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # no slow core dumps on segfault
        except (ValueError, OSError):
            pass
        _CHILD_STATE["result_fd"] = result_fd
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        if spec.get("quiet_stderr", True):
            os.dup2(devnull, 2)
        os.close(devnull)
        sys.dont_write_bytecode = True
        out = os.fdopen(result_fd, "w", buffering=1)
        root = spec["root"]
        os.chdir(root)
        purged = purge_modules(spec.get("purge", "all"))
        import importlib

        importlib.invalidate_caches()
        if spec.get("spawn_redirect", True):
            install_spawn_redirect(spec.get("workers"))
        elif spec.get("workers"):
            os.process_cpu_count = lambda: spec["workers"]
        if spec.get("cpu_limit_s"):
            import resource

            lim = int(spec["cpu_limit_s"])
            resource.setrlimit(resource.RLIMIT_CPU, (lim, lim + 5))
        if spec.get("as_limit_mb"):
            import resource

            lim = int(spec["as_limit_mb"]) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

        import pytest

        wanted = spec["ids"]
        col = Collector(out, wanted, spec.get("max_failures"), spec.get("caps"))
        col.max_timeouts = spec.get("max_timeouts")
        col.emit(ev="begin", purged=purged, cpu=_cpu())
        files = list(dict.fromkeys(i.split("::", 1)[0] for i in wanted))
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:  # pytest aborts the whole session on a missing path argument
            col.emit(ev="missing", files=missing)
            files = [f for f in files if f not in missing]
        if not files:
            col.emit(ev="end", rc=0, collect_errors={}, cpu=_cpu())
            out.flush()
            os._exit(0)
        prepared = PREPARED.get("config") if spec.get("use_prepared") else None
        if prepared is not None:
            # same as pytest.main(): the config was parsed/plugin-loaded in the zygote
            col.emit(ev="log", msg="prepared config")
            os.environ["PYTEST_VERSION"] = pytest.__version__
            prepared.args = files
            prepared.pluginmanager.register(col, "br-runner-collector")
            try:
                rc = prepared.hook.pytest_cmdline_main(config=prepared)
            finally:
                prepared._ensure_unconfigure()
        else:
            args = base_pytest_args(spec.get("timeout")) + list(spec.get("pytest_args", ()))
            rc = pytest.main(args + files, plugins=[col])
        col.emit(ev="t", what="tests_done", cpu=_cpu(), vmpeak_kb=_vmpeak())
        stop_template()
        col.emit(ev="end", rc=int(rc), collect_errors=col.collect_errors, cpu=_cpu())
        out.flush()
    except BaseException as e:  # report and die; the zygote marks the rest unknown
        try:
            os.write(result_fd, (json.dumps({"ev": "fatal", "why": repr(e)[:500]}) + "\n").encode())
        except Exception:
            pass
        code = 3
    finally:
        os._exit(code)
