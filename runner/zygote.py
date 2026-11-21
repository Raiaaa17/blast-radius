"""Warm verification server: `python -m runner.zygote --pristine DIR --lane-dir DIR`.

Protocol: one JSON request per line on stdin, one JSON response per line on stdout
(stdout carries nothing else; logs go to stderr).

Request:  {"rid": any, "patch": "<diff text>", "ids": ["tests/x.py::C::t", ...],
           "timeout": 20.0 | null,       # pytest-timeout seconds per test (grader's value)
           "budget_s": 60.0 | null,      # wall budget for the whole request
           "hard_cap_s": 90.0 | null,    # wall cap for a single test (signal timeout failed)
           "workers": 1,                 # max_workers for spawn pools inside tests
           "max_failures": null}         # stop early after this many failing tests
Response: {"rid", "failing": {id: why}, "passed": [...], "skipped": [...], "unknown": {id: why},
           "cpu_s", "wall_s", "runs", "error": null | str, "purge": level}
"""

from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import select
import signal
import sys
import time

from . import driver
from .overlay import Overlay, resolve_sensitive_tests
from .patch import PatchError, apply_patch, parse_patch


_LOG = None


def log(*a):
    out = _LOG or sys.stderr
    try:
        print("[runner.zygote]", *a, file=out, flush=True)
    except Exception:
        pass


class Zygote:
    def __init__(self, pristine: str, lane_dir: str, preload: bool, quiet: bool = True,
                 timeout: float | None = None, prepare: bool = True):
        self.pristine = os.path.abspath(pristine)
        self.lane_dir = lane_dir
        self.root = os.path.join(lane_dir, "root")
        self.preload = preload
        self.quiet = quiet
        self.preload_files: set[str] = set()  # repo-relative files read/imported at preload
        os.makedirs(lane_dir, exist_ok=True)
        self.overlay = Overlay(self.pristine, self.root, resolve_sensitive_tests(self.pristine))
        self.overlay.reset()
        self.preload_s = 0.0
        self.proto_fds = ()
        self.timeout = timeout
        self.prepare = prepare

    # ------------------------------------------------------------------ warmup
    def warm(self) -> None:
        t0 = time.process_time()
        os.environ.setdefault("SKIP_INTEGRATION", "0")
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        os.chdir(self.root)
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        # third-party modules the tests import; never patched
        # (pytest plugins such as pytest_timeout are left to pytest so it rewrites them)
        for mod in ("pytest", "_pytest.unittest", "_pytest.subtests",
                    "numpy", "pandas", "pandas.testing", "duckdb", "sqlite3",
                    "concurrent.futures.process", "multiprocessing.forkserver",
                    "multiprocessing.popen_forkserver", "multiprocessing.resource_tracker",
                    "unittest.mock", "decimal", "json", "pickle"):
            try:
                __import__(mod)
            except Exception as e:
                log(f"preload {mod} failed: {e!r}")
        import multiprocessing.resource_tracker as rt

        rt.ensure_running()  # shared by every child; avoids a tracker spawn per commit
        if self.preload:
            self._preload_suite()
        else:
            self._pytest_warm()
        if self.prepare:
            try:
                driver.prepare_config(self.timeout)
            except Exception as e:
                log(f"prepare config failed: {e!r}")
        import gc

        gc.collect()
        gc.freeze()  # children never scan (or copy-on-write) the warm heap during gc
        self.preload_s = time.process_time() - t0
        log(f"warm in {self.preload_s:.2f}s cpu, preload={self.preload}, "
            f"{len(self.preload_files)} files read at import")

    def _pytest_warm(self) -> None:
        """Run pytest once on an empty dir so its plugins and internals are imported."""
        import contextlib

        import pytest

        d = os.path.join(self.lane_dir, "warm")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "test_warm.py"), "w") as f:
            f.write("import unittest\nclass T(unittest.TestCase):\n"
                    "    def test_a(self):\n        with self.subTest(1):\n            pass\n")
        cwd = os.getcwd()
        try:
            os.chdir(d)
            with contextlib.redirect_stdout(io.StringIO()):
                pytest.main(["-q", "-p", "no:cacheprovider", "test_warm.py"])
        finally:
            os.chdir(cwd)
            sys.modules.pop("test_warm", None)

    def _preload_suite(self) -> None:
        """Import sqlglot and collect every test module (with pytest's rewriting)."""
        import contextlib

        import pytest

        self._pytest_warm()
        opened: set[str] = set()
        real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

        def note(p):
            try:
                opened.add(os.path.abspath(os.fspath(p)))
            except Exception:
                pass

        def b_open(file, *a, **k):
            if not isinstance(file, int):
                note(file)
            return real_open(file, *a, **k)

        def o_open(path, *a, **k):
            note(path)
            return real_os_open(path, *a, **k)

        builtins.open = b_open
        io.open = b_open
        os.open = o_open
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pytest.main(["-q", "--collect-only", "-p", "no:cacheprovider", "tests"])
        finally:
            builtins.open, io.open, os.open = real_open, real_io_open, real_os_open
        rootp = self.root + os.sep
        for p in opened:
            if p.startswith(rootp):
                self.preload_files.add(p[len(rootp):])
        for m in list(sys.modules.values()):
            f = getattr(m, "__file__", None)
            if f and f.startswith(rootp):
                self.preload_files.add(f[len(rootp):])

    # ------------------------------------------------------------------ requests
    def purge_level(self, paths: list[str]) -> str:
        if not self.preload:
            return "all"
        level = "none"
        for p in paths:
            if p.startswith("sqlglot/") or p in self.preload_files and not p.startswith("tests/"):
                return "all"
            if p.startswith("tests/"):
                if p in self.preload_files and not p.endswith(".py"):
                    return "all"  # data read while importing: may be cached anywhere
                if p.endswith(".py") or p in self.preload_files:
                    level = "tests"
            elif p in self.preload_files or p.endswith(".py"):
                return "all"
        return level

    def handle(self, req: dict) -> dict:
        t0 = time.monotonic()
        ids = list(dict.fromkeys(req.get("ids") or []))
        resp = {"rid": req.get("rid"), "failing": {}, "passed": [], "skipped": [],
                "unknown": {}, "cpu_s": 0.0, "wall_s": 0.0, "runs": 0, "error": None,
                "purge": None}
        if not ids:
            return resp
        try:
            changes = parse_patch(req.get("patch") or "")
            files = apply_patch(changes, self.overlay.read_pristine)
            self.overlay.apply(files)
        except (PatchError, OSError, UnicodeError, ValueError, IndexError, KeyError) as e:
            resp["error"] = f"patch: {e}"
            resp["unknown"] = {i: "patch did not apply" for i in ids}
            self._reset()
            return resp
        level = self.purge_level(list(files))
        resp["purge"] = level
        budget = req.get("budget_s")
        deadline = t0 + budget if budget else None
        hard_cap = req.get("hard_cap_s")
        remaining = ids
        max_failures = req.get("max_failures")
        streak = 0  # consecutive aborted runs that died on their first test
        try:
            while remaining:
                if deadline is not None and time.monotonic() >= deadline:
                    for i in remaining:
                        resp["unknown"].setdefault(i, "budget exhausted")
                    break
                res = self._run_once(req, remaining, level, deadline, hard_cap, files)
                resp["runs"] += 1
                resp["cpu_s"] += res["cpu_s"]
                resp.setdefault("diag", []).append({"timeline": res.get("timeline"),
                                                    "minflt": res.get("minflt"),
                                                    "vmpeak_kb": res.get("vmpeak_kb")})
                resp["failing"].update(res["failing"])
                for i, why in res["timeout"].items():
                    resp["unknown"][i] = f"verification timeout: {why[:200]}"
                resp["passed"] += res["passed"]
                resp["skipped"] += res["skipped"]
                done = (set(res["failing"]) | set(res["passed"]) | set(res["skipped"])
                        | set(res["timeout"]))
                if res.get("collect_errors"):
                    for i in remaining:
                        if i in done:
                            continue
                        for node, why in res["collect_errors"].items():
                            if i == node or i.startswith(node + "::") or node == "":
                                resp["failing"][i] = f"collect error: {why}"
                                done.add(i)
                                break
                if res["missing"]:
                    gone = set(res["missing"])
                    for i in remaining:
                        if i not in done and i.split("::", 1)[0] in gone:
                            resp["unknown"][i] = "test file missing after patch"
                            done.add(i)
                if res["aborted"] is None:
                    for i in remaining:
                        if i not in done:
                            resp["unknown"][i] = "not collected"
                    break
                if res["aborted"] in ("max_failures", "timeouts"):
                    why = ("not run: too many verification timeouts (patch likely hangs)"
                           if res["aborted"] == "timeouts" else "not run (early exit)")
                    for i in remaining:
                        if i not in done:
                            resp["unknown"][i] = why
                    break
                # crash / hard cap / budget: the running test is charged, the rest rerun
                cur = res.get("current")
                first = remaining[0] if remaining else None
                streak = streak + 1 if (cur == first and len(done) == 0) else 0
                if cur and cur not in done:
                    if res["aborted"] == "budget" or res["aborted"].startswith("hard timeout"):
                        resp["unknown"][cur] = f"{res['aborted']} while running"
                    else:
                        resp["failing"][cur] = res["aborted"]
                    done.add(cur)
                new_remaining = [i for i in remaining if i not in done]
                if len(new_remaining) == len(remaining):  # no progress: give up
                    for i in remaining:
                        resp["unknown"].setdefault(i, f"no progress ({res['aborted']})")
                    break
                remaining = new_remaining
                if streak >= 2 and remaining:  # every run dies at once: stop paying for it
                    for i in remaining:
                        resp["unknown"].setdefault(i, f"not run: repeated {res['aborted']}")
                    break
                if max_failures and len(resp["failing"]) >= max_failures:
                    for i in remaining:
                        resp["unknown"].setdefault(i, "not run (early exit)")
                    break
        finally:
            self._reset()
        resp["wall_s"] = round(time.monotonic() - t0, 4)
        resp["cpu_s"] = round(resp["cpu_s"], 4)
        return resp

    def _reset(self):
        try:
            self.overlay.reset()
        except OSError as e:
            log(f"overlay reset failed: {e!r}")
            raise

    def _can_use_prepared(self, req, files) -> bool:
        if "config" not in driver.PREPARED or req.get("pytest_args"):
            return False
        if req.get("timeout", self.timeout) != driver.PREPARED.get("timeout"):
            return False
        return not any(p.rsplit("/", 1)[-1] in driver.CONFIG_FILES for p in files)

    def _run_once(self, req, ids, level, deadline, hard_cap, files=()) -> dict:
        idset = set(ids)
        caps = {i: c for i, c in (req.get("caps") or {}).items() if i in idset}
        r_fd, w_fd = os.pipe()
        spec = {
            "root": self.root,
            "ids": ids,
            "purge": level,
            "timeout": req.get("timeout"),
            "workers": req.get("workers", 1),
            "cpu_limit_s": req.get("cpu_limit_s"),
            "as_limit_mb": req.get("as_limit_mb"),
            "pytest_args": req.get("pytest_args", ()),
            "quiet_stderr": self.quiet,
            "max_failures": req.get("max_failures"),
            "use_prepared": self._can_use_prepared(req, files),
            "caps": caps,
            "max_timeouts": req.get("max_timeouts", 3),
            "spawn_redirect": req.get("spawn_redirect", True),
        }
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            os.close(r_fd)
            for fd in self.proto_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                nul = os.open(os.devnull, os.O_RDONLY)
                os.dup2(nul, 0)
                os.close(nul)
            except OSError:
                pass
            driver.run_child(spec, w_fd)  # never returns
            os._exit(4)
        os.close(w_fd)
        try:
            os.setpgid(pid, pid)
        except OSError:
            pass
        res = {"failing": {}, "timeout": {}, "passed": [], "skipped": [], "collect_errors": {},
               "missing": [],
               "aborted": "crash", "current": None, "cpu_s": 0.0}
        buf = b""
        cur_start = None
        cur_hard = hard_cap
        ended = False
        status = None
        try:
            while True:
                now = time.monotonic()
                waits = [1.0]
                if deadline is not None:
                    waits.append(deadline - now)
                if cur_hard and cur_start is not None:
                    waits.append(cur_start + cur_hard - now)
                wait = max(0.0, min(waits))
                rl, _, _ = select.select([r_fd], [], [], wait)
                if rl:
                    chunk = os.read(r_fd, 65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        kind = ev.get("ev")
                        if kind == "start":
                            res["current"] = ev["id"]
                            cur_start = time.monotonic()
                            c = caps.get(ev["id"])
                            # pytest-timeout (signal) should fire at the cap; this is the
                            # backstop for code stuck in C where the signal cannot land
                            cur_hard = min(hard_cap or 1e9, c * 1.5 + 15) if c else hard_cap
                        elif kind == "done":
                            v = ev.get("v")
                            if v == "failing":
                                res["failing"][ev["id"]] = ev.get("why", "")
                            elif v == "timeout":
                                res["timeout"][ev["id"]] = ev.get("why", "")
                            elif v == "skipped":
                                res["skipped"].append(ev["id"])
                            else:
                                res["passed"].append(ev["id"])
                            res["current"] = None
                            cur_start = None
                        elif kind in ("t", "begin"):
                            res.setdefault("timeline", []).append((ev.get("what", kind), ev.get("cpu")))
                            if ev.get("vmpeak_kb"):
                                res["vmpeak_kb"] = ev["vmpeak_kb"]
                        elif kind == "missing":
                            res["missing"] += ev.get("files") or []
                        elif kind == "collect_error":
                            res["collect_errors"][ev["node"]] = ev.get("why", "")
                        elif kind == "end":
                            ended = True
                            res["rc"] = ev.get("rc")
                            res["collect_errors"].update(ev.get("collect_errors") or {})
                        elif kind == "fatal":
                            res["fatal"] = ev.get("why")
                            log(f"child fatal: {ev.get('why')}")
                        elif kind == "early_exit":
                            res["aborted"] = ("timeouts" if ev.get("why") == "timeouts"
                                              else "max_failures")
                    continue
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    res["aborted"] = "budget"
                    self._kill(pid)
                    break
                if cur_hard and cur_start is not None and now - cur_start >= cur_hard:
                    res["aborted"] = f"hard timeout after {cur_hard:.0f}s"
                    self._kill(pid)
                    break
                # child gone without closing the pipe (a grandchild holds it)?
                try:
                    wpid, st, ru = os.wait4(pid, os.WNOHANG)
                except ChildProcessError:
                    wpid = pid
                    st, ru = 0, None
                if wpid:
                    status = (st, ru)
                    break
        finally:
            os.close(r_fd)
        if status is None:
            try:
                _, st, ru = os.wait4(pid, 0)
                status = (st, ru)
            except ChildProcessError:
                status = (0, None)
        self._kill(pid)  # stragglers in the child's process group
        st, ru = status
        if ru is not None:
            res["cpu_s"] = ru.ru_utime + ru.ru_stime
            res["minflt"] = ru.ru_minflt
        if ended and res["aborted"] == "crash":
            rc = res.get("rc")
            # 0 ok, 1 tests failed, 5 nothing collected; 2 interrupted, 3 internal error,
            # 4 usage error: the session stopped early -> rerun what did not finish
            res["aborted"] = None if rc in (0, 1, 5, None) else f"pytest exit code {rc}"
        elif res["aborted"] == "crash":
            if os.WIFSIGNALED(st):
                res["aborted"] = f"crash (signal {os.WTERMSIG(st)})"
            else:
                res["aborted"] = f"crash (exit {os.WEXITSTATUS(st) if os.WIFEXITED(st) else st})"
            if res.get("fatal"):
                res["aborted"] += f": {res['fatal']}"
        return res

    @staticmethod
    def _kill(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        except OSError:
            pass

    # ------------------------------------------------------------------ main loop
    def serve(self, inp, out) -> None:
        out.write(json.dumps({"ready": True, "warm_cpu_s": round(self.preload_s, 3),
                              "pid": os.getpid()}) + "\n")
        out.flush()
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError:
                continue
            if req.get("op") == "quit":
                break
            try:
                resp = self.handle(req)
            except Exception as e:  # never die on one bad request
                resp = {"rid": req.get("rid"), "error": f"internal: {e!r}", "failing": {},
                        "passed": [], "skipped": [],
                        "unknown": {i: "internal error" for i in req.get("ids") or []},
                        "cpu_s": 0.0, "wall_s": 0.0, "runs": 0}
            out.write(json.dumps(resp) + "\n")
            out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pristine", required=True)
    ap.add_argument("--lane-dir", required=True)
    ap.add_argument("--preload", action="store_true",
                    help="also import sqlglot + every test module in the zygote")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--no-prepare", action="store_true")
    a = ap.parse_args(argv)
    # protocol channel = the original stdout; everything else printed goes to stderr
    global _LOG
    proto_out = os.fdopen(os.dup(1), "w", buffering=1)
    proto_in = os.fdopen(os.dup(0), "r")
    _LOG = os.fdopen(os.dup(2), "w", buffering=1)
    nul = os.open(os.devnull, os.O_RDWR)
    os.dup2(nul, 0)
    os.dup2(nul if not a.verbose else 2, 1)
    if not a.verbose:
        os.dup2(nul, 2)  # pytest's fd capture saves these: children inherit /dev/null
    os.close(nul)
    z = Zygote(a.pristine, a.lane_dir, preload=a.preload, quiet=not a.verbose,
               timeout=a.timeout, prepare=not a.no_prepare)
    z.proto_fds = (proto_out.fileno(), proto_in.fileno(), _LOG.fileno())
    z.warm()
    z.serve(proto_in, proto_out)
    sys.stderr.flush()
    os._exit(0)  # skip interpreter teardown (duckdb/nanobind leak noise, CPU)


if __name__ == "__main__":
    main()
