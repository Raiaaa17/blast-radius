"""Runtime verification: run candidate tests against a patched tree.

The patched tree is a symlink farm over the pristine checkout in /tmp: only
the directories on the way to a changed file are real, everything else
(including __pycache__ with the build's bytecode) links to the pristine copy.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHILD = HERE / "runner_child.py"


def build_tree(pristine: Path, root: Path, files: dict[str, bytes | None]) -> None:
    """files: relative path -> new content (None deletes)."""
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    # Directories that must be real (they contain a changed entry).
    real_dirs = {Path(".")}
    for rel in files:
        parent = Path(rel).parent
        while True:
            real_dirs.add(parent)
            if parent == Path("."):
                break
            parent = parent.parent
    for rel_dir in sorted(real_dirs, key=lambda p: len(p.parts)):
        src_dir = pristine / rel_dir
        dst_dir = root / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.is_dir():
            continue
        for entry in os.scandir(src_dir):
            rel = rel_dir / entry.name
            if rel in real_dirs or str(rel) in files:
                continue
            os.symlink(entry.path, dst_dir / entry.name)
    for rel, content in files.items():
        if content is None:
            continue
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.write_bytes(content)


def run_tests(
    root: Path,
    test_ids: list[str],
    allowance_s: list[float],
    deadline: float,
    log=None,
    cpu_out: dict[str, float] | None = None,
    env_extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Run tests in fresh children; returns id -> status for every test that got a verdict.

    Status is passed/failed/error/skipped from the child, or "timeout"/"crash"
    when the watchdog had to intervene. Tests without an entry got no verdict.
    """
    verdicts: dict[str, str] = {}
    remaining = list(range(len(test_ids)))
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SKIP_INTEGRATION"] = "0"
    env.pop("PYTHONPATH", None)
    if env_extra:
        env.update(env_extra)
    while remaining and time.monotonic() < deadline:
        batch = [test_ids[i] for i in remaining]
        r, w = os.pipe()
        proc = subprocess.Popen(
            [sys.executable, str(CHILD), str(root), str(w)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=[w],
            env=env,
            cwd=str(root),
        )
        os.close(w)
        try:
            proc.stdin.write(json.dumps(batch).encode())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        buf = b""
        started: int | None = None  # index into batch of the running test
        limit = time.monotonic() + 30.0  # interpreter start and imports
        done_upto = set()
        finished = False
        with os.fdopen(r, "rb", buffering=0) as pipe:
            while True:
                now = time.monotonic()
                wait = min(limit, deadline) - now
                if wait <= 0:
                    break
                ready, _, _ = select.select([pipe], [], [], wait)
                if not ready:
                    continue
                chunk = pipe.read(65536)
                if not chunk:
                    break  # child exited
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    parts = line.decode().split()
                    if not parts:
                        continue
                    if parts[0] == "DONE":
                        finished = True
                    elif parts[0] == "S":
                        started = int(parts[1])
                        limit = time.monotonic() + allowance_s[remaining[started]]
                    elif parts[0] == "E":
                        i = int(parts[1])
                        verdicts[batch[i]] = parts[2]
                        if cpu_out is not None and len(parts) > 3:
                            cpu_out[batch[i]] = float(parts[3])
                        done_upto.add(i)
                        started = None
                        limit = time.monotonic() + 30.0
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if finished:
            break
        # The child hung or died. Blame the running test (or, if none was
        # running, give up on what is left: something below the tests broke).
        if started is not None:
            verdicts[batch[started]] = "timeout" if time.monotonic() >= limit else "crash"
            done_upto.add(started)
            remaining = [remaining[i] for i in range(len(batch)) if i not in done_upto]
            if log:
                log(f"child stopped on {batch[started]}: {verdicts[batch[started]]}")
        else:
            if log:
                log("child died outside a test; leaving the rest unverified")
            break
    return verdicts


# --------------------------------------------------------------------------
# Fork-based verification: one warm, single-threaded parent forks a child per
# commit; up to `parallel` children run at once, multiplexed with select().
# --------------------------------------------------------------------------

BETWEEN_TESTS_S = 10.0  # module import / class fixture allowance between tests

WARM_MODULES = (
    "unittest",
    "unittest.mock",
    "concurrent.futures",
    "concurrent.futures.process",
    "multiprocessing",
    "json",
    "decimal",
    "datetime",
    "numpy",
    "pandas",
    "pandas.testing",
    "duckdb",
)


def warm_up() -> None:
    """Import the third-party modules tests use, once, before any fork.

    Only modules a patch cannot change are imported here: never sqlglot or the
    tests. BLAS thread pools are disabled first so the process stays
    single-threaded and safe to fork.
    """
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import importlib

    for name in WARM_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            pass
    import gc

    gc.collect()
    gc.freeze()


class Job:
    def __init__(self, key: str, root: Path, ids: list[str], allowance: list[float]):
        self.key = key
        self.root = root
        self.ids = ids
        self.allowance = allowance
        self.verdicts: dict[str, str] = {}
        self.cpu: dict[str, float] = {}
        self.gave_up = False


def _fork_child(root: Path, ids: list[str]) -> tuple[int, int]:
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(r)
            devnull = os.open(os.devnull, os.O_RDWR)
            for fd in (0, 1, 2):
                os.dup2(devnull, fd)
            out = os.fdopen(w, "w", buffering=1)

            def emit(line: str) -> None:
                out.write(line + "\n")
                out.flush()

            os.environ["SKIP_INTEGRATION"] = "0"
            from runner_child import run

            run(str(root), ids, emit)
        except BaseException:  # noqa: BLE001
            pass
        finally:
            os._exit(0)
    os.close(w)
    return pid, r


def run_jobs(jobs: list[Job], parallel: int, deadline: float, log=None) -> None:
    import signal
    from collections import deque

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
            os.waitpid(st["pid"], 0)
        except ChildProcessError:
            pass
        job: Job = st["job"]
        if st["finished"]:
            return
        if st["started"] is not None:
            tid = job.ids[st["batch"][st["started"]]]
            job.verdicts[tid] = "timeout" if timed_out else "crash"
            if log:
                log(f"{job.key}: child stopped on {tid}: {job.verdicts[tid]}")
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
            pid, fd = _fork_child(job.root, [job.ids[i] for i in batch])
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
        now = time.monotonic()
        for fd in [fd for fd, st in active.items() if st["limit"] <= now]:
            finish(fd, True)
