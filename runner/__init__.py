"""br-runner: verify which of a given set of sqlglot tests fail under a patch.

    from runner import Verifier
    v = Verifier()                      # starts warm zygote lane(s) once per batch
    r = v.verify(patch_text, [12, 845, 901])
    r.failing   # {12}                  test numbers (or ids, matching the input type)
    r.passed    # {845}                 cleanly passed or skipped: cannot fail
    r.unknown   # {901}                 not decided (crash budget, not collected, ...):
                #                       NEVER treat as passed
    r.cpu_s     # CPU seconds of the verification child(ren)
    v.close()

Batch API (lanes run in parallel, results in input order):
    results = v.verify_many([(patch_text, tests), ...])

See runner/README.md for semantics, costs and configuration.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass, field

DEFAULT_PRISTINE_CANDIDATES = ("/opt/sqlglot", "/app/sqlglot", "/sqlglot")
DEFAULT_TESTS_JSON = "/data/tests.json"
DEFAULT_TMP = "/tmp/br-runner"


def _find_pristine() -> str:
    env = os.environ.get("BR_SQLGLOT_ROOT")
    if env:
        return env
    for c in DEFAULT_PRISTINE_CANDIDATES:
        if os.path.isdir(os.path.join(c, "sqlglot")) and os.path.isdir(os.path.join(c, "tests")):
            return c
    raise FileNotFoundError("pristine sqlglot checkout not found; set BR_SQLGLOT_ROOT")


@dataclass
class Result:
    failing: set = field(default_factory=set)
    passed: set = field(default_factory=set)
    unknown: set = field(default_factory=set)
    skipped: set = field(default_factory=set)  # subset of passed
    why: dict = field(default_factory=dict)  # test -> reason (failing and unknown)
    cpu_s: float = 0.0
    wall_s: float = 0.0
    runs: int = 0
    error: str | None = None
    purge: str | None = None

    def as_dict(self) -> dict:
        return {"failing": self.failing, "passed": self.passed, "unknown": self.unknown,
                "cpu_s": self.cpu_s}


class _Lane:
    def __init__(self, idx, pristine, tmp_root, preload, verbose, python, timeout, prepare):
        lane_dir = os.path.join(tmp_root, f"lane{idx}")
        args = [python, "-m", "runner.zygote", "--pristine", pristine, "--lane-dir", lane_dir]
        if preload:
            args.append("--preload")
        if timeout:
            args += ["--timeout", str(timeout)]
        if not prepare:
            args.append("--no-prepare")
        if verbose:
            args.append("--verbose")
        env = dict(os.environ)
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # the zygote must not see the pristine tree (or anything else named sqlglot/tests)
        env["PYTHONPATH"] = here
        env.setdefault("SKIP_INTEGRATION", "0")
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     env=env, cwd="/", bufsize=0)
        self.busy = None
        self.buf = b""
        self.warm_cpu_s = None

    def send(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        self.proc.stdin.flush()

    def read_line(self, timeout=None):
        """Blocking read of one JSON line."""
        deadline = None if timeout is None else time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.buf:
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("zygote did not answer")
                import select

                r, _, _ = select.select([fd], [], [], left)
                if not r:
                    continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise EOFError("zygote exited")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)


class Verifier:
    def __init__(self, pristine: str | None = None, tests_json: str | None = DEFAULT_TESTS_JSON,
                 lanes: int = 1, timeout: float | None = None, hard_cap_s: float | None = 120.0,
                 workers: int = 1, preload: bool = False, tmp_root: str = DEFAULT_TMP,
                 verbose: bool = False, python: str = sys.executable, start: bool = True,
                 prepare: bool = True, cap_factor: float | None = 10.0, cap_base_s: float = 2.0,
                 as_limit_mb: int | None = None, default_budget_s: float | None = None):
        self.pristine = pristine or _find_pristine()
        self.timeout = timeout
        self.hard_cap_s = hard_cap_s
        self.workers = workers
        self.num_to_id: dict[int, str] = {}
        self.id_to_num: dict[str, int] = {}
        self.seconds: dict[str, float] = {}
        self.cap_factor = cap_factor
        self.cap_base_s = cap_base_s
        self.as_limit_mb = as_limit_mb
        self.default_budget_s = default_budget_s
        if tests_json and os.path.exists(tests_json):
            with open(tests_json) as f:
                for t in json.load(f)["tests"]:
                    self.num_to_id[t["number"]] = t["id"]
                    self.id_to_num[t["id"]] = t["number"]
                    self.seconds[t["id"]] = t["seconds"]
        self._lane_args = (self.pristine, tmp_root, preload, verbose, python, timeout, prepare)
        self.lanes: list[_Lane] = []
        self.warm_cpu_s = 0.0
        self._n_lanes = lanes
        if start:
            self.start()

    def start(self):
        if self.lanes:
            return
        self.lanes = [_Lane(i, *self._lane_args) for i in range(self._n_lanes)]
        for lane in self.lanes:
            hello = lane.read_line(timeout=300)
            lane.warm_cpu_s = hello.get("warm_cpu_s")
            self.warm_cpu_s += hello.get("warm_cpu_s") or 0.0

    # ------------------------------------------------------------------ helpers
    def _to_ids(self, tests) -> tuple[list[str], bool]:
        tests = list(tests)
        numeric = bool(tests) and all(isinstance(t, int) for t in tests)
        if numeric:
            ordered = sorted(set(tests))
            return [self.num_to_id[t] for t in ordered], True
        ids = list(dict.fromkeys(tests))
        if self.id_to_num:
            ids.sort(key=lambda i: self.id_to_num.get(i, 1 << 30))
        return ids, False

    def _request(self, rid, patch, ids, budget_s, **kw) -> dict:
        req = {"rid": rid, "patch": patch, "ids": ids, "timeout": kw.get("timeout", self.timeout),
               "budget_s": budget_s, "hard_cap_s": kw.get("hard_cap_s", self.hard_cap_s),
               "workers": kw.get("workers", self.workers),
               "max_failures": kw.get("max_failures"),
               "max_timeouts": kw.get("max_timeouts", 3)}
        factor = kw.get("cap_factor", self.cap_factor)
        base = kw.get("cap_base_s", self.cap_base_s)
        if factor:
            # per-test verification cap: slower than this -> reported unknown, never passed
            req["caps"] = {i: round(base + factor * self.seconds.get(i, 10.0), 3) for i in ids}
        if budget_s is None:
            req["budget_s"] = self.default_budget_s
        req["as_limit_mb"] = kw.get("as_limit_mb", self.as_limit_mb)
        for k in ("cpu_limit_s", "pytest_args", "spawn_redirect"):
            if kw.get(k) is not None:
                req[k] = kw[k]
        return req

    def _result(self, resp: dict, numeric: bool) -> Result:
        conv = (lambda i: self.id_to_num[i]) if numeric else (lambda i: i)
        r = Result(cpu_s=resp.get("cpu_s", 0.0), wall_s=resp.get("wall_s", 0.0),
                   runs=resp.get("runs", 0), error=resp.get("error"), purge=resp.get("purge"))
        for i, why in (resp.get("failing") or {}).items():
            r.failing.add(conv(i))
            r.why[conv(i)] = why
        for i, why in (resp.get("unknown") or {}).items():
            if conv(i) not in r.failing:
                r.unknown.add(conv(i))
                r.why[conv(i)] = why
        for i in resp.get("passed") or []:
            c = conv(i)
            if c not in r.failing and c not in r.unknown:
                r.passed.add(c)
        for i in resp.get("skipped") or []:
            c = conv(i)
            if c not in r.failing and c not in r.unknown:
                r.passed.add(c)
                r.skipped.add(c)
        return r

    @staticmethod
    def _read_patch(patch) -> str:
        if isinstance(patch, bytes):
            return patch.decode("utf-8", "surrogateescape")
        if isinstance(patch, os.PathLike) or (isinstance(patch, str) and "\n" not in patch
                                              and os.path.exists(patch)):
            with open(patch, "rb") as f:
                return f.read().decode("utf-8", "surrogateescape")
        return patch

    # ------------------------------------------------------------------ public
    def verify(self, patch, tests, budget_s: float | None = None, **kw) -> Result:
        """Run exactly `tests` (numbers or ids) with `patch` applied. Blocking."""
        return self.verify_many([(patch, tests)], budget_s=budget_s, **kw)[0]

    def verify_many(self, items, budget_s: float | None = None, **kw) -> list[Result]:
        """Verify several (patch, tests) pairs, spread over the lanes. Input order kept."""
        self.start()
        items = list(items)
        results: list[Result | None] = [None] * len(items)
        prepared = []
        for idx, (patch, tests) in enumerate(items):
            ids, numeric = self._to_ids(tests)
            if not ids:
                results[idx] = Result()
                continue
            prepared.append((idx, self._read_patch(patch), ids, numeric))
        pending = list(reversed(prepared))
        sel = selectors.DefaultSelector()
        inflight = 0
        try:
            for lane in self.lanes:
                sel.register(lane.proc.stdout, selectors.EVENT_READ, lane)
            while pending or inflight:
                for lane in self.lanes:
                    if lane.busy is None and pending:
                        idx, patch, ids, numeric = pending.pop()
                        lane.busy = (idx, numeric)
                        lane.send(self._request(idx, patch, ids, budget_s, **kw))
                        inflight += 1
                for key, _ in sel.select(timeout=None):
                    lane = key.data
                    chunk = os.read(lane.proc.stdout.fileno(), 65536)
                    if not chunk:
                        raise EOFError("zygote exited unexpectedly")
                    lane.buf += chunk
                    while b"\n" in lane.buf and lane.busy is not None:
                        line, lane.buf = lane.buf.split(b"\n", 1)
                        resp = json.loads(line)
                        idx, numeric = lane.busy
                        results[idx] = self._result(resp, numeric)
                        lane.busy = None
                        inflight -= 1
        finally:
            sel.close()
        return results  # type: ignore[return-value]

    def close(self):
        for lane in self.lanes:
            try:
                lane.send({"op": "quit"})
                lane.proc.stdin.close()
                lane.proc.wait(timeout=10)
            except Exception:
                try:
                    lane.proc.kill()
                except Exception:
                    pass
        self.lanes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def verify(patch_path, test_numbers, tests_json=DEFAULT_TESTS_JSON, budget_s=None, **kw) -> dict:
    """One-shot convenience wrapper (starts and stops a zygote: prefer Verifier for batches)."""
    with Verifier(tests_json=tests_json, **{k: v for k, v in kw.items()
                                             if k in ("pristine", "lanes", "timeout", "workers",
                                                      "preload", "tmp_root", "hard_cap_s")}) as v:
        return v.verify(patch_path, test_numbers, budget_s=budget_s).as_dict()
