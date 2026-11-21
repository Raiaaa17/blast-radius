"""CLI for br-runner (logs and reports go to stderr unless --json-out is given).

  python -m runner verify PATCH TEST [TEST ...]       # numbers or node ids
  python -m runner sample [--commits DIR] [--answers FILE] [--extra N] [--max-seconds S]
        validate verdicts on labeled commits: every labeled failing test must report
        failing, every sampled non-failing test must pass; prints CPU per commit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time

from . import Verifier


def _cpu_all():
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def _parse_tests(vals):
    out = []
    for v in vals:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.append(int(part) if part.isdigit() else part)
    return out


def cmd_verify(a):
    with Verifier(pristine=a.pristine, tests_json=a.tests_json, lanes=1, timeout=a.timeout,
                  workers=a.workers, preload=a.preload, verbose=a.verbose,
                  prepare=not a.no_prepare) as v:
        r = v.verify(a.patch, _parse_tests(a.tests), budget_s=a.budget)
    out = {"failing": sorted(r.failing, key=str), "passed": sorted(r.passed, key=str),
           "unknown": sorted(r.unknown, key=str), "cpu_s": r.cpu_s, "wall_s": r.wall_s,
           "why": {str(k): v for k, v in r.why.items()}, "error": r.error, "purge": r.purge,
           "warm_cpu_s": v.warm_cpu_s}
    print(json.dumps(out, indent=1, default=str), file=sys.stderr)


def cmd_sample(a):
    with open(a.tests_json) as f:
        tests = json.load(f)["tests"]
    with open(a.answers) as f:
        answers = json.load(f)["commits"]
    secs = {t["number"]: t["seconds"] for t in tests}
    rng = random.Random(a.seed)
    cheap = [t["number"] for t in tests if t["seconds"] <= a.max_seconds]
    commits = sorted(answers)
    if a.only:
        commits = [c for c in commits if c in set(a.only.split(","))]
    items = []
    for cid in commits:
        failing = set(answers[cid]["failing"])
        extra = set(rng.sample(cheap, min(a.extra, len(cheap)))) - failing
        if a.all:
            extra = {t["number"] for t in tests} - failing
        items.append((cid, failing, extra))
    t0, c0 = time.monotonic(), _cpu_all()
    v = Verifier(pristine=a.pristine, tests_json=a.tests_json, lanes=a.lanes, timeout=a.timeout,
                 workers=a.workers, preload=a.preload, verbose=a.verbose,
                  prepare=not a.no_prepare)
    warm_wall = time.monotonic() - t0
    print(f"warm: {v.warm_cpu_s:.2f}s cpu (zygote), {warm_wall:.2f}s wall", file=sys.stderr)
    results = v.verify_many(
        [(os.path.join(a.commits, f"{cid}.patch"), sorted(f | e)) for cid, f, e in items],
        budget_s=a.budget)
    v.close()
    total_cpu = _cpu_all() - c0
    bad = 0
    report = []
    sum_child = 0.0
    for (cid, failing, extra), r in zip(items, results):
        missed = failing - r.failing  # labeled failing, not reported failing
        false_fail = (extra & r.failing)
        unk = r.unknown
        rec_s = sum(secs[t] for t in failing | extra)
        unsafe = missed - unk  # labeled failing but reported passed: would be a 300 s miss
        ok = not missed and not false_fail and not unk
        bad += bool(unsafe or false_fail)
        status = "OK" if ok else ("UNSAFE" if unsafe else ("FALSE-FAIL" if false_fail else "UNKNOWN"))
        sum_child += r.cpu_s
        line = (f"{cid} {answers[cid]['kind']:8s} n={len(failing | extra):4d} fail={len(failing):3d} "
                f"cpu={r.cpu_s:6.2f}s wall={r.wall_s:6.2f}s rec={rec_s:6.2f}s purge={r.purge} "
                f"runs={r.runs} {status}")
        print(line, file=sys.stderr)
        if not ok:
            for t in sorted(missed):
                print(f"   labeled-failing but {('unknown' if t in unk else 'passed')}: {t} "
                      f"{r.why.get(t, '')[:200]}", file=sys.stderr)
            for t in sorted(false_fail):
                print(f"   reported failing but labeled passing: {t} {r.why.get(t, '')[:300]}",
                      file=sys.stderr)
            for t in sorted(unk - missed):
                print(f"   unknown: {t} {r.why.get(t, '')[:200]}", file=sys.stderr)
        report.append({"commit": cid, "cpu_s": r.cpu_s, "wall_s": r.wall_s, "ok": ok,
                       "missed": sorted(missed), "false_fail": sorted(false_fail),
                       "unknown": sorted(unk), "n": len(failing | extra)})
    wall = time.monotonic() - t0
    print(f"commits={len(items)} unsafe_or_false_fail={bad} child_cpu={sum_child:.2f}s "
          f"total_cpu(incl zygote+api)={total_cpu:.2f}s wall={wall:.2f}s", file=sys.stderr)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(report, f, indent=1)
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m runner")
    ap.add_argument("--pristine", default=None)
    ap.add_argument("--tests-json", default="/data/tests.json")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--no-prepare", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("verify")
    p.add_argument("patch")
    p.add_argument("tests", nargs="+")
    p = sub.add_parser("sample")
    p.add_argument("--commits", default="/data/commits")
    p.add_argument("--answers", default="/app/sample/answers.json")
    p.add_argument("--extra", type=int, default=40)
    p.add_argument("--max-seconds", type=float, default=0.3)
    p.add_argument("--all", action="store_true", help="run every test on every commit")
    p.add_argument("--only", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lanes", type=int, default=1)
    p.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        return cmd_verify(a)
    return cmd_sample(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
