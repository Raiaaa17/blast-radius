"""Build-time timing pass: run every test through the runtime runner on the
pristine tree, untraced, recording its verdict and CPU seconds.

This is what verifying a test at runtime costs, and it checks that the runner
passes every test on the unpatched tree (a test that does not is marked
unreliable and is never trusted to have passed).

Usage: python build_timing.py <sqlglot-root> <maps-dir> [--jobs N]
"""

from __future__ import annotations

import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import verify  # noqa: E402


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    with open(out / "maps.pkl", "rb") as handle:
        maps = pickle.load(handle)
    ids: list[str] = maps["test_ids"]
    traced = dict(zip(ids, maps["seconds"]))
    # Keep each file's tests together (one import per file per shard).
    by_file: dict[str, list[str]] = {}
    for tid in ids:
        by_file.setdefault(tid.split("::", 1)[0], []).append(tid)
    shards: list[list[str]] = [[] for _ in range(jobs)]
    loads = [0.0] * jobs
    for rel, tids in sorted(by_file.items(), key=lambda kv: -sum(traced[t] for t in kv[1])):
        i = loads.index(min(loads))
        shards[i].extend(tids)
        loads[i] += sum(traced[t] for t in tids)

    started = time.perf_counter()
    deadline = time.monotonic() + 400.0

    def run(shard: list[str]):
        cpu: dict[str, float] = {}
        verdicts = verify.run_tests(
            src, shard, [max(60.0, 5 * traced[t]) for t in shard], deadline, cpu_out=cpu
        )
        return verdicts, cpu

    timing: dict[str, tuple[str, float]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for verdicts, cpu in pool.map(run, [s for s in shards if s]):
            for tid, status in verdicts.items():
                timing[tid] = (status, cpu.get(tid, traced.get(tid, 1.0)))
    with open(out / "timing.pkl", "wb") as handle:
        pickle.dump(timing, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bad = sorted(tid for tid, (status, _) in timing.items() if status != "passed")
    missing = sorted(set(ids) - set(timing))
    total = sum(c for _, c in timing.values())
    print(
        f"[timing] {len(timing)} tests, {total:.1f} cpu-s, {time.perf_counter() - started:.1f}s wall; "
        f"not passing: {bad}; no verdict: {missing}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
