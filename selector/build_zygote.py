"""Build-time pass through the runtime's warm verifier, on the pristine tree.

* Imports sqlglot and every test module under an import trace, the way the
  runtime's warm process does: which code runs at import (a change there
  cannot be swapped into a live process).
* Forks children that swap EVERY function's code object for a fresh compile
  of the same source (an identity patch, which exercises exactly the runtime's
  hot-patch mechanics) and run every test, untraced: each test's verdict and
  CPU seconds, and each test class's fixture CPU. A test that does not pass
  here is never trusted to have passed at run time.
* Measures what a fresh child (sqlglot and one test module imported from
  disk) costs, per test module.

Usage: python build_zygote.py <sqlglot-root> <maps-dir> [--jobs N]
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import zygote  # noqa: E402


def test_modules(src: Path) -> list[str]:
    names = []
    for path in sorted((src / "tests").rglob("test_*.py")):
        rel = path.relative_to(src)
        names.append(".".join(rel.with_suffix("").parts))
    return names


def identity_patches(warm: zygote.Warm, src: Path) -> list:
    patches = []
    for base in ("sqlglot", "tests"):
        for path in sorted((src / base).rglob("*.py")):
            data = path.read_bytes()
            try:
                code = compile(data, str(path), "exec", dont_inherit=True)
            except SyntaxError:
                continue
            stack = [code]
            while stack:
                c = stack.pop()
                stack.extend(k for k in c.co_consts if isinstance(k, type(code)))
                funcs = warm.functions.get(zygote.code_key(c))
                if funcs:
                    patches.append((funcs, c))
    return patches


def fresh_costs(src: Path, modules: list[str]) -> dict[str, float]:
    """CPU of importing sqlglot plus one test module from disk, in a forked child."""
    import resource

    costs: dict[str, float] = {}
    for name in [""] + modules:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(r)
                t0 = time.process_time()
                sys.path.insert(0, str(src))
                os.chdir(src)
                from inline_pool import install_inline_pool

                install_inline_pool()
                import importlib

                importlib.import_module("sqlglot")
                if name:
                    try:
                        importlib.import_module(name)
                    except BaseException:  # noqa: BLE001
                        pass
                os.write(w, f"{time.process_time() - t0:.4f}".encode())
            finally:
                os._exit(0)
        os.close(w)
        data = os.read(r, 100)
        os.close(r)
        os.waitpid(pid, 0)
        try:
            costs[name] = float(data.decode())
        except ValueError:
            costs[name] = 2.0
    base = costs.pop("")
    for name in costs:
        costs[name] = max(0.0, costs[name] - base)
    costs[""] = base
    return costs


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    with open(out / "maps.pkl", "rb") as handle:
        maps = pickle.load(handle)
    ids: list[str] = maps["test_ids"]
    traced = dict(zip(ids, maps["seconds"]))
    started = time.perf_counter()

    zygote.warm_third_party()
    modules = test_modules(src)
    costs = fresh_costs(src, modules)
    print(f"[zygote] fresh import costs measured in {time.perf_counter() - started:.1f}s; sqlglot {costs['']:.2f}s", file=sys.stderr)

    warm = zygote.Warm(src)
    warm.load(modules)
    patches = identity_patches(warm, src)
    print(
        f"[zygote] warm: {len(warm.import_keys)} code objects ran at import, {len(warm.functions)} live "
        f"function codes, {len(patches)} identity patches, failed imports {sorted(warm.failed_imports)}",
        file=sys.stderr,
    )

    # Shards keep each file's tests together, balanced by traced seconds.
    by_file: dict[str, list[str]] = {}
    for tid in ids:
        by_file.setdefault(tid.split("::", 1)[0], []).append(tid)
    shards: list[list[str]] = [[] for _ in range(jobs)]
    loads = [0.0] * jobs
    for rel, tids in sorted(by_file.items(), key=lambda kv: -sum(traced[t] for t in kv[1])):
        i = loads.index(min(loads))
        shards[i].extend(tids)
        loads[i] += sum(traced[t] for t in tids)
    job_list = []
    for i, shard in enumerate(s for s in shards if s):
        job_list.append(
            zygote.Job(f"shard{i}", shard, [max(120.0, 6 * traced[t]) for t in shard], "hot", src, patches=patches)
        )
    zygote.run_jobs(job_list, parallel=jobs, deadline=time.monotonic() + 480.0, log=lambda m: print(m, file=sys.stderr))
    timing: dict[str, tuple[str, float]] = {}
    class_cpu: dict[str, float] = {}
    for job in job_list:
        for tid, status in job.verdicts.items():
            timing[tid] = (status, job.cpu.get(tid, traced.get(tid, 1.0)))
        for cls, cpu in job.class_cpu.items():
            class_cpu[cls] = max(class_cpu.get(cls, 0.0), cpu)
    data = {
        "import_keys": {(os.path.relpath(f, src), q, n) for f, q, n in warm.import_keys},
        "timing": timing,
        "class_cpu": class_cpu,
        "fresh_import": costs,
    }
    with open(out / "zygote.pkl", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    bad = sorted(tid for tid, (status, _) in timing.items() if status != "passed")
    missing = sorted(set(ids) - set(timing))
    total = sum(c for _, c in timing.values())
    print(
        f"[zygote] {len(timing)} tests, {total:.1f} cpu-s, {sum(class_cpu.values()):.1f} fixture cpu-s, "
        f"{time.perf_counter() - started:.1f}s wall; not passing: {bad}; no verdict: {missing}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
