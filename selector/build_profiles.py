"""Build-time comparison-site profiles of the expensive tests (see profiles.py).

Usage: python build_profiles.py <sqlglot-root> <maps-dir> [--jobs N] [--min-seconds S]

Each worker is a fresh interpreter that imports sqlglot through an
instrumenting loader and runs its share of the expensive tests. Comparisons
evaluated between tests (class fixtures, test-module imports) are attributed
to the test classes on both sides, like the coverage map does.
"""

from __future__ import annotations

import ast
import builtins
import importlib.abc
import importlib.machinery
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import profiles  # noqa: E402


class InstrumentingLoader(importlib.machinery.SourceFileLoader):
    recorder: profiles.Recorder
    root: str

    def source_to_code(self, data, path, *, _optimize=-1):
        rel = os.path.relpath(path, self.root)
        try:
            tree = ast.parse(data, filename=path)
            tree = profiles.Instrument(self.recorder, rel).visit(tree)
            ast.fix_missing_locations(tree)
            return compile(tree, path, "exec", dont_inherit=True)
        except Exception:  # noqa: BLE001 - then nothing in this file can be proven
            self.recorder.uninstrumented.add(rel)
            return compile(data, path, "exec", dont_inherit=True)

    def get_code(self, fullname):  # never use the cached bytecode
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class InstrumentingFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root: str) -> None:
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sqlglot" and not fullname.startswith("sqlglot."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            return spec
        spec.loader = InstrumentingLoader(spec.loader.name, spec.loader.path)
        return spec


def worker(root: Path, ids: list[str], out: Path) -> None:
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    recorder = profiles.Recorder()
    InstrumentingLoader.recorder = recorder
    InstrumentingLoader.root = str(root)
    sys.meta_path.insert(0, InstrumentingFinder(str(root)))
    builtins.__brcmp__ = recorder.make_hook()
    from runner_child import run

    own: dict[int, dict] = {}
    status: dict[int, str] = {}
    gaps: list[tuple] = []
    state = {"prev": None, "cur": None, "caches": None}
    from cachereset import reset as reset_caches
    recorder.bucket = {}

    def emit(line: str) -> None:
        parts = line.split()
        if parts[0] == "S":
            i = int(parts[1])
            if state["caches"] is None:
                from cachereset import find_caches

                state["caches"] = find_caches()
            reset_caches(state["caches"])
            gaps.append((state["prev"], i, recorder.bucket))
            recorder.bucket = {}
            recorder.seen = set()
            state["cur"] = i
        elif parts[0] == "E":
            i = int(parts[1])
            status[i] = parts[2]
            if state["cur"] == i:
                own[i] = recorder.bucket
                recorder.bucket = {}
                recorder.seen = set()
                state["prev"], state["cur"] = i, None

    started = time.process_time()
    setter_files, order_of = [], {}
    if os.environ.get("BR_SETTERS"):
        with open(os.environ["BR_SETTERS"]) as handle:
            got = json.load(handle)
        setter_files, order_of = got["setters"], got["order"]
    run(str(root), ids, emit, fresh=False, setters=setter_files, order_of=order_of)
    gaps.append((state["prev"], None, recorder.bucket))
    recorder.bucket = None

    def cls(i):
        return ids[i].rsplit("::", 1)[0]

    extra: dict[str, dict] = {}
    for a, b, bucket in gaps:
        for i in (a, b):
            if i is None:
                continue
            target = extra.setdefault(cls(i), {})
            for site, sigs in bucket.items():
                target.setdefault(site, set()).update(sigs)
    result = {}
    for i, tid in enumerate(ids):
        prof: dict[tuple, frozenset] = {}
        for source in (own.get(i, {}), extra.get(cls(i), {})):
            for site, sigs in source.items():
                key = recorder.sites[site]
                prof[key] = frozenset(prof.get(key, frozenset()) | sigs)
        result[tid] = (status.get(i, "missing"), prof)
    with open(out, "wb") as handle:
        pickle.dump({"tests": result, "uninstrumented": recorder.uninstrumented}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[profiles] {len(ids)} tests in {time.process_time() - started:.1f} cpu-s, "
        f"{len(recorder.sites)} sites; {[(t, s[0]) for t, s in result.items()]}",
        file=sys.stderr,
    )


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--worker":
        worker(Path(args[1]), json.loads(args[3]), Path(args[2]))
        return
    root, out = Path(args[0]), Path(args[1])
    jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 4
    min_s = float(args[args.index("--min-seconds") + 1]) if "--min-seconds" in args else 0.5
    tests_json = Path(args[args.index("--tests") + 1]) if "--tests" in args else HERE / "sample_tests.json"
    recorded = {t["id"]: t["seconds"] for t in json.loads(tests_json.read_text())["tests"]}
    cpu = {}
    zfile = out / "zygote.pkl"
    if zfile.exists():
        with open(zfile, "rb") as handle:
            cpu = {tid: c for tid, (st, c) in pickle.load(handle)["timing"].items()}
    # test_max_nodes is left out: instrumented it costs 13x (392 cpu-s here),
    # more than the whole build can spare; it is kept whenever it is a candidate.
    skip = {"tests/test_parser.py::TestParser::test_max_nodes"}
    chosen = sorted(
        (tid for tid in recorded
         if tid not in skip and (recorded[tid] >= min_s or cpu.get(tid, 0.0) >= 2 * min_s)),
        key=lambda t: -max(recorded.get(t, 0.0), cpu.get(t, 0.0)),
    )
    weights = {t: max(recorded.get(t, 0.0), cpu.get(t, 0.0)) + 0.002 for t in chosen}
    # Shards keep each file's tests together (one import, one class fixture per worker).
    by_file: dict[str, list[str]] = {}
    for tid in chosen:
        by_file.setdefault(tid.split("::", 1)[0], []).append(tid)
    groups = []
    for rel, tids in by_file.items():
        # A file with one dominant test (tpcds) is split so that test runs alone.
        big = [t for t in tids if weights[t] > 10.0]
        rest = [t for t in tids if t not in big]
        groups += [[t] for t in big]
        chunk: list[str] = []
        for t in rest:  # a heavy file is cut into chunks of about 10 s
            chunk.append(t)
            if sum(weights[x] for x in chunk) > 10.0:
                groups.append(chunk)
                chunk = []
        if chunk:
            groups.append(chunk)
    shards: list[list[str]] = [[] for _ in range(jobs)]
    loads = [0.0] * jobs
    for group in sorted(groups, key=lambda g: -sum(weights[t] for t in g)):
        i = loads.index(min(loads))
        shards[i].extend(group)
        loads[i] += sum(weights[t] for t in group)
    started = time.perf_counter()
    # State setters (setters.py) replay in each shard before the files that follow them,
    # so every shard runs its tests in suite order, in the state the grader's run has.
    position = {tid: i for i, tid in enumerate(recorded)}
    setters_env = {}
    mfile = out / "maps.pkl"
    if mfile.exists():
        with open(mfile, "rb") as handle:
            setter_files = pickle.load(handle).get("setters", [])
        order_of: dict[str, int] = {}
        for tid in recorded:
            order_of.setdefault(tid.split("::", 1)[0], len(order_of))
        sfile = out / "setters_profiles.json"
        sfile.write_text(json.dumps({"setters": setter_files, "order": order_of}))
        setters_env = {"BR_SETTERS": str(sfile)}
    procs = []
    for i, shard in enumerate(s for s in shards if s):
        shard = sorted(shard, key=lambda t: position.get(t, 1 << 30))
        path = out / f"profile{i}.pkl"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(root), str(path), json.dumps(shard)]
        procs.append((subprocess.Popen(cmd, env={**os.environ, "SKIP_INTEGRATION": "0", **setters_env}), path))
    merged: dict = {}
    uninstrumented: set[str] = set()
    for proc, path in procs:
        if proc.wait() != 0:
            raise SystemExit("a profile worker failed")
        with open(path, "rb") as handle:
            part = pickle.load(handle)
        merged.update(part["tests"])
        uninstrumented |= part["uninstrumented"]
        path.unlink()
    (out / "setters_profiles.json").unlink(missing_ok=True)
    # Invert: per site, each distinct signature set and the tests (bits over the
    # coverage map's test order) that produced it.
    with open(out / "maps.pkl", "rb") as handle:
        bit = {tid: i for i, tid in enumerate(pickle.load(handle)["test_ids"])}
    profiled = 0
    by_site: dict[tuple, dict[frozenset, int]] = {}
    for tid, (st, prof) in merged.items():
        if st != "passed" or tid not in bit:
            continue
        b = 1 << bit[tid]
        profiled |= b
        for key, sigs in prof.items():
            entry = by_site.setdefault(key, {})
            entry[sigs] = entry.get(sigs, 0) | b
    sites = {key: list(entry.items()) for key, entry in by_site.items()}
    with open(out / "profiles.pkl", "wb") as handle:
        pickle.dump({"sites": sites, "profiled": profiled, "uninstrumented": uninstrumented}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    bad = [t for t, (st, _) in merged.items() if st != "passed"]
    print(
        f"[profiles] {len(merged)} tests profiled in {time.perf_counter() - started:.1f}s wall; "
        f"not passing (never used): {bad}; size {(out / 'profiles.pkl').stat().st_size / 1e6:.2f} MB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
