"""Build-time generation of staticeq/data/: the per-file segment index and the
file-access map (see datamap.py for the map's shape).

    python -m staticeq.build_map --root /opt/sqlglot --out staticeq/data \
        [--tests sample/tests.json] [--jobs 3] [--no-trace]

The trace runs the pristine suite once, sharded across --jobs processes, with
the labeling environment (SKIP_INTEGRATION=0, sqlglot on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)


def run_trace(root: str, work: str, jobs: int, tests_json: str | None, timeout: int,
              python: str) -> list[str]:
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([root, APP] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["SKIP_INTEGRATION"] = "0"
    env.setdefault("PYTHONHASHSEED", "0")
    procs = []
    for k in range(jobs):
        cmd = [python, "-m", "pytest", "-p", "staticeq.trace_plugin", "-p", "no:cacheprovider",
               "-q", "--continue-on-collection-errors", "-o", "addopts=",
               "--staticeq-out", work, "--staticeq-shard", f"{k}/{jobs}", "tests"]
        if tests_json:
            cmd[cmd.index("tests"):cmd.index("tests")] = ["--staticeq-tests", tests_json]
        log = open(os.path.join(work, f"pytest-{k}.log"), "w")
        procs.append((subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    deadline = time.time() + timeout
    for p, log in procs:
        try:
            p.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        log.close()
    return sorted(os.path.join(work, f) for f in os.listdir(work) if f.startswith("trace-"))


def merge(trace_files: list[str], root: str) -> dict:
    items: list[str] = []
    events: list = []
    missing: list = []
    outcomes: dict[str, str] = {}
    imported: set[str] = set()
    for tf in trace_files:
        with open(tf, encoding="utf-8") as f:
            d = json.load(f)
        items.extend(d["items"])
        events.extend(tuple(e) for e in d["events"])
        missing.extend(tuple(m) for m in d["missing"])
        outcomes.update(d["outcomes"])
        imported.update(d["imported"])
    tests = sorted(set(items) | set(outcomes))
    index = {t: i for i, t in enumerate(tests)}

    def expand(phase: str, node: str, reader: str, rel: str) -> list[int] | None:
        """Tests a read in this context can affect; None = every test."""
        if phase == "call":
            return [index[node]] if node in index else []
        if phase in ("setup", "teardown"):
            if node not in index:
                return []
            # class-level fixtures (setUpClass) -> the whole class; setUpModule -> module
            prefix = node.rsplit("::", 1)[0] + "::"
            if "setUpModule" in reader or "tearDownModule" in reader:
                prefix = node.split("::", 1)[0] + "::"
            return [index[t] for t in tests if t.startswith(prefix)] or [index[node]]
        if phase == "collect":
            if reader == "<no repo frame>":
                return []  # pytest's own collection walk (new test files are new tests)
            # import-time read by a shared (non-test) module -> every test
            qual = reader.rsplit(":", 1)[-1]
            reader_file = reader.split(":", 1)[0]
            if qual == "<module>" and not os.path.basename(reader_file).startswith("test_"):
                return None
            prefix = node.rstrip("/") + ("::" if node.endswith(".py") else "/")
            return [index[t] for t in tests if t.startswith(prefix)] or None
        return []

    reads: dict[str, dict] = {}
    source_reads: dict[str, dict] = {}
    listdirs: dict[str, dict] = {}
    writes: dict[str, dict] = {}
    all_tests_reads: set[str] = set()
    for phase, node, kind, rel, reader, tags in events:
        if phase in ("init", "idle"):
            continue
        tgt = expand(phase, node, reader, rel)
        table = {"read": reads, "source-read": source_reads, "listdir": listdirs}.get(kind, writes)
        if tgt is None:
            all_tests_reads.add(rel)
            continue
        e = table.setdefault(rel, {"tests": set(), "readers": set()})
        e["tests"].update(tgt)
        e["readers"].add(reader + (f" [{tags}]" if tags else ""))
    miss: dict[str, set] = {}
    for rec in missing:
        phase, node, name = rec[:3]
        reader = rec[3] if len(rec) > 3 else "<no repo frame>"
        if phase == "collect" and reader == "<no repo frame>":
            reader = "<module>"  # import machinery during collection: treat as shared import
        tgt = expand(phase, node, reader if ":" in reader else f"x:{reader}", "")
        if tgt is None:
            tgt = list(range(len(tests)))
        miss.setdefault(name, set()).update(tgt)

    def fin(table: dict) -> dict:
        return {k: {"tests": sorted(v["tests"]), "readers": sorted(v["readers"])}
                for k, v in sorted(table.items()) if v["tests"]}

    failed = sorted(t for t, o in outcomes.items() if o not in ("passed", "skipped"))
    return {
        "version": 2,
        "tests": tests,
        "reads": fin(reads),
        "source_reads": fin(source_reads),
        "listdirs": fin(listdirs),
        "writes": fin(writes),
        "missing_imports": {k: sorted(v) for k, v in sorted(miss.items())},
        "imported": sorted(imported),
        "all_tests_reads": sorted(all_tests_reads),
        "outcomes": {"run": len(outcomes), "failed": failed,
                     "lost": sorted(t for t, o in outcomes.items() if o in ("lost", "?"))},
    }


def mentions(root: str) -> dict[str, list[str]]:
    """Static safety net for readers the audit hook cannot see (e.g. DuckDB's
    native file reader): test-tree sources that spell a repo file's basename
    (or its path relative to tests/fixtures) literally."""
    sources: dict[str, str] = {}
    data_files: list[str] = []
    for top, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__"))
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(top, fn), root).replace(os.sep, "/")
            if fn.endswith(".py"):
                if rel.startswith("tests/"):
                    with open(os.path.join(top, fn), encoding="utf-8", errors="replace") as f:
                        sources[rel] = f.read()
            else:
                data_files.append(rel)
    out: dict[str, list[str]] = {}
    for rel in data_files:
        needles = {os.path.basename(rel)}
        if rel.startswith("tests/fixtures/"):
            needles.add(rel[len("tests/fixtures/"):])
        hits = sorted(src for src, text in sources.items() if any(n in text for n in needles))
        if hits:
            out[rel] = hits
    return out


def typing_guard(root: str, python: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([root, APP])
    try:
        out = subprocess.run([python, "-m", "staticeq.typing_guard"], cwd=root, env=env,
                             capture_output=True, text=True, timeout=300)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--tests", default=None)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=480)
    ap.add_argument("--work", default=None)
    ap.add_argument("--trace", action="store_true",
                    help="re-run the suite trace and rewrite filemap.json (else keep the committed one)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    from .segindex import build_index

    t = time.time()
    n = build_index(root, os.path.join(out, "segments"))
    print(f"staticeq: indexed {n} python files in {time.time() - t:.1f}s", file=sys.stderr)

    if args.trace:
        work = args.work or os.path.join(out, "trace-work")
        t = time.time()
        files = run_trace(root, work, args.jobs, args.tests, args.timeout, args.python)
        filemap = merge(files, root)
        print(f"staticeq: traced {filemap['outcomes']['run']} tests in {time.time() - t:.1f}s; "
              f"failed={len(filemap['outcomes']['failed'])} lost={len(filemap['outcomes']['lost'])}",
              file=sys.stderr)
        if args.work is None:
            import shutil

            shutil.rmtree(work, ignore_errors=True)
        filemap["mentions"] = mentions(root)
        filemap["typing_guard"] = typing_guard(root, args.python)
        filemap["python"] = sys.version.split()[0]
        with open(os.path.join(out, "filemap.json"), "w", encoding="utf-8") as f:
            json.dump(filemap, f, separators=(",", ":"), sort_keys=True)
    elif not os.path.exists(os.path.join(out, "filemap.json")):
        print("staticeq: WARNING no filemap.json; data-file verdicts will be conservative",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
