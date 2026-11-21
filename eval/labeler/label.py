#!/usr/bin/env python3
"""Reference labeler: which numbered tests does each patch break?

For every `<id>.patch` in a corpus directory it copies a pristine export of
sqlglot at the pinned commit, applies the patch with `git apply`, runs the full
sqlglot suite inside the reference image (eval/ref/Dockerfile: the organizers'
base image digest plus requirements.lock, sqlglot on PYTHONPATH,
SKIP_INTEGRATION=0) and writes the failing test numbers in the kit's
answers.json format.

    eval/labeler/label.py --corpus DIR [--tests sample/tests.json] [--jobs 3]

Writes DIR/answers.json ({"commits": {id: {"failing": [...], "kind": k}}}) and
DIR/labels/<id>.json (full per-patch detail: per-test real seconds, collection
errors, timeouts, vanished tests). Kinds come from DIR/kinds.json when present
({id: kind}), else "unknown". Already-labeled patches are skipped unless
--force, so the labeler can be stopped and resumed.

Semantics (see eval/README.md and findings.md for how they were validated):
a numbered test fails when any of its pytest reports failed (setup/call/
teardown error or failure, a failing subtest, a pytest-timeout timeout) or when
its file could not be collected; a numbered test that no longer exists after
the patch (renamed/deleted) is reported separately as "vanished".
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = Path(os.environ.get("BR_EVAL_DATA", "/home/rayana/firstmate/data/br-eval"))
PINNED = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
IMAGE = os.environ.get("BR_EVAL_IMAGE", "br-eval-ref")


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def ensure_pristine(sqlglot_clone: Path, pristine: Path) -> None:
    if (pristine / ".complete").exists():
        return
    if pristine.exists():
        shutil.rmtree(pristine)
    pristine.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(sqlglot_clone), "archive", PINNED],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(pristine)], input=archive, check=True)
    (pristine / ".complete").write_text(PINNED)


def pytest_command(args, report_path: str) -> list[str]:
    command = [
        "python",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "brreport",
        "-q",
        "--continue-on-collection-errors",
        f"--timeout={args.timeout}",
    ]
    if args.xdist:
        command += ["-n", str(args.xdist)]
    command += ["tests"]
    return command


def per_file_command(args, tree: Path) -> list[str]:
    """One pytest process per test file, in collection order (results go to the stream)."""
    files = sorted(
        str(p.relative_to(tree)) for p in (tree / "tests").rglob("test_*.py") if "fixtures" not in p.parts
    )
    single = " ".join(pytest_command(args, "")[:-1])
    script = f'for f in {" ".join(files)}; do {single} "$f" >/dev/null 2>&1; done; true'
    return ["sh", "-c", script]


def run_suite(args, tree: Path, out_dir: Path, name: str, cpu: str | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o777)
    report = out_dir / f"{name}.report.json"
    for stale in (report, out_dir / f"{name}.report.jsonl"):
        if stale.exists():
            stale.unlink()
    container = f"br-eval-label-{os.getpid()}-{name}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        "--network",
        "none",
    ] + (["--cpuset-cpus", cpu] if cpu else ["--cpus", str(args.cpus)]) + [
        "--memory",
        args.memory,
        "--memory-swap",
        args.memory,
        "--pids-limit",
        "512",
        "--user",
        "65534:65534",
        "-e",
        "HOME=/tmp",
        "-e",
        "PYTHONPATH=/src:/plug",
        "-e",
        "SKIP_INTEGRATION=0",
        "-e",
        f"BR_REPORT=/out/{name}.report.json",
        "-v",
        f"{tree}:/src:ro",
        "-v",
        f"{HERE}:/plug:ro",
        "-v",
        f"{out_dir}:/out",
        IMAGE,
    ] + (
        per_file_command(args, tree) if args.per_file else pytest_command(args, f"/out/{name}.report.json")
    )
    started = time.time()
    killed = False
    with open(out_dir / f"{name}.log", "w") as log_handle:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=args.wall)
        except subprocess.TimeoutExpired:
            killed = True
            subprocess.run(["docker", "kill", container], capture_output=True)
            process.wait()
    result = {"returncode": process.returncode, "wall": time.time() - started, "killed": killed}
    stream = out_dir / f"{name}.report.jsonl"
    if report.exists() and not args.per_file:
        result["report"] = json.loads(report.read_text())
        report.unlink()
    elif stream.exists():
        # Per-file mode, or killed/crashed: rebuild from the stream; when the run did
        # not finish, the tests it never reached are reported as not run.
        tests, errors = {}, []
        for line in stream.read_text().splitlines():
            entry = json.loads(line)
            if "collect_error" in entry:
                errors.append(entry["collect_error"])
                continue
            tests[entry.pop("nodeid")] = entry
        finished = args.per_file and not killed
        result["report"] = {"exitstatus": 0 if finished else -1, "partial": not finished, "tests": tests, "collect_errors": errors}
    if report.exists():
        report.unlink()
    if stream.exists():
        stream.unlink()
    return result


def label_one(args, tests: list[dict], patch: Path, kind: str, cpu: str | None) -> dict:
    commit = patch.stem
    work = DATA / "work" / "trees" / commit
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(DATA / "work" / "pristine", work, symlinks=True)
    detail: dict = {"commit": commit, "kind": kind, "timeout": args.timeout}
    # The tree is not a git repository but may sit inside one; without a
    # ceiling, `git apply` would resolve paths against that outer repository
    # and silently skip every file ("outside the directory").
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(work.parent))
    if patch.read_text().strip():
        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch)],
            cwd=work,
            capture_output=True,
            text=True,
            env=env,
        )
        if applied.returncode == 0:
            # Proof that it applied: the reverse must now apply.
            reverse = subprocess.run(
                ["git", "apply", "--reverse", "--check", str(patch)],
                cwd=work,
                capture_output=True,
                text=True,
                env=env,
            )
            if reverse.returncode != 0:
                applied = subprocess.CompletedProcess([], 1, "", "not applied: " + reverse.stderr)
    else:  # an empty patch is the no-op control
        applied = subprocess.CompletedProcess([], 0, "", "")
    if applied.returncode != 0:
        detail["apply_error"] = applied.stderr[-2000:]
        shutil.rmtree(work)
        return detail
    # The container runs as nobody: make the tree readable.
    subprocess.run(["chmod", "-R", "a+rX", str(work)], check=True)
    run = run_suite(args, work, DATA / "work" / "runs", commit, cpu)
    shutil.rmtree(work)
    detail.update({k: v for k, v in run.items() if k != "report"})
    report = run.get("report")
    by_file: dict[str, list[dict]] = {}
    for test in tests:
        by_file.setdefault(test["file"], []).append(test)
    failing: set[int] = set()
    reasons: dict[int, str] = {}
    vanished: list[int] = []
    seconds: dict[int, float] = {}
    if report is None:
        detail["no_report"] = True
        failing = {test["number"] for test in tests}
        reasons = {n: "no_report" for n in failing}
    else:
        detail["exitstatus"] = report["exitstatus"]
        detail["partial"] = bool(report.get("partial"))
        detail["collect_errors"] = report["collect_errors"]
        broken_files = set()
        for error in report["collect_errors"]:
            node = error["nodeid"].split("::")[0]
            broken_files.add(node)
            if node in ("", "tests"):
                broken_files.update(by_file)
        for test in tests:
            number = test["number"]
            entry = report["tests"].get(test["id"])
            if test["file"] in broken_files:
                failing.add(number)
                reasons[number] = "collect_error"
            elif entry is None and report.get("partial"):
                failing.add(number)
                reasons[number] = "not_run_killed"
            elif entry is None:
                vanished.append(number)
            else:
                seconds[number] = round(entry["seconds"], 3)
                if entry["failed"]:
                    failing.add(number)
                    reasons[number] = "timeout" if entry.get("timeout") else ",".join(
                        o for o in entry["outcomes"] if ":passed" not in o and "skipped" not in o
                    )
        known = {t["id"] for t in tests}
        detail["new_tests"] = sorted(
            nodeid
            for nodeid, entry in report["tests"].items()
            if nodeid not in known and not all("skipped" in o for o in entry["outcomes"])
        )[:200]
    detail["failing"] = sorted(failing)
    detail["reasons"] = {str(k): v for k, v in sorted(reasons.items())}
    detail["vanished"] = vanished
    detail["seconds"] = {str(k): v for k, v in seconds.items()}
    return detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, required=True, help="holds commits/*.patch")
    parser.add_argument("--tests", type=Path, default=REPO / "sample" / "tests.json")
    parser.add_argument("--sqlglot", type=Path, default=DATA / "sqlglot")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--cpus", type=float, default=1.0, help="docker --cpus per run (without --cpuset)")
    parser.add_argument(
        "--cpuset",
        default="9,10,11",
        help="comma list of host CPUs, one pinned per job ('' = use --cpus quota). Pinning matters: "
        "TestOptimizer/TestExecutor spawn os.process_cpu_count() workers, and under a CPU quota on a "
        "many-core host they thrash (test_canonicalize 18 s instead of 1 s)",
    )
    parser.add_argument("--memory", default="3g")
    parser.add_argument("--timeout", type=int, default=300, help="pytest-timeout per test")
    parser.add_argument("--wall", type=int, default=1800, help="hard wall limit per suite run")
    parser.add_argument("--xdist", type=int, default=0, help="pytest -n N (0 = serial)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--per-file", action="store_true", help="one pytest process per test file")
    parser.add_argument("--labels-dir", default="labels", help="label subdirectory name")
    parser.add_argument("--only", nargs="*", help="label only these ids")
    parser.add_argument("--kinds", nargs="*", help="label only these kinds (or kind prefixes like perf)")
    parser.add_argument("--skip-kinds", nargs="*", default=[], help="do not label these kinds/prefixes")
    args = parser.parse_args()

    tests = json.loads(args.tests.read_text())["tests"]
    ensure_pristine(args.sqlglot, DATA / "work" / "pristine")
    commits_dir = args.corpus / "commits"
    labels_dir = args.corpus / args.labels_dir
    labels_dir.mkdir(exist_ok=True)
    kinds_path = args.corpus / "kinds.json"
    kinds = json.loads(kinds_path.read_text()) if kinds_path.exists() else {}
    patches = sorted(commits_dir.glob("*.patch"))
    if args.only:
        patches = [p for p in patches if p.stem in set(args.only)]
    todo = [p for p in patches if args.force or not (labels_dir / f"{p.stem}.json").exists()]

    def kind_matches(kind: str, wanted: list[str]) -> bool:
        return any(kind == w or kind.startswith(w + ":") for w in wanted)

    if args.kinds:
        todo = [p for p in todo if kind_matches(kinds.get(p.stem, "unknown"), args.kinds)]
    todo = [p for p in todo if not kind_matches(kinds.get(p.stem, "unknown"), args.skip_kinds)]
    # Round-robin over kinds so a partial run already covers every kind.
    rank: dict[str, int] = {}
    order = []
    for patch in todo:
        kind = kinds.get(patch.stem, "unknown")
        rank[kind] = rank.get(kind, -1) + 1
        order.append((rank[kind], kind, patch.stem, patch))
    todo = [item[-1] for item in sorted(order)]
    log(f"{len(patches)} patches, {len(todo)} to label, {args.jobs} jobs")

    slots: queue.Queue = queue.Queue()
    cpus = [c for c in args.cpuset.split(",") if c]
    for index in range(args.jobs):
        slots.put(cpus[index % len(cpus)] if cpus else None)

    def work(patch: Path) -> str:
        cpu = slots.get()
        try:
            detail = label_one(args, tests, patch, kinds.get(patch.stem, "unknown"), cpu)
        finally:
            slots.put(cpu)
        (labels_dir / f"{patch.stem}.json").write_text(json.dumps(detail, indent=1))
        status = detail.get("apply_error") and "APPLY-ERROR" or f"{len(detail['failing'])} failing"
        log(f"{patch.stem} [{detail['kind']}] {status} wall={detail.get('wall', 0):.0f}s")
        return patch.stem

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in concurrent.futures.as_completed([pool.submit(work, p) for p in todo]):
            future.result()

    write_answers(args.corpus, labels_dir)
    return 0


def write_answers(corpus: Path, labels_dir: Path) -> None:
    """Collect labels/<id>.json into answers.json, skipping unappliable patches."""
    commits = {}
    for path in sorted(labels_dir.glob("*.json")):
        detail = json.loads(path.read_text())
        # Unappliable patches and runs killed at the wall limit have no exact label.
        if "apply_error" in detail or "failing" not in detail or detail.get("partial") or detail.get("no_report"):
            continue
        commits[detail["commit"]] = {"failing": detail["failing"], "kind": detail["kind"]}
    name = "answers.json" if labels_dir.name == "labels" else f"answers-{labels_dir.name}.json"
    (corpus / name).write_text(json.dumps({"commits": commits}, indent=1) + "\n")
    log(f"answers.json: {len(commits)} labeled commits")


if __name__ == "__main__":
    sys.exit(main())
