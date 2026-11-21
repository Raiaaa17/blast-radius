#!/usr/bin/env python3
"""Score a selector image on a labeled corpus, the way the board does.

    eval/score_image.sh <image> <corpus-dir> [--batch N] [--runs 3] [--limit S]

<corpus-dir> holds commits/<id>.patch and answers.json (the kit's format,
{"commits": {id: {"failing": [...], "kind": k}}}); tests.json is taken from
<corpus-dir>/tests.json when present, else the kit's sample/tests.json.

Each batch is mounted into the image under the scorer's exact docker flags
(copied from the root Makefile) and its stdout is scored with the kit's own
score.py (evaluate_run, imported unchanged), so simulated seconds and misses
are exact. On top of what `make sample` prints it reports:

- CPU seconds of every process in the container, read from the cgroup v2
  cpu.stat of a dedicated systemd slice the container runs under
  (--cgroup-parent), so it is exact and includes the last instant before exit;
- the wall time of each run, against the batch limit (180 s practice,
  270 s final; --limit), since a run over the limit is rejected;
- every miss: commit id, kind, and the failing tests that were left out;
- cost per kind (commits, misses, simulated seconds, mean per commit);
- total = simulated + CPU per run, and the middle of --runs runs.

--batch N splits the corpus into consecutive batches of N commits (sorted by
id), one container run each, which mimics hidden batches of ~40-60 commits;
the default is the whole corpus in one run. Logs go to stderr; a JSON summary
goes to --json when given.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# The scorer's run flags, copied from the root Makefile's RUN_FLAGS.
RUN_FLAGS = [
    "--rm", "--platform", "linux/amd64", "--network", "none", "--cpus", "2",
    "--memory", "4096m", "--memory-swap", "4096m", "--pids-limit", "256",
    "--read-only", "--tmpfs", "/tmp:rw,size=64m,mode=1777", "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges", "--user", "65534:65534", "--env", "HOME=/tmp",
]  # fmt: skip


def load_kit_scorer():
    spec = importlib.util.spec_from_file_location("kit_score", REPO / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_makefile_flags() -> None:
    """Warn when the Makefile's RUN_FLAGS drift from the copy above."""
    text = (REPO / "Makefile").read_text()
    start = text.index("RUN_FLAGS :=")
    block = text[start : text.index("--volume", start)]
    flags = block.replace("RUN_FLAGS :=", "").replace("\\\n", " ").split()
    if flags != RUN_FLAGS:
        print(f"WARNING: Makefile RUN_FLAGS changed: {flags}", file=sys.stderr)


def cgroup_usage_usec(slice_name: str) -> int | None:
    # systemd nests a dashed slice name (a-b.slice lives in a.slice); ours has no dash.
    path = Path("/sys/fs/cgroup") / slice_name / "cpu.stat"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        key, value = line.split()
        if key == "usage_usec":
            return int(value)
    return None


def cgroup_memory_peak_mb(slice_name: str) -> int | None:
    path = Path("/sys/fs/cgroup") / slice_name / "memory.peak"
    try:
        return round(int(path.read_text().split()[0]) / 2**20)
    except (OSError, ValueError, IndexError):
        return None


def run_batch(image: str, batch_dir: Path, limit: float) -> dict:
    slice_name = f"brevalscore{os.getpid()}x{time.time_ns()}.slice"
    name = f"br-eval-score-{os.getpid()}-{time.time_ns()}"
    command = (
        ["docker", "run", "--name", name, "--cgroup-parent", slice_name]
        + RUN_FLAGS
        + [
            "--volume", f"{batch_dir / 'commits'}:/data/commits:ro",
            "--volume", f"{batch_dir / 'tests.json'}:/data/tests.json:ro",
            image,
        ]
    )  # fmt: skip
    started = time.monotonic()
    timed_out = False
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        try:
            # Grace beyond the limit so an over-limit run is still measured.
            process.wait(timeout=limit + 120)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["docker", "kill", name], capture_output=True)
            process.wait()
        wall = time.monotonic() - started
        out.seek(0)
        err.seek(0)
        stdout = out.read()
        stderr = err.read().decode("utf-8", "replace")
    usage = cgroup_usage_usec(slice_name)
    peak = cgroup_memory_peak_mb(slice_name)
    return {
        "memory_peak_mb": peak,
        "stdout": stdout,
        "stderr_tail": stderr[-4000:],
        "returncode": process.returncode,
        "wall": wall,
        "over_limit": wall > limit or timed_out,
        "cpu_usec": usage,
    }


def per_commit(kit, stdout: bytes, tests: list[dict], answers: dict) -> dict:
    """Cost and missed tests per commit, using the kit's own duration rounding."""
    duration = {t["number"]: kit._duration_milliseconds(t["seconds"]) for t in tests}
    result = {}
    for line in kit._split_lines(stdout.decode("utf-8")):
        value = json.loads(line)
        workers = [w for w in value["workers"] if w]
        selected = {n for w in workers for n in w}
        cost = (
            max(sum(duration[n] for n in w) for w in workers) + kit.WORKER_COST_MILLISECONDS * len(workers)
            if workers
            else 0
        )
        missed = sorted(set(answers[value["commit"]]["failing"]) - selected)
        result[value["commit"]] = {"cost_ms": cost, "missed": missed, "workers": len(workers)}
    return result


def oracle_cost_ms(kit, failing: list[int], duration: dict[int, int]) -> int:
    """Floor for one commit: run exactly its failing tests, packed longest-first
    into the best number of workers (1..8). 0 when nothing fails."""
    if not failing:
        return 0
    loads_by_k = []
    ordered = sorted((duration[n] for n in failing), reverse=True)
    for k in range(1, min(kit.MAXIMUM_WORKERS, len(ordered)) + 1):
        loads = [0] * k
        for d in ordered:
            loads[loads.index(min(loads))] += d
        loads_by_k.append(max(loads) + kit.WORKER_COST_MILLISECONDS * k)
    return min(loads_by_k)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--batch", type=int, default=0, help="commits per run (0 = all)")
    parser.add_argument("--runs", type=int, default=1, help="runs per batch; the middle total counts")
    parser.add_argument("--limit", type=float, default=180.0, help="wall seconds allowed per run")
    parser.add_argument("--json", type=Path, help="write a JSON summary here")
    parser.add_argument("--keep-output", type=Path, help="save each run's stdout under this dir")
    args = parser.parse_args()

    check_makefile_flags()
    kit = load_kit_scorer()
    tests_path = args.corpus / "tests.json"
    if not tests_path.exists():
        tests_path = REPO / "sample" / "tests.json"
    tests = json.loads(tests_path.read_text())["tests"]
    answers = json.loads((args.corpus / "answers.json").read_text())["commits"]
    patches = {p.stem: p for p in (args.corpus / "commits").glob("*.patch")}
    ids = sorted(c for c in answers if c in patches)
    unlabeled = sorted(set(patches) - set(answers))
    if unlabeled:
        print(f"note: {len(unlabeled)} patches have no label and are skipped", file=sys.stderr)
    size = args.batch or len(ids)
    batches = [ids[i : i + size] for i in range(0, len(ids), size)]

    summary = {"image": args.image, "corpus": str(args.corpus), "batches": []}
    commit_rows: dict[str, dict] = {}
    totals_by_run = [0] * args.runs
    with tempfile.TemporaryDirectory(prefix="br-eval-score-") as scratch:
        for index, batch in enumerate(batches):
            batch_dir = Path(scratch) / f"batch{index}"
            (batch_dir / "commits").mkdir(parents=True)
            for commit in batch:
                shutil.copyfile(patches[commit], batch_dir / "commits" / f"{commit}.patch")
            shutil.copyfile(tests_path, batch_dir / "tests.json")
            os.chmod(batch_dir, 0o755)
            batch_answers = {c: answers[c] for c in batch}
            runs = []
            for run_index in range(args.runs):
                run = run_batch(args.image, batch_dir, args.limit)
                if args.keep_output:
                    args.keep_output.mkdir(parents=True, exist_ok=True)
                    (args.keep_output / f"batch{index}.run{run_index}.jsonl").write_bytes(run["stdout"])
                evaluation = kit.evaluate_run(run["stdout"], tests, batch_answers)
                run["evaluation"] = evaluation
                cpu = run["cpu_usec"] or 0
                if evaluation["ok"]:
                    run["total_usec"] = kit.run_total_microseconds(evaluation["simulated_milliseconds"], cpu)
                    if run_index == 0:
                        for commit, row in per_commit(kit, run["stdout"], tests, batch_answers).items():
                            commit_rows[commit] = row
                else:
                    run["total_usec"] = None
                print(
                    f"batch {index + 1}/{len(batches)} run {run_index + 1}: {len(batch)} commits, "
                    + (
                        f"simulated {evaluation['simulated_milliseconds'] / 1000:.3f} s, "
                        f"misses {len(evaluation['missed_commits'])}, "
                        if evaluation["ok"]
                        else f"REJECTED {evaluation['code']} line {evaluation['line']}, "
                    )
                    + f"cpu {cpu / 1e6:.3f} s, wall {run['wall']:.1f} s, peak {run['memory_peak_mb']} MB"
                    + (" OVER LIMIT" if run["over_limit"] else ""),
                    file=sys.stderr,
                )
                if not evaluation["ok"] or run["returncode"] != 0:
                    print(run["stderr_tail"], file=sys.stderr)
                runs.append(run)
                totals_by_run[run_index] += run["total_usec"] or 0
            summary["batches"].append(
                {
                    "commits": batch,
                    "runs": [
                        {k: v for k, v in r.items() if k not in ("stdout",)} for r in runs
                    ],
                }
            )

    all_runs = [r for b in summary["batches"] for r in b["runs"]]
    rejected = [r for r in all_runs if not r["evaluation"]["ok"]]
    over = [r for r in all_runs if r["over_limit"]]
    print("", file=sys.stderr)
    if rejected:
        print(f"REJECTED runs: {len(rejected)}", file=sys.stderr)
    if over:
        print(f"runs OVER the {args.limit:.0f} s wall limit: {len(over)}", file=sys.stderr)

    # Per-kind table and miss list come from the first run of each batch.
    kinds: dict[str, dict] = {}
    misses = []
    duration = {t["number"]: kit._duration_milliseconds(t["seconds"]) for t in tests}
    oracle_total = 0
    for commit, row in sorted(commit_rows.items()):
        kind = answers[commit].get("kind", "unknown")
        entry = kinds.setdefault(kind, {"commits": 0, "breaking": 0, "misses": 0, "cost_ms": 0, "oracle_ms": 0})
        row["oracle_ms"] = oracle_cost_ms(kit, answers[commit]["failing"], duration)
        entry["oracle_ms"] += row["oracle_ms"]
        oracle_total += row["oracle_ms"]
        entry["commits"] += 1
        entry["breaking"] += bool(answers[commit]["failing"])
        entry["cost_ms"] += row["cost_ms"]
        if row["missed"]:
            entry["misses"] += 1
            misses.append({"commit": commit, "kind": kind, "missed": row["missed"]})
    simulated_ms = sum(r["cost_ms"] for r in commit_rows.values()) + 300_000 * len(misses)
    cpu_first = sum((b["runs"][0]["cpu_usec"] or 0) for b in summary["batches"])
    lines = [
        f"commits {len(commit_rows)}  batches {len(batches)}",
        f"simulated {simulated_ms / 1000:.3f} s  (selection {(simulated_ms - 300_000 * len(misses)) / 1000:.3f} s"
        f" + misses {len(misses)} x 300 s)",
        f"oracle    {oracle_total / 1000:.3f} s  (only the failing tests, best packing; the floor)",
        f"cpu       {cpu_first / 1e6:.3f} s  (run 1)",
        f"total     {(simulated_ms * 1000 + cpu_first) / 1e6:.3f} s  (run 1)",
    ]
    if args.runs > 1:
        middle = statistics.median_low(totals_by_run) if args.runs % 2 == 0 else statistics.median(totals_by_run)
        lines.append(f"middle of {args.runs} runs: {middle / 1e6:.1f} s  (all: {[round(t / 1e6, 1) for t in totals_by_run]})")
    lines.append("")
    lines.append(f"{'kind':<24}{'commits':>8}{'breaking':>9}{'misses':>7}{'sim s':>10}{'mean s':>8}{'oracle s':>10}")
    for kind, entry in sorted(kinds.items()):
        sim = entry["cost_ms"] / 1000 + 300 * entry["misses"]
        lines.append(
            f"{kind:<24}{entry['commits']:>8}{entry['breaking']:>9}{entry['misses']:>7}{sim:>10.1f}"
            f"{sim / entry['commits']:>8.2f}{entry['oracle_ms'] / 1000:>10.1f}"
        )
    if misses:
        lines.append("")
        lines.append("misses:")
        for miss in misses:
            shown = miss["missed"][:12]
            more = f" (+{len(miss['missed']) - 12})" if len(miss["missed"]) > 12 else ""
            lines.append(f"  {miss['commit']} [{miss['kind']}] missed {shown}{more}")
    print("\n".join(lines))
    summary.update(
        {
            "simulated_ms": simulated_ms,
            "oracle_ms": oracle_total,
            "cpu_usec_run1": cpu_first,
            "totals_usec_by_run": totals_by_run,
            "kinds": kinds,
            "misses": misses,
            "per_commit": commit_rows,
        }
    )
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1, default=str))
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
