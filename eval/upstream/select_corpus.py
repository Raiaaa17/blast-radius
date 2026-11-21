#!/usr/bin/env python3
"""Pick a diverse, prioritized subset of a mined pool (mine.py output) for labeling.

Labeling runs the full sqlglot suite per patch (~3 min of one core), so the pool is
ranked and the corpus holds the top --limit patches, in priority order:

  1. every fwd patch (real changes made after the pin: the most realistic kind);
  2. rev patches of the --boost-kinds (comment, fixture: rare upstream, common in the
     hidden sets), newest first;
  3. the other rev patches, newest origin commit first, picked greedily so each new pick adds an
     (area, portion) pair not yet covered, then round-robin until --limit.

The ranking uses only patch metadata (paths, portion, direction, recency), never a
test outcome. Writes OUT/commits/*.patch, OUT/manifest.json (with "rank"),
OUT/kinds.json ({id: kind}, read by br-eval's labeler) and OUT/tests.json.

    select_corpus.py --pool POOL --out OUT [--limit 500] [--max-files 40] [--boost-kinds comment,fixture]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-files", type=int, default=40, help="skip patches touching more files")
    ap.add_argument("--boost-kinds", default="comment,fixture", help="rev patches of these kinds rank right after fwd")
    ap.add_argument("--tests-json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sample", "tests.json"))
    args = ap.parse_args()

    pool = json.load(open(os.path.join(args.pool, "manifest.json")))
    patches = pool["patches"]
    # recency: mine.py walks fwd newest-first, then rev newest-first; recover origin order
    origin_order = {}
    for pid, meta in patches.items():
        origin_order.setdefault(meta["origin"], len(origin_order))
    eligible = {pid: m for pid, m in patches.items() if len(m["files"]) <= args.max_files}

    fwd = sorted((p for p, m in eligible.items() if m["direction"] == "fwd"), key=lambda p: (origin_order[eligible[p]["origin"]], p))
    rev = sorted((p for p, m in eligible.items() if m["direction"] == "rev"), key=lambda p: (origin_order[eligible[p]["origin"]], p))

    boost = set(args.boost_kinds.split(",")) - {""}
    chosen = list(fwd[: args.limit])
    chosen += [p for p in rev if eligible[p]["kind"] in boost][: max(0, args.limit - len(chosen))]
    rev = [p for p in rev if p not in set(chosen)]
    covered = {(f["area"], eligible[p]["portion"]) for p in chosen for f in eligible[p]["files"]}
    rest = []
    for p in rev:
        if len(chosen) >= args.limit:
            break
        pairs = {(f["area"], eligible[p]["portion"]) for f in eligible[p]["files"]}
        if pairs - covered:
            chosen.append(p)
            covered |= pairs
        else:
            rest.append(p)
    for p in rest:
        if len(chosen) >= args.limit:
            break
        chosen.append(p)

    os.makedirs(os.path.join(args.out, "commits"), exist_ok=True)
    keep = set(chosen)
    for name in os.listdir(os.path.join(args.out, "commits")):
        if name.endswith(".patch") and name[:-6] not in keep:
            os.remove(os.path.join(args.out, "commits", name))
    out_manifest = {}
    for rank, pid in enumerate(chosen):
        shutil.copyfile(os.path.join(args.pool, "commits", pid + ".patch"), os.path.join(args.out, "commits", pid + ".patch"))
        out_manifest[pid] = dict(eligible[pid], rank=rank)
    json.dump({k: v for k, v in pool.items() if k != "patches"} | {"patches": out_manifest}, open(os.path.join(args.out, "manifest.json"), "w"), indent=1, sort_keys=True)
    json.dump({pid: m["kind"] for pid, m in out_manifest.items()}, open(os.path.join(args.out, "kinds.json"), "w"), indent=1, sort_keys=True)
    tests = os.path.join(args.pool, "tests.json")
    shutil.copyfile(tests if os.path.exists(tests) else args.tests_json, os.path.join(args.out, "tests.json"))
    with open(os.path.join(args.out, "priority.txt"), "w") as f:
        f.write("\n".join(chosen) + "\n")
    print(f"pool={len(patches)} eligible={len(eligible)} chosen={len(chosen)} fwd={len(fwd)}")


if __name__ == "__main__":
    main()
