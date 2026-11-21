#!/usr/bin/env python3
"""Export the labeled part of a labeling dir as a kit-shaped corpus.

    export_labeled.py <labeling-dir> <out-dir> [--ids a,b,...] [--max-bytes N]

Copies every patch that has a successful label (labels/<id>.json with "failing", no
apply error) to OUT/commits/, and writes OUT/answers.json in the kit's format
({"commits": {id: {"failing": [...], "kind": k}}}), OUT/manifest.json (same entries as
the labeling dir's, for the exported ids), OUT/tests.json and OUT/labels/. Removes
patches in OUT/commits that are no longer exported, so OUT stays consistent.
--ids restricts the export (used for the small in-repo sample).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--ids", help="comma list: export only these")
    ap.add_argument("--no-labels", action="store_true", help="do not copy labels/<id>.json")
    args = ap.parse_args()

    manifest_all = json.load(open(os.path.join(args.src, "manifest.json")))
    meta = manifest_all["patches"]
    wanted = set(args.ids.split(",")) if args.ids else None
    commits, manifest = {}, {}
    for name in sorted(os.listdir(os.path.join(args.src, "labels"))):
        d = json.load(open(os.path.join(args.src, "labels", name)))
        pid = d.get("commit")
        if "apply_error" in d or "failing" not in d or pid not in meta:
            continue
        if wanted is not None and pid not in wanted:
            continue
        commits[pid] = {"failing": d["failing"], "kind": meta[pid]["kind"]}
        manifest[pid] = meta[pid]

    os.makedirs(os.path.join(args.out, "commits"), exist_ok=True)
    for name in os.listdir(os.path.join(args.out, "commits")):
        if name.endswith(".patch") and name[:-6] not in commits:
            os.remove(os.path.join(args.out, "commits", name))
    if not args.no_labels:
        os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    for pid in commits:
        shutil.copyfile(os.path.join(args.src, "commits", pid + ".patch"), os.path.join(args.out, "commits", pid + ".patch"))
        if not args.no_labels:
            shutil.copyfile(os.path.join(args.src, "labels", pid + ".json"), os.path.join(args.out, "labels", pid + ".json"))
    with open(os.path.join(args.out, "answers.json"), "w") as f:
        json.dump({"commits": commits}, f, indent=1, sort_keys=True)
        f.write("\n")
    header = {k: v for k, v in manifest_all.items() if k != "patches"}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(header | {"patches": manifest}, f, indent=1, sort_keys=True)
        f.write("\n")
    shutil.copyfile(os.path.join(args.src, "tests.json"), os.path.join(args.out, "tests.json"))
    breaking = sum(1 for c in commits.values() if c["failing"])
    print(f"exported {len(commits)} labeled patches ({breaking} break at least one test) to {args.out}")


if __name__ == "__main__":
    main()
