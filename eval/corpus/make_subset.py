#!/usr/bin/env python3
"""Copy a small labeled subset of a corpus into the repo (eval/corpus/subset).

    eval/corpus/make_subset.py <corpus-dir> [--out eval/corpus/subset] [--per-kind 1] [--max-bytes 20000]

Takes up to --per-kind exactly-labeled patches of every kind (smallest first,
breaking and non-breaking alternated), and writes subset/commits/*.patch,
subset/answers.json and subset/meta.json, so `eval/score_image.sh <image>
eval/corpus/subset` works from a fresh checkout.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "subset")
    parser.add_argument("--per-kind", type=int, default=1)
    parser.add_argument("--max-bytes", type=int, default=20000)
    args = parser.parse_args()
    answers = json.loads((args.corpus / "answers.json").read_text())["commits"]
    meta_path = args.corpus / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    by_kind: dict[str, list[str]] = {}
    for commit, answer in answers.items():
        patch = args.corpus / "commits" / f"{commit}.patch"
        if patch.stat().st_size <= args.max_bytes:
            by_kind.setdefault(answer["kind"], []).append(commit)
    chosen = []
    for kind, commits in sorted(by_kind.items()):
        commits.sort(key=lambda c: (args.corpus / "commits" / f"{c}.patch").stat().st_size)
        breaking = [c for c in commits if answers[c]["failing"]]
        clean = [c for c in commits if not answers[c]["failing"]]
        merged = [c for pair in zip(breaking, clean) for c in pair] + breaking[len(clean):] + clean[len(breaking):]
        chosen += merged[: args.per_kind]
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "commits").mkdir(parents=True)
    for commit in chosen:
        shutil.copyfile(args.corpus / "commits" / f"{commit}.patch", args.out / "commits" / f"{commit}.patch")
    (args.out / "answers.json").write_text(
        json.dumps({"commits": {c: answers[c] for c in sorted(chosen)}}, indent=1) + "\n"
    )
    (args.out / "meta.json").write_text(
        json.dumps({c: meta[c] for c in sorted(chosen) if c in meta}, indent=1, sort_keys=True) + "\n"
    )
    print(f"{len(chosen)} patches from {len(by_kind)} kinds -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
