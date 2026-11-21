#!/usr/bin/env python3
"""Summarize a labeled corpus per kind: how often patches break tests and how.

    eval/labeler/summary.py <corpus-dir> [--labels-dir labels]

Per kind: patches labeled, patches breaking >= 1 test, median/max failing
count, failure reasons (call failure, subtest, collection error, timeout,
killed), vanished tests, unappliable patches, and the slowest real suite wall.
Aggregate only: it deliberately prints no test lists per kind.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--labels-dir", default="labels")
    args = parser.parse_args()
    rows: dict[str, list[dict]] = collections.defaultdict(list)
    for path in sorted((args.corpus / args.labels_dir).glob("*.json")):
        detail = json.loads(path.read_text())
        rows[detail["kind"]].append(detail)
    print(f"{'kind':<24}{'n':>4}{'break':>6}{'med':>6}{'max':>6}{'vanish':>7}{'apply!':>7}{'wall':>6}  reasons")
    totals = collections.Counter()
    for kind in sorted(rows):
        details = rows[kind]
        applied = [d for d in details if "failing" in d]
        counts = [len(d["failing"]) for d in applied]
        reasons = collections.Counter()
        for d in applied:
            for reason in set(d.get("reasons", {}).values()):
                reasons[reason.split(",")[0]] += 1
        vanished = sum(1 for d in applied if d.get("vanished"))
        breaking = sum(1 for c in counts if c)
        totals.update(n=len(details), breaking=breaking)
        print(
            f"{kind:<24}{len(details):>4}{breaking:>6}"
            f"{(statistics.median(counts) if counts else 0):>6.0f}{(max(counts) if counts else 0):>6}"
            f"{vanished:>7}{len(details) - len(applied):>7}{max((d.get('wall', 0) for d in applied), default=0):>6.0f}  "
            + ", ".join(f"{r}={n}" for r, n in reasons.most_common(4))
        )
    print(f"total {totals['n']} labeled, {totals['breaking']} break at least one test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
