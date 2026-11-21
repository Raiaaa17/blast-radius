#!/usr/bin/env python3
"""Compare a labeled corpus against reference answers (e.g. the kit's sample/answers.json).

    eval/labeler/compare.py <corpus-dir> <answers.json>

Prints, per commit present in both, whether the failing sets are identical, and
the extra/missing test numbers otherwise. Exit 1 on any difference.
"""

import json
import sys
from pathlib import Path


def main() -> int:
    corpus, reference = Path(sys.argv[1]), Path(sys.argv[2])
    expected = json.loads(reference.read_text())["commits"]
    labels = {p.stem: json.loads(p.read_text()) for p in (corpus / "labels").glob("*.json")}
    same = differ = 0
    for commit in sorted(expected):
        if commit not in labels:
            continue
        got = set(labels[commit].get("failing", []))
        want = set(expected[commit]["failing"])
        if got == want:
            same += 1
            print(f"OK    {commit} {len(want)} failing")
        else:
            differ += 1
            print(f"DIFF  {commit} extra={sorted(got - want)} missing={sorted(want - got)}")
    print(f"{same} identical, {differ} different, {len(set(expected) - set(labels))} not labeled")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
