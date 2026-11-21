"""CLI: python -m staticeq --root SQLGLOT [--map filemap.json] PATCH... (JSON lines on stdout)."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .analyze import analyze
from .datamap import load_map


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m staticeq")
    ap.add_argument("patches", nargs="+")
    ap.add_argument("--root", required=True, help="pristine sqlglot checkout")
    ap.add_argument("--map", default=None, help="filemap.json (default: built-in location)")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--strict-docstrings", action="store_true",
                    help="treat docstring-only edits as not neutral")
    ap.add_argument("--brief", action="store_true", help="print only neutral + reasons")
    args = ap.parse_args(argv)
    dm = None if args.no_map else load_map(args.map)
    for p in args.patches:
        with open(p, "rb") as f:
            text = f.read()
        t = time.process_time()
        r = analyze(text, args.root, data_map=dm, docstrings_neutral=not args.strict_docstrings)
        r["patch"] = p
        r["cpu_ms"] = round((time.process_time() - t) * 1000, 2)
        if args.brief:
            r = {k: r[k] for k in ("patch", "neutral", "reasons", "cpu_ms")} | {
                "units": [(u["file"], u["qualname"], u["where"]) for u in r["units"]],
                "data_files": [(d["path"], d["status"], len(d["tests"])) for d in r["data_files"]],
            }
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
