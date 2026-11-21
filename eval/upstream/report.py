#!/usr/bin/env python3
"""Summarize a labeled upstream corpus: counts by kind/area/portion, how many break
something, and failures far from the edited code.

    report.py <corpus-dir> [--far N]

"Near" test files for an edited path are a coarse naming guess (dialect/parser/
generator/tokenizer/typing module X -> tests/dialects/test_X.py; sqlglot/<mod>.py ->
tests/test_<mod>.py; optimizer/* -> test_optimizer.py; executor/* -> test_executor.py;
a test file -> itself; a fixture under tests/fixtures/<d>/ -> any test). A failing
test outside every near file of the patch is reported as "far". This is a description
of the corpus for humans, not a rule to select by.
"""

from __future__ import annotations

import argparse
import collections
import json
import os


def near_files(path: str) -> set[str]:
    out = set()
    parts = path.split("/")
    if path.startswith("tests/fixtures/"):
        return {"*"}
    if path.startswith("tests/"):
        return {path}
    if path.startswith("sqlglot/") and len(parts) == 3 and parts[1] in ("dialects", "parsers", "generators", "tokenizers", "typing"):
        name = parts[2][:-3]
        out.add(f"tests/dialects/test_{name}.py")
        if parts[1] == "typing":
            out.add("tests/test_optimizer.py")
            out.add("tests/test_typing.py")
    elif path.startswith("sqlglot/optimizer/"):
        out.add("tests/test_optimizer.py")
    elif path.startswith("sqlglot/executor/"):
        out.add("tests/test_executor.py")
    elif path.startswith("sqlglot/expressions"):
        out.add("tests/test_expressions.py")
    elif path.startswith("sqlglot/") and len(parts) == 2:
        out.add(f"tests/test_{parts[1][:-3].lstrip('_')}.py")
        if parts[1] in ("tokens.py", "tokenizer_core.py"):
            out.add("tests/test_tokens.py")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--far", type=int, default=25, help="list up to N far examples")
    args = ap.parse_args()
    c = args.corpus
    manifest = json.load(open(os.path.join(c, "manifest.json")))["patches"]
    tests = json.load(open(os.path.join(c, "tests.json")))["tests"]
    by_num = {t["number"]: t for t in tests}
    answers = json.load(open(os.path.join(c, "answers.json")))["commits"] if os.path.exists(os.path.join(c, "answers.json")) else {}
    labels_dir = os.path.join(c, "labels")
    apply_err = []
    for name in os.listdir(labels_dir) if os.path.isdir(labels_dir) else []:
        d = json.load(open(os.path.join(labels_dir, name)))
        if "apply_error" in d:
            apply_err.append(d["commit"])

    def table(title, keyf):
        rows = collections.defaultdict(lambda: [0, 0, 0, 0.0])
        for pid, m in manifest.items():
            for key in keyf(m):
                r = rows[key]
                r[0] += 1
                if pid in answers:
                    r[1] += 1
                    f = answers[pid]["failing"]
                    if f:
                        r[2] += 1
                        r[3] += len(f)
        print(f"\n### {title}\n\n| {title} | patches | labeled | break >=1 | mean failing (breaking) |\n| --- | --- | --- | --- | --- |")
        for key, (n, lab, brk, nf) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
            print(f"| {key} | {n} | {lab} | {brk} | {nf / brk if brk else 0:.1f} |")

    print(f"patches {len(manifest)}, labeled {len(answers)}, apply errors {len(apply_err)} {apply_err[:5]}")
    brk = sum(1 for a in answers.values() if a["failing"])
    print(f"break at least one test: {brk} of {len(answers)}")
    table("kind", lambda m: [m["kind"]])
    table("portion", lambda m: [f'{m["direction"]}-{m["portion"]}'])
    table("area", lambda m: sorted({f["area"] for f in m["files"]}))

    sizes = collections.Counter()
    for m in manifest.values():
        n = m["added"] + m["removed"]
        sizes["<=2" if n <= 2 else "<=10" if n <= 10 else "<=50" if n <= 50 else "<=200" if n <= 200 else ">200"] += 1
    print("\nchanged lines:", dict(sizes))

    far = []
    big = []
    for pid, a in answers.items():
        m = manifest.get(pid)
        if not m or not a["failing"]:
            continue
        near = set().union(*(near_files(f["path"]) for f in m["files"]))
        files = collections.Counter(by_num[n]["file"] for n in a["failing"])
        if "*" not in near:
            off = {f: k for f, k in files.items() if f not in near}
            if off:
                far.append((pid, [f["path"] for f in m["files"]], off))
        if len(a["failing"]) >= 20:
            big.append((pid, len(a["failing"]), len(files)))
    print(f"\n### far failures: {len(far)} breaking patches fail a test outside their near files")
    for pid, paths, off in far[: args.far]:
        print(f"- {pid} [{manifest[pid]['kind']}] edits {', '.join(paths[:4])}{' ...' if len(paths) > 4 else ''} -> {dict(off)}")
    print(f"\n### wide breakage (>=20 failing tests): {len(big)}")
    for pid, n, nf in sorted(big, key=lambda x: -x[1])[: args.far]:
        print(f"- {pid} [{manifest[pid]['kind']}]: {n} tests in {nf} files; edits {[f['path'] for f in manifest[pid]['files']][:4]}")
    expensive = {t["number"] for t in tests if t["seconds"] >= 3}
    hits = collections.Counter(by_num[n]["id"] for a in answers.values() for n in a["failing"] if n in expensive)
    print("\n### expensive tests (>=3 s) that fail:", dict(hits))


if __name__ == "__main__":
    main()
