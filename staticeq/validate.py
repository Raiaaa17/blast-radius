"""Developer check on the real tree: random mutations, expected verdicts, and
agreement between chunk mode and whole-file mode.

    python -m staticeq.validate --root SQLGLOT --segments DIR [--n 2000] [--seed 1]

Mutation kinds and the verdict each must get:
  comment     full-line comment at a token-safe line boundary   -> neutral
  inline      trailing comment on a line that ends outside strings -> neutral
  blank       blank line at a token-safe boundary                -> neutral
  docstring   reword an existing docstring                       -> docstring-only
  code        flip an operator / bump an int / edit a string const -> not neutral
"""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import json
import os
import random
import sys
import time
import tokenize
from collections import Counter

from .analyze import analyze
from .segindex import load_index


def _diff(path: str, old: str, new: str) -> str:
    lines = difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}")
    body = "".join(x if x.endswith("\n") else x + "\n\\ No newline at end of file\n" for x in lines)
    return f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n" + body


def _string_lines(text: str) -> tuple[set[int], set[int]]:
    """(lines strictly inside or ending a multi-line string, lines with a backslash continuation)."""
    inside: set[int] = set()
    cont: set[int] = set()
    depth_fstring = 0
    fstart = 0
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    for tok in toks:
        if tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
            inside.update(range(tok.start[0] + 1, tok.end[0] + 1))
        if tok.type == getattr(tokenize, "FSTRING_START", -1):
            if depth_fstring == 0:
                fstart = tok.start[0]
            depth_fstring += 1
        if tok.type == getattr(tokenize, "FSTRING_END", -1):
            depth_fstring -= 1
            if depth_fstring == 0 and tok.end[0] != fstart:
                inside.update(range(fstart + 1, tok.end[0] + 1))
    for i, line in enumerate(text.split("\n"), 1):
        if line.rstrip().endswith("\\"):
            cont.add(i)
    return inside, cont


def mutate(text: str, kind: str, rng: random.Random) -> str | None:
    lines = text.split("\n")
    n = len(lines) - (1 if text.endswith("\n") else 0)
    if n < 4:
        return None
    inside, cont = _string_lines(text)
    if kind in ("comment", "blank"):
        for _ in range(50):
            k = rng.randint(3, n)  # insert before line k (keep lines 1-2 for cookies)
            if k in inside or (k - 1) in cont or (k - 1) in inside and k in inside:
                continue
            if kind == "blank":
                ins = ""
            else:
                nxt = lines[k - 1]
                indent = nxt[: len(nxt) - len(nxt.lstrip())] if rng.random() < 0.7 else " " * rng.randint(0, 12)
                ins = f"{indent}# {rng.choice(['note', 'TODO: check', 'Keep in sync', 'x = 1  # not code'])}"
            return "\n".join(lines[: k - 1] + [ins] + lines[k - 1 :])
        return None
    if kind == "inline":
        safe = []
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.NEWLINE, tokenize.NL) and tok.start[0] not in cont and tok.start[0] > 2:
                line = lines[tok.start[0] - 1]
                if line.strip() and "#" not in line:
                    safe.append(tok.start[0])
        if not safe:
            return None
        k = rng.choice(safe)
        lines[k - 1] = lines[k - 1] + "  # inline note"
        return "\n".join(lines)
    tree = ast.parse(text)
    if kind == "docstring":
        docs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                        getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
                    d = node.body[0]
                    if d.lineno > 2:
                        docs.append(d)
        if not docs:
            return None
        d = rng.choice(docs)
        line = lines[d.lineno - 1]
        for q in ('"""', "'''", '"', "'"):
            pos = line.find(q, d.col_offset)
            if pos >= 0:
                lines[d.lineno - 1] = line[: pos + len(q)] + "Reworded. " + line[pos + len(q):]
                return "\n".join(lines)
        return None
    if kind == "annotation":
        cands = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.returns is not None and n.returns.lineno == n.returns.end_lineno and n.lineno > 2]
        if not cands:
            return None
        node = rng.choice(cands).returns
        line = lines[node.lineno - 1]
        lines[node.lineno - 1] = line[: node.col_offset] + "t.Any" + line[node.end_col_offset:]
        return "\n".join(lines)
    if kind == "code":
        cands = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and node.lineno == node.end_lineno and node.lineno > 2:
                cands.append(("cmp", node))
            elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
                    node.value, bool) and node.lineno == node.end_lineno and node.lineno > 2:
                cands.append(("int", node))
        if not cands:
            return None
        what, node = rng.choice(cands)
        line = lines[node.lineno - 1]
        seg = line[node.col_offset: node.end_col_offset]
        if what == "cmp":
            for a, b in ((" == ", " != "), (" != ", " == "), (" < ", " <= "), (" > ", " >= "),
                         (" is not ", " is "), (" not in ", " in "), (" in ", " not in "), (" is ", " is not ")):
                if a in seg:
                    seg2 = seg.replace(a, b, 1)
                    break
            else:
                return None
        else:
            try:
                seg2 = str(int(seg, 0) + 1)
            except ValueError:
                return None
        lines[node.lineno - 1] = line[: node.col_offset] + seg2 + line[node.end_col_offset:]
        return "\n".join(lines)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--segments", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kinds", default="comment,inline,blank,docstring,code,annotation")
    args = ap.parse_args(argv)
    seg = load_index(args.segments)
    if seg is None:
        print("no segment index", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    files = []
    for top, dirs, fs in os.walk(args.root):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__"))
        for f in sorted(fs):
            rel = os.path.relpath(os.path.join(top, f), args.root)
            if f.endswith(".py") and rel.startswith(("sqlglot/", "tests/")):
                files.append(rel)
    kinds = args.kinds.split(",")
    expect = {"comment": "neutral", "inline": "neutral", "blank": "neutral",
              "docstring": "docstring-only", "code": "changed", "annotation": "neutral"}
    stats: Counter = Counter()
    bad: list = []
    cpu_chunk = cpu_whole = 0.0
    done = 0
    while done < args.n:
        rel = rng.choice(files)
        kind = rng.choice(kinds)
        with open(os.path.join(args.root, rel), encoding="utf-8") as f:
            old = f.read()
        try:
            new = mutate(old, kind, rng)
        except (SyntaxError, tokenize.TokenError):
            new = None
        if new is None or new == old:
            continue
        done += 1
        diff = _diff(rel, old, new)
        t = time.process_time()
        a = analyze(diff, args.root, data_map=None, segments=seg)
        cpu_chunk += time.process_time() - t
        t = time.process_time()
        b = analyze(diff, args.root, data_map=None, segments=None)
        cpu_whole += time.process_time() - t
        fa, fb = a["files"][0], b["files"][0]
        la = fa.get("level") if not fa.get("neutral") or fa.get("level") != "docstring-only" else "docstring-only"
        got = "neutral" if a["neutral"] and fa.get("level") != "docstring-only" else (
            "docstring-only" if fa.get("level") == "docstring-only" else "changed")
        ua = sorted((u["qualname"], u["where"]) for u in a["units"])
        ub = sorted((u["qualname"], u["where"]) for u in b["units"])
        stats[(kind, got, fa.get("mode"))] += 1
        if a["neutral"] != b["neutral"] or ua != ub or fa.get("level") != fb.get("level"):
            bad.append({"why": "chunk/whole disagree", "kind": kind, "file": rel, "a": (a["neutral"], la, ua),
                        "b": (b["neutral"], fb.get("level"), ub), "diff": diff[:1500]})
        elif kind == "annotation" and "from __future__ import annotations" not in old:
            if got == "neutral":
                bad.append({"why": "annotation edit without future import called neutral", "file": rel})
        elif got != expect[kind] and not (kind == "code" and got != "neutral"):
            bad.append({"why": f"expected {expect[kind]} got {got}", "kind": kind, "file": rel,
                        "reasons": a["reasons"], "diff": diff[:1500]})
    for k, v in sorted(stats.items()):
        print(k, v)
    print(f"cpu chunk {cpu_chunk * 1000 / done:.2f} ms/patch, whole {cpu_whole * 1000 / done:.2f} ms/patch")
    print(f"problems: {len(bad)}")
    for b in bad[:12]:
        print(json.dumps(b)[:2500])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
