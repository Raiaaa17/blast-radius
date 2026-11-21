#!/usr/bin/env python3
"""Mine real sqlglot history into standalone evaluation patches against the pinned commit.

Evaluation data only: nothing here may feed selection rules.

Two sources of changes, both turned into a unified diff against PIN:
  fwd  commits after PIN on upstream main, applied forward (C^ -> C);
  rev  commits before PIN, reverse-applied (C -> C^), i.e. "undo that commit".

Each commit is split into portions and every distinct portion that applies cleanly
to PIN becomes one patch:
  whole     every in-scope file of the commit (only when all of them apply)
  src       sqlglot/ and sqlglotc/ files only
  tests     tests/**/*.py only
  fixtures  tests/ non-.py files only (tests/fixtures/...)
  partial   whichever in-scope files apply, when "whole" does not
  squash    fwd only, when "whole" does not apply: PIN -> C for the files C touched
            (lands the intermediate upstream history of those files too)

A portion is applied with `git apply --cached` onto a private index of PIN, the
resulting tree is diffed against PIN with `git diff --no-renames --abbrev=7`, which
is the same shape as the kit's sample/commits/*.patch. Paths outside sqlglot/,
sqlglotc/ and tests/ (docs, CI, pyproject, the integration-tests submodule) and
binary files are dropped. Identical patch texts are emitted once.

The coarse kind is computed from the patch alone, never from a test outcome:
  code / comment (src files whose AST is unchanged, or only docstrings changed)
  / test / fixture, and "multi" when more than one of those appears.

Usage:
  mine.py --repo <sqlglot clone with history> --out <corpus dir>
          [--rev-limit N] [--max-patch-bytes B] [--tests-json sample/tests.json]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize

PIN = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
SCOPE = ("sqlglot/", "sqlglotc/", "tests/")


def git(repo, *args, env=None, input=None, check=True):
    p = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        env=env,
        input=input,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.decode(errors='replace')}")
    return p


def split_diff(text: str) -> list[tuple[str, str]]:
    """Split a multi-file git diff into (path, file_patch)."""
    parts = re.split(r"(?m)^(?=diff --git )", text)
    out = []
    for part in parts:
        if not part.startswith("diff --git "):
            continue
        m = re.match(r"diff --git a/(\S+) b/(\S+)", part)
        if not m:
            continue
        out.append((m.group(2), part))
    return out


def in_scope(path: str, fpatch: str) -> bool:
    if not path.startswith(SCOPE):
        return False
    if "Binary files" in fpatch or "GIT binary patch" in fpatch:
        return False
    header = fpatch.split("\n@@", 1)[0]
    if "160000" in header:  # submodule pointer
        return False
    return True


def category(path: str) -> str:
    if path.startswith("tests/"):
        return "test" if path.endswith(".py") else "fixture"
    return "src"


def area(path: str) -> str:
    p = path
    if p.startswith("tests/fixtures/"):
        rest = p[len("tests/fixtures/"):]
        return "fixtures/" + rest.split("/")[0]
    if p.startswith("tests/"):
        rest = p[len("tests/"):]
        if rest.startswith("dialects/"):
            return "tests/dialects"
        return "tests/" + rest.split("/")[0]
    if p.startswith("sqlglotc/"):
        return "sqlglotc"
    rest = p[len("sqlglot/"):]
    if "/" in rest:
        top = rest.split("/")[0]
        return top  # dialects, parsers, generators, optimizer, executor, expressions, typing, ...
    return rest[:-3] if rest.endswith(".py") else rest  # parser, generator, tokens, __init__, ...


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def py_change_class(before: str | None, after: str | None) -> str:
    """'code', 'comment' (AST identical: comments/whitespace/formatting), 'docstring'."""
    if before is None or after is None:
        return "code"  # file added or removed
    try:
        a = ast.parse(before)
        b = ast.parse(after)
    except SyntaxError:
        return "code"
    if ast.dump(a) == ast.dump(b):
        return "comment"
    if ast.dump(strip_docstrings(a)) == ast.dump(strip_docstrings(b)):
        return "docstring"
    return "code"


class Miner:
    def __init__(self, repo, out, max_bytes):
        self.repo = repo
        self.out = out
        self.max_bytes = max_bytes
        # private index files must sit on the repo's filesystem (git renames into them)
        gitdir = git(repo, "rev-parse", "--absolute-git-dir").stdout.decode().strip()
        self.tmp = tempfile.mkdtemp(prefix="br-upstream-", dir=gitdir)
        self.base_index = os.path.join(self.tmp, "base.idx")
        git(self.repo, "read-tree", f"--index-output={self.base_index}", PIN)
        self.seen: dict[str, str] = {}
        self.manifest: dict[str, dict] = {}
        os.makedirs(os.path.join(out, "commits"), exist_ok=True)

    def env(self, idx):
        e = dict(os.environ)
        e["GIT_INDEX_FILE"] = idx
        return e

    def applies(self, patch: str) -> bool:
        p = git(self.repo, "apply", "--cached", "--check", env=self.env(self.base_index), input=patch.encode(), check=False)
        return p.returncode == 0

    def build(self, patch: str) -> str | None:
        """Apply patch on a copy of PIN's index; return the regenerated diff against PIN."""
        idx = os.path.join(self.tmp, "work.idx")
        shutil.copyfile(self.base_index, idx)
        env = self.env(idx)
        p = git(self.repo, "apply", "--cached", env=env, input=patch.encode(), check=False)
        if p.returncode != 0:
            return None
        tree = git(self.repo, "write-tree", env=env).stdout.decode().strip()
        d = git(self.repo, "diff", "--no-renames", "--no-ext-diff", "--abbrev=7", PIN, tree).stdout
        return d.decode("utf-8", errors="surrogateescape") if d else None

    def show(self, rev, path):
        p = git(self.repo, "show", f"{rev}:{path}", check=False)
        return p.stdout.decode("utf-8", errors="replace") if p.returncode == 0 else None

    def classify(self, patch: str) -> dict:
        files = split_diff(patch)
        cats = set()
        per_file = []
        added = removed = 0
        for path, fp in files:
            for line in fp.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    removed += 1
            c = category(path)
            detail = c
            if c == "src":
                if path.endswith(".py"):
                    before = None if "\n--- /dev/null" in fp else self.show(PIN, path)
                    after = None if "\n+++ /dev/null" in fp else self._after(path, fp)
                    detail = py_change_class(before, after)
                else:
                    detail = "code"
                cats.add("comment" if detail in ("comment", "docstring") else "code")
            else:
                cats.add(c)
            per_file.append({"path": path, "category": c, "change": detail, "area": area(path)})
        kind = cats.pop() if len(cats) == 1 else "multi"
        return {"kind": kind, "files": per_file, "added": added, "removed": removed}

    def _after(self, path, fp):
        idx = os.path.join(self.tmp, "cls.idx")
        shutil.copyfile(self.base_index, idx)
        env = self.env(idx)
        git(self.repo, "apply", "--cached", env=env, input=fp.encode(), check=True)
        blob = git(self.repo, "ls-files", "-s", "--", path, env=env).stdout.decode().split()
        if not blob:
            return None
        return git(self.repo, "cat-file", "blob", blob[1]).stdout.decode("utf-8", errors="replace")

    def emit(self, pid, patch, meta):
        if len(patch.encode()) > self.max_bytes:
            return False
        h = hashlib.sha256(patch.encode()).hexdigest()
        if h in self.seen:
            self.manifest[self.seen[h]].setdefault("duplicates", []).append(pid)
            return False
        self.seen[h] = pid
        meta.update(self.classify(patch))
        meta["bytes"] = len(patch.encode())
        with open(os.path.join(self.out, "commits", pid + ".patch"), "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(patch)
        self.manifest[pid] = meta
        return True

    def mine_commit(self, sha, direction, subject):
        if direction == "fwd":
            a, b = f"{sha}^", sha
        else:
            a, b = sha, f"{sha}^"
        raw = git(self.repo, "diff", "--no-renames", "--no-ext-diff", "--abbrev=7", a, b, check=False).stdout.decode("utf-8", errors="surrogateescape")
        files = split_diff(raw)
        scoped = [(p, fp) for p, fp in files if in_scope(p, fp)]
        dropped = [p for p, fp in files if not in_scope(p, fp)]
        if not scoped:
            return 0
        ok = [(p, fp) for p, fp in scoped if self.applies(fp)]
        short = sha[:7]
        base_meta = {"origin": sha, "direction": direction, "subject": subject, "dropped_paths": dropped}
        portions: list[tuple[str, list]] = []
        all_apply = len(ok) == len(scoped)
        if all_apply:
            portions.append(("whole", scoped))
        elif ok:
            portions.append(("partial", ok))
        by_cat = {"src": [], "test": [], "fixture": []}
        for p, fp in ok:
            by_cat[category(p)].append((p, fp))
        for name, cat in (("src", "src"), ("tests", "test"), ("fixtures", "fixture")):
            sel = by_cat[cat]
            if sel and len(sel) < len(ok):
                portions.append((name, sel))
        n = 0
        for name, sel in portions:
            patch = self.build("".join(fp for _, fp in sel))
            if patch and self.emit(f"{short}-{direction}-{name}", patch, dict(base_meta, portion=name, files_in_origin=len(scoped))):
                n += 1
        if direction == "fwd" and not all_apply:
            paths = [p for p, _ in scoped]
            d = git(self.repo, "diff", "--no-renames", "--no-ext-diff", "--abbrev=7", PIN, sha, "--", *paths).stdout.decode("utf-8", errors="surrogateescape")
            d = "".join(fp for p, fp in split_diff(d) if in_scope(p, fp))
            if d:
                patch = self.build(d)
                if patch and self.emit(f"{short}-fwd-squash", patch, dict(base_meta, portion="squash", files_in_origin=len(scoped))):
                    n += 1
        return n

    def commits(self, rng):
        out = git(self.repo, "log", "--first-parent", "--no-merges", "--format=%H%x09%s", rng).stdout.decode()
        return [line.split("\t", 1) for line in out.splitlines() if line]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--upstream", default="origin/main")
    ap.add_argument("--rev-limit", type=int, default=600, help="most recent N commits before PIN to revert")
    ap.add_argument("--max-patch-bytes", type=int, default=200_000)
    ap.add_argument("--tests-json", help="copied into the corpus so it is shaped like sample/")
    args = ap.parse_args()

    m = Miner(args.repo, args.out, args.max_patch_bytes)
    try:
        fwd = m.commits(f"{PIN}..{args.upstream}")
        rev = m.commits(PIN)[: args.rev_limit]
        stats = {"fwd": [0, 0], "rev": [0, 0]}
        for direction, lst in (("fwd", fwd), ("rev", rev)):
            for i, (sha, subject) in enumerate(lst):
                n = m.mine_commit(sha, direction, subject)
                stats[direction][0] += 1
                stats[direction][1] += n
                if i % 50 == 0:
                    print(f"{direction} {i}/{len(lst)} patches={len(m.manifest)}", file=sys.stderr)
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump({"pin": PIN, "upstream": git(args.repo, "rev-parse", args.upstream).stdout.decode().strip(), "patches": m.manifest}, f, indent=1, sort_keys=True)
        if args.tests_json:
            shutil.copyfile(args.tests_json, os.path.join(args.out, "tests.json"))
        print(json.dumps({"commits_scanned": stats, "patches": len(m.manifest)}), file=sys.stderr)
    finally:
        shutil.rmtree(m.tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
