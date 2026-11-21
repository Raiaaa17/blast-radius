"""Self-tests for runner.patch: `python -m runner.selftest [N]`.

Differential test against real git (skipped when git is unavailable): builds a small repo,
makes N random change sets (edits, hunks near each other, CRLF files, files without a final
newline, new/deleted/renamed/copied files, mode changes, binary files, quoted paths), takes
`git diff --binary -M -C`, applies it with runner.patch to the old tree and compares every
file byte for byte with git's new tree.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile

from .patch import PatchError, apply_patch, parse_patch


def _git(cwd, *args, input=None):
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=os.path.dirname(cwd), GIT_CONFIG_NOSYSTEM="1",
               HOME=cwd, XDG_CONFIG_HOME=cwd, GIT_CONFIG_GLOBAL=os.devnull, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", *args], cwd=cwd, env=env, input=input, check=True,
                          capture_output=True).stdout


def _snapshot(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != ".git"]
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if os.path.islink(p):
                out[rel] = ("link", os.readlink(p).encode())
            else:
                with open(p, "rb") as f:
                    out[rel] = ("x" if os.access(p, os.X_OK) else "f", f.read())
    return out


def _rand_text(rng, n, crlf=False, eol=True):
    words = ["SELECT", "a", "b", "FROM", "x", "WHERE", "1", "=", "--", "#", "(", ")", "é", "\t"]
    lines = [" ".join(rng.choice(words) for _ in range(rng.randint(0, 8))) for _ in range(n)]
    sep = "\r\n" if crlf else "\n"
    s = sep.join(lines)
    return (s + sep if eol else s).encode()


def _mutate(rng, data: bytes) -> bytes:
    lines = data.split(b"\n")
    for _ in range(rng.randint(1, 4)):
        op = rng.random()
        i = rng.randrange(len(lines) + 1)
        if op < 0.35 and lines:
            lines[min(i, len(lines) - 1)] += b" mutated"
        elif op < 0.6:
            lines.insert(i, b"inserted line %d" % rng.randint(0, 99))
        elif op < 0.8 and len(lines) > 1:
            del lines[min(i, len(lines) - 1)]
        else:
            lines[i:i] = [b"block %d" % k for k in range(rng.randint(1, 6))]
    out = b"\n".join(lines)
    if rng.random() < 0.15:
        out = out.rstrip(b"\n") if out.endswith(b"\n") else out + b"\n"
    return out


def differential(n: int = 200, seed: int = 0, verbose: bool = False) -> int:
    if shutil.which("git") is None:
        print("selftest: git not found, differential test skipped", file=sys.stderr)
        return 0
    rng = random.Random(seed)
    failures = 0
    base = tempfile.mkdtemp(prefix="br-runner-selftest-")
    try:
        for case in range(n):
            root = os.path.join(base, f"c{case}")
            os.makedirs(root)
            _git(root, "init", "-q")
            files = {}
            for k in range(rng.randint(2, 6)):
                d = rng.choice(["", "sub/", "sub/deep/", "q dir/"])
                name = f"{d}f{k}{rng.choice(['.py', '.sql', '.txt', ''])}"
                files[name] = _rand_text(rng, rng.randint(0, 60), crlf=rng.random() < 0.2,
                                         eol=rng.random() < 0.85)
            if rng.random() < 0.3:
                files["bin.dat"] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 3000)))
            for name, data in files.items():
                p = os.path.join(root, name)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(data)
            _git(root, "add", "-A")
            _git(root, "commit", "-qm", "base")
            old = _snapshot(root)
            names = list(files)
            for name in names:
                p = os.path.join(root, name)
                r = rng.random()
                if r < 0.45:
                    with open(p, "rb") as f:
                        data = f.read()
                    if name == "bin.dat":
                        data = data[: len(data) // 2] + bytes([7, 7, 7]) + data[len(data) // 2:]
                    else:
                        data = _mutate(rng, data)
                    with open(p, "wb") as f:
                        f.write(data)
                elif r < 0.55:
                    os.unlink(p)
                elif r < 0.65:
                    newp = os.path.join(root, "renamed", os.path.basename(name) + ".new")
                    os.makedirs(os.path.dirname(newp), exist_ok=True)
                    os.rename(p, newp)
                    if rng.random() < 0.5 and name != "bin.dat":
                        with open(newp, "rb") as f:
                            data = f.read()
                        with open(newp, "wb") as f:
                            f.write(_mutate(rng, data) if data.count(b"\n") > 8 else data)
                elif r < 0.72:
                    os.chmod(p, 0o755)
                elif r < 0.78:
                    shutil.copy(p, p + ".copy")
            if rng.random() < 0.4:
                p = os.path.join(root, "new dir", "added.py")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(_rand_text(rng, rng.randint(0, 20), eol=rng.random() < 0.8))
            _git(root, "add", "-A")
            diff = _git(root, "diff", "--cached", "--binary", "-M", "-C", "--find-copies-harder")
            new = _snapshot(root)
            try:
                result = apply_patch(parse_patch(diff), lambda path: old.get(path, (None, None))[1])
            except PatchError as e:
                failures += 1
                print(f"case {case}: PatchError {e}", file=sys.stderr)
                continue
            got = {k: v for k, v in old.items()}
            for path, nf in result.items():
                if nf is None:
                    got.pop(path, None)
                else:
                    kind = "x" if (nf.mode or "").endswith("755") else "f"
                    got[path] = (kind, nf.content)
            if got != new:
                failures += 1
                bad = sorted(set(got) ^ set(new)) + [k for k in got if k in new and got[k] != new[k]]
                print(f"case {case}: mismatch on {bad[:5]}", file=sys.stderr)
                if verbose:
                    sys.stderr.write(diff.decode("utf-8", "replace")[:3000])
            shutil.rmtree(root, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print(f"selftest: differential {n} cases, {failures} failures", file=sys.stderr)
    return failures


def unit() -> int:
    """Hand-written cases that git does not generate by default."""
    fails = 0
    # drifted hunk header (line numbers off) still applies by exact context
    old = b"".join(b"line %d\n" % i for i in range(30))
    patch = (b"--- a/x\n+++ b/x\n@@ -2,3 +2,3 @@\n line 9\n-line 10\n+LINE 10\n line 11\n")
    r = apply_patch(parse_patch(patch), lambda p: old)
    fails += r["x"].content != old.replace(b"line 10\n", b"LINE 10\n")
    # hunk that does not apply
    try:
        apply_patch(parse_patch(b"--- a/x\n+++ b/x\n@@ -1 +1 @@\n-nope\n+yes\n"), lambda p: old)
        fails += 1
    except PatchError:
        pass
    # binary without data cannot be applied
    try:
        apply_patch(parse_patch(b"diff --git a/b b/b\nindex 1..2 100644\nBinary files a/b and b/b differ\n"),
                    lambda p: b"\0")
        fails += 1
    except PatchError:
        pass
    print(f"selftest: unit cases, {fails} failures", file=sys.stderr)
    return fails


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    sys.exit(1 if (unit() + differential(n)) else 0)
