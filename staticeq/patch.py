"""Unified-diff parsing and exact application against a pristine tree.

Only the standard library is used, so this runs in the slim run image. The
parser accepts `git diff` output (with extended headers) and plain unified
diffs. Anything it cannot interpret precisely is recorded on the FilePatch as
a reason, and the analyzer then refuses to call the patch neutral.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GIT_HEADER_RE = re.compile(r"^diff --git (.+)$")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    # (tag, text) with tag in " ", "-", "+"; text excludes the trailing "\n".
    lines: list[tuple[str, str]] = field(default_factory=list)
    old_no_eol: bool = False  # last old line has no trailing newline
    new_no_eol: bool = False  # last new line has no trailing newline


@dataclass
class FilePatch:
    old_path: str | None  # repo-relative, None for /dev/null
    new_path: str | None
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_rename: bool = False
    is_copy: bool = False
    is_binary: bool = False
    old_mode: str | None = None
    new_mode: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def mode_change(self) -> bool:
        return bool(self.old_mode and self.new_mode and self.old_mode != self.new_mode)

    def touched_old_lines(self) -> tuple[set[int], set[int]]:
        """Old-file line numbers removed, and insertion points.

        An insertion point k means new lines were inserted between old lines
        k and k+1 (k=0: before the first line).
        """
        removed: set[int] = set()
        inserts: set[int] = set()
        for h in self.hunks:
            old_no = h.old_start if h.old_len else h.old_start + 1
            for tag, _ in h.lines:
                if tag == " ":
                    old_no += 1
                elif tag == "-":
                    removed.add(old_no)
                    old_no += 1
                else:
                    inserts.add(old_no - 1)
        return removed, inserts


def _unquote(path: str) -> str:
    path = path.strip()
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        body = path[1:-1]
        out = bytearray()
        i = 0
        while i < len(body):
            ch = body[i]
            if ch == "\\" and i + 1 < len(body):
                nxt = body[i + 1]
                simple = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "a": "\a",
                          "b": "\b", "f": "\f", "r": "\r", "v": "\v"}
                if nxt in simple:
                    out += simple[nxt].encode()
                    i += 2
                    continue
                if nxt in "01234567" and i + 3 < len(body) + 1:
                    octal = body[i + 1:i + 4]
                    try:
                        out.append(int(octal, 8))
                        i += 4
                        continue
                    except ValueError:
                        pass
            out += ch.encode("utf-8", "surrogateescape")
            i += 1
        return out.decode("utf-8", "surrogateescape")
    return path


def _strip_prefix(path: str) -> str | None:
    path = _unquote(path)
    # "--- a/file\tTIMESTAMP" in non-git diffs
    if "\t" in path:
        path = path.split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def _split_git_header(rest: str) -> tuple[str | None, str | None]:
    rest = rest.strip()
    if rest.startswith('"'):
        # quoted a-path
        end = 1
        while end < len(rest):
            if rest[end] == "\\":
                end += 2
                continue
            if rest[end] == '"':
                break
            end += 1
        a = rest[: end + 1]
        b = rest[end + 1 :].strip()
        return _strip_prefix(a), _strip_prefix(b)
    # Unquoted: "a/X b/X" -- split on " b/" choosing the symmetric split.
    candidates = [m.start() for m in re.finditer(r" b/", rest)]
    for pos in candidates:
        a, b = rest[:pos], rest[pos + 1 :]
        if a[2:] == b[2:]:
            return _strip_prefix(a), _strip_prefix(b)
    if candidates:
        pos = candidates[0]
        return _strip_prefix(rest[:pos]), _strip_prefix(rest[pos + 1 :])
    return None, None


def parse_patch(text: str | bytes) -> list[FilePatch]:
    """Parse a unified diff into FilePatch records (problems recorded, never raised)."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "surrogateescape")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    patches: list[FilePatch] = []
    cur: FilePatch | None = None
    in_git_header = False
    i = 0
    n = len(lines)

    def start(old: str | None, new: str | None) -> FilePatch:
        fp = FilePatch(old_path=old, new_path=new)
        patches.append(fp)
        return fp

    while i < n:
        line = lines[i]
        m = _GIT_HEADER_RE.match(line)
        if m:
            a, b = _split_git_header(m.group(1))
            cur = start(a, b)
            if a is None or b is None:
                cur.problems.append("unparseable diff --git header")
            i += 1
            # extended header lines
            while i < n and not lines[i].startswith(("--- ", "@@", "diff --git ")):
                h = lines[i]
                if h.startswith("new file mode "):
                    cur.is_new = True
                    cur.new_mode = h.split()[-1]
                elif h.startswith("deleted file mode "):
                    cur.is_deleted = True
                    cur.old_mode = h.split()[-1]
                elif h.startswith("old mode "):
                    cur.old_mode = h.split()[-1]
                elif h.startswith("new mode "):
                    cur.new_mode = h.split()[-1]
                elif h.startswith("rename from "):
                    cur.is_rename = True
                    cur.old_path = _unquote(h[len("rename from "):])
                elif h.startswith("rename to "):
                    cur.is_rename = True
                    cur.new_path = _unquote(h[len("rename to "):])
                elif h.startswith("copy from "):
                    cur.is_copy = True
                    cur.old_path = _unquote(h[len("copy from "):])
                elif h.startswith("copy to "):
                    cur.is_copy = True
                    cur.new_path = _unquote(h[len("copy to "):])
                elif h.startswith(("similarity index", "dissimilarity index")):
                    pass
                elif h.startswith("index "):
                    parts = h.split()
                    if len(parts) >= 3 and not cur.old_mode:
                        cur.old_mode = cur.new_mode = parts[2]
                elif h.startswith("Binary files ") or h.startswith("GIT binary patch"):
                    cur.is_binary = True
                    if h.startswith("GIT binary patch"):
                        # skip the base85 payload
                        i += 1
                        while i < n and not lines[i].startswith("diff --git "):
                            i += 1
                        continue
                else:
                    cur.problems.append(f"unknown extended header: {h[:60]!r}")
                i += 1
            if cur.is_new:
                cur.old_path = None
            if cur.is_deleted:
                cur.new_path = None
            in_git_header = True
            continue
        if line.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            old = _strip_prefix(line[4:])
            new = _strip_prefix(lines[i + 1][4:])
            if cur is not None and in_git_header:
                exp_old = None if cur.is_new else cur.old_path
                exp_new = None if cur.is_deleted else cur.new_path
                if (old, new) != (exp_old, exp_new):
                    cur.problems.append("---/+++ paths disagree with diff --git header")
            else:
                cur = start(old, new)
            in_git_header = False
            if old is None:
                cur.is_new = True
                cur.old_path = None
            if new is None:
                cur.is_deleted = True
                cur.new_path = None
            i += 2
            continue
        if line.startswith("@@"):
            hm = _HUNK_RE.match(line)
            if cur is None:
                cur = start(None, None)
                cur.problems.append("hunk without file header")
            if not hm:
                cur.problems.append(f"bad hunk header {line[:60]!r}")
                i += 1
                continue
            h = Hunk(
                old_start=int(hm.group(1)),
                old_len=int(hm.group(2)) if hm.group(2) is not None else 1,
                new_start=int(hm.group(3)),
                new_len=int(hm.group(4)) if hm.group(4) is not None else 1,
            )
            cur.hunks.append(h)
            in_git_header = False
            i += 1
            old_seen = new_seen = 0
            last_tag = None
            while i < n and (old_seen < h.old_len or new_seen < h.new_len or
                             (i < n and lines[i].startswith("\\"))):
                body = lines[i]
                if body.startswith("\\"):
                    # "\ No newline at end of file" applies to the previous line
                    if last_tag == "-":
                        h.old_no_eol = True
                    elif last_tag == "+":
                        h.new_no_eol = True
                    elif last_tag == " ":
                        h.old_no_eol = h.new_no_eol = True
                    i += 1
                    continue
                if old_seen >= h.old_len and new_seen >= h.new_len:
                    break
                tag = body[:1] if body else " "
                if tag not in (" ", "-", "+"):
                    cur.problems.append("hunk body shorter than its header says")
                    break
                text = body[1:]
                h.lines.append((tag, text))
                if tag in (" ", "-"):
                    old_seen += 1
                if tag in (" ", "+"):
                    new_seen += 1
                last_tag = tag
                i += 1
            if old_seen != h.old_len or new_seen != h.new_len:
                cur.problems.append("hunk line counts do not match header")
            continue
        # anything else between files (e.g. commit message text) is ignored,
        # but text inside a file section after hunks is suspicious.
        if cur is not None and cur.hunks and line.strip():
            cur.problems.append(f"unexpected line in diff: {line[:60]!r}")
        i += 1
    return patches


class ApplyError(Exception):
    pass


def split_lines(data: bytes) -> tuple[list[str], bool]:
    """Split file bytes into lines without terminators; report trailing newline."""
    text = data.decode("utf-8", "surrogateescape")
    if text == "":
        return [], True
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
        return parts, True
    return parts, False


def join_lines(lines: list[str], final_newline: bool) -> bytes:
    if not lines:
        return b""
    text = "\n".join(lines) + ("\n" if final_newline else "")
    return text.encode("utf-8", "surrogateescape")


def apply_hunks(old: bytes | None, hunks: list[Hunk]) -> tuple[bytes, list[int]]:
    """Apply hunks exactly at their stated positions.

    Returns (new_bytes, old_to_new) where old_to_new[k] (1-based old line k)
    gives the 1-based new line number of an unchanged old line, or 0 if the
    old line was removed. Raises ApplyError if any context does not match.
    """
    old_lines, old_eol = split_lines(old or b"")
    out: list[str] = []
    mapping = [0] * (len(old_lines) + 1)
    pos = 0  # 0-based index into old_lines of next unconsumed line
    new_eol = old_eol
    for h in hunks:
        start = h.old_start - 1 if h.old_len else h.old_start
        if start < pos or start > len(old_lines):
            raise ApplyError(f"hunk at old line {h.old_start} out of order or range")
        while pos < start:
            out.append(old_lines[pos])
            mapping[pos + 1] = len(out)
            pos += 1
        for tag, text in h.lines:
            if tag in (" ", "-"):
                if pos >= len(old_lines) or old_lines[pos] != text:
                    raise ApplyError(f"context mismatch at old line {pos + 1}")
                if tag == " ":
                    out.append(text)
                    mapping[pos + 1] = len(out)
                pos += 1
            else:
                out.append(text)
        if pos == len(old_lines):
            # hunk reached EOF: its no-eol markers decide the final newline
            consumed_last = any(tag != "+" for tag, _ in h.lines) and old_lines
            if consumed_last and h.old_no_eol == old_eol:
                raise ApplyError("no-newline marker disagrees with file")
            if not consumed_last and not old_eol and old_lines:
                raise ApplyError("lines appended after a final line without newline")
            new_eol = not h.new_no_eol
    while pos < len(old_lines):
        out.append(old_lines[pos])
        mapping[pos + 1] = len(out)
        pos += 1
    return join_lines(out, new_eol), mapping


def read_pristine(root: str, rel: str) -> bytes | None:
    path = os.path.join(root, rel)
    try:
        with open(path, "rb") as f:
            return f.read()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None
