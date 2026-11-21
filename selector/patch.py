"""Parse and apply git-style unified diffs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HUNK_RE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    pass


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[bytes] = field(default_factory=list)  # with their ' ', '-', '+' or '\\' prefix


@dataclass
class FileDiff:
    old_path: str | None  # None when the file is new
    new_path: str | None  # None when the file is deleted
    hunks: list[Hunk] = field(default_factory=list)
    binary: bool = False
    mode_change: bool = False
    rename: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    def changed_old_lines(self) -> list[tuple[int, int]]:
        """Old-file line ranges [a, b] touched by each hunk (insertions get [a, a-1])."""
        ranges = []
        for hunk in self.hunks:
            line = hunk.old_start
            for raw in hunk.lines:
                tag = raw[:1]
                if tag == b" ":
                    line += 1
                elif tag == b"-":
                    ranges.append((line, line))
                    line += 1
                elif tag == b"+":
                    ranges.append((line, line - 1))  # insertion before `line`
        return ranges


def _strip_prefix(path: bytes) -> str:
    text = path.decode("utf-8", "surrogateescape")
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
    text = text.split("\t", 1)[0]
    if text == "/dev/null":
        return ""
    if text[:2] in ("a/", "b/"):
        return text[2:]
    return text


def parse(data: bytes) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk: Hunk | None = None
    old_left = new_left = 0
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    for line in lines:
        if hunk is not None and (old_left > 0 or new_left > 0):
            tag = line[:1]
            if tag == b"" :
                line, tag = b" ", b" "  # a blank context line whose space was stripped
            if tag == b" ":
                old_left -= 1
                new_left -= 1
            elif tag == b"-":
                old_left -= 1
            elif tag == b"+":
                new_left -= 1
            elif tag != b"\\":
                raise PatchError("short hunk")
            hunk.lines.append(line)
            continue
        if hunk is not None and line.startswith(b"\\"):
            hunk.lines.append(line)
            continue
        hunk = None
        if line.startswith(b"diff --git "):
            current = FileDiff(old_path=None, new_path=None)
            files.append(current)
            parts = line[len(b"diff --git ") :].split(b" b/")
            if len(parts) == 2:
                current.old_path = _strip_prefix(parts[0]) or None
                current.new_path = _strip_prefix(b"b/" + parts[1]) or None
            continue
        if current is None:
            continue
        if line.startswith(b"@@"):
            m = HUNK_RE.match(line)
            if not m:
                raise PatchError("bad hunk header")
            hunk = Hunk(
                int(m.group(1)),
                int(m.group(2)) if m.group(2) is not None else 1,
                int(m.group(3)),
                int(m.group(4)) if m.group(4) is not None else 1,
            )
            old_left, new_left = hunk.old_len, hunk.new_len
            current.hunks.append(hunk)
        elif line.startswith(b"--- "):
            current.old_path = _strip_prefix(line[4:]) or None
        elif line.startswith(b"+++ "):
            current.new_path = _strip_prefix(line[4:]) or None
        elif line.startswith(b"new file mode"):
            current.old_path = None
        elif line.startswith(b"deleted file mode"):
            current.new_path = None
        elif line.startswith((b"old mode", b"new mode")):
            current.mode_change = True
        elif line.startswith((b"rename from", b"rename to", b"copy from", b"copy to", b"similarity index")):
            current.rename = True
            if line.startswith(b"rename from "):
                current.old_path = line[len(b"rename from ") :].decode("utf-8", "surrogateescape")
            elif line.startswith(b"rename to "):
                current.new_path = line[len(b"rename to ") :].decode("utf-8", "surrogateescape")
        elif line.startswith((b"Binary files", b"GIT binary patch")):
            current.binary = True
    if hunk is not None and (old_left > 0 or new_left > 0):
        raise PatchError("truncated hunk")
    return files


def apply(old: bytes | None, diff: FileDiff) -> bytes | None:
    """Return the new content of one file, or None when the diff deletes it."""
    if diff.new_path is None:
        return None
    if diff.binary:
        raise PatchError("binary diff")
    src = (old or b"").split(b"\n")
    # Keep track of whether the old file ended with a newline.
    old_lines = [line + b"\n" for line in src[:-1]]
    if src and src[-1] != b"":
        old_lines.append(src[-1])
    out: list[bytes] = []
    pos = 0  # index into old_lines
    for hunk in diff.hunks:
        start = hunk.old_start - 1 if hunk.old_len > 0 else hunk.old_start
        if start < pos or start > len(old_lines):
            raise PatchError("hunk out of order")
        out.extend(old_lines[pos:start])
        pos = start
        last_kind = None
        for raw in hunk.lines:
            tag, body = raw[:1], raw[1:]
            if tag == b"\\":
                # "\ No newline at end of file": strip the newline of the previous line.
                if last_kind in (b" ", b"+") and out and out[-1].endswith(b"\n"):
                    out[-1] = out[-1][:-1]
                continue
            if tag == b" ":
                if pos >= len(old_lines) or old_lines[pos].rstrip(b"\n") != body:
                    raise PatchError("context mismatch")
                out.append(old_lines[pos] if old_lines[pos].endswith(b"\n") else body + b"\n")
                pos += 1
            elif tag == b"-":
                if pos >= len(old_lines) or old_lines[pos].rstrip(b"\n") != body:
                    raise PatchError("removed line mismatch")
                pos += 1
            elif tag == b"+":
                out.append(body + b"\n")
            last_kind = tag
    out.extend(old_lines[pos:])
    return b"".join(out)
