"""Parse and apply unified / git diffs in pure Python (no git or patch binary needed).

The run image (python:3.13-slim) has neither `git` nor `patch`, so the runner applies
patches itself. Everything works on bytes so CRLF files, odd encodings and missing final
newlines round-trip exactly.

Public API:
    parse_patch(data: bytes | str) -> list[FileChange]
    apply_patch(changes, read_old) -> dict[str, NewFile | None]
        read_old(path) -> bytes | None returns the pristine content (None when missing);
        the result maps every touched path to its new content (None = deleted).

Supported: modified/new/deleted files, renames and copies (with or without hunks),
mode-only changes, symlinks (mode 120000), "\\ No newline at end of file", hunks that
drifted by some lines (exact-context search, like `git apply` without fuzz), quoted
paths, and GIT binary patches (literal and delta). "Binary files ... differ" without
data cannot be applied and raises PatchError.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field


class PatchError(Exception):
    """The patch is malformed or does not apply to the pristine tree."""


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    # each line: (tag, content) with tag in b" ", b"-", b"+"; content excludes the
    # trailing b"\n"; a None content marks "\ No newline at end of file" for the prior line
    lines: list = field(default_factory=list)


@dataclass
class FileChange:
    old_path: str | None  # None for a new file
    new_path: str | None  # None for a deleted file
    hunks: list = field(default_factory=list)
    old_mode: str | None = None
    new_mode: str | None = None
    is_rename: bool = False
    is_copy: bool = False
    binary: bool = False  # binary change
    binary_patch: tuple | None = None  # ("literal"|"delta", size, data) for new side
    binary_no_data: bool = False  # "Binary files ... differ"

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def paths(self) -> list[str]:
        return [p for p in dict.fromkeys([self.old_path, self.new_path]) if p]


@dataclass
class NewFile:
    content: bytes
    mode: str | None = None  # e.g. "100644", "100755", "120000" (symlink: content = target)


_HUNK_RE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_ESCAPES = {
    b"a": b"\a", b"b": b"\b", b"f": b"\f", b"n": b"\n", b"r": b"\r",
    b"t": b"\t", b"v": b"\v", b"\\": b"\\", b'"': b'"',
}


def _unquote(raw: bytes) -> bytes:
    """Undo git's C-style path quoting ("a/x\\tb")."""
    if not (raw.startswith(b'"') and raw.endswith(b'"') and len(raw) >= 2):
        return raw
    s = raw[1:-1]
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i : i + 1]
        if c == b"\\" and i + 1 < len(s):
            n = s[i + 1 : i + 2]
            if n in _ESCAPES:
                out += _ESCAPES[n]
                i += 2
                continue
            if len(s[i + 1 : i + 4]) == 3 and s[i + 1 : i + 4].isdigit():
                out.append(int(s[i + 1 : i + 4], 8))
                i += 4
                continue
        out += c
        i += 1
    return bytes(out)


def _strip_prefix(path: bytes) -> str | None:
    path = _unquote(path.split(b"\t")[0] if not path.startswith(b'"') else path)
    if path == b"/dev/null":
        return None
    text = path.decode("utf-8", "surrogateescape")
    # a/ b/, or git's diff.mnemonicPrefix variants (c/ i/ w/ o/)
    if len(text) > 2 and text[1] == "/" and text[0] in "abciwo":
        text = text[2:]
    return text


def _split_git_header(rest: bytes) -> tuple[str | None, str | None]:
    """Paths from 'diff --git a/x b/y' (best effort; ---/+++ and rename lines override)."""
    rest = rest.strip()
    if rest.startswith(b'"'):
        end = rest.index(b'"', 1)
        while rest[end - 1 : end] == b"\\":
            end = rest.index(b'"', end + 1)
        a, b = rest[: end + 1], rest[end + 1 :].strip()
        return _strip_prefix(a), _strip_prefix(b)
    # unquoted: "a/<p> b/<p>" - split at the middle " X/" when both halves match
    for sep in (b" b/", b" i/", b" w/", b" o/", b" c/"):
        parts = rest.split(sep)
        if len(parts) == 2:
            return _strip_prefix(parts[0]), _strip_prefix(sep[1:] + parts[1])
        for i in range(1, len(parts)):
            a = sep.join(parts[:i])
            b = sep.join(parts[i:])
            if a[2:] == b:
                return _strip_prefix(a), _strip_prefix(sep[1:] + b)
    half = len(rest) // 2
    return _strip_prefix(rest[:half]), _strip_prefix(rest[half + 1 :])


def parse_patch(data: bytes | str) -> list[FileChange]:
    if isinstance(data, str):
        data = data.encode("utf-8", "surrogateescape")
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    changes: list[FileChange] = []
    cur: FileChange | None = None
    i = 0
    n = len(lines)

    def finish():
        nonlocal cur
        if cur is not None:
            changes.append(cur)
        cur = None

    while i < n:
        line = lines[i]
        if line.endswith(b"\r") and not line.startswith((b" ", b"+", b"-")):
            line = line[:-1]
        if line.startswith(b"diff --git "):
            finish()
            a, b = _split_git_header(line[len(b"diff --git ") :])
            cur = FileChange(old_path=a, new_path=b)
            i += 1
            continue
        if line.startswith(b"--- ") and i + 1 < n and lines[i + 1].startswith(b"+++ "):
            if cur is None or cur.hunks:
                finish()
                cur = FileChange(old_path=None, new_path=None)
            old = _strip_prefix(line[4:].rstrip(b"\r"))
            new = _strip_prefix(lines[i + 1][4:].rstrip(b"\r"))
            if not cur.is_rename and not cur.is_copy:
                cur.old_path, cur.new_path = old, new
            else:
                cur.old_path = old if old is not None else cur.old_path
                cur.new_path = new if new is not None else cur.new_path
            if old is None:
                cur.old_path = None
            if new is None:
                cur.new_path = None
            i += 2
            continue
        if cur is not None and line.startswith(b"@@ "):
            m = _HUNK_RE.match(line)
            if not m:
                raise PatchError(f"bad hunk header: {line[:80]!r}")
            h = Hunk(
                int(m.group(1)),
                int(m.group(2)) if m.group(2) is not None else 1,
                int(m.group(3)),
                int(m.group(4)) if m.group(4) is not None else 1,
            )
            i += 1
            old_left, new_left = h.old_len, h.new_len
            while i < n and (old_left > 0 or new_left > 0 or lines[i].startswith(b"\\")):
                hl = lines[i]
                if hl.startswith(b"\\"):
                    h.lines.append((b"\\", None))
                    i += 1
                    continue
                tag = hl[:1]
                body = hl[1:]
                if tag == b"" and hl == b"":
                    # some tools drop the space of empty context lines
                    tag, body = b" ", b""
                if tag == b" ":
                    old_left -= 1
                    new_left -= 1
                elif tag == b"-":
                    old_left -= 1
                elif tag == b"+":
                    new_left -= 1
                else:
                    raise PatchError(f"bad hunk line: {hl[:80]!r}")
                h.lines.append((tag, body))
                i += 1
            if old_left != 0 or new_left != 0:
                raise PatchError("truncated hunk")
            # a trailing "\ No newline" after the counted lines
            while i < n and lines[i].startswith(b"\\"):
                h.lines.append((b"\\", None))
                i += 1
            cur.hunks.append(h)
            continue
        if cur is not None:
            if line.startswith(b"old mode "):
                cur.old_mode = line[9:].strip().decode()
            elif line.startswith(b"new mode "):
                cur.new_mode = line[9:].strip().decode()
            elif line.startswith(b"deleted file mode "):
                cur.old_mode = line[18:].strip().decode()
                cur.new_path = None
            elif line.startswith(b"new file mode "):
                cur.new_mode = line[14:].strip().decode()
                cur.old_path = None
            elif line.startswith(b"rename from "):
                cur.is_rename = True
                cur.old_path = _unquote(line[12:]).decode("utf-8", "surrogateescape")
            elif line.startswith(b"rename to "):
                cur.is_rename = True
                cur.new_path = _unquote(line[10:]).decode("utf-8", "surrogateescape")
            elif line.startswith(b"copy from "):
                cur.is_copy = True
                cur.old_path = _unquote(line[10:]).decode("utf-8", "surrogateescape")
            elif line.startswith(b"copy to "):
                cur.is_copy = True
                cur.new_path = _unquote(line[8:]).decode("utf-8", "surrogateescape")
            elif line.startswith(b"index "):
                parts = line.split()
                if len(parts) >= 3 and cur.new_mode is None and cur.old_mode is None:
                    cur.old_mode = cur.new_mode = parts[2].decode()
            elif line.startswith(b"Binary files "):
                cur.binary = True
                cur.binary_no_data = True
                if b" and /dev/null differ" in line:
                    cur.new_path = None
                elif line.startswith(b"Binary files /dev/null and"):
                    cur.old_path = None
            elif line.startswith(b"GIT binary patch"):
                cur.binary = True
                i += 1
                cur.binary_patch, i = _read_binary(lines, i)
                # the reverse patch (if present) follows; skip it
                if i < n and lines[i].startswith((b"literal ", b"delta ")):
                    _, i = _read_binary(lines, i)
                continue
        i += 1
    finish()
    return [c for c in changes if c.old_path or c.new_path]


# ---------------------------------------------------------------- binary patches

_B85 = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
_B85_DEC = {c: i for i, c in enumerate(_B85)}


def _b85_line(line: bytes) -> bytes:
    c = line[0]
    if 65 <= c <= 90:
        length = c - 65 + 1
    elif 97 <= c <= 122:
        length = c - 97 + 27
    else:
        raise PatchError("bad binary line length")
    enc = line[1:]
    out = bytearray()
    for j in range(0, len(enc), 5):
        acc = 0
        for ch in enc[j : j + 5]:
            acc = acc * 85 + _B85_DEC[ch]
        out += acc.to_bytes(4, "big")
    return bytes(out[:length])


def _read_binary(lines, i):
    header = lines[i].strip()
    kind, size = header.split()
    i += 1
    raw = bytearray()
    while i < len(lines) and lines[i].strip():
        raw += _b85_line(lines[i].rstrip(b"\r"))
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
        break
    return (kind.decode(), int(size), zlib.decompress(bytes(raw))), i


def _apply_delta(src: bytes, delta: bytes) -> bytes:
    pos = 0

    def varint():
        nonlocal pos
        val = shift = 0
        while True:
            b = delta[pos]
            pos += 1
            val |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return val

    src_size = varint()
    dst_size = varint()
    if src_size != len(src):
        raise PatchError("binary delta base size mismatch")
    out = bytearray()
    while pos < len(delta):
        op = delta[pos]
        pos += 1
        if op & 0x80:
            off = size = 0
            for k in range(4):
                if op & (1 << k):
                    off |= delta[pos] << (8 * k)
                    pos += 1
            for k in range(3):
                if op & (1 << (4 + k)):
                    size |= delta[pos] << (8 * k)
                    pos += 1
            size = size or 0x10000
            out += src[off : off + size]
        elif op:
            out += delta[pos : pos + op]
            pos += op
        else:
            raise PatchError("bad binary delta opcode")
    if len(out) != dst_size:
        raise PatchError("binary delta size mismatch")
    return bytes(out)


# ---------------------------------------------------------------- applying


def _split_keep(content: bytes) -> list[bytes]:
    """Lines with their b"\\n" kept (last line may lack it)."""
    if not content:
        return []
    parts = content.split(b"\n")
    out = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _hunk_sides(h: Hunk) -> tuple[list[bytes], list[bytes]]:
    old: list[bytes] = []
    new: list[bytes] = []
    last = None
    for tag, body in h.lines:
        if tag == b"\\":
            if last == b" ":
                old[-1] = old[-1][:-1]
                new[-1] = new[-1][:-1]
            elif last == b"-":
                old[-1] = old[-1][:-1]
            elif last == b"+":
                new[-1] = new[-1][:-1]
            continue
        line = body + b"\n"
        if tag == b" ":
            old.append(line)
            new.append(line)
        elif tag == b"-":
            old.append(line)
        else:
            new.append(line)
        last = tag
    return old, new


def apply_hunks(content: bytes, hunks: list[Hunk], max_offset: int = 10**9) -> bytes:
    lines = _split_keep(content)
    out: list[bytes] = []
    pos = 0  # index into lines of the next unconsumed line
    delta = 0  # accumulated offset between header line numbers and reality
    for h in hunks:
        old, new = _hunk_sides(h)
        want = (h.old_start - 1 if h.old_len else h.old_start) + delta
        if not old:
            start = max(pos, min(want, len(lines)))
        else:
            start = _find(lines, old, want, pos, max_offset)
            if start is None:
                # tolerate a final line whose newline differs only at EOF
                raise PatchError(f"hunk @@ -{h.old_start},{h.old_len} does not apply")
        delta = start - (h.old_start - 1 if h.old_len else h.old_start)
        out.extend(lines[pos:start])
        out.extend(new)
        pos = start + len(old)
    out.extend(lines[pos:])
    return b"".join(out)


def _find(lines, old, want, lo, max_offset):
    n = len(old)
    hi = len(lines) - n
    if hi < lo:
        return None

    def match(s):
        return lines[s : s + n] == old

    if lo <= want <= hi and match(want):
        return want
    for off in range(1, max(hi - lo, 0) + 1):
        if off > max_offset:
            break
        for s in (want - off, want + off):
            if lo <= s <= hi and match(s):
                return s
        if want - off < lo and want + off > hi:
            break
    return None


def apply_patch(changes: list[FileChange], read_old) -> dict[str, NewFile | None]:
    """Compute the new content of every path the patch touches.

    read_old(path) returns pristine bytes or None if the file does not exist.
    Returns {path: NewFile or None (deleted)}. Raises PatchError when it does not apply.
    """
    result: dict[str, NewFile | None] = {}

    for ch in changes:
        if ch.binary and ch.binary_no_data:
            raise PatchError(f"binary change without data: {ch.path}")
        # every file patch applies to the preimage (git semantics): a copy/rename source
        # edited elsewhere in the same diff is still copied from its original content
        src = read_old(ch.old_path) if ch.old_path else b""
        if ch.old_path and src is None:
            raise PatchError(f"missing file: {ch.old_path}")
        if ch.new_path is None:
            # deletion (hunks, if any, must remove everything; trust the header)
            result[ch.old_path] = None
            continue
        if ch.binary_patch is not None:
            kind, size, data = ch.binary_patch
            if len(data) != size:  # literal: file size; delta: inflated delta size
                raise PatchError(f"binary size mismatch: {ch.path}")
            content = data if kind == "literal" else _apply_delta(src or b"", data)
        elif ch.hunks:
            content = apply_hunks(src or b"", ch.hunks)
        else:
            content = src or b""
        if ch.is_rename and ch.old_path and ch.old_path != ch.new_path:
            result.setdefault(ch.old_path, None)
        mode = ch.new_mode or ch.old_mode
        result[ch.new_path] = NewFile(content, mode)
    return result
