"""Statement-segment index of the pristine tree, so a patch is compiled and
parsed only around the lines it touches.

For each pristine .py file the index partitions its lines into *segments*:
segment k covers the blank/comment gap after statement k-1 plus statement k
(decorators included), and a final "tail" segment holds trailing comments.
Class statements whose body starts on a later line than the header colon
carry child segments partitioning their body the same way, plus the indent
of the body and the line of the header colon.

Why chunk compilation is sound: a segment boundary in the old file is a
point where the tokenizer is at depth 0 with no pending indentation, so for
a patch whose touched lines all fall inside one run of segments, the new
file is P + N + S with P and S byte-identical to the old file. If N compiles
on its own as complete statements (same __future__ flags; wrapped in the
enclosing class headers for class-body segments, which preserves qualnames,
name mangling and the __class__ cell), P and S tokenize and parse exactly as
before, so the module's behavior differs only through N. Anything that does
not fit (encoding cookie or BOM, CR or form-feed characters, __future__
text in the chunk, touched header lines, a chunk that does not compile) falls
back to whole-file comparison.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import tokenize

INDEX_VERSION = 1


def _first_line(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", None) or []
    return min([node.lineno] + [d.lineno for d in decos])


def _class_colon_line(lines: list[str], node: ast.ClassDef) -> int | None:
    """Line of the ':' ending the class header (tokenize over header lines)."""
    start = node.lineno
    body_first = _first_line(node.body[0])
    text = "\n".join(lines[start - 1 : body_first]) + "\n"
    depth = 0
    seen_class = False
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.NAME and tok.string == "class" and depth == 0:
                seen_class = True
            elif tok.type == tokenize.OP:
                if tok.string in "([{":
                    depth += 1
                elif tok.string in ")]}":
                    depth -= 1
                elif tok.string == ":" and depth == 0 and seen_class:
                    return start + tok.start[0] - 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    return None


def _segments(body: list[ast.stmt], region_start: int, region_end: int, lines: list[str]) -> list:
    segs: list = []
    prev_end = region_start - 1
    for stmt in body:
        end = stmt.end_lineno or stmt.lineno
        first = _first_line(stmt)
        if end <= prev_end or first <= prev_end:
            # shares a line with the previous statement: merge into it
            if segs:
                segs[-1][1] = max(segs[-1][1], end)
                segs[-1][2] = "merged"
                segs[-1][6] = None
            prev_end = max(prev_end, end)
            continue
        kind = type(stmt).__name__
        name = getattr(stmt, "name", None)
        children = None
        if isinstance(stmt, ast.ClassDef) and stmt.body:
            colon = _class_colon_line(lines, stmt)
            body_first = _first_line(stmt.body[0])
            if colon is not None and colon < body_first:
                indent_line = lines[body_first - 1]
                indent = indent_line[: len(indent_line) - len(indent_line.lstrip(" \t"))]
                children = {
                    "colon": colon,
                    "indent": indent,
                    "segs": _segments(stmt.body, colon + 1, end, lines),
                }
        segs.append([prev_end + 1, end, kind, name, first, stmt.lineno, children])
        prev_end = end
    if prev_end < region_end:
        segs.append([prev_end + 1, region_end, "tail", None, prev_end + 1, prev_end + 1, None])
    return segs


def index_source(data: bytes, filename: str = "<file>") -> dict:
    sha = hashlib.sha1(data).hexdigest()
    entry: dict = {"sha1": sha, "simple": False}
    if data.startswith(b"\xef\xbb\xbf") or b"\r" in data or b"\f" in data or b"\x00" in data:
        return entry
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return entry
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for ln in lines[:2]:
        s = ln.lstrip()
        if s.startswith("#") and "coding" in s:
            return entry
    try:
        tree = ast.parse(data, filename)
    except (SyntaxError, ValueError):
        return entry
    flags = 0
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            import __future__

            for alias in stmt.names:
                feat = getattr(__future__, alias.name, None)
                if feat is not None:
                    flags |= feat.compiler_flag
    entry.update(simple=True, nlines=len(lines), future_flags=flags,
                 segs=_segments(tree.body, 1, len(lines), lines))
    return entry


def _entry_name(rel: str) -> str:
    return hashlib.sha1(rel.encode("utf-8", "surrogateescape")).hexdigest()[:20] + ".json"


def build_index(root: str, out_dir: str) -> int:
    """Write one small JSON per pristine .py file into out_dir (loaded lazily)."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for top, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__", "node_modules"))
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            full = os.path.join(top, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            with open(full, "rb") as f:
                data = f.read()
            entry = index_source(data, rel)
            entry["path"] = rel
            with open(os.path.join(out_dir, _entry_name(rel)), "w", encoding="utf-8") as f:
                json.dump(entry, f, separators=(",", ":"))
            count += 1
    with open(os.path.join(out_dir, "VERSION"), "w", encoding="utf-8") as f:
        f.write(f"{INDEX_VERSION}\n")
    return count


class SegmentIndex:
    """Lazy per-file view of a build_index() directory."""

    def __init__(self, directory: str):
        self.directory = directory
        self._cache: dict[str, dict | None] = {}

    def get(self, rel: str) -> dict | None:
        if rel not in self._cache:
            try:
                with open(os.path.join(self.directory, _entry_name(rel)), encoding="utf-8") as f:
                    entry = json.load(f)
                self._cache[rel] = entry if entry.get("path") == rel else None
            except (OSError, ValueError):
                self._cache[rel] = None
        return self._cache[rel]


_INDEX_CACHE: dict[str, SegmentIndex | None] = {}


def default_index_dir() -> str:
    return os.environ.get("STATICEQ_SEGMENTS") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "segments")


def load_index(directory: str | None = None) -> SegmentIndex | None:
    directory = directory or default_index_dir()
    if directory not in _INDEX_CACHE:
        try:
            with open(os.path.join(directory, "VERSION"), encoding="utf-8") as f:
                ok = f.read().strip() == str(INDEX_VERSION)
        except OSError:
            ok = False
        _INDEX_CACHE[directory] = SegmentIndex(directory) if ok else None
    return _INDEX_CACHE[directory]


# ---------------------------------------------------------------------------
# chunk selection


class Chunk:
    """A run of segments [start, end] (old lines) plus its class context."""

    def __init__(self, start: int, end: int, classes: list,
                 body_start: bool):
        self.start = start
        self.end = end
        # (name, header indent, def line, first line, end line, body indent)
        self.classes = classes
        self.body_start = body_start  # chunk begins at the first statement of its body


def _touched_segments(segs: list, touched: set[int]) -> tuple[int, int] | None:
    idx = [k for k, s in enumerate(segs) if any(s[0] <= t <= s[1] for t in touched)]
    if not idx:
        return None
    return idx[0], idx[-1]


def select_chunk(entry: dict, removed: set[int], inserts: set[int], widen: int = 0) -> Chunk | None:
    """Smallest segment run containing every touched line, descending into
    class bodies when possible. widen=1 also includes the segment before an
    insertion point (an insertion may extend the previous statement)."""
    if not entry.get("simple"):
        return None
    n = entry["nlines"]
    touched = set(removed)
    for k in inserts:
        touched.add(min(k + 1, n) if n else 1)
        if widen and k >= 1:
            touched.add(k)
    if not touched or min(touched) <= 2:
        return None  # lines 1-2 may carry an encoding cookie
    if max(touched) > n:
        return None
    segs = entry["segs"]
    classes: list = []
    header_indent = ""
    body_start = True
    while True:
        run = _touched_segments(segs, touched)
        if run is None:
            return None
        a, b = run
        start, end = segs[a][0], segs[b][1]
        if a == b and segs[a][6] is not None:
            seg = segs[a]
            ch = seg[6]
            if all(ch["colon"] < t <= seg[1] for t in touched):
                # (name, header indent, def line, first line, end line, body indent)
                classes.append((seg[3], header_indent, seg[5], seg[4], seg[1], ch["indent"]))
                header_indent = ch["indent"]
                segs = ch["segs"]
                continue
        body_start = a == 0
        return Chunk(start, end, classes, body_start)


def wrap(chunk_lines: list[str], classes: list) -> tuple[bytes, int]:
    """Source: enclosing class headers, the chunk, and (inside a class) a
    trailing ``pass`` so a comment-only chunk still forms a body. The sentinel
    cannot complete an unfinished chunk, so it never hides a syntax error.
    Returns (source bytes, header line count)."""
    head = [f"{c[1]}class {c[0]}:" for c in classes]
    tail = [f"{classes[-1][5]}pass"] if classes else []
    text = "\n".join(head + chunk_lines + tail) + "\n"
    return text.encode("utf-8", "surrogateescape"), len(head)
