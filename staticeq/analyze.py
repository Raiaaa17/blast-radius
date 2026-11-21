"""analyze(patch_text, pristine_root) -> verdict dict. See README.md."""

from __future__ import annotations

import os
import posixpath

from .datamap import DataMap, load_map
from .patch import ApplyError, FilePatch, apply_hunks, parse_patch, read_pristine
from .pycompare import FileVerdict, Unit, compare_chunk, compare_sources
from .segindex import load_index, select_chunk, wrap

# Files that configure how pytest or Python itself runs: never neutral.
CONFIG_BASENAMES = {
    "conftest.py",
    "pytest.ini",
    ".pytest.ini",
    "tox.ini",
    "sitecustomize.py",
    "usercustomize.py",
}
# pytest reads these from the rootdir (repo root) at startup: neutral only when
# neither version has a pytest section and the new one still parses.
ROOT_INI_FILES = {"pyproject.toml", "setup.cfg"}
# Readers of .py source text whose result depends only on the compiled
# program (so line-only / docstring-only edits cannot change it):
# - test_typing parses sqlglot/typing/*.py with ast and looks only at
#   ImportFrom nodes; its code-object comparison is re-checked by the
#   build-time typing guard (filemap "typing_guard").
# - test_sqlglotc executes sqlglotc/setup.py with runpy.run_path.
CODE_LEVEL_SOURCE_READERS = {
    "tests/test_typing.py:TestTyping.test_no_redundant_expression_metadata",
    "tests/test_typing.py:TestTyping.test_no_redundant_expression_metadata [pkgutil]",
    "tests/test_sqlglotc.py:TestSQLGlotC.setUpClass",
    "tests/test_sqlglotc.py:TestSQLGlotC.setUpClass [pkgutil]",
    "tests/test_sqlglotc.py:TestSQLGlotC.setUpClass [runpy]",
}
IMPORTABLE_ROOTS = ("sqlglot/", "tests/")
FIXTURE_FILTER_READERS = {
    "tests/helpers.py:load_sql_fixtures",
    "tests/helpers.py:load_sql_fixture_pairs",
}


def module_name(path: str) -> str | None:
    if not path.endswith(".py") or not path.startswith(IMPORTABLE_ROOTS):
        return None
    mod = path[:-3].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


def _filter_comments(text: str) -> str:
    """tests/helpers.py::_filter_comments, applied after universal newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join([line for line in text.splitlines() if line and not line.startswith("--")])


def _is_config(path: str) -> bool:
    base = posixpath.basename(path)
    return base in CONFIG_BASENAMES or base.endswith(".pth")


def _root_ini_risky(path: str, old: bytes, new: bytes) -> str | None:
    """Why a root pyproject.toml / setup.cfg edit could change pytest, else None."""
    markers = (b"[tool:pytest]", b"[pytest]", b"tool.pytest", b"[tool.pytest")
    if any(m in old or m in new for m in markers):
        return "file carries pytest configuration"
    try:
        text = new.decode("utf-8")
    except UnicodeDecodeError:
        return "not UTF-8"
    if path == "pyproject.toml":
        try:
            import tomllib

            tomllib.loads(text)
        except Exception as e:  # noqa: BLE001
            return f"pyproject.toml no longer parses: {e}"
    else:
        import configparser

        try:
            configparser.ConfigParser(interpolation=None, strict=False).read_string(text)
        except Exception as e:  # noqa: BLE001
            return f"setup.cfg no longer parses: {e}"
    return None


def _test_ids_for_unit(u: Unit, tests: set[str] | None) -> list[str]:
    """Direct test ids for a unit that is itself a test method/class body."""
    if not u.file.startswith("tests/") or not u.qualname or tests is None:
        return []
    parts = u.qualname.split(".")
    if "<locals>" in parts:
        parts = parts[: parts.index("<locals>")]
    if len(parts) >= 2 and parts[-1].startswith("test"):
        tid = f"{u.file}::{'::'.join(parts)}"
        return [tid] if tid in tests else []
    if len(parts) == 1 and parts[0].startswith("test"):
        tid = f"{u.file}::{parts[0]}"
        return [tid] if tid in tests else []
    return []


class _Result:
    def __init__(self) -> None:
        self.neutral = True
        self.files: list[dict] = []
        self.units: list[dict] = []
        self.data_files: list[dict] = []
        self.reasons: list[str] = []

    def not_neutral(self, reason: str) -> None:
        self.neutral = False
        self.reasons.append(reason)

    def as_dict(self) -> dict:
        return {
            "neutral": self.neutral,
            "files": self.files,
            "units": self.units,
            "data_files": self.data_files,
            "reasons": self.reasons,
        }


def analyze(
    patch_text: str | bytes,
    pristine_root: str,
    *,
    data_map: DataMap | None | bool = True,
    docstrings_neutral: bool = True,
    segments=True,
) -> dict:
    """Classify one patch. Never raises on bad input: uncertainty -> not neutral.

    data_map: True loads the build-time map from its default location; None or
    False runs without it (every non-Python file then counts as relevant).
    docstrings_neutral: treat docstring-only edits as neutral (see README for
    the evidence: no test or sqlglot code reads __doc__ and no doctest is
    collected); the evidence is re-checked against the map's source readers.
    """
    res = _Result()
    dm = load_map() if data_map is True else (data_map or None)
    try:
        files = parse_patch(patch_text)
    except Exception as e:  # pragma: no cover - parser is defensive
        res.not_neutral(f"patch parse crashed: {type(e).__name__}: {e}")
        return res.as_dict()
    if not files:
        res.not_neutral("no file changes found in patch")
        return res.as_dict()
    seen_paths: set[str] = set()
    for fp in files:
        for p in {fp.old_path, fp.new_path} - {None}:
            if p in seen_paths:
                fp.problems.append(f"path {p!r} appears in more than one file section")
            seen_paths.add(p)
    test_ids = set(dm.tests) if dm else None
    seg_index = load_index() if segments is True else (segments or None)
    for fp in files:
        try:
            _analyze_file(fp, pristine_root, dm, docstrings_neutral, res, test_ids, seg_index)
        except Exception as e:  # defensive: never let analysis crash the selector
            res.not_neutral(f"{fp.path}: analysis error {type(e).__name__}: {e}")
            res.files.append({"path": fp.path, "kind": "error", "neutral": False})
    return res.as_dict()


def compare_python(fp: FilePatch, old: bytes, new: bytes, mapping: list[int], path: str,
                   mod: str | None, seg_index) -> FileVerdict:
    """Chunk-level comparison when the segment index allows it, else whole file."""
    entry = seg_index.get(path) if seg_index else None
    if entry and entry.get("simple"):
        import hashlib

        if hashlib.sha1(old).hexdigest() == entry["sha1"]:
            removed, inserts = fp.touched_old_lines()
            old_lines = old.decode("utf-8", "surrogateescape").split("\n")
            new_lines = new.decode("utf-8", "surrogateescape").split("\n")
            if old_lines and old_lines[-1] == "":
                old_lines.pop()
            if new_lines and new_lines[-1] == "":
                new_lines.pop()
            n = entry["nlines"]
            tried = set()
            for widen in (0, 1):
                chunk = select_chunk(entry, removed, inserts, widen)
                if chunk is None:
                    break
                key = (chunk.start, chunk.end, len(chunk.classes))
                if key in tried:
                    continue
                tried.add(key)
                if chunk.start > 1 and mapping[chunk.start - 1] == 0:
                    break
                if chunk.end < n and mapping[chunk.end + 1] == 0:
                    break
                new_start = mapping[chunk.start - 1] + 1 if chunk.start > 1 else 1
                new_end = mapping[chunk.end + 1] - 1 if chunk.end < n else len(new_lines)
                o_lines = old_lines[chunk.start - 1 : chunk.end]
                n_lines = new_lines[new_start - 1 : new_end]
                joined = "\n".join(o_lines + n_lines)
                if "__future__" in joined or "\r" in joined or "\f" in joined or "\x00" in joined:
                    break
                o_src, hdr = wrap(o_lines, chunk.classes)
                n_src, _ = wrap(n_lines, chunk.classes)
                docstrings = chunk.body_start if chunk.classes else chunk.start == 1
                v = compare_chunk(o_src, n_src, path, mod, flags=entry["future_flags"],
                                  classes=chunk.classes, header_lines=hdr,
                                  old_first_line=chunk.start, docstrings=docstrings)
                if v is not None:
                    return v
    return compare_sources(old, new, path, mod)


def _analyze_file(fp: FilePatch, root: str, dm: DataMap | None, docstrings_neutral: bool,
                  res: _Result, test_ids: set[str] | None, seg_index) -> None:
    path = fp.path
    entry: dict = {"path": path, "old_path": fp.old_path, "new_path": fp.new_path}
    res.files.append(entry)

    def fail(kind: str, reason: str) -> None:
        entry.update(kind=kind, neutral=False, reason=reason)
        res.not_neutral(f"{path}: {reason}")

    if fp.problems:
        return fail("unparsed", "; ".join(fp.problems))
    for p in (fp.old_path, fp.new_path):
        if p is not None and (p.startswith("/") or ".." in p.split("/")):
            return fail("unparsed", f"suspicious path {p!r}")
    is_py = path.endswith(".py")
    mod = module_name(path)

    # --- structural changes: never neutral, but say what they touch ---------
    if fp.is_binary or fp.is_rename or fp.is_copy or fp.is_new or fp.is_deleted or fp.mode_change:
        what = ("binary" if fp.is_binary else "rename" if fp.is_rename else "copy" if fp.is_copy
                else "added" if fp.is_new else "deleted" if fp.is_deleted else "mode")
        reason = {"binary": "binary change", "rename": "file renamed", "copy": "file copied",
                  "added": "new file", "deleted": "file deleted",
                  "mode": f"mode change {fp.old_mode}->{fp.new_mode}"}[what]
        fail(f"structural-{what}", reason)
        for p in {fp.old_path, fp.new_path} - {None}:
            _record_structural(p, what, dm, res, test_ids)
        return

    old = read_pristine(root, path)
    if old is None:
        return fail("missing", "patched file not found in pristine tree")
    try:
        new, _mapping = apply_hunks(old, fp.hunks)
    except ApplyError as e:
        return fail("unapplied", f"hunks do not apply: {e}")
    if new == old:
        entry.update(kind="no-op", neutral=True, reason="content unchanged")
        return

    if _is_config(path):
        return fail("config", "pytest/interpreter configuration file changed")
    if path in ROOT_INI_FILES:
        risk = _root_ini_risky(path, old, new)
        if risk:
            return fail("config", risk)

    if is_py:
        verdict = compare_python(fp, old, new, _mapping, path, mod, seg_index)
        entry["mode"] = verdict.mode
        src_ids, src_readers = dm.source_readers_of(path) if dm else ([], [])
        code_level_readers = _source_readers_are_code_level(src_readers)
        guarded = dm is not None and (path in dm.typing_guard_files or (
            "<all sqlglot/typing>" in dm.typing_guard_files and path.startswith("sqlglot/typing/")))
        line_or_doc_only = verdict.neutral or (
            verdict.level == "docstring-only" and docstrings_neutral)
        if line_or_doc_only and not guarded and (not src_ids or code_level_readers):
            why = {"code-equal": "code objects equal modulo line numbers",
                   "ast-equal": "ASTs equal modulo positions",
                   "annotation-only": "only def annotations changed under "
                                      "'from __future__ import annotations' (never evaluated)",
                   "docstring-only": "only docstrings changed; nothing reads __doc__"}[verdict.level]
            entry.update(kind="python", level=verdict.level, neutral=True, reason=why)
            return
        if dm is not None and not dm.is_relevant(path):
            entry.update(kind="python-unused", level=verdict.level, neutral=True,
                         reason="file is never imported or read by any test")
            return
        entry.update(kind="python", level=verdict.level, neutral=False)
        if line_or_doc_only:
            res.not_neutral(f"{path}: {verdict.level} edit, but tests read this file's text "
                            f"({', '.join(src_readers[:2]) or 'typing guard'})")
        else:
            res.not_neutral(f"{path}: code changed" + (
                f" ({'; '.join(verdict.reasons)})" if verdict.reasons else ""))
        for u in verdict.units:
            d = u.as_dict()
            d["package_init"] = path.endswith("__init__.py")
            d["test_ids"] = _test_ids_for_unit(u, test_ids)
            res.units.append(d)
        if src_ids:
            res.data_files.append({"path": path, "status": "source-read", "tests": src_ids,
                                   "readers": src_readers, "dir_listers": [], "known": True})
        if mod is None:
            ids, readers = dm.readers_of(path) if dm else ([], [])
            res.data_files.append({"path": path, "status": "modified-script", "tests": ids,
                                   "readers": readers, "dir_listers": [],
                                   "known": dm is not None})
        return

    # --- non-Python file modified -------------------------------------------
    if dm is None:
        res.data_files.append({"path": path, "status": "modified", "tests": [], "readers": [],
                               "dir_listers": [], "known": False})
        return fail("data", "data file changed and no file map is available")
    ids, readers = dm.readers_of(path)
    if not ids:
        mention_ids, mention_files = dm.mention_tests(path)
        if not mention_ids:
            entry.update(kind="data-unused", neutral=True,
                         reason="no test opens this file and no test source names it")
            return
        res.data_files.append({"path": path, "status": "mentioned", "tests": mention_ids,
                               "readers": [f"named in {f}" for f in mention_files],
                               "dir_listers": [], "known": True})
        return fail("data", "data file no test was seen opening, but test sources name it")
    if readers and set(readers) <= FIXTURE_FILTER_READERS and not helpers_changed(res):
        try:
            same = _filter_comments(old.decode("utf-8")) == _filter_comments(new.decode("utf-8"))
        except UnicodeDecodeError:
            same = False
        if same:
            entry.update(kind="data", neutral=True,
                         reason="only blank or '--' comment lines changed; every reader filters them")
            return
    res.data_files.append({"path": path, "status": "modified", "tests": ids, "readers": readers,
                           "dir_listers": [], "known": True})
    fail("data", f"data file read by {len(ids)} test(s) changed")


def helpers_changed(res: "_Result") -> bool:
    return any(f.get("path") == "tests/helpers.py" for f in res.files)


def _source_readers_are_code_level(readers: list[str]) -> bool:
    return bool(readers) and set(readers) <= CODE_LEVEL_SOURCE_READERS


def _record_structural(path: str, what: str, dm: DataMap | None, res: _Result,
                       test_ids: set[str] | None) -> None:
    mod = module_name(path)
    ids, readers = dm.readers_of(path) if dm else ([], [])
    listers, list_readers = dm.listers_of_parents(path) if dm else ([], [])
    attempts = dm.import_attempts(mod) if (dm and mod) else []
    res.data_files.append({
        "path": path,
        "status": what,
        "tests": sorted(set(ids) | set(attempts)),
        "readers": readers + ([f"import attempt of {mod}"] if attempts else []),
        "dir_listers": listers,
        "known": dm is not None,
    })
    if path.endswith(".py"):
        res.units.append({
            "file": path, "module": mod, "qualname": "", "scope": "module",
            "where": f"file-{what}", "old_lines": None, "def_line": None, "names": [],
            "detail": f"python file {what}", "package_init": path.endswith("__init__.py"),
            "test_ids": [],
        })
