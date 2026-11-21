"""Find which code units of a Python file a change touches.

Both versions are compiled and their code objects compared recursively,
ignoring only line and column tables. A difference confined to one function's
own code (its nested functions compared separately) is a function-body
change; a difference in a class body is a class-level change; anything else
is module-level. Identical code (comments, blank lines, formatting) is
neutral.
"""

from __future__ import annotations

import ast
import difflib
import math
import types
from dataclasses import dataclass, field

CO_OPTIMIZED = 0x0001


@dataclass
class Unit:
    kind: str  # "function" | "class" | "module"
    qualname: str  # function or class qualname; "" for the module
    old_code: types.CodeType | None = None
    names: set[str] | None = field(default=None)  # bound names a class/module change rebinds; None = unknown


@dataclass
class FileAnalysis:
    neutral: bool
    syntax_error: bool = False
    units: list[Unit] = field(default_factory=list)


def _const_eq(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a == b and math.copysign(1, a) == math.copysign(1, b) or (a != a and b != b)
    if isinstance(a, complex):
        return _const_eq(a.real, b.real) and _const_eq(a.imag, b.imag)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_const_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, frozenset):
        return a == b and sorted(map(repr, a)) == sorted(map(repr, b))
    if isinstance(a, types.CodeType):
        return False  # compared structurally by the caller
    return a == b


def _shallow_equal(a: types.CodeType, b: types.CodeType) -> bool:
    return (
        a.co_code == b.co_code
        and a.co_names == b.co_names
        and a.co_varnames == b.co_varnames
        and a.co_freevars == b.co_freevars
        and a.co_cellvars == b.co_cellvars
        and a.co_flags == b.co_flags
        and a.co_argcount == b.co_argcount
        and a.co_posonlyargcount == b.co_posonlyargcount
        and a.co_kwonlyargcount == b.co_kwonlyargcount
        and a.co_name == b.co_name
        and a.co_qualname == b.co_qualname
        and a.co_exceptiontable == b.co_exceptiontable
        and a.co_stacksize == b.co_stacksize
    )


def _const_pairs(a: types.CodeType, b: types.CodeType, kind: str):
    """Nested code pairs when every non-code constant matches, else None."""
    if len(a.co_consts) != len(b.co_consts):
        return None
    pairs = []
    for x, y in zip(a.co_consts, b.co_consts):
        if isinstance(x, types.CodeType) or isinstance(y, types.CodeType):
            if not (isinstance(x, types.CodeType) and isinstance(y, types.CodeType)):
                return None
            pairs.append((x, y))
        elif not _const_eq(x, y):
            # Python 3.13 stores a class's first line number in its body.
            if (
                kind == "class"
                and isinstance(x, int)
                and isinstance(y, int)
                and x == a.co_firstlineno
                and y == b.co_firstlineno
                and "__firstlineno__" in a.co_names
            ):
                continue
            return None
    return pairs


def _kind_of(code: types.CodeType) -> str:
    return "function" if code.co_flags & CO_OPTIMIZED else "class"


def diff_code(old: types.CodeType, new: types.CodeType, kind: str, out: list) -> None:
    pairs = _const_pairs(old, new, kind) if _shallow_equal(old, new) else None
    if pairs is None:
        out.append((kind, old, new))
        return
    for a, b in pairs:
        diff_code(a, b, _kind_of(a), out)


# ---------------------------------------------------------------- names


def _bound_names(stmt: ast.stmt) -> set[str] | None:
    """Names a simple binding statement binds, or None when it is not one."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, ast.Assign):
        names: set[str] = set()
        for target in stmt.targets:
            got = _target_names(target)
            if got is None:
                return None
            names |= got
        return names
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        return _target_names(stmt.target)
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {(alias.asname or alias.name).split(".")[0] for alias in stmt.names}
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return {"__doc__"}
    if isinstance(stmt, ast.Pass):
        return set()
    return None


def _target_names(target: ast.expr) -> set[str] | None:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            got = _target_names(elt)
            if got is None:
                return None
            names |= got
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return None


def _dump(stmt: ast.stmt) -> str:
    return ast.dump(stmt, include_attributes=False)


def changed_names(old_body: list[ast.stmt], new_body: list[ast.stmt]) -> set[str] | None:
    """Names rebound by the statements that differ between two bodies."""
    a = [_dump(s) for s in old_body]
    b = [_dump(s) for s in new_body]
    names: set[str] = set()
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for stmt in [*old_body[i1:i2], *new_body[j1:j2]]:
            got = _bound_names(stmt)
            if got is None:
                return None
            names |= got
    return names


def _find_class_body(tree: ast.Module, qualname: str) -> list[ast.stmt] | None:
    body = tree.body
    for part in qualname.split("."):
        if part == "<locals>":
            return None
        found = None
        for stmt in body:
            if isinstance(stmt, ast.ClassDef) and stmt.name == part:
                found = stmt
                break
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == part:
                found = stmt
                break
        if found is None:
            return None
        body = found.body
    return body


# ---------------------------------------------------------------- entry


def analyze_python(old_text: str, new_text: str, filename: str) -> FileAnalysis:
    try:
        old_code = compile(old_text, filename, "exec", dont_inherit=True)
    except SyntaxError:
        return FileAnalysis(neutral=False, syntax_error=True)
    try:
        new_code = compile(new_text, filename, "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        return FileAnalysis(neutral=False, syntax_error=True)
    diffs: list = []
    diff_code(old_code, new_code, "module", diffs)
    if not diffs:
        return FileAnalysis(neutral=True)
    units: list[Unit] = []
    trees = None
    for kind, old, _new in diffs:
        if kind == "function":
            units.append(Unit("function", old.co_qualname, old))
            continue
        if trees is None:
            trees = (ast.parse(old_text), ast.parse(new_text))
        if kind == "module":
            units.append(Unit("module", "", old, changed_names(trees[0].body, trees[1].body)))
        else:
            old_body = _find_class_body(trees[0], old.co_qualname)
            new_body = _find_class_body(trees[1], old.co_qualname)
            names = None
            if old_body is not None and new_body is not None:
                names = changed_names(old_body, new_body)
            units.append(Unit("class", old.co_qualname, old, names))
    return FileAnalysis(neutral=False, units=units)
