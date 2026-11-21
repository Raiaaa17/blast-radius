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
    new_code: types.CodeType | None = None
    ranges: list | None = None  # function: old-file line ranges a test must reach; None = whole function
    old_node: object = None  # function: its ast node in the old and new trees (None if not found)
    new_node: object = None


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
    for kind, old, new in diffs:
        if trees is None:
            trees = (ast.parse(old_text), ast.parse(new_text))
        if kind == "function":
            try:
                ranges = changed_ranges(trees[0], trees[1], old, new)
            except Exception:  # noqa: BLE001 - fall back to the whole function
                ranges = None
            try:
                old_node, new_node = _find_def(trees[0], old), _find_def(trees[1], new)
            except Exception:  # noqa: BLE001
                old_node = new_node = None
            units.append(Unit("function", old.co_qualname, old, new_code=new, ranges=ranges,
                              old_node=old_node, new_node=new_node))
            continue
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


# ---------------------------------------------------------------- changed statements
#
# A function-body change is observable by a test only if the test reaches a
# changed statement. Execution under the new code is identical to the old one
# until control reaches the first changed instruction, so the tests that never
# executed any line of a changed statement (in the pristine run) cannot see
# the change. changed_ranges() returns the old-file line ranges that must have
# executed for a change to be reached: changed statements' spans, the header of
# a compound statement whose header changed, and, for inserted statements, the
# statements around the insertion point (control reaches it from the one before
# or the block's header). None means "use the whole function".

_NO_CODE = (ast.Pass, ast.Global, ast.Nonlocal)


def _is_noop(stmt: ast.stmt) -> bool:
    """Statements that may compile to no instruction (so no line event)."""
    if isinstance(stmt, _NO_CODE):
        return True
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)


def _span(node) -> tuple[int, int]:
    return (node.lineno, getattr(node, "end_lineno", None) or node.lineno)


_BLOCK_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")


def _header_dump(stmt: ast.stmt) -> str:
    """The statement with its nested blocks blanked out."""
    fields = {}
    for name, value in ast.iter_fields(stmt):
        if name in _BLOCK_FIELDS:
            continue
        fields[name] = ast.dump(value, include_attributes=False) if isinstance(value, ast.AST) else (
            [ast.dump(v, include_attributes=False) if isinstance(v, ast.AST) else repr(v) for v in value]
            if isinstance(value, list)
            else repr(value)
        )
    return f"{type(stmt).__name__}{sorted(fields.items())}"


def _header_span(stmt: ast.stmt) -> tuple[int, int]:
    """Lines of a compound statement's header (up to the start of its first block)."""
    end = stmt.lineno
    for name, value in ast.iter_fields(stmt):
        if name in _BLOCK_FIELDS:
            continue
        nodes = value if isinstance(value, list) else [value]
        for node in nodes:
            if isinstance(node, ast.AST) and getattr(node, "end_lineno", None):
                end = max(end, node.end_lineno)
    return (stmt.lineno, end)


def _stmt_ranges(stmt: ast.stmt, anchors: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Lines whose execution means control reached `stmt`."""
    if _is_noop(stmt):
        return list(anchors)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        start = min([stmt.lineno] + [d.lineno for d in stmt.decorator_list])
        return [(start, stmt.end_lineno or stmt.lineno)]
    return [_span(stmt)]


def _diff_stmt(x: ast.stmt, y: ast.stmt, anchors) -> list | None:
    if type(x) is not type(y):
        return _stmt_ranges(x, anchors)
    if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        # A nested function: its body is its own code unit; here only the
        # def statement itself (defaults, decorators) matters.
        if _header_dump(x) != _header_dump(y) or [ast.dump(d) for d in x.decorator_list] != [
            ast.dump(d) for d in y.decorator_list
        ]:
            return _stmt_ranges(x, anchors)
        return []
    if isinstance(x, ast.ClassDef):
        return _stmt_ranges(x, anchors)
    blocks = [name for name in _BLOCK_FIELDS if isinstance(getattr(x, name, None), list)]
    if not blocks:
        return _stmt_ranges(x, anchors)
    out: list = []
    header = _header_span(x)
    if _header_dump(x) != _header_dump(y):
        out.append(header)
    # What proves control reached this statement's blocks: its header when the
    # header surely has instructions; a try statement has none, nor has an
    # if/while on a constant (folded away), so fall back to the whole statement
    # or the enclosing anchors.
    if isinstance(x, (ast.Try, getattr(ast, "TryStar", ast.Try))):
        inner = [_span(x)] + list(anchors)
    elif isinstance(x, (ast.If, ast.While)) and isinstance(x.test, ast.Constant):
        inner = list(anchors)
    else:
        inner = [header]
    for name in blocks:
        a, b = getattr(x, name), getattr(y, name)
        if name in ("handlers", "cases"):
            if len(a) != len(b):
                return _stmt_ranges(x, anchors)
            for h, k in zip(a, b):
                if _header_dump(h) != _header_dump(k):
                    out.append(_header_span(h) if hasattr(h, "lineno") else header)
                hanchor = ([_header_span(h)] if hasattr(h, "lineno") else []) + inner
                got = _diff_block(h.body, k.body, hanchor)
                if got is None:
                    return None
                out += got
            continue
        got = _diff_block(a, b, inner)
        if got is None:
            return None
        out += got
    return out


def _diff_block(a: list, b: list, anchors: list) -> list | None:
    da = [ast.dump(s, include_attributes=False) for s in a]
    db = [ast.dump(s, include_attributes=False) for s in b]
    if da == db:
        return []
    out: list = []
    matcher = difflib.SequenceMatcher(a=da, b=db, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:
            for x, y in zip(a[i1:i2], b[j1:j2]):
                got = _diff_stmt(x, y, anchors)
                if got is None:
                    return None
                out += got
            continue
        for x in a[i1:i2]:
            out += _stmt_ranges(x, anchors)
        if j2 > j1:
            # Inserted statements: reached from the statement before them, or
            # from the block's header when they open the block; the statement
            # after them is reached only through them.
            if i1 > 0:
                out += _stmt_ranges(a[i1 - 1], anchors)
            else:
                out += list(anchors)
            if i2 < len(a):
                out += _stmt_ranges(a[i2], anchors)
    return out


def _find_def(tree: ast.Module, code: types.CodeType):
    """The FunctionDef/Lambda node of a code object, by qualname and first line."""
    target = code.co_qualname
    found = []

    def visit(body_owner, prefix: str) -> None:
        for node in ast.iter_child_nodes(body_owner):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{node.name}"
                if q == target:
                    found.append(node)
                visit(node, q + ".<locals>.")
            elif isinstance(node, ast.ClassDef):
                visit(node, f"{prefix}{node.name}.")
            elif isinstance(node, ast.Lambda):
                if f"{prefix}<lambda>" == target and node.lineno == code.co_firstlineno:
                    found.append(node)
                visit(node, f"{prefix}<lambda>.<locals>.")
            else:
                visit(node, prefix)

    visit(tree, "")
    if len(found) == 1:
        return found[0]
    # Several defs with one qualname (conditional definitions): match the line.
    for node in found:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        if start == code.co_firstlineno or node.lineno == code.co_firstlineno:
            return node
    return None


def changed_ranges(old_tree, new_tree, old_code, new_code) -> list[tuple[int, int]] | None:
    old_node = _find_def(old_tree, old_code)
    new_node = _find_def(new_tree, new_code)
    if old_node is None or new_node is None or type(old_node) is not type(new_node):
        return None
    if isinstance(old_node, ast.Lambda):
        return None
    if ast.dump(old_node.args, include_attributes=False) != ast.dump(new_node.args, include_attributes=False):
        return None
    if (old_node.returns is None) != (new_node.returns is None):
        return None
    body_a, body_b = old_node.body, new_node.body
    # Reaching the top level of the body = the function ran at all.
    entry = [_span(old_node)]
    got = _diff_block(body_a, body_b, entry)
    if got is None or not got:
        return None
    return sorted(set(got))
