"""Behavior-equivalence of two versions of one Python source file.

Two proofs are tried, cheapest first:

1. Code objects. Both versions are compiled exactly as the import system
   does it (bytes, so encoding cookies apply; dont_inherit; optimize=0) and
   compared recursively, ignoring only what depends on line positions:
   co_firstlineno, the line/position tables, and the class-body constant
   stored into ``__firstlineno__`` (new in 3.13). Bytecode, constants,
   names, docstrings, defaults and annotations are all compared.
2. ASTs. Both sources are parsed and dumped without position attributes.
   Equal dumps mean the compiler sees the same program, so only line
   tables (and position-only NOPs) can differ.

When neither holds, a structural AST diff lists the changed units: each
function, lambda, class or module whose behavior may differ, with where the
change sits (body, header, class body, module level, docstring).
"""

from __future__ import annotations

import ast
import dis
import warnings
from dataclasses import dataclass, field
from types import CodeType

_CO_OPTIMIZED = 0x0001  # function-like code (not module / class body)


# ---------------------------------------------------------------------------
# compilation


@dataclass
class CompileResult:
    code: CodeType | None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def compile_source(data: bytes, filename: str) -> CompileResult:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            code = compile(data, filename, "exec", dont_inherit=True, optimize=0)
        except (SyntaxError, ValueError, UnicodeDecodeError, MemoryError, RecursionError) as e:
            return CompileResult(None, f"{type(e).__name__}: {e}")
    return CompileResult(code, None, [f"{w.category.__name__}: {w.message}" for w in caught])


# ---------------------------------------------------------------------------
# code-object comparison


def _const_key(value: object) -> object:
    """Type-exact key for a constant (1, 1.0 and True must differ)."""
    if isinstance(value, CodeType):
        return ("<code>",)  # compared structurally by the caller
    if isinstance(value, tuple):
        return (tuple, tuple(_const_key(v) for v in value))
    if isinstance(value, frozenset):
        return (frozenset, frozenset(_const_key(v) for v in value))
    if isinstance(value, (float, complex)):
        return (type(value), repr(value))
    return (type(value), value)


_SCALAR_FIELDS = (
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_flags",
    "co_stacksize",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
    "co_name",
    "co_qualname",
)


def _normalized_instructions(code: CodeType) -> list[tuple] | None:
    """Instruction stream with constants resolved and __firstlineno__ masked.

    Used only when the raw comparison fails, to absorb constant-index shifts
    caused by the masked line constant. Jump targets are mapped to
    instruction indices so offsets need not match.
    """
    try:
        instrs = list(dis.get_instructions(code))
    except Exception:  # pragma: no cover - dis failure means "not provable"
        return None
    index_of = {ins.offset: k for k, ins in enumerate(instrs)}
    out: list[tuple] = []
    for k, ins in enumerate(instrs):
        if ins.opname in ("EXTENDED_ARG", "CACHE"):
            continue
        arg: object
        if ins.opname in ("LOAD_CONST", "RETURN_CONST"):
            nxt = instrs[k + 1] if k + 1 < len(instrs) else None
            if (
                nxt is not None
                and nxt.opname == "STORE_NAME"
                and nxt.argval == "__firstlineno__"
                and not (code.co_flags & _CO_OPTIMIZED)
            ):
                arg = ("<firstlineno>",)
            elif isinstance(ins.argval, CodeType):
                arg = ("<code>", ins.argval.co_qualname)
            else:
                arg = _const_key(ins.argval)
        elif getattr(ins, "jump_target", None) is not None:
            arg = ("<jump>", index_of.get(ins.jump_target, -1))
        elif ins.opcode in dis.hasjrel or ins.opcode in dis.hasjabs:
            arg = ("<jump>", index_of.get(ins.argval, -1))
        else:
            arg = ins.argval if not isinstance(ins.argval, CodeType) else ("<code>",)
            try:
                hash(arg)
            except TypeError:
                arg = repr(arg)
        out.append((ins.opname, arg))
    return out


def code_equal(a: CodeType, b: CodeType) -> bool:
    """True when a and b behave identically (line positions ignored)."""
    for name in _SCALAR_FIELDS:
        if getattr(a, name) != getattr(b, name):
            return False
    ca, cb = a.co_consts, b.co_consts
    codes_a = [c for c in ca if isinstance(c, CodeType)]
    codes_b = [c for c in cb if isinstance(c, CodeType)]
    if len(codes_a) != len(codes_b):
        return False
    for x, y in zip(codes_a, codes_b):
        if not code_equal(x, y):
            return False
    if a.co_code == b.co_code and len(ca) == len(cb) and all(
        _const_key(x) == _const_key(y) for x, y in zip(ca, cb)
    ):
        return a.co_exceptiontable == b.co_exceptiontable
    # Slow path: class bodies differ in the __firstlineno__ constant, which can
    # also shift constant indices. Compare resolved instruction streams.
    if a.co_flags & _CO_OPTIMIZED:
        return False
    na, nb = _normalized_instructions(a), _normalized_instructions(b)
    if na is None or nb is None or na != nb:
        return False
    # exception tables hold instruction offsets; equal streams of equal-sized
    # instructions give equal tables unless EXTENDED_ARG sizes moved.
    if len(a.co_code) == len(b.co_code):
        return a.co_exceptiontable == b.co_exceptiontable
    return False


# ---------------------------------------------------------------------------
# AST helpers


class _Fingerprints:
    """Position-free structural fingerprints of AST nodes, memoized per node.

    Equal fingerprints <=> equal ``ast.dump(..., include_attributes=False)``
    (constants are keyed type-exactly), at a fraction of the cost.
    """

    def __init__(self) -> None:
        self.memo: dict[int, tuple] = {}
        self.keep: list = []  # keep nodes alive so id() stays unique

    def __call__(self, node, mask_lambdas: bool = False):
        if isinstance(node, list):
            return ("[]",) + tuple(self(x, mask_lambdas) for x in node)
        if not isinstance(node, ast.AST):
            return _const_key(node)
        key = id(node)
        if not mask_lambdas:
            got = self.memo.get(key)
            if got is not None:
                return got
        if mask_lambdas and isinstance(node, ast.Lambda):
            fp = ("Lambda", self(node.args), "<masked>")
        else:
            fp = (type(node).__name__,) + tuple(
                self(getattr(node, f, None), mask_lambdas) for f in node._fields)
        if not mask_lambdas:
            self.memo[key] = fp
            self.keep.append(node)
        return fp


_FP = _Fingerprints()


def _reset_fp() -> None:
    _FP.memo.clear()
    _FP.keep.clear()


def _dump(node) -> tuple:
    return _FP(node)


def _docstring_node(body: list[ast.stmt]) -> ast.Expr | None:
    if body and isinstance(body[0], ast.Expr):
        v = body[0].value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return body[0]
    return None


def _first_line(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", None) or []
    lines = [node.lineno] + [d.lineno for d in decos]
    return min(lines)


def _bound_names(stmt: ast.stmt) -> list[str]:
    names: list[str] = []

    def targets(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                targets(e)
        elif isinstance(t, ast.Starred):
            targets(t.value)
        elif isinstance(t, (ast.Attribute, ast.Subscript)):
            names.append(ast.unparse(t) if hasattr(ast, "unparse") else "<target>")

    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.append(stmt.name)
    elif isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            targets(t)
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        targets(stmt.target)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for a in stmt.names:
            names.append((a.asname or a.name).split(".")[0])
    elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.If, ast.While,
                           ast.Try, getattr(ast, "TryStar", ast.Try), getattr(ast, "Match", ast.If))):
        # one walk, not recursive: compound statements bind whatever they contain
        seen: set[str] = set()
        for sub in ast.walk(stmt):
            name = None
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                name = sub.id
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = sub.name
            elif isinstance(sub, ast.alias):
                name = (sub.asname or sub.name).split(".")[0]
            if name is not None and name not in seen:
                seen.add(name)
                names.append(name)
    elif hasattr(ast, "TypeAlias") and isinstance(stmt, ast.TypeAlias):
        targets(stmt.name)
    return names


def _stmt_key(stmt: ast.stmt) -> tuple:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (type(stmt).__name__, stmt.name)
    return (type(stmt).__name__, tuple(_bound_names(stmt)))


# ---------------------------------------------------------------------------
# units


@dataclass
class Unit:
    file: str
    module: str | None
    qualname: str  # "" for module level; else co_qualname style
    scope: str  # "module" | "class" | "function"
    where: str  # body | header | class-body | class-header | module-level | docstring
    old_lines: tuple[int, int] | None  # span of the unit in the pristine file
    def_line: int | None = None  # pristine co_firstlineno-style line of the unit
    names: list[str] = field(default_factory=list)
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "file": self.file,
            "module": self.module,
            "qualname": self.qualname,
            "scope": self.scope,
            "where": self.where,
            "old_lines": list(self.old_lines) if self.old_lines else None,
            "def_line": self.def_line,
            "names": self.names,
            "detail": self.detail,
        }


@dataclass
class _Ctx:
    kind: str  # module | class | function
    qualname: str  # "" for module
    def_line: int | None
    old_lines: tuple[int, int] | None


def _child_qual(ctx: _Ctx, name: str) -> str:
    if ctx.kind == "function":
        return f"{ctx.qualname}.<locals>.{name}"
    return f"{ctx.qualname}.{name}" if ctx.qualname else name


def _span(node: ast.AST) -> tuple[int, int]:
    return (_first_line(node), getattr(node, "end_lineno", node.lineno) or node.lineno)


def _mask_lambdas(stmt: ast.stmt) -> tuple:
    return _FP(stmt, mask_lambdas=True)


class _Differ:
    def __init__(self, file: str, module: str | None):
        self.file = file
        self.module = module
        self.units: list[Unit] = []

    def add(self, **kw) -> None:
        u = Unit(file=self.file, module=self.module, **kw)
        for existing in self.units:
            if (existing.qualname, existing.where, existing.old_lines) == (u.qualname, u.where, u.old_lines):
                for n in u.names:
                    if n not in existing.names:
                        existing.names.append(n)
                return
        self.units.append(u)

    def scope_change(self, ctx: _Ctx, names: list[str], detail: str) -> None:
        """A statement directly in ctx's body changed (or was added/removed)."""
        if ctx.kind == "function":
            self.add(qualname=ctx.qualname, scope="function", where="body", old_lines=ctx.old_lines,
                     def_line=ctx.def_line, names=[], detail=detail)
        elif ctx.kind == "class":
            self.add(qualname=ctx.qualname, scope="class", where="class-body", old_lines=ctx.old_lines,
                     def_line=ctx.def_line, names=names, detail=detail)
        else:
            self.add(qualname="", scope="module", where="module-level", old_lines=None,
                     def_line=None, names=names, detail=detail)

    def diff_body(self, old: list[ast.stmt], new: list[ast.stmt], ctx: _Ctx,
                  docstrings: bool = True) -> None:
        import difflib

        if docstrings:
            old_doc, new_doc = _docstring_node(old), _docstring_node(new)
            if old_doc is not None or new_doc is not None:
                od = old_doc.value.value if old_doc is not None else None
                nd = new_doc.value.value if new_doc is not None else None
                if od != nd:
                    self.add(qualname=ctx.qualname, scope=ctx.kind, where="docstring",
                             old_lines=ctx.old_lines, def_line=ctx.def_line, names=[],
                             detail="docstring changed")
                if old_doc is not None:
                    old = old[1:]
                if new_doc is not None:
                    new = new[1:]
        da = [_dump(st) for st in old]
        db = [_dump(st) for st in new]
        sm = difflib.SequenceMatcher(None, da, db, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                self.diff_block(old[i1:i2], new[j1:j2], ctx)

    def diff_block(self, old: list[ast.stmt], new: list[ast.stmt], ctx: _Ctx) -> None:
        import difflib

        ka = [_stmt_key(st) for st in old]
        kb = [_stmt_key(st) for st in new]
        sm = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
                for o, n in zip(old[i1:i2], new[j1:j2]):
                    self.diff_stmt(o, n, ctx)
            else:
                for o in old[i1:i2]:
                    self.scope_change(ctx, _bound_names(o), f"statement removed: {type(o).__name__}")
                for n in new[j1:j2]:
                    self.scope_change(ctx, _bound_names(n), f"statement added: {type(n).__name__}")

    def diff_stmt(self, o: ast.stmt, n: ast.stmt, ctx: _Ctx) -> None:
        if _dump(o) == _dump(n):
            return
        if isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef)) and type(o) is type(n) and o.name == n.name:
            self.diff_function(o, n, ctx)
            return
        if isinstance(o, ast.ClassDef) and isinstance(n, ast.ClassDef) and o.name == n.name:
            self.diff_class(o, n, ctx)
            return
        if ctx.kind != "function" and self.diff_lambdas_only(o, n, ctx):
            return
        self.scope_change(ctx, sorted(set(_bound_names(o)) | set(_bound_names(n))),
                          f"statement changed: {type(o).__name__}")

    def diff_function(self, o, n, ctx: _Ctx) -> None:
        qual = _child_qual(ctx, o.name)

        def header(f) -> tuple:
            return (_dump(list(f.decorator_list)), _dump(f.args), _dump(f.returns),
                    _dump(getattr(f, "type_params", None) or []))

        if header(o) != header(n):
            if ctx.kind == "function":
                self.scope_change(ctx, [], f"nested def header changed: {o.name}")
            else:
                self.add(qualname=qual, scope="function", where="header", old_lines=_span(o),
                         def_line=o.lineno, names=[o.name],
                         detail="decorator/signature/default/annotation changed")
        if _dump(o.body) != _dump(n.body):
            self.diff_body(o.body, n.body, _Ctx("function", qual, o.lineno, _span(o)))

    def diff_class(self, o: ast.ClassDef, n: ast.ClassDef, ctx: _Ctx) -> None:
        qual = _child_qual(ctx, o.name)

        def header(c) -> tuple:
            return (_dump(c.decorator_list), _dump(c.bases), _dump(c.keywords),
                    _dump(getattr(c, "type_params", None) or []))

        if header(o) != header(n):
            if ctx.kind == "function":
                self.scope_change(ctx, [], f"nested class header changed: {o.name}")
            else:
                self.add(qualname=qual, scope="class", where="class-header", old_lines=_span(o),
                         def_line=o.lineno, names=[o.name],
                         detail="class decorator/bases/keywords changed")
        inner = _Ctx("class", qual, o.lineno, _span(o))
        if ctx.kind == "function":
            # Rebuilt on every call: class-level changes belong to the enclosing
            # function's body; its methods are still deferred code.
            sub = _Differ(self.file, self.module)
            sub.diff_body(o.body, n.body, inner)
            for u in sub.units:
                if u.where in ("class-body", "class-header", "docstring"):
                    self.scope_change(ctx, [], f"nested class body changed: {o.name}")
                else:
                    self.units.append(u)
            return
        self.diff_body(o.body, n.body, inner)

    def diff_lambdas_only(self, o: ast.stmt, n: ast.stmt, ctx: _Ctx) -> bool:
        """If o and n differ only inside lambda bodies, report those lambdas
        (deferred code: only callers of the lambda can observe the change)."""
        lo = [x for x in ast.walk(o) if isinstance(x, ast.Lambda)]
        ln = [x for x in ast.walk(n) if isinstance(x, ast.Lambda)]
        if not lo or len(lo) != len(ln) or _mask_lambdas(o) != _mask_lambdas(n):
            return False
        found = False
        for a, b in zip(lo, ln):
            if _dump(a.body) != _dump(b.body):
                found = True
                self.add(qualname=_child_qual(ctx, "<lambda>"), scope="function", where="body",
                         old_lines=_span(a), def_line=a.lineno, names=[], detail="lambda body changed")
        return found


# ---------------------------------------------------------------------------
# entry points


@dataclass
class FileVerdict:
    neutral: bool
    level: str  # "code-equal" | "ast-equal" | "docstring-only" | "changed" | "error"
    units: list[Unit] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    mode: str = "file"  # "file" or "chunk"


def _import_failure(file: str, module: str | None, why: str) -> FileVerdict:
    unit = Unit(file=file, module=module, qualname="", scope="module", where="module-level",
                old_lines=None, names=[], detail=f"import fails: {why}")
    return FileVerdict(False, "changed", [unit], [f"{file}: patched file does not compile ({why})"])


def _finish(differ: _Differ, extra: list[Unit], file: str, module: str | None, mode: str) -> FileVerdict:
    units = extra + differ.units
    if not units:
        units = [Unit(file=file, module=module, qualname="", scope="module", where="module-level",
                      old_lines=None, names=[], detail="code differs but no AST unit isolated")]
    if all(u.where == "docstring" for u in units):
        return FileVerdict(False, "docstring-only", units, [f"{file}: only docstrings changed"], mode)
    return FileVerdict(False, "changed", units, [], mode)


def _warning_unit(file: str, module: str | None, new_warnings: list[str]) -> list[Unit]:
    if not new_warnings:
        return []
    return [Unit(file=file, module=module, qualname="", scope="module", where="module-level",
                 old_lines=None, names=[], detail="new compile warnings: " + "; ".join(new_warnings[:3]))]


def compare_sources(old: bytes, new: bytes, file: str, module: str | None) -> FileVerdict:
    """Compare two full versions of one Python file."""
    _reset_fp()
    co = compile_source(old, file)
    cn = compile_source(new, file)
    if co.code is None:
        return FileVerdict(False, "error", [], [f"pristine file does not compile: {co.error}"])
    if cn.code is None:
        return _import_failure(file, module, cn.error or "")
    warn_units = _warning_unit(file, module, [w for w in cn.warnings if w not in co.warnings])
    if not warn_units and code_equal(co.code, cn.code):
        return FileVerdict(True, "code-equal")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            to = ast.parse(old, file)
            tn = ast.parse(new, file)
    except (SyntaxError, ValueError) as e:  # pragma: no cover - compile succeeded
        return FileVerdict(False, "error", warn_units, [f"{file}: ast parse failed: {e}"])
    differ = _Differ(file, module)
    differ.diff_body(to.body, tn.body, _Ctx("module", "", None, None))
    if not warn_units and not differ.units:
        return FileVerdict(True, "ast-equal")
    return _finish(differ, warn_units, file, module, "file")


def compare_chunk(old_src: bytes, new_src: bytes, file: str, module: str | None, *,
                  flags: int, classes: list, header_lines: int, old_first_line: int,
                  docstrings: bool) -> FileVerdict | None:
    """Compare one chunk (see segindex). None means "not decidable here":
    the caller must widen the chunk or compare whole files."""
    _reset_fp()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            code_o = compile(old_src, file, "exec", flags=flags, dont_inherit=True, optimize=0)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return None
        warn_o = [f"{w.category.__name__}: {w.message}" for w in caught]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            code_n = compile(new_src, file, "exec", flags=flags, dont_inherit=True, optimize=0)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return None
        warn_n = [f"{w.category.__name__}: {w.message}" for w in caught]
    warn_units = _warning_unit(file, module, [w for w in warn_n if w not in warn_o])
    if not warn_units and code_equal(code_o, code_n):
        return FileVerdict(True, "code-equal", mode="chunk")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            to = ast.parse(old_src, file)
            tn = ast.parse(new_src, file)
    except (SyntaxError, ValueError):
        return None
    ast.increment_lineno(to, old_first_line - 1 - header_lines)
    body_o, body_n = to.body, tn.body
    for _ in classes:
        if len(body_o) != 1 or len(body_n) != 1:
            return None
        body_o, body_n = body_o[0].body, body_n[0].body
    if classes:
        body_o, body_n = body_o[:-1], body_n[:-1]  # drop the sentinel `pass`
        c = classes[-1]
        ctx = _Ctx("class", ".".join(k[0] for k in classes), c[2], (c[3], c[4]))
    else:
        ctx = _Ctx("module", "", None, None)
    differ = _Differ(file, module)
    differ.diff_body(body_o, body_n, ctx, docstrings=docstrings)
    if not warn_units and not differ.units:
        return FileVerdict(True, "ast-equal", mode="chunk")
    return _finish(differ, warn_units, file, module, "chunk")
