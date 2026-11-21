"""Comparison-site value profiles: proving an operator edit invisible to a test.

At build time the expensive tests run once more against an instrumented
sqlglot: every single-operator comparison ``a OP b`` in sqlglot's source is
compiled as ``__brcmp__(site, a, b)``, which computes exactly ``a OP b`` and
records, per test and site, a signature of the operand pair: the result of
every comparison operator on it, where that result is known to be free of
side effects (builtin values, enum members, objects with identity equality),
or "unknown" otherwise.

At run time, when a changed function differs from the pristine one only in
the operators of some single-operator comparisons, a profiled test is
unaffected if at every changed site and for every operand pair the test
produced, the old and the new operator give the same result: the test then
executes exactly as before (it passed on the pristine tree). A site the test
never evaluated cannot matter either. Anything else keeps the test.
"""

from __future__ import annotations

import ast
import enum
import operator

# Operator name -> index into the signature, plus negations.
BASE = ("is", "eq", "lt", "le", "gt", "ge", "in")
NEGATED = {"is not": "is", "ne": "eq", "not in": "in"}
AST_OPS = {
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.In: "in",
    ast.NotIn: "not in",
}
UNKNOWN = 2
OVERFLOW = (UNKNOWN,) * 7


def result(sig: tuple, op: str):
    """True/False, or None when the signature does not know it."""
    neg = op in NEGATED
    base = NEGATED.get(op, op)
    v = sig[BASE.index(base)]
    if v == UNKNOWN:
        return None
    return (not v) if neg else bool(v)


def equivalent(sigs, old_op: str, new_op: str) -> bool:
    for sig in sigs:
        a, b = result(sig, old_op), result(sig, new_op)
        if a is None or b is None or a != b:
            return False
    return True


# ---------------------------------------------------------------- recording side

_SCALARS = (type(None), bool, int, float, str, bytes)
_ORDERABLE_NUM = (bool, int, float)


def _plain(x, depth=0) -> bool:
    """A value whose ==, ordering and hashing are builtin and side-effect free."""
    t = type(x)
    if t in _SCALARS:
        return True
    if t is tuple and depth < 3 and len(x) <= 16:
        return all(_plain(y, depth + 1) for y in x)
    if t is frozenset and depth < 3 and len(x) <= 16:
        return all(_plain(y, depth + 1) for y in x)
    return False


def _eq_safe(x) -> bool:
    """== on x has no side effects and a well-defined result."""
    if _plain(x):
        return True
    t = type(x)
    if isinstance(x, enum.Enum):
        return t.__eq__ is object.__eq__ or t.__eq__ in (str.__eq__, int.__eq__, enum.Enum.__eq__)
    if isinstance(x, type):
        return type(x).__eq__ is type.__eq__
    return t.__eq__ is object.__eq__ and t.__ne__ is object.__ne__


def _order_kind(x):
    t = type(x)
    if t in _ORDERABLE_NUM:
        return "n"
    if t is str:
        return "s"
    if t is bytes:
        return "b"
    if t is tuple and len(x) <= 16 and all(type(y) in _ORDERABLE_NUM or type(y) is str for y in x):
        return "t"
    return None


def signature(a, b) -> tuple:
    out = [1 if a is b else 0]
    if _eq_safe(a) and _eq_safe(b):
        try:
            out.append(1 if a == b else 0)
        except Exception:  # noqa: BLE001
            out.append(UNKNOWN)
    else:
        out.append(UNKNOWN)
    ka, kb = _order_kind(a), _order_kind(b)
    if ka is not None and ka == kb and ka != "t":
        out += [1 if a < b else 0, 1 if a <= b else 0, 1 if a > b else 0, 1 if a >= b else 0]
    elif ka == "t" and kb == "t":
        try:
            out += [1 if a < b else 0, 1 if a <= b else 0, 1 if a > b else 0, 1 if a >= b else 0]
        except TypeError:
            out += [UNKNOWN] * 4
    else:
        out += [UNKNOWN] * 4
    tb = type(b)
    if tb in (str, bytes) and type(a) is tb:
        out.append(1 if a in b else 0)
    elif tb in (tuple, list, frozenset, set, dict) and len(b) <= 64 and _eq_safe(a):
        try:
            hash(a) if tb in (frozenset, set, dict) else None
            elems = b.keys() if tb is dict else b
            if all(_eq_safe(e) for e in elems):
                out.append(1 if a in b else 0)
            else:
                out.append(UNKNOWN)
        except TypeError:
            out.append(UNKNOWN)
    else:
        out.append(UNKNOWN)
    return tuple(out)


_OPFN = {
    "is": operator.is_,
    "is not": operator.is_not,
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "in": lambda a, b: a in b,
    "not in": lambda a, b: a not in b,
}


class Recorder:
    """Holds the site table and the current test's bucket."""

    def __init__(self) -> None:
        self.sites: list[tuple] = []  # (rel, lineno, col, end_lineno, end_col)
        self.site_ops: list = []
        self.bucket: dict[int, set] | None = None
        self.seen: set = set()  # (site, operand keys) already in the current bucket
        self.uninstrumented: set[str] = set()

    def site(self, key: tuple, op: str) -> int:
        self.sites.append(key)
        self.site_ops.append(_OPFN[op])
        return len(self.sites) - 1

    def make_hook(self):
        ops = self.site_ops
        rec = self
        sig = signature
        fast = {type(None), bool, int, str}
        plain_obj = {}  # type -> == is identity and no container semantics

        def __brcmp__(site, a, b):
            r = ops[site](a, b)
            bucket = rec.bucket
            if bucket is None:
                return r
            ta, tb = type(a), type(b)
            if ta in fast and tb in fast:
                key = (site, ta, a, tb, b, a is b)
            else:
                pa = plain_obj.get(ta)
                if pa is None:
                    pa = plain_obj[ta] = ta not in fast and ta.__eq__ is object.__eq__ and ta.__ne__ is object.__ne__ \
                        and not isinstance(a, (type, tuple, list, set, frozenset, dict, str, bytes))
                pb = plain_obj.get(tb)
                if pb is None:
                    pb = plain_obj[tb] = tb not in fast and tb.__eq__ is object.__eq__ and tb.__ne__ is object.__ne__ \
                        and not isinstance(b, (type, tuple, list, set, frozenset, dict, str, bytes))
                if (pa or ta in fast) and (pb or tb in fast) and (pa or pb):
                    # Identity-equality objects: the signature depends only on the types and a is b.
                    key = (site, ta, a if ta in fast else None, tb, b if tb in fast else None, a is b)
                else:
                    key = None
            seen = rec.seen
            if key is not None:
                if key in seen:
                    return r
                seen.add(key)
            s = bucket.get(site)
            if s is None:
                s = bucket[site] = set()
            if len(s) < 64:
                s.add(sig(a, b))
            else:
                s.add(OVERFLOW)  # too many kinds of operands: prove nothing here
            return r

        return __brcmp__

class Instrument(ast.NodeTransformer):
    def __init__(self, recorder: Recorder, rel: str) -> None:
        self.recorder = recorder
        self.rel = rel

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if len(node.ops) != 1:
            return node
        op = AST_OPS.get(type(node.ops[0]))
        if op is None:
            return node
        key = (self.rel, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
        site = self.recorder.site(key, op)
        call = ast.Call(
            func=ast.Name(id="__brcmp__", ctx=ast.Load()),
            args=[ast.Constant(site), node.left, node.comparators[0]],
            keywords=[],
        )
        return ast.copy_location(call, node)


# ---------------------------------------------------------------- run-time side


def compare_only_diff(old_node: ast.AST, new_node: ast.AST, rel: str) -> list | None:
    """[(site_key, old_op, new_op)] when the trees differ only in single comparison operators."""
    out: list = []

    def walk(a, b) -> bool:
        if type(a) is not type(b):
            return False
        if isinstance(a, ast.Compare):
            if len(a.ops) != len(b.ops):
                return False
            if len(a.ops) == 1 and type(a.ops[0]) is not type(b.ops[0]):
                oa, ob = AST_OPS.get(type(a.ops[0])), AST_OPS.get(type(b.ops[0]))
                if oa is None or ob is None:
                    return False
                out.append(((rel, a.lineno, a.col_offset, a.end_lineno, a.end_col_offset), oa, ob))
                return walk(a.left, b.left) and walk(a.comparators[0], b.comparators[0])
        if isinstance(a, ast.AST):
            for (fa, va), (fb, vb) in zip(ast.iter_fields(a), ast.iter_fields(b)):
                if fa != fb or not walk(va, vb):
                    return False
            return True
        if isinstance(a, list):
            return len(a) == len(b) and all(walk(x, y) for x, y in zip(a, b))
        return a == b and type(a) is type(b)

    if not walk(old_node, new_node):
        return None
    return out
