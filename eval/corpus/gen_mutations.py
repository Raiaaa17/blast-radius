#!/usr/bin/env python3
"""Generate a stress corpus of patches against sqlglot at the pinned commit.

    eval/corpus/gen_mutations.py --sqlglot CLONE --out DIR [--per-kind 8] [--seed 1]

Every patch is a real `git diff` (sample format: `diff --git`, `index`, `---`,
`+++`, hunks; renames as `rename from/to`) of one generated edit, made in the
clone's working tree, which is reset after each one. The clone must be checked
out at the pinned commit with a clean tree.

Writes DIR/commits/<id>.patch, DIR/kinds.json ({id: kind}) and DIR/meta.json
({id: {kind, desc, files}}). Ids are random 4-character names that say nothing
about the kind. Kinds aim at what the sample does not show (renames, deletions,
added files, test-file edits, import/syntax errors, class/module tables,
signatures, perf and loops, no-op refactors, multi-file mixes) as well as the
sample's own (operator flips, constants, comment-only, fixture edits).

The corpus is for evaluation only: never derive selection rules from it.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import random
import re
import string
import subprocess
import sys
from pathlib import Path

PINNED = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"


# --------------------------------------------------------------------------- tree


class Tree:
    """The clone's working tree; edits are diffed against HEAD then reset."""

    def __init__(self, root: Path):
        self.root = root
        head = self.git("rev-parse", "HEAD").strip()
        if head != PINNED:
            raise SystemExit(f"{root} is at {head}, not {PINNED}")
        if self.git("status", "--porcelain").strip():
            raise SystemExit(f"{root} has local changes")
        self.src = sorted(
            p for p in self.files("sqlglot") if p.endswith(".py") and not p.endswith("__main__.py")
        )
        self.fixtures = sorted(p for p in self.files("tests/fixtures") if p.endswith(".sql"))
        self.tests = sorted(
            p for p in self.files("tests") if p.endswith(".py") and "/fixtures/" not in p
        )

    def git(self, *args: str, check: bool = True) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args], check=check, capture_output=True, text=True
        ).stdout

    def files(self, prefix: str) -> list[str]:
        return self.git("ls-files", prefix).split()

    def read(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    def write(self, path: str, text: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def diff(self) -> str:
        self.git("add", "-A")
        # --binary implies --full-index, so use it only when a binary file changed: text patches
        # keep the sample's abbreviated `index` lines.
        binary = any(line.startswith("-\t-\t") for line in self.git("diff", "--cached", "--numstat").splitlines())
        text = self.git(
            "-c", "core.abbrev=7", "-c", "core.quotePath=true", "diff", "--cached", "-M",
            *(["--binary"] if binary else []), "--no-color", "--no-ext-diff",
            "--src-prefix=a/", "--dst-prefix=b/",
        )  # fmt: skip
        self.git("reset", "-q")
        return text

    def reset(self) -> None:
        self.git("reset", "-q", "--hard", "HEAD")
        self.git("clean", "-fdq")


# --------------------------------------------------------------------------- source helpers


class Source:
    """A Python file with byte-accurate AST spans."""

    def __init__(self, text: str):
        self.text = text
        self.data = text.encode("utf-8")
        self.tree = ast.parse(text)
        self.line_starts = [0]
        for line in self.data.splitlines(keepends=True):
            self.line_starts.append(self.line_starts[-1] + len(line))
        self.lines = text.splitlines(keepends=True)
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                child._parent = node  # type: ignore[attr-defined]

    def offset(self, line: int, col: int) -> int:
        return self.line_starts[line - 1] + col

    def span(self, node: ast.AST) -> tuple[int, int]:
        return (
            self.offset(node.lineno, node.col_offset),
            self.offset(node.end_lineno, node.end_col_offset),
        )

    def segment(self, start: int, end: int) -> str:
        return self.data[start:end].decode("utf-8")

    def node_text(self, node: ast.AST) -> str:
        return self.segment(*self.span(node))

    def replace(self, edits: list[tuple[int, int, str]]) -> str:
        data = self.data
        for start, end, new in sorted(edits, reverse=True):
            data = data[:start] + new.encode("utf-8") + data[end:]
        return data.decode("utf-8")

    def nodes(self, kind) -> list:
        return [n for n in ast.walk(self.tree) if isinstance(n, kind)]

    def is_docstring(self, node: ast.AST) -> bool:
        parent = getattr(node, "_parent", None)
        if not isinstance(parent, ast.Expr):
            return False
        holder = getattr(parent, "_parent", None)
        return isinstance(holder, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and (
            holder.body and holder.body[0] is parent
        )

    def in_fstring(self, node: ast.AST) -> bool:
        parent = getattr(node, "_parent", None)
        while parent is not None:
            if isinstance(parent, ast.JoinedStr):
                return True
            parent = getattr(parent, "_parent", None)
        return False

    def whole_lines(self, node: ast.AST) -> bool:
        """The node starts its line (after indentation) and ends it (before a comma/comment)."""
        first = self.lines[node.lineno - 1]
        last = self.lines[node.end_lineno - 1]
        before = first.encode("utf-8")[: node.col_offset].decode("utf-8", "replace")
        after = last.encode("utf-8")[node.end_col_offset :].decode("utf-8", "replace")
        return before.strip() == "" and re.fullmatch(r"\s*,?\s*(#.*)?\n?", after) is not None

    def indent_of(self, node: ast.AST) -> str:
        line = self.lines[node.lineno - 1]
        return line[: len(line) - len(line.lstrip())]

    def line_range_text(self, first: int, last: int) -> str:
        return "".join(self.lines[first - 1 : last])

    def without_lines(self, first: int, last: int) -> str:
        return "".join(self.lines[: first - 1] + self.lines[last:])

    def insert_lines(self, before_line: int, text: str) -> str:
        return "".join(self.lines[: before_line - 1] + [text] + self.lines[before_line - 1 :])

    def functions(self) -> list:
        return self.nodes((ast.FunctionDef, ast.AsyncFunctionDef))

    def body_start(self, function) -> ast.stmt | None:
        body = function.body
        index = 1 if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str) else 0
        if index >= len(body):
            return None
        return body[index]


def dotted(path: str) -> str:
    return path[:-3].replace("/", ".").removesuffix(".__init__")


# --------------------------------------------------------------------------- mutation operators
# Each operator takes (rng, tree) and returns (desc, files) after editing the
# tree, or None when it found nothing to do (the driver retries elsewhere).

SRC_GROUPS = {
    "dialects": "sqlglot/dialects/",
    "parsers": "sqlglot/parsers/",
    "generators": "sqlglot/generators/",
    "typing": "sqlglot/typing/",
    "optimizer": "sqlglot/optimizer/",
    "expressions": "sqlglot/expressions/",
    "executor": "sqlglot/executor/",
}


def pick_src(rng: random.Random, tree: Tree, prefix: str | None = None) -> str:
    """A source file, spread over the package's areas rather than by file count."""
    if prefix:
        pool = [p for p in tree.src if p.startswith(prefix)]
        return rng.choice(pool)
    groups = list(SRC_GROUPS.values()) + ["core"]
    group = rng.choice(groups)
    if group == "core":
        pool = [p for p in tree.src if p.count("/") == 1]
    else:
        pool = [p for p in tree.src if p.startswith(group)]
    return rng.choice(pool)


COMPARE_SWAP = {
    ast.Eq: ("==", "!="),
    ast.NotEq: ("!=", "=="),
    ast.Lt: ("<", "<="),
    ast.LtE: ("<=", "<"),
    ast.Gt: (">", ">="),
    ast.GtE: (">=", ">"),
    ast.Is: ("is", "is not"),
    ast.IsNot: ("is not", "is"),
    ast.In: ("in", "not in"),
    ast.NotIn: ("not in", "in"),
}


def _edit_file(tree: Tree, path: str, text: str, desc: str):
    if text == tree.read(path):
        return None
    try:
        compile(text, path, "exec")
    except SyntaxError:
        return None
    tree.write(path, text)
    return desc, [path]


def op_compare(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    candidates = [n for n in source.nodes(ast.Compare) if len(n.ops) == 1 and type(n.ops[0]) in COMPARE_SWAP]
    if not candidates:
        return None
    node = rng.choice(candidates)
    old, new = COMPARE_SWAP[type(node.ops[0])]
    start = source.span(node.left)[1]
    end = source.span(node.comparators[0])[0]
    segment = source.segment(start, end)
    if re.sub(r"\s+", " ", segment.strip(" ()\n\\")).strip() != old:
        return None
    replaced = segment.replace(old, new, 1) if old != "in" else re.sub(r"\bin\b", "not in", segment, count=1)
    if old == "is":
        replaced = re.sub(r"\bis\b", "is not", segment, count=1)
    text = source.replace([(start, end, replaced)])
    return _edit_file(tree, path, text, f"{path}:{node.lineno} `{old}` -> `{new}`")


def op_bool(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    candidates = [n for n in source.nodes(ast.BoolOp)]
    if not candidates:
        return None
    node = rng.choice(candidates)
    index = rng.randrange(len(node.values) - 1)
    start = source.span(node.values[index])[1]
    end = source.span(node.values[index + 1])[0]
    segment = source.segment(start, end)
    old = "and" if isinstance(node.op, ast.And) else "or"
    new = "or" if old == "and" else "and"
    if not re.search(rf"\b{old}\b", segment):
        return None
    text = source.replace([(start, end, re.sub(rf"\b{old}\b", new, segment, count=1))])
    return _edit_file(tree, path, text, f"{path}:{node.lineno} `{old}` -> `{new}`")


def op_arith(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    swaps = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "+"), ast.FloorDiv: ("//", "/")}
    candidates = [n for n in source.nodes(ast.BinOp) if type(n.op) in swaps]
    if not candidates:
        return None
    node = rng.choice(candidates)
    old, new = swaps[type(node.op)]
    start, end = source.span(node.left)[1], source.span(node.right)[0]
    segment = source.segment(start, end)
    if segment.strip(" ()\n\\") != old:
        return None
    text = source.replace([(start, end, segment.replace(old, new, 1))])
    return _edit_file(tree, path, text, f"{path}:{node.lineno} `{old}` -> `{new}`")


def _mutate_constant(rng, source: Source, node: ast.Constant) -> str | None:
    original = source.node_text(node)
    value = node.value
    if isinstance(value, bool):
        return "False" if value else "True"
    if isinstance(value, int) and re.fullmatch(r"-?\d+", original):
        return str(value + rng.choice([1, -1])) if value else "1"
    if isinstance(value, str) and not source.is_docstring(node) and not source.in_fstring(node):
        if len(original) < 2 or original[0] not in "'\"" or original[:3] in ('"""', "'''"):
            return None
        body = original[1:-1]
        letters = [i for i, ch in enumerate(body) if ch.isalpha()]
        if letters and "\\" not in body:
            i = rng.choice(letters)
            body = body[:i] + body[i].swapcase() + body[i + 1 :]
        else:
            body = body + "_"
        return original[0] + body + original[-1]
    if value is None and original == "None":
        return rng.choice(["0", "''", "False"])
    return None


def op_constant(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    candidates = [n for n in source.nodes(ast.Constant) if not source.is_docstring(n)]
    rng.shuffle(candidates)
    for node in candidates[:20]:
        new = _mutate_constant(rng, source, node)
        if new is None:
            continue
        old = source.node_text(node)
        text = source.replace([(*source.span(node), new)])
        result = _edit_file(tree, path, text, f"{path}:{node.lineno} {old[:40]} -> {new[:40]}")
        if result:
            return result
    return None


def _statements_in_functions(source: Source):
    for function in source.functions():
        for stmt in function.body:
            yield function, stmt
        for node in ast.walk(function):
            for field in ("body", "orelse"):
                block = getattr(node, field, None)
                if node is not function and isinstance(block, list):
                    for stmt in block:
                        if isinstance(stmt, ast.stmt):
                            yield node, stmt


def op_stmt_delete(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    candidates = [
        (holder, stmt)
        for holder, stmt in _statements_in_functions(source)
        if isinstance(stmt, (ast.Expr, ast.Assign, ast.AugAssign, ast.Return, ast.If, ast.AnnAssign))
        and not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
        and source.whole_lines(stmt)
    ]
    if not candidates:
        return None
    holder, stmt = rng.choice(candidates)
    block = holder.body if stmt in getattr(holder, "body", []) else holder.orelse
    if len(block) == 1:
        text = source.replace([(*source.span(stmt), "pass")])
    else:
        text = source.without_lines(stmt.lineno, stmt.end_lineno)
    return _edit_file(tree, path, text, f"{path}:{stmt.lineno} delete `{source.lines[stmt.lineno - 1].strip()[:60]}`")


def _insert_at_function_start(rng, tree, path, make_lines, label, filter_function=None):
    source = Source(tree.read(path))
    functions = [f for f in source.functions() if (filter_function is None or filter_function(f))]
    rng.shuffle(functions)
    for function in functions:
        first = source.body_start(function)
        if first is None or first.lineno == function.lineno:
            continue
        indent = source.indent_of(first)
        text = source.insert_lines(first.lineno, "".join(indent + line + "\n" for line in make_lines(function)))
        result = _edit_file(tree, path, text, f"{path}:{first.lineno} {label} in {function.name}()")
        if result:
            return result
    return None


def op_return_early(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    return _insert_at_function_start(rng, tree, path, lambda f: ["return None"], "early `return None`")


def op_raise(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    return _insert_at_function_start(
        rng, tree, path, lambda f: ['raise RuntimeError("unexpected")'], "`raise RuntimeError`"
    )


def op_cond_negate(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    candidates = source.nodes((ast.If, ast.While, ast.IfExp))
    if not candidates:
        return None
    node = rng.choice(candidates)
    test = source.node_text(node.test)
    text = source.replace([(*source.span(node.test), f"not ({test})")])
    return _edit_file(tree, path, text, f"{path}:{node.lineno} negate condition `{test[:50]}`")


def op_arg_swap(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    candidates = [
        n
        for n in source.nodes(ast.Call)
        if len(n.args) >= 2 and not any(isinstance(a, ast.Starred) for a in n.args[:2])
        and source.node_text(n.args[0]) != source.node_text(n.args[1])
    ]
    if not candidates:
        return None
    node = rng.choice(candidates)
    first, second = node.args[0], node.args[1]
    text = source.replace(
        [(*source.span(first), source.node_text(second)), (*source.span(second), source.node_text(first))]
    )
    return _edit_file(tree, path, text, f"{path}:{node.lineno} swap first two arguments")


def op_sig_default(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    defaults = [
        d
        for f in source.functions()
        for d in list(f.args.defaults) + [k for k in f.args.kw_defaults if k is not None]
        if isinstance(d, ast.Constant)
    ]
    rng.shuffle(defaults)
    for node in defaults:
        new = _mutate_constant(rng, source, node)
        if new is None:
            continue
        old = source.node_text(node)
        text = source.replace([(*source.span(node), new)])
        result = _edit_file(tree, path, text, f"{path}:{node.lineno} default {old} -> {new}")
        if result:
            return result
    return None


def op_sig_add_param(rng, tree, path=None):
    """Add a trailing optional parameter: callers unaffected (should be neutral)."""
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    functions = [
        f
        for f in source.functions()
        if f.args.args and not (f.args.vararg or f.args.kwarg or f.args.kwonlyargs or f.args.posonlyargs)
    ]
    if not functions:
        return None
    function = rng.choice(functions)
    last = function.args.args[-1]
    anchor = function.args.defaults[-1] if function.args.defaults else last
    end = source.span(anchor)[1]
    text = source.replace([(end, end, ", _extra=None" if function.args.defaults else ", _extra=None")])
    return _edit_file(tree, path, text, f"{path}:{function.lineno} add param _extra=None to {function.name}()")


def op_sig_remove_param(rng, tree, path=None):
    """Drop the last optional parameter: callers passing it break."""
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    functions = [
        f
        for f in source.functions()
        if len(f.args.args) >= 2 and f.args.defaults and not (f.args.kwonlyargs or f.args.posonlyargs)
        and not f.args.vararg
    ]
    if not functions:
        return None
    function = rng.choice(functions)
    previous = function.args.args[-2]
    if len(function.args.defaults) >= 2:
        previous = function.args.defaults[-2]
    start = source.span(previous)[1]
    end = source.span(function.args.defaults[-1])[1]
    name = function.args.args[-1].arg
    text = source.replace([(start, end, "")])
    # The body still uses the name: bind it locally so only callers break.
    first = source.body_start(function)
    if first is None:
        return None
    edited = Source(text) if _compiles(text) else None
    if edited is None:
        return None
    target = next(f for f in edited.functions() if f.lineno == function.lineno and f.name == function.name)
    body_first = edited.body_start(target)
    default_text = source.node_text(function.args.defaults[-1])
    text = edited.insert_lines(body_first.lineno, f"{edited.indent_of(body_first)}{name} = {default_text}\n")
    return _edit_file(tree, path, text, f"{path}:{function.lineno} remove param {name} from {function.name}()")


def _compiles(text: str) -> bool:
    try:
        compile(text, "<x>", "exec")
        return True
    except SyntaxError:
        return False


def _table_assigns(source: Source, scope: str, name_filter=None):
    """Assignments of dict/set/list/tuple literals at class or module level."""
    holders = source.nodes(ast.ClassDef) if scope == "class" else [source.tree]
    out = []
    for holder in holders:
        for stmt in holder.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(
                stmt.value, (ast.Dict, ast.Set, ast.List, ast.Tuple)
            ):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if not names or (name_filter and not name_filter(names[0], holder)):
                    continue
                out.append((holder, names[0], stmt.value))
    return out


def _table_elements(source: Source, value):
    if isinstance(value, ast.Dict):
        elements = []
        for key, item in zip(value.keys, value.values):
            if key is None:  # **spread
                continue
            first = key
            node = ast.Tuple(elts=[], ctx=ast.Load())
            node.lineno, node.col_offset = first.lineno, first.col_offset
            node.end_lineno, node.end_col_offset = item.end_lineno, item.end_col_offset
            elements.append((node, key, item))
        return elements
    return [(e, None, e) for e in value.elts if not isinstance(e, ast.Starred)]


def _table_delete(rng, tree, path, scope, name_filter=None, label="delete entry"):
    source = Source(tree.read(path))
    tables = [t for t in _table_assigns(source, scope, name_filter) if len(_table_elements(source, t[2])) >= 2]
    rng.shuffle(tables)
    for holder, name, value in tables:
        elements = [e for e in _table_elements(source, value) if source.whole_lines(e[0])]
        if not elements:
            continue
        node, key, item = rng.choice(elements)
        shown = source.lines[node.lineno - 1].strip()[:60]
        text = source.without_lines(node.lineno, node.end_lineno)
        where = f"{holder.name}." if isinstance(holder, ast.ClassDef) else ""
        result = _edit_file(tree, path, text, f"{path}:{node.lineno} {label} {where}{name}: `{shown}`")
        if result:
            return result
    return None


def _table_modify(rng, tree, path, scope, name_filter=None, label="change entry"):
    source = Source(tree.read(path))
    tables = [t for t in _table_assigns(source, scope, name_filter) if len(_table_elements(source, t[2])) >= 2]
    rng.shuffle(tables)
    for holder, name, value in tables:
        elements = _table_elements(source, value)
        node, key, item = rng.choice(elements)
        where = f"{holder.name}." if isinstance(holder, ast.ClassDef) else ""
        if isinstance(item, ast.Constant):
            new = _mutate_constant(rng, source, item)
            if new is None:
                continue
            edit = (*source.span(item), new)
        else:
            others = [e[2] for e in elements if source.node_text(e[2]) != source.node_text(item)]
            if not others:
                continue
            edit = (*source.span(item), source.node_text(rng.choice(others)))
        text = source.replace([edit])
        result = _edit_file(
            tree, path, text, f"{path}:{item.lineno} {label} {where}{name}: `{source.node_text(item)[:40]}` -> `{edit[2][:40]}`"
        )
        if result:
            return result
    return None


def _table_swap_adjacent(rng, tree, path, scope):
    """Swap two adjacent one-line entries: usually neutral, sometimes order matters."""
    source = Source(tree.read(path))
    for holder, name, value in rng.sample(_table_assigns(source, scope), k=len(_table_assigns(source, scope))):
        elements = [e for e in _table_elements(source, value) if e[0].lineno == e[0].end_lineno and source.whole_lines(e[0])]
        pairs = [(a, b) for a, b in zip(elements, elements[1:]) if b[0].lineno == a[0].lineno + 1]
        if not pairs:
            continue
        a, b = rng.choice(pairs)
        lines = list(source.lines)
        i = a[0].lineno - 1
        lines[i], lines[i + 1] = lines[i + 1], lines[i]
        text = "".join(lines)
        result = _edit_file(tree, path, text, f"{path}:{a[0].lineno} swap adjacent entries of {name}")
        if result:
            return result
    return None


def _dialect_area(rng):
    return rng.choice(["sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/", "sqlglot/"])


def op_class_table_delete(rng, tree, path=None):
    return _table_delete(rng, tree, path or pick_src(rng, tree, _dialect_area(rng)), "class")


def op_class_table_modify(rng, tree, path=None):
    return _table_modify(rng, tree, path or pick_src(rng, tree, _dialect_area(rng)), "class")


def op_module_table(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    if rng.random() < 0.5:
        return _table_delete(rng, tree, path, "module")
    return _table_modify(rng, tree, path, "module")


def op_tokenizer_keywords(rng, tree, path=None):
    path = path or rng.choice(["sqlglot/tokens.py"] + [p for p in tree.src if p.startswith("sqlglot/dialects/")])
    keep = lambda name, holder: name in ("KEYWORDS", "SINGLE_TOKENS", "COMMANDS") or "TOKEN" in name
    if rng.random() < 0.5:
        return _table_delete(rng, tree, path, "class", keep, "delete keyword")
    return _table_modify(rng, tree, path, "class", keep, "remap keyword")


def op_class_attr(rng, tree, path=None):
    path = path or pick_src(rng, tree, _dialect_area(rng))
    source = Source(tree.read(path))
    candidates = [
        stmt.value
        for holder in source.nodes(ast.ClassDef)
        for stmt in holder.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant)
    ]
    rng.shuffle(candidates)
    for node in candidates:
        new = _mutate_constant(rng, source, node)
        if new is None:
            continue
        text = source.replace([(*source.span(node), new)])
        result = _edit_file(tree, path, text, f"{path}:{node.lineno} class attribute {source.node_text(node)[:30]} -> {new[:30]}")
        if result:
            return result
    return None


def op_reorder_entries(rng, tree, path=None):
    path = path or pick_src(rng, tree, _dialect_area(rng))
    return _table_swap_adjacent(rng, tree, path, rng.choice(["class", "module"]))


def op_import_remove(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    candidates = [
        s
        for s in source.tree.body
        if isinstance(s, (ast.Import, ast.ImportFrom))
        and not (isinstance(s, ast.ImportFrom) and s.module == "__future__")
        and s.lineno == s.end_lineno
    ]
    if not candidates:
        return None
    stmt = rng.choice(candidates)
    text = source.without_lines(stmt.lineno, stmt.end_lineno)
    return _edit_file(tree, path, text, f"{path}:{stmt.lineno} remove `{source.lines[stmt.lineno - 1].strip()}`")


def _after_imports_line(source: Source) -> int:
    line = 1
    for stmt in source.tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            line = stmt.end_lineno + 1
        elif line > 1:
            break
    return line


def op_import_missing(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    text = source.insert_lines(_after_imports_line(source), "import sqlglot._compat_shim\n")
    return _edit_file(tree, path, text, f"{path} import of a module that does not exist")


def op_import_unused(rng, tree, path=None):
    """Add an unused stdlib import (neutral unless it shadows a name)."""
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    module = rng.choice(["functools", "itertools", "collections", "textwrap", "operator"])
    text = source.insert_lines(_after_imports_line(source), f"import {module}\n")
    return _edit_file(tree, path, text, f"{path} add unused `import {module}`")


def op_import_cycle(rng, tree, path=None):
    """Add an eager import of a heavy sibling module at the top (may create a cycle)."""
    path = path or rng.choice([p for p in tree.src if p.count("/") <= 2])
    source = Source(tree.read(path))
    target = rng.choice(["sqlglot.optimizer", "sqlglot.executor", "sqlglot.dialects.duckdb", "sqlglot.parser", "sqlglot.generator", "sqlglot.lineage"])
    if dotted(path).startswith(target):
        return None
    text = source.insert_lines(_after_imports_line(source), f"import {target}  # noqa: F401\n")
    return _edit_file(tree, path, text, f"{path} add top-level `import {target}`")


INIT_FILES = [
    "sqlglot/__init__.py",
    "sqlglot/dialects/__init__.py",
    "sqlglot/expressions/__init__.py",
    "sqlglot/optimizer/__init__.py",
    "sqlglot/executor/__init__.py",
    "sqlglot/parsers/__init__.py",
    "sqlglot/generators/__init__.py",
    "sqlglot/typing/__init__.py",
]


def op_init_edit(rng, tree, path=None):
    path = path or rng.choice([p for p in INIT_FILES if p in tree.src])
    operator = rng.choice([op_import_remove, op_module_table, op_constant, op_stmt_delete, op_compare])
    return operator(rng, tree, path)


def op_syntax_error(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    statements = [s for s in ast.walk(source.tree) if isinstance(s, (ast.Assign, ast.Return, ast.Expr)) and s.lineno == s.end_lineno]
    if not statements:
        return None
    stmt = rng.choice(statements)
    line = source.lines[stmt.lineno - 1]
    broken = rng.choice([line.rstrip("\n") + ")\n", line.rstrip("\n") + " :\n", line.replace("(", "", 1) if "(" in line else line.rstrip("\n") + " +\n"])
    lines = list(source.lines)
    lines[stmt.lineno - 1] = broken
    text = "".join(lines)
    if _compiles(text):
        return None
    tree.write(path, text)
    return f"{path}:{stmt.lineno} syntax error", [path]


def op_indent_error(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    statements = [s for _, s in _statements_in_functions(source) if s.lineno == s.end_lineno]
    if not statements:
        return None
    stmt = rng.choice(statements)
    lines = list(source.lines)
    lines[stmt.lineno - 1] = " " + lines[stmt.lineno - 1]
    text = "".join(lines)
    if _compiles(text):
        return None
    tree.write(path, text)
    return f"{path}:{stmt.lineno} unexpected indent", [path]


def op_file_rename_src(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    if path.endswith("__init__.py"):
        return None
    new = path[:-3] + rng.choice(["_impl", "_old", "2"]) + ".py"
    tree.git("mv", path, new)
    return f"rename {path} -> {new}", [path, new]


def op_file_rename_fixture(rng, tree, path=None):
    path = path or rng.choice(tree.fixtures)
    new = path[:-4] + "_v2.sql"
    tree.git("mv", path, new)
    return f"rename {path} -> {new}", [path, new]


def op_file_rename_test(rng, tree, path=None):
    candidates = [p for p in tree.tests if Path(p).name.startswith("test_")]
    path = path or rng.choice(candidates)
    new = str(Path(path).with_name(Path(path).stem.replace("test_", "check_") + ".py"))
    tree.git("mv", path, new)
    return f"rename {path} -> {new} (no longer collected)", [path, new]


def op_file_delete_src(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    tree.git("rm", "-q", path)
    return f"delete {path}", [path]


def op_file_delete_fixture(rng, tree, path=None):
    path = path or rng.choice(tree.fixtures)
    tree.git("rm", "-q", path)
    return f"delete {path}", [path]


def op_file_delete_test(rng, tree, path=None):
    path = path or rng.choice([p for p in tree.tests if Path(p).name.startswith("test_")])
    tree.git("rm", "-q", path)
    return f"delete {path}", [path]


NEW_MODULE = '''"""Helpers for {name}."""

from __future__ import annotations

import typing as t

from sqlglot import exp


def {func}(expression: exp.Expr) -> t.List[str]:
    """Return the names of the columns referenced by `expression`."""
    return sorted({{column.name for column in expression.find_all(exp.Column)}})
'''

NEW_TEST = '''import unittest

from sqlglot import parse_one


class Test{cls}(unittest.TestCase):
    def test_{func}(self):
        self.assertEqual(parse_one("SELECT a FROM b").sql(), "SELECT a FROM b")

    def test_{func}_columns(self):
        self.assertEqual(len(list(parse_one("SELECT a, b FROM c").find_all(parse_one("a").__class__))), 2)
'''


def _word(rng, n=6):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def op_file_add_src(rng, tree, path=None):
    area = rng.choice(["sqlglot/", "sqlglot/optimizer/", "sqlglot/dialects/", "sqlglot/executor/"])
    name = _word(rng)
    path = f"{area}{name}_utils.py"
    tree.write(path, NEW_MODULE.format(name=name, func=f"{name}_columns"))
    return f"add unused module {path}", [path]


def op_file_add_test(rng, tree, path=None):
    name = _word(rng)
    path = f"tests/test_{name}.py"
    tree.write(path, NEW_TEST.format(cls=name.capitalize(), func=name))
    return f"add new test file {path}", [path]


def op_file_add_fixture(rng, tree, path=None):
    name = _word(rng)
    path = f"tests/fixtures/optimizer/{name}.sql"
    tree.write(path, "SELECT a FROM x;\nSELECT x.a AS a FROM x AS x;\n")
    return f"add unused fixture {path}", [path]


def op_file_mode(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    (tree.root / path).chmod(0o755)
    return f"chmod +x {path} (mode-only diff)", [path]


# fixtures ------------------------------------------------------------------


def _fixture_lines(text: str) -> list[int]:
    return [i for i, line in enumerate(text.splitlines(keepends=True)) if line.strip() and not line.startswith("--") and not line.startswith("#")]


TOKENS = ["42", "x", "NULL", "a", "1", "AS", "b", "0", "y", ","]


def _mutate_sql_line(rng, line: str) -> str:
    words = line.rstrip("\n").split(" ")
    choice = rng.random()
    if choice < 0.45:
        i = rng.randrange(len(words) + 1)
        words.insert(i, rng.choice(TOKENS))
    elif choice < 0.75 and len(words) > 1:
        del words[rng.randrange(len(words))]
    else:
        i = rng.randrange(len(words))
        word = words[i]
        if word:
            j = rng.randrange(len(word))
            words[i] = word[:j] + rng.choice(string.ascii_lowercase + "0123456789") + word[j + 1 :]
    return " ".join(words) + ("\n" if line.endswith("\n") else "")


def pick_fixture(rng, tree):
    # Optimizer fixtures dominate the sample; still spread over the others.
    if rng.random() < 0.65:
        return rng.choice([f for f in tree.fixtures if "/optimizer/" in f])
    return rng.choice(tree.fixtures)


def op_fixture_modify(rng, tree, path=None):
    path = path or pick_fixture(rng, tree)
    text = tree.read(path)
    lines = text.splitlines(keepends=True)
    candidates = _fixture_lines(text)
    if not candidates:
        return None
    i = rng.choice(candidates)
    lines[i] = _mutate_sql_line(rng, lines[i])
    new = "".join(lines)
    if new == text:
        return None
    tree.write(path, new)
    return f"{path}:{i + 1} edit SQL line", [path]


def op_fixture_add_valid(rng, tree, path=None):
    """Duplicate an existing statement (or pair): new subtests that pass."""
    path = path or pick_fixture(rng, tree)
    text = tree.read(path)
    statements = [s for s in re.split(r"(?<=;)\n", text)]
    if len(statements) < 4:
        return None
    if "/optimizer/" in path:
        # Pairs: find a blank-line separated block "input;\nexpected;\n" and repeat it.
        blocks = text.split("\n\n")
        good = [b for b in blocks if b.count(";") == 2 and not b.lstrip().startswith("--")]
        if not good:
            return None
        block = rng.choice(good)
        new = text.rstrip("\n") + "\n\n" + block.strip("\n") + "\n"
    else:
        lines = text.splitlines(keepends=True)
        candidates = _fixture_lines(text)
        i = rng.choice(candidates)
        new = "".join(lines[: i + 1] + [lines[i]] + lines[i + 1 :])
    tree.write(path, new)
    return f"{path} add a duplicate statement", [path]


def op_fixture_add_invalid(rng, tree, path=None):
    """Add a new statement whose expected output is wrong (or SQL that does not round-trip)."""
    path = path or pick_fixture(rng, tree)
    text = tree.read(path)
    lines = text.splitlines(keepends=True)
    candidates = _fixture_lines(text)
    if not candidates:
        return None
    i = rng.choice(candidates)
    added = _mutate_sql_line(rng, lines[i])
    position = rng.choice([i + 1, len(lines)])
    new = "".join(lines[:position] + ([added] if added.endswith("\n") else [added + "\n"]) + lines[position:])
    tree.write(path, new)
    return f"{path}:{position + 1} add a new SQL line", [path]


def op_fixture_delete(rng, tree, path=None):
    path = path or pick_fixture(rng, tree)
    text = tree.read(path)
    lines = text.splitlines(keepends=True)
    candidates = _fixture_lines(text)
    if len(candidates) < 3:
        return None
    i = rng.choice(candidates)
    count = rng.choice([1, 2])
    new = "".join(lines[:i] + lines[i + count :])
    tree.write(path, new)
    return f"{path}:{i + 1} delete {count} line(s)", [path]


def op_fixture_comment(rng, tree, path=None):
    """A `--` comment line: dropped by the fixture loaders."""
    path = path or pick_fixture(rng, tree)
    text = tree.read(path)
    lines = text.splitlines(keepends=True)
    i = rng.randrange(len(lines) + 1)
    new = "".join(lines[:i] + ["-- " + rng.choice(["TODO: revisit", "regression test", "see issue", "edge case"]) + "\n"] + lines[i:])
    tree.write(path, new)
    return f"{path}:{i + 1} add `--` comment line", [path]


def op_fixture_hash_comment(rng, tree, path=None):
    """A `#` line: the optimizer fixture loader parses it as `# key: value` meta."""
    path = path or rng.choice([f for f in tree.fixtures if "/optimizer/" in f])
    text = tree.read(path)
    lines = text.splitlines(keepends=True)
    starts = [i for i in _fixture_lines(text) if i == 0 or not lines[i - 1].strip() or lines[i - 1].startswith("--")]
    if not starts:
        return None
    i = rng.choice(starts)
    comment = rng.choice(["# note this one is slow", "# title: renamed case", "# dialect: duckdb", "# execute: false"])
    new = "".join(lines[:i] + [comment + "\n"] + lines[i:])
    tree.write(path, new)
    return f"{path}:{i + 1} add `{comment}` line", [path]


# test files ----------------------------------------------------------------


def _test_sources(tree):
    return [p for p in tree.tests if Path(p).name.startswith("test_")]


def op_test_assert_edit(rng, tree, path=None):
    path = path or rng.choice(_test_sources(tree))
    source = Source(tree.read(path))
    calls = [
        c
        for c in source.nodes(ast.Call)
        if isinstance(c.func, ast.Attribute)
        and c.func.attr in ("validate_identity", "validate_all", "assertEqual", "assertIn", "assertTrue", "assertFalse", "assertIsNone")
        and c.args
    ]
    rng.shuffle(calls)
    for call in calls[:20]:
        constants = [a for a in call.args if isinstance(a, ast.Constant)]
        if not constants:
            continue
        node = rng.choice(constants)
        new = _mutate_constant(rng, source, node)
        if new is None:
            continue
        text = source.replace([(*source.span(node), new)])
        result = _edit_file(tree, path, text, f"{path}:{node.lineno} change expected value in {call.func.attr}")
        if result:
            return result
    return None


def op_test_weaken(rng, tree, path=None):
    """Delete an assertion line from a test (tests still pass)."""
    path = path or rng.choice(_test_sources(tree))
    source = Source(tree.read(path))
    statements = [
        (holder, s)
        for holder, s in _statements_in_functions(source)
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and source.whole_lines(s)
    ]
    if not statements:
        return None
    holder, stmt = rng.choice(statements)
    block = holder.body if stmt in getattr(holder, "body", []) else holder.orelse
    if len(block) == 1:
        text = source.replace([(*source.span(stmt), "pass")])
    else:
        text = source.without_lines(stmt.lineno, stmt.end_lineno)
    return _edit_file(tree, path, text, f"{path}:{stmt.lineno} delete a test statement")


def op_test_add(rng, tree, path=None):
    path = path or rng.choice(_test_sources(tree))
    source = Source(tree.read(path))
    classes = [c for c in source.tree.body if isinstance(c, ast.ClassDef)]
    if not classes:
        return None
    cls = rng.choice(classes)
    indent = source.indent_of(cls.body[0])
    failing = rng.random() < 0.5
    body = 'self.assertEqual(1 + 1, 3)' if failing else 'self.assertEqual(1 + 1, 2)'
    method = f"\n{indent}def test_added_{_word(rng, 4)}(self):\n{indent}    {body}\n"
    text = source.insert_lines(cls.end_lineno + 1, method) if cls.end_lineno < len(source.lines) else source.text + method
    return _edit_file(tree, path, text, f"{path} add a new {'failing' if failing else 'passing'} test to {cls.name}")


def op_test_rename(rng, tree, path=None):
    path = path or rng.choice(_test_sources(tree))
    source = Source(tree.read(path))
    methods = [f for f in source.functions() if f.name.startswith("test_")]
    if not methods:
        return None
    method = rng.choice(methods)
    lines = list(source.lines)
    lines[method.lineno - 1] = lines[method.lineno - 1].replace(f"def {method.name}(", f"def {method.name}_renamed(", 1)
    return _edit_file(tree, path, "".join(lines), f"{path}:{method.lineno} rename {method.name}")


def op_test_delete_method(rng, tree, path=None):
    path = path or rng.choice(_test_sources(tree))
    source = Source(tree.read(path))
    methods = [f for f in source.functions() if f.name.startswith("test_") and isinstance(getattr(f, "_parent", None), ast.ClassDef) and len(f._parent.body) > 1]
    if not methods:
        return None
    method = rng.choice(methods)
    first = min([method.lineno] + [d.lineno for d in method.decorator_list])
    text = source.without_lines(first, method.end_lineno)
    return _edit_file(tree, path, text, f"{path}:{first} delete {method.name}")


def op_test_syntax(rng, tree, path=None):
    path = path or rng.choice(_test_sources(tree))
    return op_syntax_error(rng, tree, path)


def op_test_helper(rng, tree, path=None):
    path = "tests/helpers.py"
    return rng.choice([op_compare, op_constant, op_stmt_delete, op_bool, op_module_table])(rng, tree, path)


def op_test_skip(rng, tree, path=None):
    """Decorate a test with unittest.skip: it stops running (cannot fail)."""
    path = path or rng.choice(_test_sources(tree))
    source = Source(tree.read(path))
    if "import unittest" not in source.text:
        return None
    methods = [f for f in source.functions() if f.name.startswith("test_")]
    if not methods:
        return None
    method = rng.choice(methods)
    first = min([method.lineno] + [d.lineno for d in method.decorator_list])
    text = source.insert_lines(first, f'{source.indent_of(method)}@unittest.skip("flaky")\n')
    return _edit_file(tree, path, text, f"{path}:{first} skip {method.name}")


# comments / docstrings / no-ops -------------------------------------------

COMMENTS = [
    "# Keep in sync with the generator",
    "# TODO: handle the remaining edge cases",
    "# Cheap check first",
    "# Intentionally not cached",
    "# See the dialect docs for details",
    "# NOTE: order matters here",
]


def op_comment(rng, tree, path=None):
    path = path or rng.choice(tree.src + _test_sources(tree)[:10])
    source = Source(tree.read(path))
    statements = [s for s in ast.walk(source.tree) if isinstance(s, ast.stmt) and source.whole_lines(s)]
    if not statements:
        return None
    stmt = rng.choice(statements)
    style = rng.random()
    if style < 0.5:
        text = source.insert_lines(stmt.lineno, source.indent_of(stmt) + rng.choice(COMMENTS) + "\n")
    elif style < 0.8 and stmt.lineno == stmt.end_lineno:
        lines = list(source.lines)
        lines[stmt.lineno - 1] = lines[stmt.lineno - 1].rstrip("\n") + "  " + rng.choice(COMMENTS) + "\n"
        text = "".join(lines)
    else:
        comments = [i for i, line in enumerate(source.lines) if line.lstrip().startswith("#")]
        if not comments:
            return None
        i = rng.choice(comments)
        lines = list(source.lines)
        lines[i] = lines[i].rstrip("\n") + " (updated)\n"
        text = "".join(lines)
    return _edit_file(tree, path, text, f"{path}:{stmt.lineno} comment-only edit")


def op_comment_in_string(rng, tree, path=None):
    """A `#`/`--` "comment" inside a multi-line string literal changes the string."""
    pool = tree.src + _test_sources(tree)
    path = path or rng.choice(pool)
    source = Source(tree.read(path))
    strings = [
        n
        for n in source.nodes(ast.Constant)
        if isinstance(n.value, str) and n.lineno < n.end_lineno and not source.is_docstring(n)
    ]
    if not strings:
        return None
    node = rng.choice(strings)
    line = rng.randrange(node.lineno, node.end_lineno)  # a line inside the literal
    indent = re.match(r"\s*", source.lines[line]).group(0)
    marker = "--" if "SELECT" in node.value.upper() else "#"
    text = source.insert_lines(line + 1, f"{indent}{marker} comment\n")
    return _edit_file(tree, path, text, f"{path}:{line + 1} comment line inside a string literal")


def op_docstring(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    docs = [n for n in source.nodes(ast.Constant) if isinstance(n.value, str) and source.is_docstring(n)]
    if docs and rng.random() < 0.7:
        node = rng.choice(docs)
        original = source.node_text(node)
        quote = original[:3] if original[:3] in ('"""', "'''") else original[0]
        body = original[len(quote) : -len(quote)]
        words = body.split(" ")
        i = rng.randrange(len(words))
        words.insert(i, rng.choice(["the", "given", "(see above)", "always"]))
        new = quote + " ".join(words) + quote
        text = source.replace([(*source.span(node), new)])
        return _edit_file(tree, path, text, f"{path}:{node.lineno} docstring wording")
    functions = [f for f in source.functions() if source.body_start(f) is source.tree and False or (f.body and not (isinstance(f.body[0], ast.Expr) and isinstance(f.body[0].value, ast.Constant)))]
    if not functions:
        return None
    function = rng.choice(functions)
    if function.body[0].lineno == function.lineno:
        return None
    indent = source.indent_of(function.body[0])
    text = source.insert_lines(function.body[0].lineno, f'{indent}"""{function.name.strip("_").replace("_", " ").capitalize()}."""\n')
    return _edit_file(tree, path, text, f"{path}:{function.lineno} add docstring to {function.name}()")


def op_noop_format(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    text = tree.read(path)
    choice = rng.random()
    if choice < 0.4:
        new = re.sub(r"\n\n\n(?=(def|class) )", "\n\n\n\n", text, count=1)
    elif choice < 0.7:
        lines = text.splitlines(keepends=True)
        code = [i for i, line in enumerate(lines) if line.strip() and not line.rstrip().endswith("\\")]
        i = rng.choice(code)
        lines[i] = lines[i].rstrip("\n") + "   \n"
        new = "".join(lines)
    else:
        new = text + "\n"
    return _edit_file(tree, path, new, f"{path} whitespace-only change")


def op_noop_rename_local(rng, tree, path=None):
    """Rename a local variable consistently inside one function."""
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    functions = source.functions()
    rng.shuffle(functions)
    for function in functions[:30]:
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Global, ast.Nonlocal, ast.JoinedStr)) for n in ast.walk(function) if n is not function):
            continue
        assigned = {n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        params = {a.arg for a in function.args.args + function.args.kwonlyargs}
        names = sorted(assigned - params)
        if not names:
            continue
        name = rng.choice(names)
        new_name = f"{name}_value"
        all_names = {n.id for n in ast.walk(source.tree) if isinstance(n, ast.Name)}
        if new_name in all_names or "locals()" in source.node_text(function):
            continue
        uses = [n for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == name]
        text = source.replace([(*source.span(n), new_name) for n in uses])
        result = _edit_file(tree, path, text, f"{path}:{function.lineno} rename local {name} -> {new_name} in {function.name}()")
        if result:
            return result
    return None


def op_noop_pass(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    return _insert_at_function_start(rng, tree, path, lambda f: ["pass"], "`pass`")


def op_noop_reorder_functions(rng, tree, path=None):
    """Swap two adjacent undecorated top-level functions."""
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    body = source.tree.body
    pairs = [
        (a, b)
        for a, b in zip(body, body[1:])
        if isinstance(a, ast.FunctionDef) and isinstance(b, ast.FunctionDef) and not a.decorator_list and not b.decorator_list
    ]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    lines = source.lines
    a_text = "".join(lines[a.lineno - 1 : a.end_lineno])
    gap = "".join(lines[a.end_lineno : b.lineno - 1])
    b_text = "".join(lines[b.lineno - 1 : b.end_lineno])
    text = "".join(lines[: a.lineno - 1]) + b_text + gap + a_text + "".join(lines[b.end_lineno :])
    return _edit_file(tree, path, text, f"{path}:{a.lineno} swap functions {a.name} and {b.name}")


def op_nonsrc(rng, tree, path=None):
    """Edit a file outside sqlglot/ and tests/ (README, CHANGELOG, setup, docs, benchmarks)."""
    pool = [p for p in tree.git("ls-files").split() if not p.startswith(("sqlglot/", "tests/")) and p.endswith((".md", ".py", ".toml", ".cfg", ".yml", ".yaml"))]
    path = path or rng.choice(pool)
    text = tree.read(path)
    new = text + ("\n# local note\n" if path.endswith((".py", ".toml", ".cfg", ".yml", ".yaml")) else "\nSee the changelog for details.\n")
    tree.write(path, new)
    return f"{path} edit outside the package and tests", [path]


# performance / loops ---------------------------------------------------------

HOT_PATHS = [
    ("sqlglot/tokenizer_core.py", None),
    ("sqlglot/parser.py", "_parse_"),
    ("sqlglot/generator.py", "_sql"),
    ("sqlglot/expressions/core.py", None),
    ("sqlglot/optimizer/scope.py", None),
    ("sqlglot/optimizer/simplify.py", None),
    ("sqlglot/optimizer/qualify_columns.py", None),
]


def op_perf(rng, tree, path=None):
    file, prefix = rng.choice(HOT_PATHS)
    if file not in tree.src:
        return None
    work = rng.choice([200, 2000, 20000])
    return _insert_at_function_start(
        rng,
        tree,
        file,
        lambda f: [f"for _ in range({work}):", "    pass"],
        f"busy loop x{work}",
        (lambda f: f.name.startswith(prefix)) if prefix else None,
    )


def op_loop(rng, tree, path=None):
    """An infinite loop in a narrow function (a dialect-specific generator/parser helper)."""
    path = path or pick_src(rng, tree, rng.choice(["sqlglot/generators/", "sqlglot/parsers/", "sqlglot/dialects/"]))
    return _insert_at_function_start(rng, tree, path, lambda f: ["while True:", "    pass"], "infinite loop")


def op_recursion(rng, tree, path=None):
    path = path or pick_src(rng, tree)

    def lines(function):
        args = ", ".join(a.arg for a in function.args.args)
        call = function.name if not function.args.args or function.args.args[0].arg != "self" else f"self.{function.name}"
        call_args = ", ".join(a.arg for a in function.args.args[1:]) if call.startswith("self.") else args
        return [f"return {call}({call_args})"]

    return _insert_at_function_start(
        rng, tree, path, lines, "unbounded recursion",
        lambda f: not (f.args.vararg or f.args.kwarg or f.args.kwonlyargs or f.decorator_list) and isinstance(getattr(f, "_parent", None), (ast.Module, ast.ClassDef)),
    )


def op_leak(rng, tree, path=None):
    """Retain every instance in a class-level list (memory/weakref tests notice)."""
    path = path or rng.choice(["sqlglot/generator.py", "sqlglot/parser.py", "sqlglot/schema.py", "sqlglot/tokens.py", "sqlglot/dialects/dialect.py", "sqlglot/optimizer/scope.py"])
    return _insert_at_function_start(
        rng, tree, path,
        lambda f: ['type(self)._instances = [*getattr(type(self), "_instances", ()), self]'],
        "retain instances", lambda f: f.name == "__init__" and f.args.args and f.args.args[0].arg == "self",
    )


# classes, decorators, annotations -------------------------------------------


def op_expr_add_class(rng, tree, path=None):
    """Add a new Func expression class: it auto-registers by name in the parser's function table."""
    path = path or rng.choice([p for p in tree.src if p.startswith("sqlglot/expressions/") and not p.endswith(("__init__.py", "core.py", "builders.py"))])
    source = Source(tree.read(path))
    if "Func" not in source.text:
        return None
    name = "".join(w.capitalize() for w in [_word(rng, 5), _word(rng, 4)])
    body = rng.choice(["    pass\n", '    arg_types = {"this": True, "expression": False}\n'])
    text = source.text.rstrip("\n") + f"\n\n\nclass {name}(Expression, Func):\n{body}"
    if "from sqlglot.expressions.core import" not in text or "Expression" not in text.split("class ")[0]:
        return None
    return _edit_file(tree, path, text, f"{path} add new function expression {name}")


def op_class_add_method(rng, tree, path=None):
    """Add an unused method to a class (a generator/parser/dialect class or an expression)."""
    path = path or pick_src(rng, tree, _dialect_area(rng))
    source = Source(tree.read(path))
    classes = [c for c in source.nodes(ast.ClassDef) if c.body and c.end_lineno]
    if not classes:
        return None
    cls = rng.choice(classes)
    last = cls.body[-1]
    indent = source.indent_of(cls.body[0])
    name = f"_{_word(rng, 5)}_helper"
    method = f"\n{indent}def {name}(self, *args, **kwargs):\n{indent}    return None\n"
    text = source.insert_lines(last.end_lineno + 1, method)
    return _edit_file(tree, path, text, f"{path}:{cls.lineno} add unused method {cls.name}.{name}")


def op_class_base(rng, tree, path=None):
    """Change a class's base to its grandparent-ish: e.g. a dialect parser inheriting the base Parser."""
    path = path or pick_src(rng, tree, rng.choice(["sqlglot/parsers/", "sqlglot/generators/", "sqlglot/dialects/"]))
    source = Source(tree.read(path))
    candidates = []
    for cls in source.nodes(ast.ClassDef):
        for base in cls.bases:
            text = source.node_text(base)
            if isinstance(base, ast.Name) and text.endswith(("Parser", "Generator", "Dialect")) and text not in ("Parser", "Generator", "Dialect"):
                replacement = {"Parser": "parser.Parser", "Generator": "generator.Generator", "Dialect": "Dialect"}[
                    next(k for k in ("Parser", "Generator", "Dialect") if text.endswith(k))
                ]
                module = replacement.split(".")[0]
                if "." in replacement and not re.search(rf"^from sqlglot import .*\b{module}\b|^import sqlglot\.{module}|^from sqlglot import {module}", source.text, re.M):
                    continue
                if replacement == "Dialect" and not re.search(r"\bDialect\b", source.text.split("class ")[0]):
                    continue
                candidates.append((cls, base, replacement))
    if not candidates:
        return None
    cls, base, replacement = rng.choice(candidates)
    text = source.replace([(*source.span(base), replacement)])
    return _edit_file(tree, path, text, f"{path}:{cls.lineno} base of {cls.name}: {source.node_text(base)} -> {replacement}")


def op_decorator_remove(rng, tree, path=None):
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    decorated = [
        (f, d)
        for f in source.functions()
        for d in f.decorator_list
        if d.lineno == d.end_lineno and source.lines[d.lineno - 1].strip().startswith("@")
    ]
    if not decorated:
        return None
    function, decorator = rng.choice(decorated)
    shown = source.lines[decorator.lineno - 1].strip()
    text = source.without_lines(decorator.lineno, decorator.lineno)
    return _edit_file(tree, path, text, f"{path}:{decorator.lineno} remove `{shown}` from {function.name}()")


def op_noop_annotation(rng, tree, path=None):
    """Change a type annotation only (modules use `from __future__ import annotations`)."""
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    if "from __future__ import annotations" not in source.text:
        return None
    annotations = [
        a.annotation
        for f in source.functions()
        for a in f.args.args + f.args.kwonlyargs
        if a.annotation is not None
    ] + [f.returns for f in source.functions() if f.returns is not None]
    if not annotations:
        return None
    node = rng.choice(annotations)
    old = source.node_text(node)
    new = "t.Any" if re.search(r"^import typing as t$", source.text, re.M) and old != "t.Any" else f"{old} | None"
    text = source.replace([(*source.span(node), new)])
    return _edit_file(tree, path, text, f"{path}:{node.lineno} annotation {old[:30]} -> {new[:30]}")


def op_fixture_inline_comment(rng, tree, path=None):
    """An inline `--` comment on a SQL line: not filtered, so it reaches the parser."""
    path = path or pick_fixture(rng, tree)
    text = tree.read(path)
    lines = text.splitlines(keepends=True)
    candidates = _fixture_lines(text)
    if not candidates:
        return None
    i = rng.choice(candidates)
    lines[i] = lines[i].rstrip("\n") + " -- " + rng.choice(["note", "keep", "see above"]) + "\n"
    tree.write(path, "".join(lines))
    return f"{path}:{i + 1} add inline `--` comment", [path]


# shared-state pollution and dispatch additions --------------------------------
# Shapes from br-research report section C: a narrowly reached function that mutates a
# class- or module-level container (victims are later tests in suite order), and newly
# added dispatch entries (`<key>_sql` methods, TRANSFORMS and parser FUNCTIONS entries)
# that attribute-level maps of the pristine code cannot see.

INTROSPECTION: dict = {}
TEST_WORDS: set = set()


def _introspection(tree: Tree) -> dict:
    if not INTROSPECTION:
        INTROSPECTION.update(json.loads(Path(tree.introspect_path).read_text()))
        words = set()
        for test in tree.tests:
            words.update(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\(", tree.read(test)))
        TEST_WORDS.update(words)
    return INTROSPECTION


def _funcs_by_class(tree):
    return {f["class"]: f for f in _introspection(tree)["funcs"]}


def _class_node(source: Source, name: str):
    return next((c for c in source.nodes(ast.ClassDef) if c.name == name), None)


def _exp_imported(source: Source) -> bool:
    return re.search(r"^from sqlglot import .*\bexp\b|^from sqlglot import exp$|^import sqlglot.expressions as exp", source.text, re.M) is not None


def _insert_in_method(rng, tree, path, class_name, line, label, prefer=None):
    source = Source(tree.read(path))
    cls = _class_node(source, class_name)
    if cls is None:
        return None
    methods = [f for f in cls.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name != "__init__"]
    if prefer:
        preferred = [f for f in methods if f.name.startswith(prefer) or f.name.endswith(prefer)]
        methods = preferred or methods
    rng.shuffle(methods)
    for function in methods:
        first = source.body_start(function)
        if first is None or first.lineno == function.lineno:
            continue
        if not function.args.args or function.args.args[0].arg != "self":
            continue
        text = source.insert_lines(first.lineno, source.indent_of(first) + line + "\n")
        result = _edit_file(tree, path, text, f"{path}:{first.lineno} {label} in {class_name}.{function.name}()")
        if result:
            return result
    return None


def op_research_c1(rng, tree, path=None):
    """br-research C1 verbatim: BigQuery _parse_make_interval pops FUNCTIONS["DATE_DIFF"]."""
    path = "sqlglot/parsers/bigquery.py"
    text = tree.read(path)
    anchor = "    def _parse_make_interval(self) -> exp.MakeInterval:\n        expr = exp.MakeInterval()\n"
    if anchor not in text:
        return None
    text = text.replace(anchor, anchor + '        self.FUNCTIONS.pop("DATE_DIFF", None)\n', 1)
    return _edit_file(tree, path, text, f"{path} research C1: _parse_make_interval pops FUNCTIONS['DATE_DIFF']")


def op_research_c2(rng, tree, path=None):
    """br-research C2 verbatim: DuckDBGenerator gains abs_sql."""
    path = "sqlglot/generators/duckdb.py"
    text = tree.read(path)
    anchor = '    _TRYCAST_DATE_MON_FMT = "%d-%b-%Y"\n\n'
    if anchor not in text:
        return None
    method = (
        "    def abs_sql(self, expression: exp.Abs) -> str:\n"
        '        return self.func("ABS", expression.this, expression.this)\n\n'
    )
    text = text.replace(anchor, anchor + method, 1)
    return _edit_file(tree, path, text, f"{path} research C2: DuckDBGenerator.abs_sql added")


def _dialect_classes(tree, kind):
    entries = _introspection(tree)["generators" if kind == "generator" else "parsers"]
    return [e for e in entries if not e["file"].endswith(("/__init__.py", "/base.py")) and e["file"] in tree.src]


def op_pollute_class_container(rng, tree, path=None):
    """A dialect method mutates a class-level table it shares (self.FUNCTIONS, TRANSFORMS, _dispatch)."""
    kind = rng.choice(["parser", "generator"])
    entry = rng.choice(_dialect_classes(tree, kind))
    source = Source(tree.read(entry["file"]))
    if kind == "parser":
        container = rng.choice(["FUNCTIONS", "FUNCTIONS", "FUNCTION_PARSERS"])
        keys = entry["functions"] if container == "FUNCTIONS" else entry["function_parsers"]
        seen = [k for k in keys if k in TEST_WORDS] or keys
        if not seen:
            return None
        key = rng.choice(seen)
        line = f"self.{container}.pop({key!r}, None)"
        return _insert_in_method(rng, tree, entry["file"], entry["class"], line, f"`{line}`", prefer="_parse_")
    if not _exp_imported(source) or not entry["transforms"]:
        return None
    key = rng.choice(entry["transforms"])
    container = rng.choice(["TRANSFORMS", "_dispatch"])
    line = f"self.{container}.pop(exp.{key}, None)"
    return _insert_in_method(rng, tree, entry["file"], entry["class"], line, f"`{line}`", prefer="_sql")


def op_pollute_module_container(rng, tree, path=None):
    """A function mutates a module-level dict/set/list of its own module."""
    path = path or pick_src(rng, tree)
    source = Source(tree.read(path))
    containers = []
    for stmt in source.tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, (ast.Dict, ast.Set, ast.List)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not names:
                continue
            value = stmt.value
            items = [k for k in value.keys if k is not None] if isinstance(value, ast.Dict) else list(value.elts)
            items = [i for i in items if isinstance(i, (ast.Constant, ast.Attribute)) and i.lineno == i.end_lineno]
            if items:
                containers.append((names[0], value, items))
    functions = [f for f in source.functions() if f.name != "__init__" and source.body_start(f) is not None]
    if not containers or not functions:
        return None
    name, value, items = rng.choice(containers)
    key = source.node_text(rng.choice(items))
    if isinstance(value, ast.Dict):
        line = f"{name}.pop({key}, None)"
    elif isinstance(value, ast.Set):
        line = f"{name}.discard({key})"
    else:
        line = f"{name}.remove({key}) if {key} in {name} else None"
    rng.shuffle(functions)
    for function in functions:
        first = source.body_start(function)
        if first.lineno == function.lineno:
            continue
        text = source.insert_lines(first.lineno, source.indent_of(first) + line + "\n")
        result = _edit_file(tree, path, text, f"{path}:{first.lineno} `{line}` in {function.name}()")
        if result:
            return result
    return None


def _pick_func(rng, tree, exclude_classes, need_key_free=None):
    funcs = [f for f in _introspection(tree)["funcs"] if not f["key"].startswith("_") and f["class"] not in exclude_classes]
    if need_key_free is not None:
        funcs = [f for f in funcs if f"{f['key']}_sql" not in need_key_free]
    exercised = [f for f in funcs if any(n in TEST_WORDS for n in f["sql_names"])]
    pool = exercised if exercised and rng.random() < 0.8 else funcs
    return rng.choice(pool) if pool else None


def _sql_name(func):
    return func["sql_names"][0] if func["sql_names"] else func["key"].upper()


def op_dispatch_add_sql_method(rng, tree, path=None):
    """A dialect generator gains `<key>_sql` for an expression it used to generate by fallback."""
    entry = rng.choice(_dialect_classes(tree, "generator"))
    source = Source(tree.read(entry["file"]))
    if not _exp_imported(source):
        return None
    func = _pick_func(rng, tree, set(entry["transforms"]), set(entry["sql_methods"]))
    cls = _class_node(source, entry["class"])
    if func is None or cls is None:
        return None
    name = _sql_name(func)
    body = rng.choice(
        [
            f'return self.func("{name}", expression.this)',
            f'return self.func("{name}", expression.this, expression.this)',
            f'return self.func("{name}", *(v for v in expression.args.values() if isinstance(v, exp.Expr)))',
            "return self.function_fallback_sql(expression)",
        ]
    )
    methods = [f for f in cls.body if isinstance(f, ast.FunctionDef)]
    indent = source.indent_of(cls.body[0])
    method = f"{indent}def {func['key']}_sql(self, expression: exp.{func['class']}) -> str:\n{indent}    {body}\n\n"
    at = (methods[0].decorator_list[0].lineno if methods and methods[0].decorator_list else methods[0].lineno) if methods else cls.end_lineno + 1
    if not methods:
        method = "\n" + method.rstrip("\n") + "\n"
    text = source.insert_lines(at, method)
    return _edit_file(tree, entry["file"], text, f"{entry['file']} add {entry['class']}.{func['key']}_sql: `{body}`")


def _dict_literal(source: Source, cls, name: str):
    for stmt in cls.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, ast.Dict):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return stmt.value
    return None


def _insert_dict_entry(source: Source, value: ast.Dict, entry_text: str) -> str | None:
    """Insert `entry_text,` as a new line after the dict's first element (often the ** spread)."""
    first = value.values[0] if value.keys and value.keys[0] is None else (value.keys[0] if value.keys else None)
    if first is None or value.lineno == value.end_lineno:
        return None
    end_line = (value.values[0] if value.keys[0] is None else value.values[0]).end_lineno
    line = source.lines[end_line - 1]
    if not line.rstrip().endswith(","):
        return None
    indent = source.indent_of(first)
    return source.insert_lines(end_line + 1, f"{indent}{entry_text},\n")


def op_dispatch_add_transform(rng, tree, path=None):
    """A dialect generator's TRANSFORMS gains an entry for an expression it did not map."""
    entries = [e for e in _dialect_classes(tree, "generator") if e["own_transforms"]]
    entry = rng.choice(entries)
    source = Source(tree.read(entry["file"]))
    cls = _class_node(source, entry["class"])
    value = _dict_literal(source, cls, "TRANSFORMS") if cls else None
    if value is None or not _exp_imported(source):
        return None
    override = rng.random() < 0.3  # TRANSFORMS take precedence over an existing <key>_sql method
    func = _pick_func(rng, tree, set(entry["transforms"]), None if override else set(entry["sql_methods"]))
    if func is None:
        return None
    other = _pick_func(rng, tree, set())
    target = _sql_name(other) if other and rng.random() < 0.5 else _sql_name(func)
    if re.search(r"\brename_func\b", source.text.split("class ")[0]):
        mapped = f'rename_func("{target}")'
    else:
        mapped = f'lambda self, e: self.func("{target}", e.this)'
    text = _insert_dict_entry(source, value, f"exp.{func['class']}: {mapped}")
    if text is None:
        return None
    return _edit_file(tree, entry["file"], text, f"{entry['file']} {entry['class']}.TRANSFORMS += exp.{func['class']}: {mapped}")


def op_dispatch_add_parser_function(rng, tree, path=None):
    """A dialect parser's FUNCTIONS gains a name that used to parse as an anonymous function."""
    entries = [e for e in _dialect_classes(tree, "parser") if e["own_functions"]]
    entry = rng.choice(entries)
    source = Source(tree.read(entry["file"]))
    cls = _class_node(source, entry["class"])
    value = _dict_literal(source, cls, "FUNCTIONS") if cls else None
    if value is None or not _exp_imported(source):
        return None
    known = set(entry["functions"]) | set(entry["function_parsers"]) | set(entry["no_paren"])
    dialect = Path(entry["file"]).stem
    test_file = f"tests/dialects/test_{dialect}.py"
    local = set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\(", tree.read(test_file))) if test_file in tree.tests else set()
    candidates = sorted((local or TEST_WORDS) - known)
    if not candidates:
        return None
    name = rng.choice(candidates)
    func = _pick_func(rng, tree, set())
    if func is None:
        return None
    text = _insert_dict_entry(source, value, f'"{name}": exp.{func["class"]}.from_arg_list')
    if text is None:
        return None
    return _edit_file(tree, entry["file"], text, f'{entry["file"]} {entry["class"]}.FUNCTIONS += "{name}": exp.{func["class"]}.from_arg_list')


# --------------------------------------------------------------------------- round 3
# Kinds the first rounds lacked: patch-format edges (CRLF, no final newline, byte order mark,
# binary data, quoted paths, a submodule bump, an empty patch), a file outside the package that
# the tests execute (sqlglotc/setup.py), a new dialect and its registries, renamed methods,
# Expression arg_types, fixture whitespace, a tests/conftest.py, and sweeps over many files.


def _write_bytes(tree: Tree, path: str, data: bytes) -> None:
    target = tree.root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _small(tree: Tree, paths: list[str], low: int, high: int) -> list[str]:
    return [p for p in paths if low < (tree.root / p).stat().st_size < high]


def op_fixture_blank_line(rng, tree, path=None):
    """A blank line inside a fixture: the optimizer loader splits its cases on blank lines."""
    path = path or pick_fixture(rng, tree)
    text = tree.read(path)
    lines = text.splitlines(keepends=True)
    candidates = [i for i in _fixture_lines(text) if i > 0]
    if not candidates:
        return None
    i = rng.choice(candidates)
    tree.write(path, "".join(lines[:i] + ["\n"] + lines[i:]))
    return f"{path}:{i + 1} insert a blank line", [path]


def op_fixture_trailing_space(rng, tree, path=None):
    path = path or pick_fixture(rng, tree)
    text = tree.read(path)
    lines = text.splitlines(keepends=True)
    candidates = [i for i in _fixture_lines(text) if lines[i].endswith("\n")]
    if not candidates:
        return None
    i = rng.choice(candidates)
    lines[i] = lines[i][:-1] + rng.choice([" ", "  ", "\t"]) + "\n"
    tree.write(path, "".join(lines))
    return f"{path}:{i + 1} trailing whitespace", [path]


def op_format_crlf(rng, tree, path=None):
    """A whole file converted to CRLF line endings (every line of it changes in the diff)."""
    if path is None:
        pool = _small(tree, tree.fixtures, 200, 30000) if rng.random() < 0.5 else _small(tree, tree.src, 500, 15000)
        path = rng.choice(pool)
    data = (tree.root / path).read_bytes()
    if b"\r" in data:
        return None
    _write_bytes(tree, path, data.replace(b"\n", b"\r\n"))
    return f"{path} converted to CRLF line endings", [path]


def op_format_no_eol(rng, tree, path=None):
    path = path or (rng.choice(tree.fixtures) if rng.random() < 0.5 else pick_src(rng, tree))
    text = tree.read(path)
    if not text.endswith("\n"):
        return None
    tree.write(path, text.rstrip("\n"))
    return f"{path} drop the final newline", [path]


def op_format_bom(rng, tree, path=None):
    path = path or (pick_fixture(rng, tree) if rng.random() < 0.5 else pick_src(rng, tree))
    data = (tree.root / path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return None
    _write_bytes(tree, path, b"\xef\xbb\xbf" + data)
    return f"{path} gains a UTF-8 byte order mark", [path]


def op_file_binary(rng, tree, path=None):
    """A gzipped TPC-H/TPC-DS table changes (GIT binary patch), or a binary image is added."""
    tables = [p for p in tree.files("tests/fixtures") if p.endswith(".csv.gz")]
    mode = rng.choice(["value", "row", "truncate", "image"])
    if mode == "image":
        path = f"posts/{_word(rng)}.png"
        _write_bytes(tree, path, b"\x89PNG\r\n\x1a\n" + bytes(rng.randrange(256) for _ in range(300)))
        return f"add binary image {path}", [path]
    path = path or rng.choice(tables)
    raw = gzip.decompress((tree.root / path).read_bytes())
    lines = raw.decode("utf-8").splitlines(keepends=True)
    if len(lines) < 3:
        return None
    if mode == "truncate":
        data = (tree.root / path).read_bytes()
        _write_bytes(tree, path, data[: len(data) // 2])
        return f"{path} truncated (corrupt gzip)", [path]
    i = rng.randrange(1, len(lines))
    if mode == "row":
        del lines[i]
        desc = f"{path} delete data row {i}"
    else:
        cells = lines[i].rstrip("\n").split("|")
        numeric = [j for j, c in enumerate(cells) if re.fullmatch(r"-?\d+(\.\d+)?", c)]
        if not numeric:
            return None
        j = rng.choice(numeric)
        cells[j] = str(int(float(cells[j])) + 1) if "." not in cells[j] else f"{float(cells[j]) + 1:.2f}"
        lines[i] = "|".join(cells) + "\n"
        desc = f"{path} row {i} column {j} value changed"
    _write_bytes(tree, path, gzip.compress("".join(lines).encode("utf-8"), mtime=0))
    return desc, [path]


def op_file_quoted_path(rng, tree, path=None):
    """A path that git quotes or that holds spaces (non-ASCII or space in the name)."""
    if rng.random() < 0.5:
        path = rng.choice(["posts/notes on café.md", "posts/größe notes.md", "posts/release notes/read me.md"])
        tree.write(path, "# Notes\n\nNothing to see here.\n")
        return f"add {path} (quoted path in the diff)", [path]
    old = path or pick_fixture(rng, tree)
    new = old[:-4] + rng.choice([" copy.sql", "_é.sql"])
    tree.git("mv", old, new)
    return f"rename {old} -> {new} (quoted path in the diff)", [old, new]


def op_file_submodule(rng, tree, path=None):
    """Bump the sqlglot-integration-tests submodule (a gitlink, mode 160000)."""
    entry = tree.git("ls-files", "-s", "sqlglot-integration-tests").split()
    if not entry or entry[0] != "160000":
        return None
    new = hashlib.sha1(_word(rng, 12).encode()).hexdigest()
    tree.git("update-index", "--cacheinfo", f"160000,{new},sqlglot-integration-tests")
    return f"bump submodule sqlglot-integration-tests {entry[1][:7]} -> {new[:7]}", ["sqlglot-integration-tests"]


SETUP = "sqlglotc/setup.py"


def op_outside_executed(rng, tree, path=None):
    """Edit sqlglotc/setup.py, outside the package, which tests/test_sqlglotc.py runs with runpy."""
    if SETUP not in tree.files("sqlglotc"):
        return None
    op = rng.choice([op_compare, op_constant, op_stmt_delete, op_return_early, op_cond_negate, op_comment, op_syntax_error, op_noop_format])
    result = op(rng, tree, SETUP)
    if not result:
        return None
    return f"[{op.__name__}] {result[0]}", result[1]


NEW_DIALECT = '''from __future__ import annotations

from sqlglot.dialects.{base_mod} import {base}
{imports}

class {name}({base}):
    """{name}: a {base} variant."""
{body}'''


def op_dialect_new(rng, tree, path=None):
    """A new dialect: its module(s) plus the registries (DIALECTS list, Dialects enum)."""
    base, base_mod = rng.choice([("Postgres", "postgres"), ("MySQL", "mysql"), ("Trino", "trino"), ("DuckDB", "duckdb"), ("Hive", "hive")])
    word = _word(rng, 5)
    name, module = word.capitalize(), word
    variant = rng.choice(["minimal", "layered", "unregistered", "missing-module", "enum-only"])
    imports, body = "", "    pass\n"
    files = []
    if variant == "layered":
        # The repository's layout: dialects/<x>.py + parsers/<x>.py + generators/<x>.py.
        if not all((tree.root / f"sqlglot/{area}/{base_mod}.py").exists() for area in ("parsers", "generators")):
            return None
        base_parser = f"{base}Parser"
        tree.write(
            f"sqlglot/parsers/{module}.py",
            f"from __future__ import annotations\n\nfrom sqlglot.parsers.{base_mod} import {base_parser}\n\n\n"
            f"class {name}Parser({base_parser}):\n    FUNCTIONS = {{**{base_parser}.FUNCTIONS}}\n",
        )
        tree.write(
            f"sqlglot/generators/{module}.py",
            f"from __future__ import annotations\n\nfrom sqlglot import exp\nfrom sqlglot.generators.{base_mod} import {base}Generator\n\n\n"
            f"class {name}Generator({base}Generator):\n    TRANSFORMS = {{\n        **{base}Generator.TRANSFORMS,\n"
            f"        exp.CurrentDate: lambda self, e: \"CURRENT_DATE()\",\n    }}\n",
        )
        imports = f"from sqlglot.generators.{module} import {name}Generator\nfrom sqlglot.parsers.{module} import {name}Parser\n"
        body = f"    Parser = {name}Parser\n    Generator = {name}Generator\n"
        files += [f"sqlglot/parsers/{module}.py", f"sqlglot/generators/{module}.py"]
    if variant != "missing-module" and variant != "enum-only":
        tree.write(f"sqlglot/dialects/{module}.py", NEW_DIALECT.format(base_mod=base_mod, base=base, imports=imports, name=name, body=body))
        files.append(f"sqlglot/dialects/{module}.py")
    if variant != "unregistered":
        if variant != "enum-only":
            init = "sqlglot/dialects/__init__.py"
            text = tree.read(init)
            entries = re.findall(r'^    "(\w+)",$', text.split("DIALECTS = [", 1)[1].split("]", 1)[0], re.M)
            after = max([e for e in entries if e.lower() < name.lower()] or [entries[0]], key=str.lower)
            text = text.replace(f'    "{after}",\n', f'    "{after}",\n    "{name}",\n', 1)
            tree.write(init, text)
            files.append(init)
        dialect_py = "sqlglot/dialects/dialect.py"
        text = tree.read(dialect_py)
        marker = '    TSQL = "tsql"\n'
        if marker not in text:
            return None
        tree.write(dialect_py, text.replace(marker, marker + f'    {name.upper()} = "{module}"\n', 1))
        files.append(dialect_py)
    return f"new dialect {name}({base}) [{variant}]", files


def op_rename_method(rng, tree, path=None):
    """Rename a method and every reference to its name in the same file (an IDE-style rename)."""
    path = path or pick_src(rng, tree, rng.choice(["sqlglot/generators/", "sqlglot/parsers/", "sqlglot/optimizer/", "sqlglot/dialects/", "sqlglot/expressions/"]))
    source = Source(tree.read(path))
    methods = [
        f
        for c in source.nodes(ast.ClassDef)
        for f in c.body
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and not f.name.startswith("__")
    ]
    if not methods:
        return None
    function = rng.choice(methods)
    old = function.name
    new = rng.choice([f"{old}_impl", f"{old}2", f"_{old}" if not old.startswith("_") else old.lstrip("_")])
    if re.search(rf"\b{re.escape(new)}\b", source.text):
        return None
    text = re.sub(rf"\b{re.escape(old)}\b", new, source.text)
    count = len(re.findall(rf"\b{re.escape(old)}\b", source.text))
    return _edit_file(tree, path, text, f"{path}:{function.lineno} rename method {old} -> {new} ({count} references in the file)")


def op_expr_arg_types(rng, tree, path=None):
    """An Expression's arg_types: a flag flips, an optional arg is added, or an arg is removed."""
    pool = [p for p in tree.src if p.startswith("sqlglot/expressions/") and "arg_types = {" in tree.read(p)]
    path = path or rng.choice(pool)
    source = Source(tree.read(path))
    assigns = [
        n
        for c in source.nodes(ast.ClassDef)
        for n in c.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "arg_types"
        and isinstance(n.value, ast.Dict)
    ]
    if not assigns:
        return None
    assign = rng.choice(assigns)
    cls = assign._parent  # type: ignore[attr-defined]
    entries = [
        (k, v)
        for k, v in zip(assign.value.keys, assign.value.values)
        if isinstance(k, ast.Constant) and isinstance(k.value, str) and isinstance(v, ast.Constant) and isinstance(v.value, bool)
    ]
    mode = rng.choice(["flip", "add", "remove"])
    if mode == "add" or not entries:
        word = _word(rng, 5)
        last = assign.value.values[-1] if assign.value.values else None
        if last is None:
            return None
        start, end = source.span(last)
        text = source.replace([(end, end, f', "{word}": False')])
        desc = f'{path}:{assign.lineno} {cls.name}.arg_types += "{word}": False'
    elif mode == "flip":
        key, value = rng.choice(entries)
        start, end = source.span(value)
        text = source.replace([(start, end, str(not value.value))])
        desc = f'{path}:{value.lineno} {cls.name}.arg_types["{key.value}"] {value.value} -> {not value.value}'
    else:
        if len(entries) < 2:
            return None
        key, value = rng.choice([e for e in entries if e[0].value != "this"] or entries)
        start, _ = source.span(key)
        _, end = source.span(value)
        tail = source.data[end:]
        match = re.match(rb"\s*,[ \t]*", tail)
        end += match.end() if match else 0
        text = source.replace([(start, end, "")])
        desc = f'{path}:{key.lineno} {cls.name}.arg_types drops "{key.value}"'
    return _edit_file(tree, path, text, desc)


CONFTESTS = [
    ("benign autouse fixture", "import pytest\n\n\n@pytest.fixture(autouse=True)\ndef _noop():\n    yield\n"),
    (
        "autouse fixture that errors in {module}",
        "import pytest\n\n\n@pytest.fixture(autouse=True)\ndef _guard(request):\n"
        "    if request.module.__name__.endswith(\"{module}\"):\n        raise RuntimeError(\"not supported here\")\n    yield\n",
    ),
    ("collect_ignore {file}", 'collect_ignore = ["{file}"]\n'),
    ("marker registration only", 'def pytest_configure(config):\n    config.addinivalue_line("markers", "slow: slow tests")\n'),
]


def op_test_conftest(rng, tree, path=None):
    """A new tests/conftest.py: hooks and fixtures that reach tests no import connects."""
    if "tests/conftest.py" in tree.tests:
        return None
    desc, body = rng.choice(CONFTESTS)
    test = rng.choice(_test_sources(tree))
    module = Path(test).stem
    file = str(Path(test).relative_to("tests"))
    tree.write("tests/conftest.py", body.format(module=module, file=file))
    return f"add tests/conftest.py: {desc.format(module=module, file=file)}", ["tests/conftest.py"]


def _sweep(rng, tree, count):
    files = rng.sample([p for p in tree.src if not p.endswith("__init__.py")], count)
    for path in files:
        text = tree.read(path)
        tree.write(path, text.rstrip("\n") + "\n\n\n# " + rng.choice(["reviewed for the 2026 cleanup", "see CONTRIBUTING.md", "end of module"]) + "\n")
    return files


def op_sweep_comment(rng, tree, path=None):
    """A comment appended to 30-60 source files (a mechanical sweep, behavior-neutral)."""
    files = _sweep(rng, tree, rng.randrange(30, 61))
    return f"append a comment line to {len(files)} source files", files


def op_sweep_mixed(rng, tree, path=None):
    """The comment sweep plus one real code mutation hidden among the files."""
    files = _sweep(rng, tree, rng.randrange(30, 61))
    for _ in range(30):
        op = rng.choice([op_compare, op_constant, op_return_early, op_cond_negate, op_bool])
        target = pick_src(rng, tree)
        if target in files:
            continue
        result = op(rng, tree, target)
        if result:
            return f"comment sweep over {len(files)} files + [{op.__name__}] {result[0]}", files + result[1]
    return None


def op_noop_large(rng, tree, path=None):
    """A large documentation-only diff (~200 KB appended to CHANGELOG.md)."""
    lines = [f"- {_word(rng, 8)}: {' '.join(_word(rng, rng.randrange(3, 9)) for _ in range(6))}\n" for _ in range(3500)]
    tree.write("CHANGELOG.md", tree.read("CHANGELOG.md") + "\n## Unreleased\n\n" + "".join(lines))
    return "append ~3500 lines to CHANGELOG.md", ["CHANGELOG.md"]



# --------------------------------------------------------------------------- kinds

KINDS = {
    # function bodies (the sample's "code" lives here)
    "code:compare": op_compare,
    "code:boolop": op_bool,
    "code:arith": op_arith,
    "code:constant": op_constant,
    "code:stmt-delete": op_stmt_delete,
    "code:return-early": op_return_early,
    "code:raise": op_raise,
    "code:negate": op_cond_negate,
    "code:arg-swap": op_arg_swap,
    # signatures
    "sig:default": op_sig_default,
    "sig:add-param": op_sig_add_param,
    "sig:remove-param": op_sig_remove_param,
    # class bodies and module level
    "table:class-delete": op_class_table_delete,
    "table:class-modify": op_class_table_modify,
    "table:class-attr": op_class_attr,
    "table:module": op_module_table,
    "table:tokenizer": op_tokenizer_keywords,
    "table:reorder": op_reorder_entries,
    # imports, packages, syntax
    "import:remove": op_import_remove,
    "import:missing": op_import_missing,
    "import:unused": op_import_unused,
    "import:eager": op_import_cycle,
    "init:edit": op_init_edit,
    "syntax:error": op_syntax_error,
    "syntax:indent": op_indent_error,
    # files
    "file:rename-src": op_file_rename_src,
    "file:rename-fixture": op_file_rename_fixture,
    "file:rename-test": op_file_rename_test,
    "file:delete-src": op_file_delete_src,
    "file:delete-fixture": op_file_delete_fixture,
    "file:delete-test": op_file_delete_test,
    "file:add-src": op_file_add_src,
    "file:add-test": op_file_add_test,
    "file:add-fixture": op_file_add_fixture,
    "file:mode": op_file_mode,
    # fixtures
    "fixture:modify": op_fixture_modify,
    "fixture:add-valid": op_fixture_add_valid,
    "fixture:add-invalid": op_fixture_add_invalid,
    "fixture:delete": op_fixture_delete,
    "fixture:dash-comment": op_fixture_comment,
    "fixture:hash-line": op_fixture_hash_comment,
    # test files
    "test:assert-edit": op_test_assert_edit,
    "test:weaken": op_test_weaken,
    "test:add": op_test_add,
    "test:rename": op_test_rename,
    "test:delete": op_test_delete_method,
    "test:syntax": op_test_syntax,
    "test:helper": op_test_helper,
    "test:skip": op_test_skip,
    # comments, docstrings, no-ops
    "comment:code": op_comment,
    "comment:in-string": op_comment_in_string,
    "docstring": op_docstring,
    "noop:format": op_noop_format,
    "noop:rename-local": op_noop_rename_local,
    "noop:pass": op_noop_pass,
    "noop:reorder-functions": op_noop_reorder_functions,
    "noop:outside-package": op_nonsrc,
    # performance and non-termination
    "perf:busy-loop": op_perf,
    "perf:infinite-loop": op_loop,
    "perf:recursion": op_recursion,
    "perf:leak": op_leak,
    # classes, decorators, annotations
    "class:add-expression": op_expr_add_class,
    "class:add-method": op_class_add_method,
    "class:base": op_class_base,
    "code:decorator-remove": op_decorator_remove,
    "noop:annotation": op_noop_annotation,
    "fixture:inline-comment": op_fixture_inline_comment,
    # shared-state pollution and dispatch additions (br-research C1/C2 shapes)
    "pollute:research-c1": op_research_c1,
    "dispatch:research-c2": op_research_c2,
    "pollute:class-container": op_pollute_class_container,
    "pollute:module-container": op_pollute_module_container,
    "dispatch:add-sql-method": op_dispatch_add_sql_method,
    "dispatch:add-transform": op_dispatch_add_transform,
    "dispatch:add-parser-function": op_dispatch_add_parser_function,
    # round 3: patch-format edges, executed files outside the package, registries, refactors
    "fixture:blank-line": op_fixture_blank_line,
    "fixture:trailing-space": op_fixture_trailing_space,
    "format:crlf": op_format_crlf,
    "format:no-eol": op_format_no_eol,
    "format:bom": op_format_bom,
    "file:binary": op_file_binary,
    "file:quoted-path": op_file_quoted_path,
    "file:submodule": op_file_submodule,
    "outside:executed": op_outside_executed,
    "dialect:new": op_dialect_new,
    "refactor:rename-method": op_rename_method,
    "expr:arg-types": op_expr_arg_types,
    "test:conftest": op_test_conftest,
    "sweep:comment": op_sweep_comment,
    "sweep:mixed": op_sweep_mixed,
    "noop:large": op_noop_large,
    "noop:empty": None,  # an empty patch file: nothing changes at all
}

# Multi-file mixes, like the sample's "multi" (code + comment + fixture) and beyond.
MULTI_RECIPES = [
    ["comment:code", "code:compare", "fixture:modify"],
    ["comment:code", "fixture:modify"],
    ["code:constant", "code:boolop"],
    ["table:class-modify", "test:assert-edit"],
    ["file:add-src", "import:unused", "code:compare"],
    ["noop:format", "comment:code", "docstring"],
    ["fixture:modify", "fixture:modify"],
    ["code:stmt-delete", "test:add"],
    ["file:rename-src", "comment:code"],
    ["table:tokenizer", "fixture:add-valid"],
]


def generate(tree: Tree, rng: random.Random, kind: str) -> tuple[str, list[str], str] | None:
    if kind == "noop:empty":
        return "empty patch (no change at all)", [], ""
    if kind.startswith("multi:"):
        recipe = MULTI_RECIPES[int(kind.split(":")[1])]
        descs, files = [], []
        for part in recipe:
            for _ in range(30):
                result = KINDS[part](rng, tree)
                if result and not (set(result[1]) & set(files)):
                    descs.append(f"[{part}] {result[0]}")
                    files += result[1]
                    break
            else:
                return None
        patch = tree.diff()
        return (" + ".join(descs), files, patch) if patch.strip() else None
    for _ in range(40):
        result = KINDS[kind](rng, tree)
        if result:
            patch = tree.diff()
            if patch.strip():
                return result[0], result[1], patch
        tree.reset()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sqlglot", type=Path, required=True, help="clone at the pinned commit, clean")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=8)
    parser.add_argument("--multi-per-recipe", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kinds", nargs="*", help="only these kinds")
    parser.add_argument(
        "--introspect",
        type=Path,
        default=Path("/home/rayana/firstmate/data/br-eval/work/introspect.json"),
        help="output of eval/corpus/introspect.py run in the reference image on the pristine tree",
    )
    args = parser.parse_args()

    tree = Tree(args.sqlglot)
    tree.introspect_path = args.introspect
    rng = random.Random(args.seed)
    (args.out / "commits").mkdir(parents=True, exist_ok=True)
    kinds_path, meta_path = args.out / "kinds.json", args.out / "meta.json"
    kinds = json.loads(kinds_path.read_text()) if kinds_path.exists() else {}
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    seen_patches = {(args.out / "commits" / f"{c}.patch").read_text() for c in kinds if (args.out / "commits" / f"{c}.patch").exists()}

    plan = [k for k in KINDS for _ in range(args.per_kind)]
    plan += [f"multi:{i}" for i in range(len(MULTI_RECIPES)) for _ in range(args.multi_per_recipe)]
    if args.kinds:
        plan = [k for k in plan if k in args.kinds or k.split(":")[0] in args.kinds]
    made = 0
    alphabet = string.ascii_lowercase + string.digits
    for kind in plan:
        try:
            result = generate(tree, rng, kind)
        finally:
            tree.reset()
        if result is None:
            print(f"skip {kind}: nothing applicable", file=sys.stderr)
            continue
        desc, files, patch = result
        if patch in seen_patches:
            continue
        seen_patches.add(patch)
        while True:
            commit = "".join(rng.choice(alphabet) for _ in range(4))
            if commit not in kinds:
                break
        (args.out / "commits" / f"{commit}.patch").write_text(patch)
        label = kind if not kind.startswith("multi:") else "multi"
        kinds[commit] = label
        meta[commit] = {"kind": label, "desc": desc, "files": sorted(set(files))}
        made += 1
        print(f"{commit} {label}: {desc}", file=sys.stderr)
    kinds_path.write_text(json.dumps(kinds, indent=1, sort_keys=True) + "\n")
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"generated {made} patches into {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
