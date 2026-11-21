"""Static index of the pristine tree, built once at image build time.

* every code object (function, class body, module body) of every .py file
  under sqlglot/ and tests/: its kind, qualname, line span and the global and
  attribute names it references;
* the module import graph (who imports whom, statically);
* which classes hold another class as a class attribute value (a dialect's
  tokenizer_class, parser_class, ...), computed from the live pristine classes.

Usage: python static_index.py <sqlglot-root> <out-dir>
"""

from __future__ import annotations

import ast
import os
import pickle
import sys
import types
from pathlib import Path

CO_OPTIMIZED = 0x0001


def module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def code_span(code: types.CodeType) -> tuple[int, int]:
    lines = [line for _, _, line in code.co_lines() if line is not None]
    first = min(lines) if lines else code.co_firstlineno
    last = max(lines) if lines else code.co_firstlineno
    return min(first, code.co_firstlineno), last


def walk_codes(code: types.CodeType, kind: str):
    yield code, kind
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            sub_kind = "function" if const.co_flags & CO_OPTIMIZED else "class"
            yield from walk_codes(const, sub_kind)


def imports_of(tree: ast.Module, this_module: str) -> set[str]:
    found: set[str] = set()
    package = this_module.rsplit(".", 1)[0] if "." in this_module else ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = this_module.split(".")
                anchor = anchor[: len(anchor) - node.level] if not this_module.endswith("__init__") else anchor
                base = ".".join([*anchor, base] if base else anchor)
            if base:
                found.add(base)
                for alias in node.names:
                    found.add(f"{base}.{alias.name}")
    # Every parent package is imported too.
    out = set()
    for name in found:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            out.add(".".join(parts[:i]))
    out.discard(this_module)
    return out


def build(src: Path) -> dict:
    codes: list[tuple[str, str, str, int, int, frozenset]] = []  # file, kind, qualname, first, last, names
    importers: dict[str, set[str]] = {}
    modules: dict[str, str] = {}
    for base in ("sqlglot", "tests"):
        for path in sorted((src / base).rglob("*.py")):
            rel = str(path.relative_to(src))
            text = path.read_text(encoding="utf-8")
            try:
                module_code = compile(text, str(path), "exec", dont_inherit=True)
                tree = ast.parse(text)
            except SyntaxError:
                continue
            name = module_name(rel)
            modules[name] = rel
            for imported in imports_of(tree, name):
                importers.setdefault(imported, set()).add(name)
            for code, kind in walk_codes(module_code, "module"):
                first, last = code_span(code)
                codes.append((rel, kind, code.co_qualname, first, last, frozenset(code.co_names)))

    holders: dict[str, set[str]] = {}
    sys.path.insert(0, str(src))
    cwd = os.getcwd()
    os.chdir(src)
    try:
        import importlib

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_maps import _collect_classes, _iter_sqlglot_modules, class_key

        live = []
        for name in _iter_sqlglot_modules(src):
            try:
                live.append(importlib.import_module(name))
            except Exception:
                pass
        for cls in _collect_classes(live):
            key = class_key(cls)
            for value in list(vars(cls).values()):
                if isinstance(value, type) and getattr(value, "__module__", "").startswith("sqlglot"):
                    for base in value.__mro__:
                        if getattr(base, "__module__", "").startswith("sqlglot"):
                            holders.setdefault(class_key(base), set()).add(key)
    finally:
        os.chdir(cwd)

    return {"codes": codes, "importers": importers, "modules": modules, "holders": holders}


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    index = build(src)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "static.pkl", "wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[static] {len(index['codes'])} code objects, {len(index['modules'])} modules",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
