"""Build-time check behind treating line-only/docstring-only edits as neutral
for tests/test_typing.py, the one test that compares code objects.

TestTyping.test_no_redundant_expression_metadata fails when a dialect's
EXPRESSION_METADATA entry equals its parent's: same keys, equal non-annotator
values, and annotators with equal co_code/co_names/co_varnames/co_consts.
A line-only or docstring-only edit changes an annotator only in co_consts[0]
(the docstring slot) or in nested code objects' line tables, so it can flip
the test only if some child/parent pair is *already* equal modulo those.
This script lists the files holding such pairs (expected: none).

    python -m staticeq.typing_guard   # run with cwd = sqlglot root on sys.path
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import sys
from pathlib import Path


def _relaxed_same(f, g) -> bool:
    from staticeq.pycompare import code_equal

    if not (hasattr(f, "__code__") and hasattr(g, "__code__")):
        return False
    a, b = f.__code__, g.__code__
    if a.co_code != b.co_code or a.co_names != b.co_names or a.co_varnames != b.co_varnames:
        return False
    ca, cb = list(a.co_consts), list(b.co_consts)
    if ca and (isinstance(ca[0], str) or ca[0] is None):
        ca = ca[1:]
    if cb and (isinstance(cb[0], str) or cb[0] is None):
        cb = cb[1:]
    if len(ca) != len(cb):
        return False
    for x, y in zip(ca, cb):
        if hasattr(x, "co_code") and hasattr(y, "co_code"):
            if not code_equal(x, y):
                return False
        elif type(x) is not type(y) or x != y:
            return False
    return True


def check(root: str) -> dict:
    sys.path.insert(0, root)
    import sqlglot.typing
    from tests.test_typing import _parent_module

    pkg = Path(sqlglot.typing.__file__).parent
    pairs = 0
    sensitive: set[str] = set()
    for info in pkgutil.iter_modules([str(pkg)]):
        parent_name = _parent_module((pkg / f"{info.name}.py").read_text("utf-8"))
        if not parent_name:
            continue
        child = importlib.import_module(f"sqlglot.typing.{info.name}").EXPRESSION_METADATA
        parent = importlib.import_module(parent_name).EXPRESSION_METADATA
        for et, cv in child.items():
            pv = parent.get(et)
            if pv is None or cv is pv or set(cv) != set(pv):
                continue
            pairs += 1
            if not all(cv[k] == pv[k] for k in cv if k != "annotator"):
                continue
            f, g = cv.get("annotator"), pv.get("annotator")
            if f is not None and g is not None and _relaxed_same(f, g):
                for fn in (f.__code__.co_filename, g.__code__.co_filename):
                    sensitive.add(os.path.relpath(fn, root).replace(os.sep, "/"))
    return {"ok": True, "pairs": pairs, "sensitive_files": sorted(sensitive)}


if __name__ == "__main__":
    try:
        result = check(os.getcwd())
    except Exception as e:  # noqa: BLE001 - an unknown guard makes analysis conservative
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(result))
