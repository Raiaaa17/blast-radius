"""State setters: pristine test files that leave sqlglot's global state changed.

The grader runs the whole suite in one process, in file order. A test file whose
fixtures change sqlglot's process-wide state and leave it so (for example
tests/test_dialect_entry_points.py empties the dialect registry in setUp and
tearDown) puts every later file in a state that a per-shard trace or a
selective run never sees: code reached only in that state (the registry's
fallback lookup) is attributed to no later test, and a verify child would pass
what the grader fails.

detect() finds them at build time: test files whose source mutates an object
imported from sqlglot are candidates; each is confirmed in a subprocess by
replaying its fixtures (setUpClass, setUp and tearDown of one test, tearDownClass)
and comparing structural digests of every sqlglot slot before and after
(an equivalent fresh value, like sqlglot.schema = MappingSchema(), is no
change). replay() applies a setter's fixtures in a process before the tests
that follow it in suite order (map build, profile build, verify children).

Usage (probe): python setters.py --probe <sqlglot-root> <test file>
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_MUTATORS = {"pop", "popitem", "clear", "update", "setdefault", "append", "extend", "insert", "remove",
             "discard", "add", "__setitem__", "__delitem__"}


def static_candidates(src: Path, files: list[str]) -> list[str]:
    """Test files whose source mutates something imported from sqlglot."""
    out = []
    for rel in files:
        try:
            tree = ast.parse((src / rel).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "sqlglot":
                        names.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "sqlglot":
                for a in node.names:
                    names.add(a.asname or a.name)

        def rooted(e) -> bool:
            while isinstance(e, (ast.Attribute, ast.Subscript)):
                e = e.value
            return isinstance(e, ast.Name) and e.id in names

        hit = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in _MUTATORS and isinstance(node.func.value, (ast.Attribute, ast.Subscript)) \
                        and rooted(node.func.value):
                    hit = True
            elif isinstance(node, (ast.Assign, ast.Delete, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                if any(isinstance(t, (ast.Attribute, ast.Subscript)) and rooted(t.value) for t in targets):
                    hit = True
            if hit:
                break
        if hit:
            out.append(rel)
    return out


def replay(module) -> None:
    """Run a test module's fixtures once (no test bodies), as its suite run leaves them."""
    import unittest

    loader = unittest.TestLoader()
    for value in list(vars(module).values()):
        if not (isinstance(value, type) and issubclass(value, unittest.TestCase)
                and value.__module__ == module.__name__):
            continue
        names = loader.getTestCaseNames(value)
        if not names:
            continue
        try:
            value.setUpClass()
        except BaseException:  # noqa: BLE001 - a failing fixture leaves what it left
            pass
        try:
            test = value(names[0])
            try:
                test.setUp()
            finally:
                try:
                    test.tearDown()
                finally:
                    test.doCleanups()
        except BaseException:  # noqa: BLE001
            pass
        try:
            value.tearDownClass()
        except BaseException:  # noqa: BLE001
            pass
        try:
            value.doClassCleanups()
        except BaseException:  # noqa: BLE001
            pass


def replay_file(rel: str) -> None:
    import importlib

    try:
        replay(importlib.import_module(rel[:-3].replace("/", ".")))
    except BaseException:  # noqa: BLE001 - it cannot even import: nothing to replay
        pass


def clears_before(setters, order: dict, rel: str) -> list:
    """The "clear" effects of the setters that precede `rel` in suite order.

    A registry the grader empties is refilled by the first lookup after it, so which later
    test runs the refilling code depends on the order of the tests around it. Emptying it
    again before every later test makes each one run that code itself, in a trace or in a
    verify child, whatever its neighbours."""
    out = []
    for setter in setters:
        if not isinstance(setter, (list, tuple)):
            continue
        f, effects = setter
        if order.get(f, 1 << 30) < order.get(rel, 1 << 30):
            out += [tuple(e[1]) for e in (effects or ()) if e[0] == "clear"]
    return out


def reclear(slots: list) -> None:
    for slot in slots:
        try:
            obj, name = _resolve(slot)
            getattr(obj, name).clear()
        except Exception:  # noqa: BLE001
            pass


def _resolve(slot):
    import sys as _sys

    kind, owner, name = slot
    if kind == "M":
        return _sys.modules[owner], name
    module, _, qual = owner.partition(":")
    obj = _sys.modules[module]
    for part in qual.split("."):
        obj = vars(obj)[part]
    return obj, name


def apply(setter) -> None:
    """Re-create a setter's effect: its recorded effects, or its fixtures when one is not simple."""
    rel, effects = setter if isinstance(setter, (list, tuple)) else (setter, None)
    if effects is None or any(e[0] == "fixtures" for e in effects):
        replay_file(rel)
        return
    import importlib

    for effect in effects:
        try:
            obj, name = _resolve(tuple(effect[1]))
            if effect[0] == "clear":
                getattr(obj, name).clear()
            elif effect[0] == "new":
                module, _, qual = effect[2].partition(":")
                cls = importlib.import_module(module)
                for part in qual.split("."):
                    cls = getattr(cls, part)
                setattr(obj, name, cls())
        except Exception:  # noqa: BLE001 - fall back to the fixtures themselves
            replay_file(rel)
            return


def _probe(src: Path, rel: str) -> list:
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(HERE))
    os.chdir(src)
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    from inline_pool import install_inline_pool

    install_inline_pool()
    import importlib

    import statediff

    for name in statediff.iter_modules(src):
        try:
            importlib.import_module(name)
        except BaseException:  # noqa: BLE001
            pass
    module = importlib.import_module(rel[:-3].replace("/", "."))
    before = statediff.snapshot(src)
    replay(module)
    after = statediff.snapshot(src)
    changed = sorted(slot for slot, digest in before.items()
                     if after.get(slot) != digest and slot[-1] != "__slotnames__")
    # Effects simple enough to re-create without the test module: an emptied dict, or a
    # rebinding to a fresh instance of a sqlglot class (no arguments, equal to a new one).
    effects = []
    for slot in changed:
        try:
            obj, name = _resolve(slot)
            value = vars(obj).get(name) if not isinstance(obj, type(sys)) else getattr(obj, name, None)
        except (KeyError, TypeError):
            effects.append(["fixtures"])
            continue
        if type(value) is dict and not value:
            effects.append(["clear", list(slot)])
            continue
        cls = type(value)
        if str(getattr(cls, "__module__", "")).startswith("sqlglot"):
            try:
                fresh = cls()
                d = statediff.Digester()
                if d.fp(fresh) == d.fp(value):
                    effects.append(["new", list(slot), f"{cls.__module__}:{cls.__qualname__}"])
                    continue
            except Exception:  # noqa: BLE001
                pass
        effects.append(["fixtures"])
    return {"changed": [repr(s) for s in changed], "effects": effects}


def detect(src: Path, files: list[str], log=None) -> list:
    """The setters among `files`, in the given (suite) order: [file, effects]."""
    found = []
    candidates = static_candidates(src, files)
    procs = [
        (rel, subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--probe", str(src), rel],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               env={**os.environ, "PYTHONHASHSEED": "0"}))
        for rel in candidates
    ]
    for rel, proc in procs:
        stdout, _ = proc.communicate()
        try:
            got = json.loads(stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            got = {"changed": ["<probe failed>"], "effects": [["fixtures"]]}
        if log:
            log(f"[setters] {rel}: {len(got['changed'])} slots left changed {got['changed'][:4]}: {got['effects'][:4]}")
        if got["changed"]:
            found.append([rel, got["effects"]])
    return found


def suite_order(tests_json: Path) -> dict[str, int]:
    """Test file -> position in the suite (tests.json order is the grader's collection order)."""
    order: dict[str, int] = {}
    for t in json.loads(tests_json.read_text())["tests"]:
        order.setdefault(t["id"].split("::", 1)[0], len(order))
    return order


if __name__ == "__main__":
    if sys.argv[1] == "--probe":
        print(json.dumps(_probe(Path(sys.argv[2]), sys.argv[3])))
