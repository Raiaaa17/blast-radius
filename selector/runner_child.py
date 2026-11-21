"""Runs a list of sqlglot tests in one interpreter and reports each verdict.

Used two ways: forked from the selector (``run``), or as a script for the
build's timing pass:

    python runner_child.py <tree-root> <report-fd>   (test ids as JSON on stdin)

Report lines: "S <i>" when test i starts, "E <i> <status> <cpu>" when it ends
(status: passed, failed, error, skipped; cpu seconds of the test), "DONE" at
the end. Tests of a module that cannot be imported end with status "error"
without starting.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(root: str, ids: list[str], emit, fresh: bool = True, preimport=(), before_tests=None,
        after_tests=None, setters=(), order_of=None) -> None:
    """Run the tests; with fresh=False the modules are already imported (warm process).

    preimport: test modules to import first, as pytest's collection would (a new
    test module's import runs before every test); their failures are ignored.
    Every test module is imported before the first test runs, as pytest's
    collection does. before_tests() runs then; after_tests() runs after the last
    test and returns a line to report (or None). setters: test files that leave
    sqlglot state changed (setters.py); each is replayed before the first module
    that follows it in suite order (`order`: file -> position), as in the grader's
    single process.
    """
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from inline_pool import install_inline_pool

    install_inline_pool()
    if fresh:
        os.chdir(root)
        sys.path.insert(0, root)
    import importlib
    import unittest

    for name in preimport:
        try:
            importlib.import_module(name)
        except BaseException:  # noqa: BLE001 - a collection error of that file only
            pass

    # Group the tests by module, keeping the requested order.
    order: dict[str, list[int]] = {}
    for i, tid in enumerate(ids):
        order.setdefault(tid.split("::", 1)[0], []).append(i)

    statuses: dict[int, str] = {}
    positions = dict(order_of or {})

    last_stop = [time.process_time()]
    last_class = [None]

    class Result(unittest.TestResult):
        index: dict[int, int] = {}
        current: int | None = None
        t0 = 0.0

        def startTest(self, test):
            super().startTest(test)
            if setters:
                rel_of_test = type(test).__module__.replace(".", "/") + ".py"
                clears = clears_of.get(rel_of_test)
                if clears is None:
                    from setters import clears_before

                    clears = clears_of[rel_of_test] = clears_before(setters, positions, rel_of_test)
                if clears:
                    from setters import reclear

                    reclear(clears)
            now = time.process_time()
            cls = f"{type(test).__module__}.{type(test).__qualname__}"
            if cls != last_class[0]:
                emit(f"C {cls} {now - last_stop[0]:.4f}")
                last_class[0] = cls
            self.current = self.index.get(id(test))
            self.t0 = now
            if self.current is not None:
                statuses[self.current] = "passed"
                emit(f"S {self.current}")

        def stopTest(self, test):
            super().stopTest(test)
            now = time.process_time()
            if self.current is not None:
                emit(f"E {self.current} {statuses[self.current]} {now - self.t0:.4f}")
            self.current = None
            last_stop[0] = now

        def _mark(self, status: str) -> None:
            if self.current is not None and statuses.get(self.current) in ("passed", "skipped"):
                statuses[self.current] = status

        def addError(self, test, err):
            super().addError(test, err)
            if isinstance(test, unittest.TestCase):
                self._mark("error")

        def addFailure(self, test, err):
            super().addFailure(test, err)
            self._mark("failed")

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err is not None:
                self._mark("failed")

        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self._mark("skipped")

        def addUnexpectedSuccess(self, test):
            super().addUnexpectedSuccess(test)
            self._mark("failed")

    modules = {}
    for rel, indexes in order.items():
        modname = rel[:-3].replace("/", ".")
        try:
            modules[rel] = importlib.import_module(modname)
        except BaseException:  # noqa: BLE001 - the file cannot import: its tests fail
            for i in indexes:
                emit(f"E {i} error 0")
    if before_tests is not None:
        before_tests()
    applied: set[str] = set()
    clears_of: dict[str, list] = {}
    setter_of = {(s[0] if isinstance(s, (list, tuple)) else s): s for s in setters}
    for rel, indexes in order.items():
        due = [f for f in setter_of if f not in applied and positions.get(f, 1 << 30) < positions.get(rel, 1 << 30)]
        if due:
            from setters import apply as apply_setter

            if not applied:
                # The grader has every dialect imported before a later file empties the
                # registry; a lazy first import afterwards would register it again.
                from zygote import _import_all_sqlglot

                _import_all_sqlglot(root)
            for f in due:
                apply_setter(setter_of[f])
                applied.add(f)
        if rel in setter_of:
            applied.add(rel)
        module = modules.get(rel)
        if module is None:
            continue
        suite = unittest.TestSuite()
        result = Result()
        result.index = {}
        missing = []
        for i in indexes:
            _, cls_name, method = ids[i].split("::")
            cls = getattr(module, cls_name, None)
            if cls is None or not hasattr(cls, method):
                missing.append(i)
                continue
            test = cls(method)
            result.index[id(test)] = i
            suite.addTest(test)
        for i in missing:
            emit(f"E {i} error 0")
        suite.run(result)
        # A class or module fixture error leaves tests without a verdict.
        for i in indexes:
            if i not in statuses and i not in missing:
                emit(f"E {i} error 0")
    if after_tests is not None:
        line = after_tests()
        if line:
            emit(line)
    emit("DONE")


def main() -> None:
    root = sys.argv[1]
    report = os.fdopen(int(sys.argv[2]), "w", buffering=1)
    ids: list[str] = json.loads(sys.stdin.read())

    def emit(line: str) -> None:
        report.write(line + "\n")
        report.flush()

    run(root, ids, emit)


if __name__ == "__main__":
    main()
