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


def run(root: str, ids: list[str], emit) -> None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from inline_pool import install_inline_pool

    install_inline_pool()
    os.chdir(root)
    sys.path.insert(0, root)
    import importlib
    import unittest

    # Group the tests by module, keeping the requested order.
    order: dict[str, list[int]] = {}
    for i, tid in enumerate(ids):
        order.setdefault(tid.split("::", 1)[0], []).append(i)

    statuses: dict[int, str] = {}

    class Result(unittest.TestResult):
        index: dict[int, int] = {}
        current: int | None = None
        t0 = 0.0

        def startTest(self, test):
            super().startTest(test)
            self.current = self.index.get(id(test))
            self.t0 = time.process_time()
            if self.current is not None:
                statuses[self.current] = "passed"
                emit(f"S {self.current}")

        def stopTest(self, test):
            super().stopTest(test)
            if self.current is not None:
                emit(f"E {self.current} {statuses[self.current]} {time.process_time() - self.t0:.4f}")
            self.current = None

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

    for rel, indexes in order.items():
        modname = rel[:-3].replace("/", ".")
        try:
            module = importlib.import_module(modname)
        except BaseException:  # noqa: BLE001 - the file cannot import: its tests fail
            for i in indexes:
                emit(f"E {i} error 0")
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
