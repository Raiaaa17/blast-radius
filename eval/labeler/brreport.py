"""pytest plugin: record per-test outcome and duration as JSON.

Load with `-p brreport` (its directory on PYTHONPATH) and set BR_REPORT to the
output path. A test counts as failed when any of its reports failed: setup or
teardown error, call failure, timeout, strict xpass, or any failing subtest
(pytest 9 prints such a test PASSED beside SUBFAILED lines). Collection errors
are recorded per collector so a caller can fail every test of that file.
"""

import json
import os
import time

_tests: dict = {}
_collect_errors: list = []
_started = time.time()


def _entry(nodeid):
    return _tests.setdefault(nodeid, {"failed": False, "seconds": 0.0, "outcomes": []})


def pytest_runtest_logreport(report):
    entry = _entry(report.nodeid)
    kind = type(report).__name__
    tag = f"{report.when}:{report.outcome}" + (":sub" if kind != "TestReport" else "")
    if tag not in entry["outcomes"]:
        entry["outcomes"].append(tag)
    if report.failed:
        entry["failed"] = True
        text = getattr(report, "longreprtext", "") or ""
        if "Timeout" in text and "timeout" not in entry:
            entry["timeout"] = True
    if kind == "TestReport":
        entry["seconds"] += report.duration
        if report.when == "teardown":
            _stream(report.nodeid, entry)


def _stream(nodeid, entry):
    """Append each finished test to BR_REPORT.jsonl so a killed run keeps its results."""
    path = os.environ.get("BR_REPORT")
    if path:
        with open(path + "l", "a") as handle:
            handle.write(json.dumps({"nodeid": nodeid, **entry}) + "\n")


def pytest_collection_modifyitems(config, items):
    """BR_TIMEOUTS = path of {"floor": F, "scale": S, "seconds": {nodeid: s}}: per-test timeout
    max(F, S * s) instead of the uniform --timeout (used to bound runs of non-terminating patches)."""
    spec = os.environ.get("BR_TIMEOUTS")
    if not spec:
        return
    import pytest

    with open(spec) as handle:
        data = json.load(handle)
    for item in items:
        seconds = data["seconds"].get(item.nodeid.split("[")[0], 0.0)
        item.add_marker(pytest.mark.timeout(max(data["floor"], data["scale"] * seconds)))


def pytest_collectreport(report):
    if report.failed:
        error = {"nodeid": report.nodeid, "text": (getattr(report, "longreprtext", "") or "")[-2000:]}
        _collect_errors.append(error)
        path = os.environ.get("BR_REPORT")
        if path:
            with open(path + "l", "a") as handle:
                handle.write(json.dumps({"collect_error": error}) + "\n")


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("BR_REPORT")
    if not path:
        return
    # Under xdist only the controller writes (workers have workerinput).
    if hasattr(session.config, "workerinput"):
        return
    with open(path, "w") as handle:
        json.dump(
            {
                "exitstatus": int(exitstatus),
                "wall": time.time() - _started,
                "tests": _tests,
                "collect_errors": _collect_errors,
            },
            handle,
        )
