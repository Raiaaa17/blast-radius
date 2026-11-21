"""Blast Radius selector: for each patch, print the tests that fail, packed.

Per commit:
1. parse the diff and classify every changed unit (analyze.py): neutral,
   function body, class body, module level, test code, data file, config;
2. map the units to candidate tests through the build-time maps (coverage of
   function bodies, class-attribute reads through each class, fixture opens,
   the static name/import index), generously, falling back to every test
   whenever a change is not understood;
3. run the cheap candidates against the patched tree (verify.py) and keep the
   ones that fail; keep every expensive or unverifiable candidate as is;
4. pack the result onto workers (pack.py).
"""

from __future__ import annotations

import json
import os
import pickle
import resource
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import pack  # noqa: E402
import patch as patchlib  # noqa: E402
import verify  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", "/data"))
PRISTINE = Path(os.environ.get("BR_PRISTINE", "/opt/sqlglot"))
MAPS = Path(os.environ.get("BR_MAPS", "/app/maps"))
TMP = Path(os.environ.get("BR_TMP", "/tmp/br"))

VERIFY_MAX_S = 8.0  # never run a candidate that costs more than this to verify
VERIFY_BUDGET_S = 12.0  # most verification CPU spent on one commit
FORK_COST_S = 0.3  # fixed CPU of verifying anything for a commit (child, sqlglot import)
CONFIG_FILES = {"pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini", "conftest.py"}

CO_OPTIMIZED = 0x0001
START = time.monotonic()


def log(message: str) -> None:
    print(f"[selector {time.monotonic() - START:6.1f}s] {message}", file=sys.stderr, flush=True)


def module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def code_lines(code: types.CodeType) -> tuple[int, int]:
    lo, hi = code.co_firstlineno, code.co_firstlineno
    stack = [code]
    while stack:
        c = stack.pop()
        for _, _, line in c.co_lines():
            if line is not None:
                lo = min(lo, line)
                hi = max(hi, line)
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return lo, hi


class Index:
    def __init__(self) -> None:
        with open(MAPS / "maps.pkl", "rb") as handle:
            m = pickle.load(handle)
        with open(MAPS / "static.pkl", "rb") as handle:
            s = pickle.load(handle)
        self.ids: list[str] = m["test_ids"]
        self.bit = {tid: i for i, tid in enumerate(self.ids)}
        self.measured: list[float] = m["seconds"]
        self.sets: list[int] = m["sets"]
        self.lines: dict[str, dict[int, int]] = m["lines"]
        self.owners: dict[str, int] = m["owners"]
        self.sub_owners: dict[str, list[str]] = defaultdict(list)
        for owner, mro in m["owner_mro"].items():
            for key in mro:
                self.sub_owners[key].append(owner)
        self.opens: dict[str, int] = m["opens"]
        self.lists: dict[str, int] = m["lists"]
        self.import_codes = m["import_codes"]
        self.import_sensitive: int = m["import_sensitive"]
        self.unreliable: int = m["unreliable"]
        self.by_file: dict[str, int] = m["by_file"]
        self.timing: dict[str, tuple[str, float]] = {}
        if (MAPS / "timing.pkl").exists():
            with open(MAPS / "timing.pkl", "rb") as handle:
                self.timing = pickle.load(handle)
        self.all = (1 << len(self.ids)) - 1
        self.codes = s["codes"]
        self.codes_by_file: dict[str, list] = defaultdict(list)
        for entry in self.codes:
            self.codes_by_file[entry[0]].append(entry)
        self.importers: dict[str, set[str]] = s["importers"]
        self.modules: dict[str, str] = s["modules"]
        self.holders: dict[str, set[str]] = s["holders"]
        self._closure: dict[str, set[str]] = {}

    # ------------------------------------------------------------ masks

    def lines_mask(self, rel: str, first: int, last: int) -> int:
        table = self.lines.get(rel)
        if not table:
            return 0
        mask = 0
        sets = self.sets
        if last - first + 1 <= len(table):
            for line in range(first, last + 1):
                sid = table.get(line)
                if sid is not None:
                    mask |= sets[sid]
        else:
            for line, sid in table.items():
                if first <= line <= last:
                    mask |= sets[sid]
        return mask

    def class_use(self, key: str) -> int:
        mask = 0
        for owner in self.sub_owners.get(key, ()):
            mask |= self.sets[self.owners[owner]]
        return mask

    def importer_closure(self, mod: str) -> set[str]:
        if mod in self._closure:
            return self._closure[mod]
        seen: set[str] = set()
        stack = [mod]
        while stack:
            for nxt in self.importers.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        self._closure[mod] = seen
        return seen


class Impact:
    def __init__(self) -> None:
        self.mask = 0
        self.broad: list[str] = []
        self.notes: list[str] = []


def class_impact(ix: Index, rel: str, qualname: str, names: set[str] | None) -> int:
    mod = module_name(rel)
    key = f"{mod}:{qualname}"
    mask = ix.class_use(key)
    parts = qualname.split(".")
    for i in range(1, len(parts)):  # enclosing classes (a dialect around its Tokenizer)
        mask |= ix.class_use(f"{mod}:{'.'.join(parts[:i])}")
    for holder in ix.holders.get(key, ()):  # classes holding this one (tokenizer_class, ...)
        mask |= ix.class_use(holder)
    simple = parts[-1]
    files = {rel} | {ix.modules[m] for m in ix.importer_closure(mod) if m in ix.modules}
    for path in files:
        for _, kind, qn, first, last, cnames in ix.codes_by_file.get(path, ()):
            if kind != "class" or simple not in cnames:
                continue
            if names is not None and not (cnames & names):
                continue
            mask |= ix.class_use(f"{module_name(path)}:{qn}")
    return mask | ix.import_sensitive | ix.unreliable


def name_impact(ix: Index, rel: str, names: set[str]) -> int:
    mod = module_name(rel)
    mask = ix.import_sensitive | ix.unreliable
    files = {rel} | {ix.modules[m] for m in ix.importer_closure(mod) if m in ix.modules}
    for path in files:
        pmod = module_name(path)
        for _, kind, qn, first, last, cnames in ix.codes_by_file.get(path, ()):
            if not (cnames & names):
                continue
            if kind == "function":
                mask |= ix.lines_mask(path, first, last)
            elif kind == "class":
                mask |= ix.class_use(f"{pmod}:{qn}")
    # The rebound objects themselves: a function's callers, a class's users.
    for _, kind, qn, first, last, _ in ix.codes_by_file.get(rel, ()):
        if qn in names:
            if kind == "function":
                mask |= ix.lines_mask(rel, first, last)
            elif kind == "class":
                mask |= class_impact(ix, rel, qn, None)
    return mask


def assess(ix: Index, diffs: list[patchlib.FileDiff], files: dict[str, bytes | None]) -> Impact:
    impact = Impact()
    for diff in diffs:
        path = diff.path
        base = os.path.basename(path)
        in_tree = path.startswith(("sqlglot/", "tests/"))
        if diff.binary:
            if in_tree:
                impact.broad.append(f"binary {path}")
            continue
        if base in CONFIG_FILES and (in_tree or "/" not in path):
            impact.broad.append(f"config {path}")
            continue
        if diff.rename or (diff.old_path and diff.new_path and diff.old_path != diff.new_path):
            impact.broad.append(f"rename {path}")
            continue
        if not path.endswith(".py"):
            # Data and fixture files: the tests that read them (or list their directory).
            mask = ix.sets[ix.opens[path]] if path in ix.opens else 0
            for rel in (diff.old_path, diff.new_path):
                if rel and rel in ix.opens:
                    mask |= ix.sets[ix.opens[rel]]
            if diff.old_path is None or diff.new_path is None:
                parent = os.path.dirname(path)
                for listed, sid in ix.lists.items():
                    if parent == listed or parent.startswith(listed + "/") or listed == ".":
                        mask |= ix.sets[sid]
            impact.mask |= mask
            impact.notes.append(f"data {path}: {bin(mask).count('1')}")
            continue
        if not in_tree:
            continue  # benchmarks, docs, scripts: not imported by any test
        if diff.old_path is None:
            if base == "__init__.py":
                impact.broad.append(f"new package {path}")
                continue
            mask = ix.import_sensitive | ix.unreliable
            parent = os.path.dirname(path)
            for listed, sid in ix.lists.items():
                if parent == listed or parent.startswith(listed + "/"):
                    mask |= ix.sets[sid]
            impact.mask |= mask
            impact.notes.append(f"new module {path}")
            continue
        if diff.new_path is None:
            impact.broad.append(f"deleted {path}")
            continue
        old = (PRISTINE / path).read_bytes()
        new = files[path]
        try:
            old_text, new_text = old.decode("utf-8"), new.decode("utf-8")
        except UnicodeDecodeError:
            impact.broad.append(f"undecodable {path}")
            continue
        result = analyze.analyze_python(old_text, new_text, str(PRISTINE / path))
        if result.syntax_error:
            impact.broad.append(f"syntax error {path}")
            continue
        if result.neutral:
            impact.notes.append(f"neutral {path}")
            continue
        is_test = path.startswith("tests/")
        mod = module_name(path)
        for unit in result.units:
            if unit.kind == "function":
                code = unit.old_code
                if not is_test and (path, code.co_qualname, code.co_firstlineno) in ix.import_codes:
                    impact.broad.append(f"import-time function {path}:{unit.qualname}")
                    continue
                first, last = code_lines(code)
                mask = ix.lines_mask(path, first, last) | ix.unreliable
                impact.mask |= mask
                impact.notes.append(f"function {path}:{unit.qualname}: {bin(mask).count('1')}")
            elif is_test:
                mask = ix.by_file.get(path, 0) | ix.unreliable
                for m in ix.importer_closure(mod):
                    rel = ix.modules.get(m)
                    if rel:
                        mask |= ix.by_file.get(rel, 0)
                impact.mask |= mask
                impact.notes.append(f"test {unit.kind} {path}:{unit.qualname}: {bin(mask).count('1')}")
            elif unit.kind == "class":
                mask = class_impact(ix, path, unit.qualname, unit.names)
                impact.mask |= mask
                impact.notes.append(f"class {path}:{unit.qualname} {unit.names}: {bin(mask).count('1')}")
            else:
                if unit.names is None:
                    impact.broad.append(f"module-level {path}")
                    continue
                mask = name_impact(ix, path, unit.names)
                impact.mask |= mask
                impact.notes.append(f"module {path} {sorted(unit.names)}: {bin(mask).count('1')}")
    return impact


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    number_of = {t["id"]: t["number"] for t in tests}
    recorded_ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
    patches = sorted((DATA / "commits").glob("*.patch"))
    n = len(patches)
    limit_s = 180.0 if n <= 40 else 270.0
    deadline = START + limit_s - 30.0

    ix = Index()
    log(f"loaded maps for {len(ix.ids)} tests; {n} commits")
    # tests.json numbers <-> map bits; unknown tests are always candidates
    bit_number: dict[int, int] = {}
    unknown: set[int] = set()
    for t in tests:
        b = ix.bit.get(t["id"])
        if b is None:
            unknown.add(t["number"])
        else:
            bit_number[b] = t["number"]

    def numbers(mask: int) -> set[int]:
        out = set()
        while mask:
            low = mask & -mask
            b = low.bit_length() - 1
            if b in bit_number:
                out.add(bit_number[b])
            mask ^= low
        return out

    plans = []
    for patch_path in patches:
        commit = patch_path.stem
        try:
            diffs = patchlib.parse(patch_path.read_bytes())
            files: dict[str, bytes | None] = {}
            for diff in diffs:
                old = None
                if diff.old_path:
                    src = PRISTINE / diff.old_path
                    old = src.read_bytes() if src.is_file() else None
                if diff.binary:
                    continue
                files[diff.new_path or diff.old_path] = patchlib.apply(old, diff)
                if diff.old_path and diff.new_path and diff.old_path != diff.new_path:
                    files[diff.old_path] = None
            impact = assess(ix, diffs, files)
        except Exception as error:  # anything unexpected: run everything we can
            log(f"{commit}: analysis failed ({error!r}); broad")
            diffs, files, impact = [], {}, Impact()
            impact.broad.append("analysis failed")
        if impact.broad:
            candidates = set(recorded_ms)
        else:
            candidates = numbers(impact.mask)
            if impact.mask or any(not n.startswith("neutral") for n in impact.notes):
                candidates |= unknown
        log(f"{commit}: {len(candidates)} candidates; broad={impact.broad} {impact.notes}")
        plans.append((commit, files, candidates, impact))

    id_of = {t["number"]: t["id"] for t in tests}
    cost_of: dict[int, float] = {}  # CPU seconds to verify, from the build's timing pass
    trusted: set[int] = set()  # the runner passes it on the pristine tree
    for t in tests:
        entry = ix.timing.get(t["id"])
        if entry is not None:
            cost_of[t["number"]] = entry[1]
            if entry[0] == "passed":
                trusted.add(t["number"])
        else:
            b = ix.bit.get(t["id"])
            cost_of[t["number"]] = ix.measured[b] / 2 if b is not None else 10.0
            if b is not None and not ix.timing:
                trusted.add(t["number"])

    def split(plan):
        """Decide per candidate: verify it (pay its CPU) or keep it (pay its packed seconds).

        Tests that cannot be verified are kept. The others are taken longest
        first; each is kept when adding it to the current packing costs less
        than running it (it fits under the busiest worker, or a worker is
        already paid for), else it is verified.
        """
        commit, files, candidates, impact = plan
        keep: set[int] = set()
        maybe: list[int] = []
        for c in candidates:
            if files and c in trusted and cost_of[c] <= VERIFY_MAX_S:
                maybe.append(c)
            else:
                keep.add(c)
        if not maybe:
            return [], keep
        bins = pack.pack([(c, recorded_ms[c]) for c in keep])
        loads = [sum(recorded_ms[c] for c in b) for b in bins]
        verify_list: list[int] = []
        for c in sorted(maybe, key=lambda c: -recorded_ms[c]):
            r = recorded_ms[c]
            top = max(loads) if loads else 0
            options = []
            if loads:
                i = loads.index(min(loads))
                options.append((max(0, loads[i] + r - top), i))
            if len(loads) < pack.MAX_WORKERS:
                options.append((pack.WORKER_MS + max(0, r - top), -1))
            extra_ms, where = min(options)
            if extra_ms <= cost_of[c] * 1000:
                keep.add(c)
                if where == -1:
                    loads.append(r)
                else:
                    loads[where] += r
            else:
                verify_list.append(c)
        verify_list.sort(key=lambda c: cost_of[c])
        spent = 0.0
        for i, c in enumerate(verify_list):
            spent += cost_of[c]
            if spent > VERIFY_BUDGET_S:
                keep.update(verify_list[i:])
                verify_list = verify_list[:i]
                break
        # Verifying has a fixed cost: not worth it to save less than that.
        if verify_list:
            saved = pack.cost_ms([(c, recorded_ms[c]) for c in keep | set(verify_list)]) - pack.cost_ms(
                [(c, recorded_ms[c]) for c in keep]
            )
            if saved <= (FORK_COST_S + sum(cost_of[c] for c in verify_list)) * 1000:
                keep.update(verify_list)
                verify_list = []
        return verify_list, keep

    jobs = []
    results: dict[str, set[int]] = {}
    pending: dict[str, list[int]] = {}
    for plan in plans:
        commit, files = plan[0], plan[1]
        cheap, keep = split(plan)
        results[commit] = keep
        if not cheap:
            continue
        root = TMP / commit
        try:
            verify.build_tree(PRISTINE, root, files)
        except Exception as error:
            log(f"{commit}: cannot build tree ({error!r}); keeping candidates")
            keep.update(cheap)
            continue
        ids = [id_of[c] for c in cheap]
        allowance = [max(0.5, 3.0 * cost_of[c] + 0.5, 2.0 * recorded_ms[c] / 1000 + 1.0) for c in cheap]
        jobs.append(verify.Job(commit, root, ids, allowance))
        pending[commit] = cheap

    if jobs:
        verify.warm_up()
        log(f"verifying {sum(len(j.ids) for j in jobs)} tests over {len(jobs)} commits")
        verify.run_jobs(jobs, parallel=2, deadline=deadline, log=log)
    import shutil

    for job in jobs:
        shutil.rmtree(job.root, ignore_errors=True)
        keep = results[job.key]
        failed = 0
        for c in pending[job.key]:
            status = job.verdicts.get(id_of[c])
            if status not in ("passed", "skipped"):
                keep.add(c)
                failed += status is not None
        log(f"{job.key}: ran {len(job.verdicts)}/{len(job.ids)}, {failed} failing; selected {len(keep)}")

    for commit, _, _, _ in plans:
        keep = results[commit]
        workers = pack.pack([(c, recorded_ms[c]) for c in sorted(keep)])
        print(json.dumps({"commit": commit, "workers": workers}, separators=(",", ":")))
    sys.stdout.flush()
    usage_self = resource.getrusage(resource.RUSAGE_SELF)
    usage_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = usage_self.ru_utime + usage_self.ru_stime + usage_kids.ru_utime + usage_kids.ru_stime
    cgroup = ""
    try:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            if line.startswith("usage_usec"):
                cgroup = f", container cgroup {int(line.split()[1]) / 1e6:.2f}s"
    except OSError:
        pass
    log(f"done: cpu {cpu:.2f}s (self {usage_self.ru_utime + usage_self.ru_stime:.2f}s){cgroup}")


if __name__ == "__main__":
    main()
