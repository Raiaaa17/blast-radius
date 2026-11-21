"""What a patch changes in sqlglot's import-time state.

A class-body, module-level or import-time edit is observable at run time only
through the state the import leaves behind: module globals, class attributes
(including everything metaclasses and class-body helpers derive from them)
and class hierarchies. ``snapshot()`` imports every sqlglot module and digests
each slot:

    ("M", module, name)    a module global
    ("C", class_key, attr) an attribute in a class's own __dict__
    ("MRO", class_key)     the class's method resolution order

The digest of a value is structural and independent of the process (no ids,
no hash-seed order): containers recursively, sets order-insensitively, classes
and modules by name, functions by everything except their bytecode (name,
defaults, closure contents, attributes) because bytecode changes are mapped
through line coverage instead, other objects through ``__reduce_ex__`` or
their ``__dict__``.

The build stores the pristine snapshot, taken the way the run time takes one
(a child forked from a selector-like process), with every dict slot's
per-entry encodings in state_keys.pkl. At run time a child imports the
patched tree, snapshots it and reports the slots that differ (added, removed
or changed) and, for a changed dict, the entries that differ; the selector
maps them to the tests that read those slots or entries.

Usage (build): python statediff.py <sqlglot-root> <out.pkl>   (also writes state_keys.pkl next to it)
"""

from __future__ import annotations

import enum
import functools
import hashlib
import importlib
import os
import re
import sys
import types
from pathlib import Path

_SKIP_MODULE_NAMES = {
    "__builtins__", "__loader__", "__spec__", "__file__", "__cached__", "__path__", "__doc__",
    "__name__", "__package__",
}
_SKIP_CLASS_NAMES = {
    "__dict__", "__weakref__", "__module__", "__doc__", "__qualname__", "__firstlineno__",
    "__static_attributes__",
}


def iter_modules(root: Path) -> list[str]:
    """Every module under root/sqlglot, following symlinked directories (patched trees are symlink farms)."""
    names = []
    base = root / "sqlglot"
    for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        rel_dir = Path(dirpath).relative_to(root)
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            parts = list((rel_dir / name[:-3]).parts)
            if parts[-1] == "__main__":
                continue
            if parts[-1] == "__init__":
                parts = parts[:-1]
            names.append(".".join(parts))
    return sorted(names)


def class_key(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


_LEN = [bytes((i,)) for i in range(255)]
_LEAF_MAX = 64  # longer scalar encodings are hashed


def _frame(p: bytes) -> bytes:
    n = len(p)
    if n < 255:
        return _LEN[n] + p
    return b"\xff" + hashlib.blake2b(p, digest_size=16).digest()


class Digester:
    """Structural digests of values, stable across processes.

    Scalars encode as short tagged bytes (no hashing); every container hashes
    the framed encodings of its parts once (blake2b), so the cost is one hash
    per container rather than one per value.
    """

    def __init__(self, ordered: bool = True) -> None:
        self.ordered = ordered  # False: dicts digest order-insensitively (hash-seed dependent slots)
        self.memo: dict[int, bytes] = {}
        self.keep: list = []  # keep memoized objects alive (ids stay unique)
        self.depth = 0
        self.strs: dict[str, bytes] = {}

    @staticmethod
    def h(*parts: bytes, _len=_LEN, _frame=_frame, _b2=hashlib.blake2b) -> bytes:
        return _b2(b"".join([_len[len(p)] + p if len(p) < 255 else _frame(p) for p in parts]),
                   digest_size=16).digest()

    def fp(self, v) -> bytes:
        t = type(v)
        if t is str:
            got = self.strs.get(v)
            if got is None:
                got = b"s" + v.encode("utf-8", "surrogatepass")
                if len(got) > _LEAF_MAX:
                    got = b"S" + hashlib.blake2b(got, digest_size=16).digest()
                self.strs[v] = got
            return got
        if v is None:
            return b"N"
        if t is bool:
            return b"T" if v else b"F"
        if t is int or t is float or t is complex:
            got = (b"i" if t is int else b"f" if t is float else b"c") + repr(v).encode()
            return got if len(got) <= _LEAF_MAX else b"I" + hashlib.blake2b(got, digest_size=16).digest()
        if t is bytes:
            got = b"b" + v
            return got if len(got) <= _LEAF_MAX else b"B" + hashlib.blake2b(got, digest_size=16).digest()
        key = id(v)
        got = self.memo.get(key)
        if got is not None:
            return got
        # Placeholder for cycles: a back reference digests as the object's type.
        self.memo[key] = b"\x00cycle " + t.__qualname__.encode()
        self.keep.append(v)
        self.depth += 1
        try:
            if self.depth > 200:
                out = self.h(b"deep", t.__qualname__.encode())
            else:
                out = self._fp(v, t)
        except Exception as error:  # noqa: BLE001 - an opaque value: its type only
            out = self.h(b"opaque", t.__module__.encode(), t.__qualname__.encode(), type(error).__name__.encode())
        finally:
            self.depth -= 1
        self.memo[key] = out
        return out

    def items(self, v: dict) -> list[tuple[bytes, bytes]]:
        """(key, value) encodings of a dict, in its order."""
        fp, strs, memo = self.fp, self.strs.get, self.memo.get
        out = []
        for k, x in list(v.items()):
            kc = strs(k) if type(k) is str else None
            xc = memo(id(x)) if type(x) is not str else strs(x)
            out.append((kc or fp(k), xc or fp(x)))
        return out

    def _fp(self, v, t) -> bytes:
        fp, h = self.fp, self.h
        if t is tuple or t is list:
            return h(t.__name__.encode(), *[fp(x) for x in v])
        if t is dict or isinstance(v, dict):
            pairs = self.items(v)
            if self.ordered:
                flat = [b"dict", t.__qualname__.encode()]
                for k, x in pairs:
                    flat.append(k)
                    flat.append(x)
                return h(*flat)
            return h(b"udict", t.__qualname__.encode(), *sorted(_frame(k) + _frame(x) for k, x in pairs))
        if t is set or t is frozenset:
            return h(t.__name__.encode(), *sorted(fp(x) for x in v))
        if isinstance(v, type):
            return h(b"class", class_key(v).encode())
        if t is types.ModuleType:
            return h(b"module", v.__name__.encode())
        if isinstance(v, enum.Enum):
            # Values too: code reads them (AutoName enums generate SQL from them). Integer
            # values (IntEnum auto()) shift when a member is inserted; nothing reads them but
            # hashing, so they stay out of the digest.
            try:
                value = v._value_
            except AttributeError:
                value = None
            keep = not isinstance(value, (enum.Enum, int)) or isinstance(value, bool)
            return h(b"enum", class_key(t).encode(), v.name.encode(), fp(value) if keep else b"")
        if t is types.FunctionType:
            if v.__defaults__ is None and v.__kwdefaults__ is None and v.__closure__ is None and not v.__dict__:
                return h(b"function", v.__module__.encode() if v.__module__ else b"", v.__qualname__.encode(),
                         v.__code__.co_qualname.encode(), b"N", b"N", b"plain")
            closure = [c.cell_contents if _cell_full(c) else _EMPTY for c in (v.__closure__ or ())]
            return h(
                b"function",
                v.__module__.encode() if v.__module__ else b"",
                v.__qualname__.encode(),
                v.__code__.co_qualname.encode(),
                fp(v.__defaults__),
                fp(v.__kwdefaults__),
                fp(closure),
                fp(dict(v.__dict__)),
            )
        if t is types.MethodType:
            return h(b"method", fp(v.__func__), fp(v.__self__))
        if t in (types.BuiltinFunctionType, types.MethodWrapperType, types.WrapperDescriptorType,
                 types.MethodDescriptorType, types.ClassMethodDescriptorType, types.GetSetDescriptorType,
                 types.MemberDescriptorType):
            owner = getattr(v, "__self__", None)
            return h(b"builtin", str(getattr(v, "__module__", "")).encode(),
                     str(getattr(v, "__qualname__", getattr(v, "__name__", ""))).encode(),
                     fp(owner) if owner is not None and not isinstance(owner, types.ModuleType) else b"")
        if t is property:
            return h(b"property", fp(v.fget), fp(v.fset), fp(v.fdel))
        if t is classmethod or t is staticmethod:
            return h(t.__name__.encode(), fp(v.__func__))
        if t is functools.partial:
            return h(b"partial", fp(v.func), fp(v.args), fp(v.keywords))
        if isinstance(v, functools._lru_cache_wrapper):  # type: ignore[attr-defined]
            return h(b"lru", fp(v.__wrapped__))
        if t is re.Pattern:
            return h(b"re", fp(v.pattern), fp(v.flags))
        if t is types.CodeType:
            return h(b"code", v.co_qualname.encode())
        if t is types.SimpleNamespace:
            return h(b"ns", fp(dict(vars(v))))
        # Any other object: its reduction, else its attributes.
        if t.__reduce_ex__ is not object.__reduce_ex__ or t.__reduce__ is not object.__reduce__:
            try:
                r = v.__reduce_ex__(4)
            except Exception:  # noqa: BLE001
                r = None
            if isinstance(r, tuple):
                return h(b"reduce", class_key(t).encode(), *[fp(x) for x in r[1:3]])
        parts = [b"object", class_key(t).encode()]
        if hasattr(v, "__dict__"):
            parts.append(fp(dict(vars(v))))
        for cls in t.__mro__:
            for slot in getattr(cls, "__slots__", ()) if isinstance(getattr(cls, "__slots__", ()), (tuple, list)) else ():
                if slot in ("__dict__", "__weakref__"):
                    continue
                parts.append(slot.encode())
                parts.append(fp(getattr(v, slot, _EMPTY)))
        return h(*parts)


class _Empty:
    def __reduce_ex__(self, protocol):  # digest stably
        return (_Empty, ())


_EMPTY = _Empty()


def _cell_full(cell) -> bool:
    try:
        cell.cell_contents
        return True
    except ValueError:
        return False


def collect_classes(modules) -> list[type]:
    seen: dict[int, type] = {}
    stack = []
    for module in modules:
        for value in list(vars(module).values()):
            if isinstance(value, type):
                stack.append(value)
    while stack:
        cls = stack.pop()
        if id(cls) in seen or not str(getattr(cls, "__module__", "")).startswith("sqlglot"):
            continue
        seen[id(cls)] = cls
        for value in list(vars(cls).values()):
            if isinstance(value, type):
                stack.append(value)
    return list(seen.values())


def snapshot(root: Path, unordered: set | frozenset = frozenset(), values: dict | None = None,
             digesters: list | None = None) -> dict[tuple, bytes]:
    """Import every sqlglot module from root (already first on sys.path) and digest all slots.

    Slots in `unordered` (their dict order depends on the hash seed) digest
    dicts order-insensitively. `values`, when given, receives each slot's
    object (to digest a changed dict's entries afterwards); `digesters`
    receives the (ordered, unordered) digesters, whose memos make that cheap.
    """
    modules = []
    try:
        importlib.import_module("sqlglot")
    except BaseException:  # noqa: BLE001 - nothing else can import either
        return {("M", "sqlglot", "<import failed>"): b"x"}
    EAGER.clear()
    EAGER.update(n for n in sys.modules if n == "sqlglot" or n.startswith("sqlglot."))
    names = iter_modules(root)
    for name in names:
        try:
            modules.append(importlib.import_module(name))
        except BaseException:  # noqa: BLE001 - recorded as a slot below
            pass
    d, du = Digester(), Digester(ordered=False)
    if digesters is not None:
        digesters[:] = [d, du]
    out: dict[tuple, bytes] = {}
    dfp, dufp = d.fp, du.fp

    def put(slot, value):
        out[slot] = (dufp if slot in unordered else dfp)(value)
        if values is not None:
            values[slot] = value

    for name in names:
        module = sys.modules.get(name)
        if module is None:
            out[("M", name, "<import failed>")] = b"x"
            continue
        for key, value in list(vars(module).items()):
            if key in _SKIP_MODULE_NAMES:
                continue
            put(("M", name, key), value)
    for cls in collect_classes([sys.modules[n] for n in names if n in sys.modules]):
        ck = class_key(cls)
        out[("MRO", ck)] = d.h(*[class_key(c).encode() for c in cls.__mro__])
        for key, value in list(vars(cls).items()):
            if key in _SKIP_CLASS_NAMES:
                continue
            put(("C", ck, key), value)
    return out


def _code_span(code: types.CodeType) -> tuple[int, int]:
    lo = hi = code.co_firstlineno
    stack = [code]
    while stack:
        c = stack.pop()
        for _, _, line in c.co_lines():
            if line is not None:
                lo, hi = min(lo, line), max(hi, line)
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return lo, hi


def callable_ranges(values: dict, root: Path) -> dict[tuple, list[tuple[str, int, int]]]:
    """Class-attribute slots holding functions (plain, classmethod, staticmethod, property parts):
    the files and line spans of the code a call of the value runs first."""
    prefix = str(root) + os.sep
    out = {}
    for slot, value in values.items():
        if slot[0] != "C":
            continue
        funcs = []
        if isinstance(value, types.FunctionType):
            funcs = [value]
        elif isinstance(value, (classmethod, staticmethod)) and isinstance(value.__func__, types.FunctionType):
            funcs = [value.__func__]
        elif isinstance(value, property):
            funcs = [f for f in (value.fget, value.fset, value.fdel) if f is not None]
            if not all(isinstance(f, types.FunctionType) for f in funcs):
                continue
        if not funcs:
            continue
        spans = []
        for f in funcs:
            code = f.__code__
            if not code.co_filename.startswith(prefix):
                spans = None
                break
            lo, hi = _code_span(code)
            spans.append((code.co_filename[len(prefix):], lo, hi))
        if spans:
            out[slot] = spans
    return out


EAGER: set[str] = set()  # the modules `import sqlglot` itself imports (set by snapshot)


def dict_keys_of(values: dict, digesters: list, unordered, slots=None) -> dict[tuple, dict[bytes, bytes]]:
    """Per-entry encodings {key: value} of every dict-valued slot (or of `slots`)."""
    d, du = digesters
    out = {}
    for slot in (values if slots is None else slots):
        value = values.get(slot)
        if isinstance(value, dict) and not isinstance(value, type):
            out[slot] = dict((du if slot in unordered else d).items(value))
    return out


def diff(old: dict, new: dict) -> list[tuple]:
    changed = [k for k, v in new.items() if old.get(k) != v]
    changed += [k for k in old if k not in new]
    return changed


def _worker(root: Path, out: Path, unordered_file: str) -> None:
    import pickle
    import time

    sys.path.insert(0, str(root))
    os.chdir(root)
    unordered = set()
    if unordered_file:
        with open(unordered_file, "rb") as handle:
            unordered = pickle.load(handle)
    t0 = time.process_time()
    snap = snapshot(root, unordered)
    mro = {class_key(c): [class_key(b) for b in c.__mro__]
           for c in collect_classes([sys.modules[n] for n in iter_modules(root) if n in sys.modules])}
    with open(out, "wb") as handle:
        pickle.dump({"slots": snap, "mro": mro}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[state] {len(snap)} slots in {time.process_time() - t0:.2f} cpu-s", file=sys.stderr)


def main() -> None:
    import pickle
    import subprocess

    if sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "")
        return
    if sys.argv[1] == "--forks":
        _forks(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
        return
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    tmp = out.parent

    def run(seed: int, name: str, unordered_file: str = "") -> dict:
        path = tmp / name
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(root), str(path)]
        if unordered_file:
            cmd.append(unordered_file)
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONHASHSEED": str(seed)})
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        path.unlink()
        return data

    # Slots whose digest varies between processes: first with dict order
    # (hash seed), then with dicts digested order-insensitively.
    runs = [run(seed, f"state_{seed}.pkl") for seed in (1, 2, 3)]
    unordered = set()
    for other in runs[1:]:
        unordered |= set(diff(runs[0]["slots"], other["slots"]))
    ufile = tmp / "state_unordered.pkl"
    with open(ufile, "wb") as handle:
        pickle.dump(unordered, handle)
    again = [run(seed, f"state_u{seed}.pkl", str(ufile)) for seed in (1, 2, 3)]
    unstable = set()
    for other in again[1:]:
        unstable |= set(diff(again[0]["slots"], other["slots"]))
    # The run-time setting: children forked from a selector-like process, the
    # tree a symlink farm in a temporary directory. They must agree with each
    # other and with the reference stored here; the reference is one of them.
    keys_file = tmp / "state_keys.pkl"
    forks = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--forks", str(root), str(ufile), str(keys_file)],
        check=True, env={**os.environ, "PYTHONHASHSEED": "4"}, capture_output=True,
    )
    got = pickle.loads(forks.stdout)
    unstable |= got["unstable"]
    unstable |= set(diff(again[0]["slots"], got["slots"]))
    ufile.unlink()
    with open(out, "wb") as handle:
        pickle.dump({"slots": got["slots"], "unordered": unordered, "unstable": unstable,
                     "mro": again[0]["mro"], "dict_slots": got["dict_slots"], "callables": got["callables"],
                     "untraced": got["untraced"], "enums": got["enums"], "eager": got["eager"]},
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[state] {len(got['slots'])} slots ({len(got['dict_slots'])} dicts); {len(unordered)} hash-seed ordered; "
          f"unstable {len(unstable)}: {sorted(unstable)[:8]}", file=sys.stderr)


def _forks(root: Path, unordered_file: str, keys_file: Path) -> None:
    """Snapshots as the selector takes them at run time; prints the reference and the unstable slots."""
    import pickle
    import tempfile

    here = str(Path(__file__).resolve().parent)
    sys.path.insert(0, here)
    # Modules the run-time selector has imported before it forks these children.
    import ast  # noqa: F401
    import difflib  # noqa: F401
    import importlib.util  # noqa: F401
    import json  # noqa: F401
    import resource  # noqa: F401
    import select  # noqa: F401
    import signal  # noqa: F401

    import verify

    with open(unordered_file, "rb") as handle:
        unordered = pickle.load(handle)
    with tempfile.TemporaryDirectory() as tmp:
        farm = Path(tmp) / "farm"
        verify.build_tree(root, farm, {})
        jobs = [("0", root), ("1", root), ("farm", farm)]
        snaps = run_snapshots(jobs, unordered, parallel=1, timeout_s=120, raw=True)
    unstable = set()
    for key in ("1", "farm"):
        if snaps["0"] is None or snaps[key] is None:
            raise SystemExit("snapshot child failed")
        unstable |= set(diff(snaps["0"]["slots"], snaps[key]["slots"]))
    with open(keys_file, "wb") as handle:
        pickle.dump(snaps["0"]["keys"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(pickle.dumps({"slots": snaps["0"]["slots"], "unstable": unstable,
                                          "dict_slots": set(snaps["0"]["keys"]),
                                          "callables": snaps["0"]["callables"],
                                          "untraced": snaps["0"]["untraced"],
                                          "enums": snaps["0"]["enums"],
                                          "eager": snaps["0"]["eager"]}))


# ---------------------------------------------------------------- run time

def _child_result(root: Path, unordered, base: dict | None, keys_file: Path | None, raw: bool) -> dict:
    """What a snapshot child reports: the slots that differ from `base` (and, for
    changed dicts, which entries), or (raw) the whole snapshot with entry encodings."""
    values: dict = {}
    digesters: list = []
    snap = snapshot(root, unordered, values=values, digesters=digesters)
    if raw:
        keys = dict_keys_of(values, digesters, unordered) if digesters else {}
        # Classes the map builder's recorders skip (their attribute reads are not recorded).
        classes = collect_classes([sys.modules[n] for n in iter_modules(root) if n in sys.modules])
        untraced = {class_key(c) for c in classes if issubclass(c, (enum.Enum, tuple, BaseException))}
        enums = {class_key(c) for c in classes if issubclass(c, enum.Enum)}
        return {"slots": snap, "keys": keys, "callables": callable_ranges(values, root), "untraced": untraced,
                "enums": enums, "eager": set(EAGER)}
    if base is None:
        return {"slots": snap}
    changed = diff(base["slots"], snap)
    entries: dict[tuple, list] = {}
    dict_changed = [s for s in changed if s in base.get("dict_slots", ()) and isinstance(values.get(s), dict)]
    if dict_changed and keys_file is not None and digesters:
        import pickle

        try:
            with open(keys_file, "rb") as handle:
                old_keys = pickle.load(handle)
        except OSError:
            old_keys = {}
        new_keys = dict_keys_of(values, digesters, unordered, dict_changed)
        for slot in dict_changed:
            old, new = old_keys.get(slot), new_keys.get(slot)
            if old is None or new is None:
                continue
            entries[slot] = [k for k in old.keys() | new.keys() if old.get(k) != new.get(k)]
    return {"changed": changed, "entries": entries, "removed": [k for k in base["slots"] if k not in snap]}


def run_snapshots(jobs: list[tuple[str, Path]], unordered: set, parallel: int = 2,
                  timeout_s: float = 20.0, log=None, base: dict | None = None,
                  keys_file: Path | None = None, raw: bool = False) -> dict[str, dict | None]:
    """jobs: (key, tree root). Returns key -> result (None: failed).

    Each child is forked from the calling process (which must not have sqlglot
    imported) and imports the tree. With `base` (the build's pristine reference,
    taken the same way) a child reports {"changed": slots, "entries": {dict slot:
    changed entry keys}}; without it, {"slots": full snapshot}.
    """
    import pickle
    import select
    import signal
    import time

    results: dict[str, dict | None] = {}
    queue = list(jobs)
    active: dict[int, tuple] = {}
    while queue or active:
        while queue and len(active) < parallel:
            key, root = queue.pop(0)
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                try:
                    os.close(r)
                    devnull = os.open(os.devnull, os.O_RDWR)
                    for fd in (0, 1, 2):
                        os.dup2(devnull, fd)
                    sys.path.insert(0, str(root))
                    os.chdir(root)
                    result = _child_result(Path(root), unordered, base, keys_file, raw)
                    data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                    view = memoryview(data)
                    while view:
                        n = os.write(w, view)
                        view = view[n:]
                finally:
                    os._exit(0)
            os.close(w)
            active[r] = (key, pid, time.monotonic() + timeout_s, [])
        wait = max(0.0, min(v[2] for v in active.values()) - time.monotonic())
        ready, _, _ = select.select(list(active), [], [], wait)
        for fd in ready:
            key, pid, limit, chunks = active[fd]
            chunk = os.read(fd, 1 << 20)
            if chunk:
                chunks.append(chunk)
                continue
            os.close(fd)
            os.waitpid(pid, 0)
            del active[fd]
            try:
                results[key] = pickle.loads(b"".join(chunks))
            except Exception:  # noqa: BLE001
                results[key] = None
        now = time.monotonic()
        for fd in [fd for fd, v in active.items() if v[2] <= now]:
            key, pid, _, _ = active.pop(fd)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
            os.close(fd)
            results[key] = None
            if log:
                log(f"{key}: state snapshot timed out")
    return results


if __name__ == "__main__":
    main()
