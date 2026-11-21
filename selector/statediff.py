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

The build stores the pristine snapshot; at run time a child imports the
patched tree, snapshots it and reports the slots that differ (added, removed
or changed), which the selector maps to the tests that read them.

Usage (build): python statediff.py <sqlglot-root> <out.pkl>
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


class Digester:
    def __init__(self, ordered: bool = True) -> None:
        self.ordered = ordered  # False: dicts digest order-insensitively (hash-seed dependent slots)
        self.memo: dict[int, bytes] = {}
        self.keep: list = []  # keep memoized objects alive (ids stay unique)
        self.depth = 0

    def h(self, *parts: bytes) -> bytes:
        m = hashlib.blake2b(digest_size=16)
        for p in parts:
            m.update(len(p).to_bytes(4, "little"))
            m.update(p)
        return m.digest()

    def fp(self, v) -> bytes:
        t = type(v)
        if v is None or t in (bool, int, float, complex):
            return self.h(t.__name__.encode(), repr(v).encode())
        if t is str:
            return self.h(b"str", v.encode("utf-8", "surrogatepass"))
        if t is bytes:
            return self.h(b"bytes", v)
        key = id(v)
        got = self.memo.get(key)
        if got is not None:
            return got
        # Placeholder for cycles: a back reference digests as the object's type.
        self.memo[key] = self.h(b"cycle", t.__qualname__.encode())
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

    def _fp(self, v, t) -> bytes:
        fp, h = self.fp, self.h
        if t is tuple or t is list:
            return h(t.__name__.encode(), *[fp(x) for x in v])
        if t is dict or isinstance(v, dict):
            items = [h(fp(k), fp(x)) for k, x in list(v.items())]
            return h(b"dict", t.__qualname__.encode(), *(items if self.ordered else sorted(items)))
        if t is set or t is frozenset:
            return h(t.__name__.encode(), *sorted(fp(x) for x in v))
        if isinstance(v, type):
            return h(b"class", class_key(v).encode())
        if t is types.ModuleType:
            return h(b"module", v.__name__.encode())
        if isinstance(v, enum.Enum):
            return h(b"enum", class_key(t).encode(), v.name.encode())
        if t is types.FunctionType:
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


def snapshot(root: Path, unordered: set | frozenset = frozenset()) -> dict[tuple, bytes]:
    """Import every sqlglot module from root (already first on sys.path) and digest all slots.

    Slots in `unordered` (their dict order depends on the hash seed) digest
    dicts order-insensitively.
    """
    modules = []
    try:
        importlib.import_module("sqlglot")
    except BaseException:  # noqa: BLE001 - nothing else can import either
        return {("M", "sqlglot", "<import failed>"): b"x"}
    for name in iter_modules(root):
        try:
            modules.append(importlib.import_module(name))
        except BaseException:  # noqa: BLE001 - recorded as a slot below
            pass
    d, du = Digester(), Digester(ordered=False)
    out: dict[tuple, bytes] = {}

    def put(slot, value):
        out[slot] = (du if slot in unordered else d).fp(value)

    for name in iter_modules(root):
        module = sys.modules.get(name)
        if module is None:
            out[("M", name, "<import failed>")] = b"x"
            continue
        for key, value in list(vars(module).items()):
            if key in _SKIP_MODULE_NAMES:
                continue
            put(("M", name, key), value)
    for cls in collect_classes([sys.modules[n] for n in iter_modules(root) if n in sys.modules]):
        ck = class_key(cls)
        out[("MRO", ck)] = d.h(*[class_key(c).encode() for c in cls.__mro__])
        for key, value in list(vars(cls).items()):
            if key in _SKIP_CLASS_NAMES:
                continue
            put(("C", ck, key), value)
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
        _forks(Path(sys.argv[2]), sys.argv[3])
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
    # Forked children of one process (the run-time setting) must agree too.
    forks = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--forks", str(root), str(ufile)],
        check=True, env={**os.environ, "PYTHONHASHSEED": "4"}, capture_output=True,
    )
    unstable |= pickle.loads(forks.stdout)
    ufile.unlink()
    with open(out, "wb") as handle:
        pickle.dump({"slots": again[0]["slots"], "unordered": unordered, "unstable": unstable,
                     "mro": again[0]["mro"]}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[state] {len(again[0]['slots'])} slots; {len(unordered)} hash-seed ordered; unstable {len(unstable)}: "
          f"{sorted(unstable)[:8]}", file=sys.stderr)


def _forks(root: Path, unordered_file: str) -> None:
    import pickle

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import zygote

    zygote.warm_third_party()
    with open(unordered_file, "rb") as handle:
        unordered = pickle.load(handle)
    snaps = run_snapshots([(str(i), root) for i in range(3)], unordered, parallel=1, timeout_s=120)
    unstable = set()
    for i in (1, 2):
        if snaps["0"] is None or snaps[str(i)] is None:
            raise SystemExit("snapshot child failed")
        unstable |= set(diff(snaps["0"], snaps[str(i)]))
    sys.stdout.buffer.write(pickle.dumps(unstable))




# ---------------------------------------------------------------- run time

def run_snapshots(jobs: list[tuple[str, Path]], unordered: set, parallel: int = 2,
                  timeout_s: float = 20.0, log=None) -> dict[str, dict | None]:
    """jobs: (key, tree root). Returns key -> snapshot of that tree (None: failed).

    Each child is forked from the calling process (which must not have sqlglot
    imported) and imports the tree, so a pristine job and patched jobs see the
    same environment and differ only by the patch.
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
                    data = pickle.dumps(snapshot(Path(root), unordered), protocol=pickle.HIGHEST_PROTOCOL)
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
