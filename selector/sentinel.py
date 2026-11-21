"""Shared-state pollution sentinel for verification children.

A changed function can mutate sqlglot's process-wide state (pop an entry of a
class's table, rebind a module global). The grader runs every test in one
process, so each later test then sees the mutation, including tests that never
run the changed code and are no candidates (PolDet, Gyori et al. 2015).

fingerprint() takes a shallow fingerprint of every sqlglot module global and
class attribute: the value's identity and, for containers, their length and the
identities of their entries (and of an instance's attribute values). A child
takes one before and one after its candidates run; slots that differ, and are
not among those the pristine suite itself changes (measured at build time,
`volatile`), were polluted by the patch. For a polluted dict the entries that
differ from the build's pristine state are named too, so they map to the tests
that read those entries.
"""

from __future__ import annotations

import sys
import types

_CONTAINERS = (dict, list, set)


def _fp(v) -> tuple | int:
    t = type(v)
    try:
        if t is dict:
            return (id(v), len(v), hash(tuple(map(id, v))), hash(tuple(map(id, v.values()))))
        if t is list:
            return (id(v), len(v), hash(tuple(map(id, v))))
        if t is set:
            return (id(v), len(v), hash(frozenset(map(id, v))))
        if isinstance(v, (type, types.ModuleType, types.FunctionType, str, int, float, bool, bytes, tuple,
                          frozenset)) or v is None:
            return id(v)
        d = getattr(v, "__dict__", None)
        if type(d) is dict:
            return (id(v), len(d), hash(tuple(map(id, d))), hash(tuple(map(id, d.values()))))
    except Exception:  # noqa: BLE001 - an object that refuses: its identity only
        pass
    return id(v)


def _classes(modules) -> list[type]:
    seen: dict[int, type] = {}
    stack = [v for m in modules for v in list(vars(m).values()) if isinstance(v, type)]
    while stack:
        cls = stack.pop()
        if id(cls) in seen or not str(getattr(cls, "__module__", "")).startswith("sqlglot"):
            continue
        seen[id(cls)] = cls
        stack.extend(v for v in list(vars(cls).values()) if isinstance(v, type))
    return list(seen.values())


def fingerprint() -> dict[tuple, object]:
    modules = [m for name, m in list(sys.modules.items())
               if m is not None and (name == "sqlglot" or name.startswith("sqlglot."))]
    out: dict[tuple, object] = {}
    for m in modules:
        name = m.__name__
        for key, value in list(vars(m).items()):
            if key.startswith("__") and key.endswith("__"):
                continue
            out[("M", name, key)] = _fp(value)
    for cls in _classes(modules):
        ck = f"{cls.__module__}:{cls.__qualname__}"
        for key, value in list(vars(cls).items()):
            if key in ("__dict__", "__weakref__", "__doc__", "__module__", "__qualname__"):
                continue
            out[("C", ck, key)] = _fp(value)
    return out


def changed(before: dict, after: dict) -> list[tuple]:
    """Slots whose fingerprint moved; new slots only on modules and classes seen before."""
    owners = {slot[:2] for slot in before}
    out = [slot for slot, fp in before.items() if after.get(slot, None) != fp]
    out += [slot for slot in after if slot not in before and slot[:2] in owners]
    return out


def resolve(slot: tuple):
    """The current value of a slot, or a sentinel when it is gone."""
    kind, owner, name = slot
    try:
        if kind == "M":
            return vars(sys.modules[owner]).get(name, _GONE)
        module, _, qual = owner.partition(":")
        obj = sys.modules[module]
        for part in qual.split("."):
            obj = vars(obj)[part]
        return vars(obj).get(name, _GONE)
    except (KeyError, TypeError):
        return _GONE


_GONE = object()


def polluted_entries(slots: list[tuple], keys_file, unordered=frozenset()) -> dict[tuple, list | None]:
    """{slot: [(entry key encoding, "value" | "added" | "removed")] for a dict slot, else None},
    entries compared with the build's pristine state."""
    out: dict[tuple, list | None] = {slot: None for slot in slots}
    dict_slots = [s for s in slots if type(resolve(s)) is dict]
    if not dict_slots or keys_file is None:
        return out
    import pickle

    from statediff import Digester

    try:
        with open(keys_file, "rb") as handle:
            pristine = pickle.load(handle)
    except OSError:
        return out
    d, du = Digester(), Digester(ordered=False)
    for slot in dict_slots:
        old = pristine.get(slot)
        if old is None:
            continue
        value = resolve(slot)
        new = dict((du if slot in unordered else d).items(value))
        out[slot] = [(k, "value" if k in old and k in new else "added" if k in new else "removed")
                     for k in old.keys() | new.keys() if old.get(k) != new.get(k)]
    return out


def classify(polluted: dict) -> dict[tuple, str]:
    """Build time: how the pristine suite changes each slot: "entries" when it only removes
    and re-adds dict entries (a registry emptied and refilled), else "full"."""
    out = {}
    for slot, entries in polluted.items():
        if entries is not None and all(how != "value" for _, how in entries):
            out[slot] = "entries"
        else:
            out[slot] = "full"
    return out


def unexpected(polluted: dict, volatile: dict) -> dict:
    """Run time: drop what the pristine suite changes by itself. A slot it only empties and
    refills still counts when an entry now holds a different value."""
    out = {}
    for slot, entries in polluted.items():
        kind = volatile.get(slot)
        if kind == "full":
            continue
        if kind == "entries":
            entries = [(k, how) for k, how in (entries or []) if how == "value"]
            if not entries:
                continue
        out[slot] = entries
    return out
