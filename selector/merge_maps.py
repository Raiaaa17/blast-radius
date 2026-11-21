"""Merge coverage shards into the compact inverted maps the selector reads.

Every test set is a Python int used as a bitmask over the collector's test
order; identical sets are interned, so each map stores small set ids.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path


class Interner:
    def __init__(self) -> None:
        self.sets: list[int] = []
        self.index: dict[int, int] = {}

    def add(self, mask: int) -> int:
        sid = self.index.get(mask)
        if sid is None:
            sid = len(self.sets)
            self.sets.append(mask)
            self.index[mask] = sid
        return sid


def _class_of(test_id: str) -> str:
    return test_id.rsplit("::", 1)[0]


def _file_of(test_id: str) -> str:
    return test_id.split("::", 1)[0]


def merge(src: Path, shard_paths: list[Path], out: Path) -> None:
    started = time.perf_counter()
    shards = []
    for path in shard_paths:
        with open(path, "rb") as handle:
            shards.append(pickle.load(handle))

    test_ids: list[str] = []
    status: list[str] = []
    seconds: list[float] = []
    for shard in shards:
        for tid, entry in shard["results"].items():
            test_ids.append(tid)
            status.append(entry["status"])
            seconds.append(round(entry["seconds"], 4))
    bit = {tid: 1 << i for i, tid in enumerate(test_ids)}

    by_class: dict[str, int] = {}
    by_file: dict[str, int] = {}
    for tid, b in bit.items():
        by_class[_class_of(tid)] = by_class.get(_class_of(tid), 0) | b
        by_file[_file_of(tid)] = by_file.get(_file_of(tid), 0) | b

    line_masks: dict[tuple[str, int], int] = {}
    attr_masks: dict[tuple[tuple[str, str], str], int] = {}
    owner_masks: dict[str, int] = {}
    open_masks: dict[str, int] = {}
    list_masks: dict[str, int] = {}
    import_sensitive = 0
    owner_mro: dict[str, list[str]] = {}
    import_lines: set = set()

    def absorb(mask: int, files: list[str], lines, attrs, events, imports_count: bool = True) -> None:
        nonlocal import_sensitive
        for encoded in lines:
            key = (files[encoded >> 20], encoded & 0xFFFFF)
            line_masks[key] = line_masks.get(key, 0) | mask
        for attr_key, owner in attrs:
            k = (tuple(attr_key), owner)
            attr_masks[k] = attr_masks.get(k, 0) | mask
            owner_masks[owner] = owner_masks.get(owner, 0) | mask
        for kind, value in events:
            if kind == "open":
                open_masks[value] = open_masks.get(value, 0) | mask
            elif kind == "list":
                list_masks[value] = list_masks.get(value, 0) | mask
            elif imports_count:  # a fresh import or a subprocess during the test
                import_sensitive |= mask

    for shard in shards:
        files = shard["files"]
        owner_mro.update(shard["owner_mro"])
        import_lines |= shard["import_lines"]
        for tid, entry in shard["results"].items():
            absorb(bit[tid], files, entry["lines"], entry["attrs"], entry["events"])
        # Fixture code between tests belongs to the classes on either side.
        for last, nxt, lines, attrs, events in shard["gaps"]:
            mask = 0
            for tid in (last, nxt):
                if tid is not None:
                    mask |= by_class.get(_class_of(tid), 0)
            if mask:
                absorb(mask, files, lines, attrs, events)
        # What a test module's import ran is shared by every test in it.
        for rel, (lines, attrs, events) in shard["module_buckets"].items():
            mask = by_file.get(rel, 0)
            if mask:
                # Importing the test module imports its helpers: not a test's own import.
                absorb(mask, files, lines, attrs, events, imports_count=False)

    unreliable = 0
    for tid, st in zip(test_ids, status):
        if st != "passed":
            unreliable |= bit[tid]

    interner = Interner()
    lines_out: dict[str, dict[int, int]] = {}
    for (path, line), mask in line_masks.items():
        lines_out.setdefault(path, {})[line] = interner.add(mask)
    attrs_out: dict[tuple[str, str], dict[str, int]] = {}
    for (attr_key, owner), mask in attr_masks.items():
        attrs_out.setdefault(attr_key, {})[owner] = interner.add(mask)
    owners_out = {owner: interner.add(mask) for owner, mask in owner_masks.items()}
    opens_out = {path: interner.add(mask) for path, mask in open_masks.items()}
    lists_out = {path: interner.add(mask) for path, mask in list_masks.items()}
    import_lines_out = set(import_lines)  # (file, qualname, firstlineno) of code run at import

    maps = {
        "version": 1,
        "test_ids": test_ids,
        "status": status,
        "seconds": seconds,
        "sets": interner.sets,
        "lines": lines_out,
        "attrs": attrs_out,
        "owners": owners_out,
        "owner_mro": owner_mro,
        "opens": opens_out,
        "lists": lists_out,
        "import_codes": import_lines_out,
        "import_sensitive": import_sensitive,
        "unreliable": unreliable,
        "by_file": by_file,
        "by_class": by_class,
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "maps.pkl", "wb") as handle:
        pickle.dump(maps, handle, protocol=pickle.HIGHEST_PROTOCOL)
    size = (out / "maps.pkl").stat().st_size
    bad = [tid for tid, st in zip(test_ids, status) if st != "passed"]
    print(
        f"[maps] merged {len(test_ids)} tests, {len(interner.sets)} distinct sets, "
        f"{len(line_masks)} lines, {len(attr_masks)} attr reads, {size / 1e6:.1f} MB "
        f"in {time.perf_counter() - started:.1f}s; not passing in collector: {bad}; "
        f"import-sensitive: {[t for t in test_ids if import_sensitive & bit[t]]}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    merge(Path(sys.argv[1]), [Path(p) for p in sys.argv[3:]], Path(sys.argv[2]))
