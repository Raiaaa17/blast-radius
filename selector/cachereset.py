"""Process-wide caches in sqlglot, emptied before every test of a build-time trace.

sqlglot memoizes some per-class work for the life of the process (e.g. the
generator dispatch table, built from dir(cls) and the TRANSFORMS table on the
first use of each generator class). Without a reset only the first test of a
trace that uses a class would show the code and the class attributes the memo
was built from, although every later test depends on them.
"""

from __future__ import annotations

import functools
import sys


def find_caches(prefix: str = "sqlglot") -> list:
    caches = []
    for name, module in list(sys.modules.items()):
        if module is None or not (name == prefix or name.startswith(prefix + ".")):
            continue
        for attr, value in list(vars(module).items()):
            if isinstance(value, functools._lru_cache_wrapper):  # type: ignore[attr-defined]
                caches.append(value)
            elif "cache" in attr.lower() and isinstance(value, (dict, set, list)):
                caches.append(value)
    return caches


def reset(caches: list) -> None:
    for cache in caches:
        if isinstance(cache, functools._lru_cache_wrapper):  # type: ignore[attr-defined]
            cache.cache_clear()
        else:
            cache.clear()
