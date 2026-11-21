"""Split selected tests over workers to minimise busiest load + 2 s per worker."""

from __future__ import annotations

WORKER_MS = 2000
MAX_WORKERS = 8


def _lpt(durations: list[tuple[int, int]], k: int) -> tuple[int, list[list[int]]]:
    loads = [0] * k
    bins: list[list[int]] = [[] for _ in range(k)]
    for ms, number in durations:
        i = loads.index(min(loads))
        loads[i] += ms
        bins[i].append(number)
    return max(loads), bins


def _improve(durations: list[tuple[int, int]], k: int, best: int, bins: list[list[int]]):
    """Small exact search when the set is tiny; otherwise keep LPT."""
    if len(durations) > 12 or k == 1:
        return best, bins
    items = durations  # already sorted descending
    lower = max(items[0][0], -(-sum(ms for ms, _ in items) // k))
    if best == lower:
        return best, bins
    assign = [0] * len(items)
    loads = [0] * k
    result = [best, None]

    def rec(i: int, current_max: int) -> None:
        if current_max >= result[0]:
            return
        if i == len(items):
            result[0] = current_max
            result[1] = assign.copy()
            return
        ms = items[i][0]
        seen = set()
        for w in range(k):
            if loads[w] in seen:
                continue
            seen.add(loads[w])
            loads[w] += ms
            assign[i] = w
            rec(i + 1, max(current_max, loads[w]))
            loads[w] -= ms
            if result[0] == lower:
                return

    rec(0, 0)
    if result[1] is None:
        return best, bins
    new_bins: list[list[int]] = [[] for _ in range(k)]
    for (ms, number), w in zip(items, result[1]):
        new_bins[w].append(number)
    return result[0], new_bins


def pack(tests: list[tuple[int, int]]) -> list[list[int]]:
    """tests: (number, duration_ms). Returns the worker lists (no empty worker)."""
    if not tests:
        return []
    durations = sorted(((ms, number) for number, ms in tests), reverse=True)
    best_cost = None
    best_bins: list[list[int]] = []
    for k in range(1, min(MAX_WORKERS, len(durations)) + 1):
        load, bins = _lpt(durations, k)
        load, bins = _improve(durations, k, load, bins)
        cost = load + WORKER_MS * k
        if best_cost is None or cost < best_cost:
            best_cost, best_bins = cost, bins
        # Adding workers cannot beat the largest single test plus their cost.
        if durations[0][0] + WORKER_MS * (k + 1) >= best_cost:
            break
    return [sorted(b) for b in best_bins if b]


def cost_ms(tests: list[tuple[int, int]]) -> int:
    bins = pack(tests)
    by_number = dict(tests)
    if not bins:
        return 0
    return max(sum(by_number[n] for n in b) for b in bins) + WORKER_MS * len(bins)
