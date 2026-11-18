"""Incident episode consolidation: Merge Intervals + Union-Find.

Overlapping/adjacent time intervals from the same spatial cell are merged into
episodes, and a Union-Find (disjoint-set) deduplicates episodes that are linked
across cells (e.g., a shockwave propagating between adjacent segments).
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Sequence, Tuple

Interval = Tuple[float, float]


def merge_intervals(intervals: Sequence[Interval], gap: float = 0.0) -> List[Interval]:
    """Merge overlapping (or within ``gap``) intervals.

    Complexity: O(N log N) time (dominated by the sort), O(N) space.
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged: List[Interval] = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = merged[-1]
        if s <= le + gap:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


class UnionFind:
    """Disjoint-set with path compression + union by rank.

    find/union run in O(alpha(N)) ~ O(1) amortized (inverse-Ackermann).
    """

    def __init__(self, items: Sequence[Hashable] = ()):
        self.parent: Dict[Hashable, Hashable] = {}
        self.rank: Dict[Hashable, int] = {}
        for it in items:
            self.add(it)

    def add(self, x: Hashable) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: Hashable) -> Hashable:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:   # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> Dict[Hashable, List[Hashable]]:
        out: Dict[Hashable, List[Hashable]] = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return out


def dedupe_episodes(
    episodes: Sequence[Tuple[Hashable, str, Interval]],
    gap: float = 1.0,
) -> Dict[Hashable, List[Hashable]]:
    """Group episodes that share a spatial cell and overlap in time.

    ``episodes`` is a sequence of ``(episode_id, cell, (t_start, t_end))``.
    Two episodes in the same cell whose intervals overlap (within ``gap``) are
    unioned. Returns ``{representative_id: [member_ids...]}``.

    Complexity: O(N log N) — sort per cell for the sweep, near-constant unions.
    """
    uf = UnionFind(ep[0] for ep in episodes)
    by_cell: Dict[str, List[Tuple[float, float, Hashable]]] = {}
    for eid, cell, (s, e) in episodes:
        by_cell.setdefault(cell, []).append((s, e, eid))
    for cell, items in by_cell.items():
        items.sort()
        prev_end, prev_id = None, None
        for s, e, eid in items:
            if prev_end is not None and s <= prev_end + gap:
                uf.union(prev_id, eid)
                prev_end = max(prev_end, e)
            else:
                prev_end, prev_id = e, eid
    return uf.groups()
