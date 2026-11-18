"""Pure-Python reference implementations of the C++ algorithmic core.

These match the ``cpp_core`` signatures and are used automatically whenever the
compiled pybind11 extension is not importable. Every routine documents its
asymptotic complexity and is covered by ``tests/test_algorithms.py``.

Routines
--------
multi_source_dijkstra_subgraph : bounded k-hop / distance causal subgraph
shockwave_detector             : monotonic-deque sliding-window shockwave finder
knapsack_pack                  : 0-1 knapsack physics-penalized token packer
"""
from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, Hashable, List, Sequence, Tuple

Node = Hashable
Graph = Dict[Node, List[Tuple[Node, float]]]


# --------------------------------------------------------------------------- #
# 1. Multi-source bounded Dijkstra -> causal subgraph                         #
# --------------------------------------------------------------------------- #
def multi_source_dijkstra_subgraph(
    graph: Graph,
    sources: Sequence[Node],
    max_hops: int,
    max_dist: float = float("inf"),
) -> Dict[Node, Tuple[float, int]]:
    """Extract the causal subgraph reachable from any source within a hop and
    distance budget, via a multi-source Dijkstra.

    Returns ``{node: (shortest_distance, hops)}`` for every node kept.

    Complexity: O((V + E) log V) time, O(V) space — each node is settled once
    and each edge relaxed once; the binary heap gives the log V factor.
    """
    best: Dict[Node, Tuple[float, int]] = {}
    pq: List[Tuple[float, int, Node]] = []
    for s in sources:
        if s in graph or s in best:
            heapq.heappush(pq, (0.0, 0, s))
    while pq:
        dist, hops, u = heapq.heappop(pq)
        if u in best and best[u][0] <= dist:
            continue
        best[u] = (dist, hops)
        if hops >= max_hops:
            continue
        for v, w in graph.get(u, ()):  # neighbours
            nd = dist + w
            if nd > max_dist:
                continue
            if v not in best or nd < best[v][0]:
                heapq.heappush(pq, (nd, hops + 1, v))
    return best


# --------------------------------------------------------------------------- #
# 2. Monotonic-deque sliding-window shockwave detector                        #
# --------------------------------------------------------------------------- #
def _sliding_extrema(values: Sequence[float], window: int, want_max: bool) -> List[float]:
    """Sliding-window max (or min) via a monotonic deque. O(n) amortized, each
    index pushed/popped once; O(window) space."""
    if window <= 0:
        raise ValueError("window must be positive")
    dq: deque[int] = deque()
    out: List[float] = []
    for i, x in enumerate(values):
        while dq and dq[0] <= i - window:
            dq.popleft()
        if want_max:
            while dq and values[dq[-1]] <= x:
                dq.pop()
        else:
            while dq and values[dq[-1]] >= x:
                dq.pop()
        dq.append(i)
        if i >= window - 1:
            out.append(values[dq[0]])
    return out


def shockwave_detector(
    speeds: Sequence[float],
    window: int,
    drop_fps: float,
) -> List[int]:
    """Detect traffic shockwaves: window end-indices where the in-window speed
    drop (max - min) meets or exceeds ``drop_fps``.

    Uses two monotonic deques (one for max, one for min) so each window's extrema
    are O(1) amortized. Returns the list of end indices (0-based) of flagged
    windows.

    Complexity: O(n) time amortized, O(window) space.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if len(speeds) < window:
        return []
    maxes = _sliding_extrema(speeds, window, True)
    mins = _sliding_extrema(speeds, window, False)
    flagged = []
    for j, (mx, mn) in enumerate(zip(maxes, mins)):
        if mx - mn >= drop_fps:
            flagged.append(j + window - 1)
    return flagged


# --------------------------------------------------------------------------- #
# 3. 0-1 Knapsack physics-penalized token packer                             #
# --------------------------------------------------------------------------- #
def knapsack_pack(
    tokens: Sequence[int],
    relevance: Sequence[float],
    physics_residual: Sequence[float],
    budget: int,
    physics_lambda: float = 4.0,
) -> Tuple[List[int], float]:
    """Select a subset of context items maximizing physics-penalized value within
    a token budget, via 0-1 knapsack dynamic programming.

    Item value = relevance - physics_lambda * physics_residual (clamped at 0 so a
    physically implausible item is never worth including). Returns the selected
    item indices (ascending) and the achieved total value.

    Complexity: O(N * B) time, O(N * B) space (N items, B = budget). A min-heap
    pre-prunes dominated items but the DP bound is the guarantee.
    """
    n = len(tokens)
    if not (n == len(relevance) == len(physics_residual)):
        raise ValueError("tokens/relevance/physics_residual length mismatch")
    if budget <= 0 or n == 0:
        return [], 0.0
    values = [max(0.0, relevance[i] - physics_lambda * physics_residual[i])
              for i in range(n)]
    # DP table: dp[i][b] = best value using first i items within budget b.
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w = tokens[i - 1]
        val = values[i - 1]
        row, prev = dp[i], dp[i - 1]
        for b in range(budget + 1):
            if w <= b and val > 0.0:
                take = prev[b - w] + val
                row[b] = take if take > prev[b] else prev[b]
            else:
                row[b] = prev[b]
    # Backtrack for the chosen items.
    chosen: List[int] = []
    b = budget
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            chosen.append(i - 1)
            b -= tokens[i - 1]
    chosen.reverse()
    return chosen, dp[n][budget]
