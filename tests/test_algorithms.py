"""Unit + Big-O stress tests for the algorithmic core (C++ or Python backend)."""
import random
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physrag.algorithms import (BACKEND, SpatialCache, UnionFind, dedupe_episodes,
                                knapsack_pack, merge_intervals,
                                multi_source_dijkstra_subgraph, shockwave_detector)


# --- Dijkstra subgraph -------------------------------------------------------

def test_dijkstra_bounded_hops_and_distance():
    g = {"a": [("b", 1.0), ("c", 5.0)], "b": [("c", 1.0), ("d", 1.0)],
         "c": [("d", 1.0)], "d": []}
    reached = multi_source_dijkstra_subgraph(g, ["a"], max_hops=2)
    assert reached["a"][0] == 0.0
    assert reached["b"][0] == 1.0
    assert reached["c"][0] == 2.0          # a->b->c cheaper than a->c (5)
    assert "d" not in reached or reached["d"][1] <= 2

    near = multi_source_dijkstra_subgraph(g, ["a"], max_hops=9, max_dist=2.0)
    assert all(dist <= 2.0 for dist, _ in near.values())


def test_dijkstra_multi_source():
    # bidirectional chain 0-1-2-3; sources at both ends meet in the middle
    g = {0: [(1, 1.0)], 1: [(0, 1.0), (2, 1.0)],
         2: [(1, 1.0), (3, 1.0)], 3: [(2, 1.0)]}
    reached = multi_source_dijkstra_subgraph(g, [0, 3], max_hops=1)
    assert reached[0][0] == 0.0 and reached[3][0] == 0.0
    assert reached[1][0] == 1.0 and reached[2][0] == 1.0    # 1 from 0, 2 from 3


def test_dijkstra_scales():
    # chain of N nodes; bounded run must stay near-linear and settle each node once
    n = 20000
    g = {i: [(i + 1, 1.0)] for i in range(n - 1)}
    g[n - 1] = []
    t0 = time.perf_counter()
    reached = multi_source_dijkstra_subgraph(g, [0], max_hops=n)
    assert len(reached) == n
    assert time.perf_counter() - t0 < 5.0


# --- shockwave / monotonic deque --------------------------------------------

def test_shockwave_detects_drop():
    speeds = [30, 30, 29, 28, 5, 4, 3, 30, 31]   # a sharp drop mid-stream
    flagged = shockwave_detector(speeds, window=3, drop_fps=20)
    assert flagged, "expected a shockwave window"
    assert all(0 <= i < len(speeds) for i in flagged)


def test_shockwave_none_on_smooth():
    speeds = [30.0 + 0.1 * i for i in range(50)]
    assert shockwave_detector(speeds, window=5, drop_fps=10) == []


def test_sliding_window_matches_bruteforce():
    from physrag.algorithms.fallback_py import _sliding_extrema
    rng = random.Random(0)
    vals = [rng.uniform(0, 100) for _ in range(500)]
    w = 7
    got_max = _sliding_extrema(vals, w, True)
    exp_max = [max(vals[i:i + w]) for i in range(len(vals) - w + 1)]
    assert got_max == exp_max


# --- knapsack packer ---------------------------------------------------------

def test_knapsack_respects_budget_and_value():
    tokens = [10, 20, 30]
    relevance = [1.0, 2.0, 3.0]
    resid = [0.0, 0.0, 0.0]
    chosen, value = knapsack_pack(tokens, relevance, resid, budget=40)
    assert sum(tokens[i] for i in chosen) <= 40
    assert value == pytest.approx(4.0)      # optimal: items 0 and 2 (10+30=40, val 4.0)
    assert chosen == [0, 2]


def test_knapsack_physics_penalty_excludes_spoof():
    tokens = [10, 10]
    relevance = [1.0, 1.0]
    resid = [0.0, 1.0]                       # second item violates physics
    chosen, _ = knapsack_pack(tokens, relevance, resid, budget=20, physics_lambda=4.0)
    assert chosen == [0]                     # spoofed item has value <= 0, dropped


def test_knapsack_matches_bruteforce():
    rng = random.Random(1)
    n = 14
    tokens = [rng.randint(1, 20) for _ in range(n)]
    rel = [rng.uniform(0, 5) for _ in range(n)]
    resid = [rng.uniform(0, 0.5) for _ in range(n)]
    B = 40
    _, dp_val = knapsack_pack(tokens, rel, resid, B, physics_lambda=2.0)
    vals = [max(0.0, rel[i] - 2.0 * resid[i]) for i in range(n)]
    best = 0.0
    for mask in range(1 << n):
        w = v = 0.0
        for i in range(n):
            if mask & (1 << i):
                w += tokens[i]; v += vals[i]
        if w <= B:
            best = max(best, v)
    assert dp_val == pytest.approx(best)


# --- union-find / interval merge --------------------------------------------

def test_merge_intervals():
    assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]
    assert merge_intervals([(1, 2), (3, 4)], gap=1.0) == [(1, 4)]
    assert merge_intervals([]) == []


def test_union_find():
    uf = UnionFind(range(6))
    uf.union(0, 1); uf.union(1, 2); uf.union(4, 5)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(4)
    groups = uf.groups()
    assert sorted(len(v) for v in groups.values()) == [1, 2, 3]


def test_dedupe_episodes():
    eps = [("e1", "cellA", (0.0, 5.0)), ("e2", "cellA", (4.0, 9.0)),
           ("e3", "cellB", (0.0, 5.0))]
    groups = dedupe_episodes(eps, gap=1.0)
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [1, 2]                   # e1+e2 merge in cellA; e3 separate


# --- spatial cache -----------------------------------------------------------

def test_cache_lru_ttl():
    clock = {"t": 0.0}
    cache = SpatialCache(capacity=2, ttl_s=10.0, clock=lambda: clock["t"])
    cache.put("a", 1); cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)                        # evicts least-frequent/old
    assert len(cache) == 2
    clock["t"] = 100.0                       # everything expired
    assert cache.get("b") is None


def test_cache_lfu_keeps_hot():
    cache = SpatialCache(capacity=2, ttl_s=0.0)
    cache.put("hot", 1); cache.put("cold", 2)
    for _ in range(5):
        cache.get("hot")
    cache.put("new", 3)                      # should evict "cold", keep "hot"
    assert cache.get("hot") == 1
    assert cache.get("cold") is None


def test_backend_name():
    assert BACKEND in ("cpp", "python")
