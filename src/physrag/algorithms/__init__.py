"""Algorithmic core with transparent C++/Python dispatch.

If the compiled pybind11 extension ``physrag_cpp`` is importable, its optimized
routines are used; otherwise the pure-Python reference implementations in
``fallback_py`` are used. The public API is identical either way.
"""
from __future__ import annotations

from . import fallback_py as _py
from .interval_merger import UnionFind, dedupe_episodes, merge_intervals
from .spatial_cache import SpatialCache

try:  # optional compiled core (make build-cpp)
    import physrag_cpp as _cpp  # type: ignore
    HAVE_CPP = True
except Exception:  # pragma: no cover - depends on build environment
    _cpp = None
    HAVE_CPP = False

BACKEND = "cpp" if HAVE_CPP else "python"


def multi_source_dijkstra_subgraph(graph, sources, max_hops, max_dist=float("inf")):
    if HAVE_CPP:
        # The C++ core is int-keyed; relabel arbitrary (e.g. string) nodes to ints,
        # run, and map the result back. Relabeling is O(V + E) and preserved exactly.
        label = {}
        def idx(n):
            if n not in label:
                label[n] = len(label)
            return label[n]
        int_graph = {}
        for u, nbrs in graph.items():
            int_graph[idx(u)] = [(idx(v), float(w)) for v, w in nbrs]
        int_sources = [idx(s) for s in sources if s in graph or s in label]
        inv = {i: n for n, i in label.items()}
        res = _cpp.multi_source_dijkstra_subgraph(int_graph, int_sources, int(max_hops),
                                                  float(max_dist))
        return {inv[i]: (d, h) for i, (d, h) in res.items()}
    return _py.multi_source_dijkstra_subgraph(graph, sources, max_hops, max_dist)


def shockwave_detector(speeds, window, drop_fps):
    if HAVE_CPP:
        return list(_cpp.shockwave_detector(list(speeds), int(window), float(drop_fps)))
    return _py.shockwave_detector(speeds, window, drop_fps)


def knapsack_pack(tokens, relevance, physics_residual, budget, physics_lambda=4.0):
    if HAVE_CPP:
        return _cpp.knapsack_pack(list(tokens), list(relevance), list(physics_residual),
                                  int(budget), float(physics_lambda))
    return _py.knapsack_pack(tokens, relevance, physics_residual, budget, physics_lambda)


__all__ = [
    "BACKEND", "HAVE_CPP",
    "multi_source_dijkstra_subgraph", "shockwave_detector", "knapsack_pack",
    "UnionFind", "dedupe_episodes", "merge_intervals", "SpatialCache",
]
