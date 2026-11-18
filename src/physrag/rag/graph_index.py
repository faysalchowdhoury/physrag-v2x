"""Redis-backed vector + spatiotemporal graph store, with an in-memory fallback.

Stores incident/context nodes with embeddings and typed edges (spatial adjacency,
temporal succession, traffic-grid coupling). If a Redis Stack server is reachable
it is used; otherwise an equivalent in-process store keeps everything runnable in
CI and tests. A deterministic hashing encoder provides embeddings without any
heavyweight model download.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np


def mock_encode(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic bag-of-hashed-tokens embedding, L2-normalized.

    Reproducible and dependency-free; good enough for ranking in CI and ablations
    while remaining swappable for a real sentence encoder.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] += 0.5
    n = np.linalg.norm(vec)
    return vec / n if n else vec


class GraphIndex:
    """Vector + graph store. ``backend`` is 'redis' or 'memory'."""

    def __init__(self, dim: int = 64, redis_url: Optional[str] = None):
        self.dim = dim
        self._emb: Dict[str, np.ndarray] = {}
        self._text: Dict[str, str] = {}
        self._meta: Dict[str, dict] = {}
        self._adj: Dict[str, List[Tuple[str, float]]] = {}
        self._redis = None
        self.backend = "memory"
        if redis_url:
            try:
                import redis  # type: ignore
                self._redis = redis.Redis.from_url(redis_url, socket_connect_timeout=0.5)
                self._redis.ping()
                self.backend = "redis"
            except Exception:
                self._redis = None
                self.backend = "memory"

    # --- writes ---
    def add_node(self, node_id: str, text: str, meta: Optional[dict] = None,
                 embedding: Optional[np.ndarray] = None) -> None:
        emb = embedding if embedding is not None else mock_encode(text, self.dim)
        self._emb[node_id] = emb.astype(np.float32)
        self._text[node_id] = text
        self._meta[node_id] = meta or {}
        self._adj.setdefault(node_id, [])
        if self._redis is not None:
            try:
                self._redis.hset(f"node:{node_id}", mapping={
                    "text": text, "emb": emb.astype(np.float32).tobytes()})
            except Exception:
                pass

    def add_edge(self, src: str, dst: str, weight: float = 1.0,
                 bidirectional: bool = True) -> None:
        self._adj.setdefault(src, []).append((dst, weight))
        if bidirectional:
            self._adj.setdefault(dst, []).append((src, weight))
        if self._redis is not None:
            try:
                self._redis.zadd(f"adj:{src}", {dst: weight})
                if bidirectional:
                    self._redis.zadd(f"adj:{dst}", {src: weight})
            except Exception:
                pass

    # --- reads ---
    @property
    def graph(self) -> Dict[str, List[Tuple[str, float]]]:
        return self._adj

    def text(self, node_id: str) -> str:
        return self._text.get(node_id, "")

    def meta(self, node_id: str) -> dict:
        return self._meta.get(node_id, {})

    def vector_search(self, query: str, top_k: int = 32) -> List[Tuple[str, float]]:
        """First-stage dense retrieval (cosine). Returns [(node_id, score)].

        Complexity: O(M * d) over M stored nodes; a true HNSW index (hnswlib) can
        be dropped in for O(log M) — the interface is identical.
        """
        if not self._emb:
            return []
        q = mock_encode(query, self.dim)
        ids = list(self._emb.keys())
        mat = np.stack([self._emb[i] for i in ids])
        scores = mat @ q
        order = np.argsort(-scores)[:top_k]
        return [(ids[i], float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self._emb)
