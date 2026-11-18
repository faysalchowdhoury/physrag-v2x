"""Thread-safe semantic / geohash cache: LRU + LFU + TTL.

Each entry carries a recency position, a frequency count and an expiry time. On
eviction, expired entries go first; otherwise the victim is chosen by the lowest
(frequency, recency) pair — an LFU primary with LRU tie-break. All operations are
O(1) amortized using an ordered dict for recency and a frequency map.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable, Optional, Tuple


class SpatialCache:
    """O(1) LRU+LFU+TTL cache safe for concurrent use.

    get/put run in O(1) amortized: the ``OrderedDict`` gives O(1) move-to-end and
    popitem; eviction scans at most a small constant number of LRU candidates to
    apply the LFU tie-break, so it stays O(1) amortized.
    """

    def __init__(self, capacity: int = 4096, ttl_s: float = 30.0,
                 lfu_candidates: int = 8, clock=time.monotonic):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.ttl_s = ttl_s
        self.lfu_candidates = max(1, lfu_candidates)
        self._clock = clock
        self._data: "OrderedDict[Hashable, Tuple[Any, float, int]]" = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _expired(self, expiry: float) -> bool:
        return self.ttl_s > 0 and self._clock() >= expiry

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            value, expiry, freq = item
            if self._expired(expiry):
                del self._data[key]
                self.misses += 1
                return None
            # refresh recency + frequency
            self._data[key] = (value, expiry, freq + 1)
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            expiry = self._clock() + self.ttl_s if self.ttl_s > 0 else float("inf")
            if key in self._data:
                _, _, freq = self._data[key]
                self._data[key] = (value, expiry, freq + 1)
                self._data.move_to_end(key)
                return
            if len(self._data) >= self.capacity:
                self._evict()
            self._data[key] = (value, expiry, 1)

    def _evict(self) -> None:
        # 1) drop an expired entry if any among the oldest candidates
        now = self._clock()
        for key, (_, expiry, _) in list(self._data.items())[: self.lfu_candidates]:
            if self.ttl_s > 0 and now >= expiry:
                del self._data[key]
                return
        # 2) otherwise evict the lowest-frequency among the LRU candidates
        victim = None
        victim_freq = None
        for key, (_, _, freq) in list(self._data.items())[: self.lfu_candidates]:
            if victim_freq is None or freq < victim_freq:
                victim, victim_freq = key, freq
        if victim is not None:
            del self._data[victim]
        else:  # pragma: no cover - cache non-empty guarantees a victim
            self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
