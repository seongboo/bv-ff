"""
Byte-bounded LRU caches (neighbor list / k-vectors).

The caches are content-keyed, so during fitting the fixed training frames give
a bounded working set — but MD/relaxation through BVFFCalculator produces a
new positions key every step, which with an unbounded dict is a memory leak
over a long trajectory. These tests pin the eviction contract of
``_LRUBytesCache`` and that the neighbor cache actually evicts under pressure.
"""
from __future__ import annotations

import numpy as np

from src.potentials import _LRUBytesCache, _NEIGHBOR_CACHE, Coulomb, clear_neighbor_cache


def test_lru_evicts_oldest_beyond_byte_budget():
    c = _LRUBytesCache(max_bytes=100)
    c.put("a", 1, 40)
    c.put("b", 2, 40)
    assert c.get("a") == 1 and c.get("b") == 2
    c.put("c", 3, 40)                      # 120 > 100 → evict LRU ("a")
    assert c.get("a") is None
    assert c.get("b") == 2 and c.get("c") == 3


def test_lru_recency_updated_on_get():
    c = _LRUBytesCache(max_bytes=100)
    c.put("a", 1, 40)
    c.put("b", 2, 40)
    c.get("a")                             # "a" becomes most-recent
    c.put("c", 3, 40)                      # eviction must hit "b", not "a"
    assert c.get("a") == 1
    assert c.get("b") is None


def test_lru_keeps_newest_even_if_oversized():
    c = _LRUBytesCache(max_bytes=10)
    c.put("big", "x", 1000)                # single oversized entry survives
    assert c.get("big") == "x"


def test_neighbor_cache_bounded_under_md_like_churn():
    """Simulate MD churn (fresh positions every step): the cache must stay
    within its byte budget instead of growing without bound."""
    clear_neighbor_cache()
    old_max = _NEIGHBOR_CACHE.max_bytes
    _NEIGHBOR_CACHE.max_bytes = 200_000    # tiny budget to force eviction
    try:
        pot = Coulomb(charges={"X": 1.0}, cutoff=6.0)
        lattice = np.eye(3) * 7.8
        rng = np.random.default_rng(0)
        for _ in range(30):
            positions = rng.random((40, 3))
            pot.energy(lattice, ["X"] * 40, positions)
        assert _NEIGHBOR_CACHE._nbytes <= 2 * 200_000   # bounded (newest may overshoot once)
        assert len(_NEIGHBOR_CACHE._data) < 30           # actually evicted
    finally:
        _NEIGHBOR_CACHE.max_bytes = old_max
        clear_neighbor_cache()
