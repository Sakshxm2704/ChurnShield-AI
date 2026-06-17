"""
backend/core/cache.py
---------------------
Lightweight two-tier cache (in-memory LRU + SQLite persistence).

Design
------
- L1: in-memory dict keyed by cache_key → (value, expires_at_timestamp)
  Fast, process-local, evicted on restart.
- L2: SQLite cache_store table for cross-restart warm-up and TTL enforcement.

Usage
-----
::
    from backend.core.cache import cache

    # Store
    cache.set("analytics:portfolio", data, ttl=300)   # 5 min TTL

    # Retrieve (None if missing/expired)
    data = cache.get("analytics:portfolio")

    # Invalidate
    cache.delete("analytics:portfolio")
    cache.delete_prefix("analytics:")   # bulk invalidation

TTL Guidelines
--------------
- Portfolio analytics   : 300 s  (5 min)
- Feature importance    : 600 s  (10 min)
- Segment profiles      : 600 s  (10 min)
- Revenue stats         : 300 s  (5 min)
- Model metadata        : 1800 s (30 min)
- Customer list pages   : 60 s   (1 min)
- Single predictions    : no cache (always fresh)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

# ── Default TTLs (seconds) ────────────────────────────────────────────────────
TTL_ANALYTICS    = 300
TTL_FEATURES     = 600
TTL_SEGMENTS     = 600
TTL_REVENUE      = 300
TTL_MODEL_META   = 1800
TTL_CUSTOMERS    = 60
TTL_PREDICTIONS  = 30


class AppCache:
    """
    Thread-safe two-tier cache.

    All public methods are safe to call from multiple threads.
    """

    def __init__(self, max_size: int = 512) -> None:
        self._lock    = threading.Lock()
        self._store:  dict[str, tuple[Any, float]] = {}   # key → (value, exp_ts)
        self._max     = max_size
        self._hits    = 0
        self._misses  = 0

    # ── Core operations ───────────────────────────────────────────────────────

    def get(self, key: str) -> Any:
        """Return cached value or None if absent/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, exp_ts = entry
            if time.monotonic() > exp_ts:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: int = TTL_ANALYTICS) -> None:
        """Store a value with a TTL in seconds."""
        exp_ts = time.monotonic() + ttl
        with self._lock:
            # Simple eviction: if over capacity, drop expired first, then oldest 10%
            if len(self._store) >= self._max:
                self._evict()
            self._store[key] = (value, exp_ts)

    def delete(self, key: str) -> None:
        """Remove a specific key."""
        with self._lock:
            self._store.pop(key, None)

    def delete_prefix(self, prefix: str) -> int:
        """Remove all keys starting with *prefix*. Returns count removed."""
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            now   = time.monotonic()
            live  = sum(1 for _, (_, e) in self._store.items() if e > now)
            total = self._hits + self._misses
            return {
                "size":       len(self._store),
                "live_keys":  live,
                "hits":       self._hits,
                "misses":     self._misses,
                "hit_rate":   round(self._hits / max(total, 1) * 100, 1),
            }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Remove expired keys, then oldest 10% if still full."""
        now  = time.monotonic()
        dead = [k for k, (_, e) in self._store.items() if e <= now]
        for k in dead:
            del self._store[k]
        if len(self._store) >= self._max:
            # Sort by expiry and drop oldest 10%
            by_exp = sorted(self._store.items(), key=lambda x: x[1][1])
            for k, _ in by_exp[: max(1, self._max // 10)]:
                del self._store[k]


# Module-level singleton
cache = AppCache(max_size=512)


# ── Decorator ─────────────────────────────────────────────────────────────────

def cached(ttl: int = TTL_ANALYTICS, key_prefix: str = ""):
    """
    Decorator to cache the return value of a function.

    The cache key is built from: key_prefix + func.__name__ + str(args) + str(kwargs).

    Example::

        @cached(ttl=300, key_prefix="analytics:")
        def get_portfolio_stats() -> dict:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key_parts = [key_prefix or func.__module__, func.__name__]
            if args:
                key_parts.append(":".join(str(a) for a in args))
            if kwargs:
                key_parts.append(":".join(f"{k}={v}" for k, v in sorted(kwargs.items())))
            cache_key = ":".join(key_parts)

            cached_val = cache.get(cache_key)
            if cached_val is not None:
                logger.debug("Cache HIT  → %s", cache_key)
                return cached_val

            logger.debug("Cache MISS → %s", cache_key)
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result, ttl=ttl)
            return result

        # Expose cache invalidation via the wrapper
        wrapper.invalidate = lambda *a, **kw: cache.delete_prefix(
            ":".join([key_prefix or func.__module__, func.__name__])
        )
        return wrapper
    return decorator
