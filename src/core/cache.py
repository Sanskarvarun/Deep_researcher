"""Cache abstraction with pluggable backends (in-memory default, Redis optional).

Provides a unified interface so application code doesn't change between
local development (no Docker) and production (Redis).
"""

from __future__ import annotations

import contextlib
import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Any, Generic, TypeVar

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class CacheBackend(ABC, Generic[T]):
    """Abstract cache backend."""

    @abstractmethod
    def get(self, key: str) -> T | None:
        """Get value by key. Returns None if not found or expired."""

    @abstractmethod
    def set(self, key: str, value: T, ttl: int | None = None) -> None:
        """Set value with optional TTL (seconds)."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key. Returns True if key existed."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""


class InMemoryCache(CacheBackend[T]):
    """Thread-safe in-memory LRU cache with TTL support.

    Suitable for local development and single-process deployments.
    Not distributed — each process has its own cache.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._data: OrderedDict[str, tuple[T, float]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        with self._lock:
            if key not in self._data:
                return None
            value, expires_at = self._data[key]
            if expires_at > 0 and time() > expires_at:
                del self._data[key]
                return None
            # Move to end (LRU)
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: T, ttl: int | None = None) -> None:
        with self._lock:
            expires_at = time() + ttl if ttl and ttl > 0 else 0
            if key in self._data:
                self._data.move_to_end(key)
            elif len(self._data) >= self._max_size:
                self._data.popitem(last=False)
            self._data[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def close(self) -> None:
        self.clear()


class RedisCache(CacheBackend[T]):
    """Redis cache backend.

    Requires redis-py and a running Redis server. Keys are namespaced with a
    prefix so ``clear()`` only removes this application's entries instead of
    flushing the whole database. The constructor pings the server so callers
    can fail fast and fall back to the in-memory backend.
    """

    def __init__(self, url: str, default_ttl: int = 3600, prefix: str = "dr:") -> None:
        import redis

        self._prefix = prefix
        self._client = redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=1,
        )
        # Fail fast: without this, a missing Redis only shows up as a warning
        # on every cache access instead of an immediate fallback.
        self._client.ping()
        self._default_ttl = default_ttl
        self._closed = False

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> T | None:
        if self._closed:
            return None
        try:
            data = self._client.get(self._key(key))
            if data is None:
                return None
            import json

            return json.loads(data)  # type: ignore[no-any-return]
        except Exception as e:
            log.warning("redis_cache_get_failed", key=key, error=str(e))
            return None

    def set(self, key: str, value: T, ttl: int | None = None) -> None:
        if self._closed:
            return
        try:
            import json

            ttl = ttl if ttl is not None else self._default_ttl
            self._client.setex(self._key(key), ttl, json.dumps(value))
        except Exception as e:
            log.warning("redis_cache_set_failed", key=key, error=str(e))

    def delete(self, key: str) -> bool:
        if self._closed:
            return False
        try:
            return self._client.delete(self._key(key)) > 0
        except Exception as e:
            log.warning("redis_cache_delete_failed", key=key, error=str(e))
            return False

    def exists(self, key: str) -> bool:
        if self._closed:
            return False
        try:
            return self._client.exists(self._key(key)) > 0
        except Exception:
            return False

    def clear(self) -> None:
        if self._closed:
            return
        try:
            batch: list[str] = []
            for redis_key in self._client.scan_iter(match=f"{self._prefix}*", count=500):
                batch.append(redis_key)
                if len(batch) >= 500:
                    self._client.delete(*batch)
                    batch.clear()
            if batch:
                self._client.delete(*batch)
        except Exception as e:
            log.warning("redis_cache_clear_failed", error=str(e))

    def close(self) -> None:
        if not self._closed:
            with contextlib.suppress(Exception):
                self._client.close()
            self._closed = True


@dataclass
class CacheConfig:
    """Cache configuration."""

    backend: str = "memory"  # "memory" or "redis"
    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 3600
    max_memory_size: int = 10_000


def create_cache(config: CacheConfig | None = None) -> CacheBackend[Any]:
    """Factory function to create cache backend from config."""
    config = config or CacheConfig(
        backend="redis" if settings.enable_cache and settings.redis_url else "memory",
        redis_url=settings.redis_url,
        default_ttl=settings.cache_ttl,
        max_memory_size=10_000,
    )

    if config.backend == "redis":
        try:
            log.info("cache_backend_selected", backend="redis", url=config.redis_url)
            return RedisCache(config.redis_url, config.default_ttl)
        except Exception as e:
            log.warning("redis_cache_unavailable_fallback_memory", error=str(e))
            return InMemoryCache(config.max_memory_size)
    else:
        log.info("cache_backend_selected", backend="memory", max_size=config.max_memory_size)
        return InMemoryCache(config.max_memory_size)


# Module-level cache instance (lazy-initialized)
_cache: CacheBackend[Any] | None = None


def get_cache() -> CacheBackend[Any]:
    """Get the global cache instance, creating it on first use."""
    global _cache
    if _cache is None:
        _cache = create_cache()
    return _cache


def close_cache() -> None:
    """Close the global cache instance."""
    global _cache
    if _cache is not None:
        _cache.close()
        _cache = None


# Convenience functions
def cache_get(key: str) -> Any | None:
    return get_cache().get(key)  # type: ignore[no-any-return]


def cache_set(key: str, value: Any, ttl: int | None = None) -> None:
    get_cache().set(key, value, ttl)


def cache_delete(key: str) -> bool:
    return get_cache().delete(key)


def make_cache_key(prefix: str, *parts: object) -> str:
    """Build a stable, process-independent cache key.

    Python's built-in ``hash()`` is randomized per process for strings, so it
    must never be used for cache keys that have to survive across worker
    processes or restarts (e.g. a shared Redis cache). BLAKE2b keeps keys
    deterministic while staying short enough for Redis.
    """
    payload = "\x1f".join(repr(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    return f"{prefix}:{digest}"
