"""Tests for the cache abstraction (Phase 2.3 Caching & Performance)."""

import hashlib

import pytest

from src.core import cache as cache_module
from src.core.cache import (
    CacheConfig,
    InMemoryCache,
    RedisCache,
    cache_delete,
    cache_get,
    cache_set,
    close_cache,
    create_cache,
    get_cache,
    make_cache_key,
)


class TestInMemoryCache:
    def test_set_and_get(self):
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_missing_key_returns_none(self):
        cache: InMemoryCache[str] = InMemoryCache()
        assert cache.get("nope") is None

    def test_delete(self):
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None
        assert cache.delete("key") is False

    def test_exists(self):
        cache: InMemoryCache[str] = InMemoryCache()
        assert cache.exists("key") is False
        cache.set("key", "value")
        assert cache.exists("key") is True

    def test_clear(self):
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_ttl_expiry(self, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(cache_module, "time", lambda: clock[0])

        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "value", ttl=10)
        assert cache.get("key") == "value"

        clock[0] += 9
        assert cache.get("key") == "value"

        clock[0] += 2  # now past expiry
        assert cache.get("key") is None
        # Expired entry must be evicted, not merely hidden.
        assert cache.exists("key") is False

    def test_no_ttl_never_expires(self, monkeypatch):
        clock = [0.0]
        monkeypatch.setattr(cache_module, "time", lambda: clock[0])

        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "value")
        clock[0] += 10_000_000
        assert cache.get("key") == "value"

    def test_lru_eviction(self):
        cache: InMemoryCache[str] = InMemoryCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        # Touch "a" so "b" becomes the least recently used entry.
        assert cache.get("a") == 1
        cache.set("c", 3)

        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("b") is None

    def test_overwrite_updates_value(self):
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "first")
        cache.set("key", "second")
        assert cache.get("key") == "second"

    def test_close_clears(self):
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "value")
        cache.close()
        assert cache.get("key") is None


class TestCacheKeys:
    def test_key_is_deterministic(self):
        assert make_cache_key("emb", "model", "hello") == make_cache_key("emb", "model", "hello")

    def test_key_changes_with_parts(self):
        base = make_cache_key("llm", "prompt", "gemini", "", 0.0)
        assert base != make_cache_key("llm", "prompt2", "gemini", "", 0.0)
        assert base != make_cache_key("llm", "prompt", "groq", "", 0.0)
        assert base != make_cache_key("ret", "prompt", "gemini", "", 0.0)

    def test_key_matches_blake2b_digest(self):
        """Guards against regressions to Python's per-process randomized hash()."""
        payload = "\x1f".join(repr(p) for p in ("model", "hello"))
        expected = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
        assert make_cache_key("emb", "model", "hello") == f"emb:{expected}"

    def test_key_is_redis_safe(self):
        key = make_cache_key("llm", "multi\nline\tprompt", "gemini")
        assert " " not in key
        assert "\n" not in key
        assert len(key) < 128


class TestCacheFactory:
    def test_memory_backend(self):
        cache = create_cache(CacheConfig(backend="memory", max_memory_size=5))
        assert isinstance(cache, InMemoryCache)

    def test_redis_unavailable_falls_back_to_memory(self):
        """No Redis/Docker required: unreachable Redis must degrade gracefully."""
        cache = create_cache(CacheConfig(backend="redis", redis_url="redis://127.0.0.1:1/0"))
        assert isinstance(cache, InMemoryCache)
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_global_singleton(self):
        close_cache()
        first = get_cache()
        second = get_cache()
        assert first is second
        close_cache()

    def test_module_level_helpers(self, monkeypatch):
        close_cache()
        monkeypatch.setattr(cache_module, "_cache", InMemoryCache())

        cache_set("k", {"nested": [1, 2]})
        assert cache_get("k") == {"nested": [1, 2]}
        assert cache_delete("k") is True
        assert cache_get("k") is None
        close_cache()


def _redis_available(url: str = "redis://localhost:6379/0") -> bool:
    try:
        import redis

        client = redis.from_url(url, socket_connect_timeout=0.2)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _redis_available(), reason="Redis not running")
class TestRedisCache:
    def test_round_trip_and_prefix_isolation(self):
        cache: RedisCache = RedisCache("redis://localhost:6379/0", prefix="dr-test:")
        cache.clear()
        try:
            cache.set("answer", {"a": [1, 2, 3]})
            assert cache.get("answer") == {"a": [1, 2, 3]}
            assert cache.exists("answer") is True
            assert cache.delete("answer") is True
            assert cache.get("answer") is None

            cache.set("k1", 1)
            cache.set("k2", 2)
            cache.clear()
            assert cache.get("k1") is None
            assert cache.get("k2") is None
        finally:
            cache.close()

    def test_ttl_expiry(self):
        import time

        cache: RedisCache = RedisCache("redis://localhost:6379/0", prefix="dr-test:")
        try:
            cache.set("short", "value", ttl=1)
            assert cache.get("short") == "value"
            time.sleep(1.2)
            assert cache.get("short") is None
        finally:
            cache.close()
