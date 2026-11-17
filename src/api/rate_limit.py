"""Per-identity, per-endpoint rate limiting.

Backends mirror the cache layer: Redis in production, in-memory locally, with
automatic fallback. Counters use fixed windows so both backends behave the
same and stay cheap (``INCR`` + ``EXPIRE`` in Redis, a dict locally).
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RateLimitResult:
    """Outcome of one rate-limit hit."""

    allowed: bool
    limit: int
    remaining: int
    reset_after: float


class RateLimiter(Protocol):
    """Interface every rate-limit backend implements."""

    def hit(self, key: str, *, limit: int, window: int) -> RateLimitResult: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class InMemoryRateLimiter:
    """Fixed-window counter stored in process memory.

    Suitable for local development and single-process deployments. Not shared
    across workers — use :class:`RedisRateLimiter` for that.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, int], int] = {}
        self._last_bucket = -1

    def hit(self, key: str, *, limit: int, window: int) -> RateLimitResult:
        window = max(1, int(window))
        now = self._clock()
        bucket = int(now // window)
        with self._lock:
            if bucket > self._last_bucket:
                self._prune(bucket)
                self._last_bucket = bucket
            counter_key = (key, bucket)
            count = self._counters.get(counter_key, 0) + 1
            self._counters[counter_key] = count
        reset_after = window - (now % window)
        return RateLimitResult(
            allowed=count <= limit,
            limit=limit,
            remaining=max(0, limit - count),
            reset_after=reset_after,
        )

    def _prune(self, current_bucket: int) -> None:
        stale = [key for key in self._counters if key[1] < current_bucket - 1]
        for key in stale:
            del self._counters[key]

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._last_bucket = -1

    def close(self) -> None:
        self.reset()


class RedisRateLimiter:
    """Fixed-window counter shared across workers via Redis."""

    PREFIX = "dr:rl:"

    def __init__(
        self,
        url: str,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        import redis

        self._client = redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        # Fail fast so the factory can fall back to in-memory.
        self._client.ping()
        self._clock = clock
        self._fallback = InMemoryRateLimiter()

    def hit(self, key: str, *, limit: int, window: int) -> RateLimitResult:
        window = max(1, int(window))
        now = int(self._clock())
        bucket = now // window
        redis_key = f"{self.PREFIX}{key}:{bucket}"
        try:
            pipeline = self._client.pipeline()
            pipeline.incr(redis_key)
            pipeline.expire(redis_key, window + 1)
            count = int(pipeline.execute()[0])
        except Exception as e:  # pragma: no cover - requires a live Redis to fail
            log.warning("rate_limit_redis_failed_fallback_memory", error=str(e))
            return self._fallback.hit(key, limit=limit, window=window)
        return RateLimitResult(
            allowed=count <= limit,
            limit=limit,
            remaining=max(0, limit - count),
            reset_after=window - (now % window),
        )

    def reset(self) -> None:
        try:
            for redis_key in self._client.scan_iter(match=f"{self.PREFIX}*", count=500):
                self._client.delete(redis_key)
        except Exception as e:  # pragma: no cover
            log.warning("rate_limit_redis_reset_failed", error=str(e))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()


_limiter: RateLimiter | None = None


def create_rate_limiter() -> RateLimiter:
    """Build the configured limiter, falling back to memory when Redis is down."""
    if settings.enable_cache and settings.redis_url:
        try:
            limiter = RedisRateLimiter(settings.redis_url)
            log.info("rate_limiter_backend_selected", backend="redis")
            return limiter
        except Exception as e:
            log.warning("rate_limiter_redis_unavailable_fallback_memory", error=str(e))
    log.info("rate_limiter_backend_selected", backend="memory")
    return InMemoryRateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Return the process-wide limiter, creating it on first use."""
    global _limiter
    if _limiter is None:
        _limiter = create_rate_limiter()
    return _limiter


def close_rate_limiter() -> None:
    """Close and forget the process-wide limiter (shutdown/tests)."""
    global _limiter
    if _limiter is not None:
        _limiter.close()
        _limiter = None


def request_identity(
    headers: Mapping[str, str],
    client_host: str | None,
    *,
    trusted_proxies: list[str] | None = None,
) -> str:
    """Derive a stable, non-reversible identity for rate limiting.

    API keys win over IPs; the raw key is hashed so it never appears in
    metrics, logs, or Redis keys. ``X-Forwarded-For`` is only honored when the
    direct peer is a trusted proxy (otherwise clients could spoof identity).
    """
    api_key = headers.get("x-api-key")
    if not api_key:
        authorization = headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            api_key = token
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"key:{digest}"

    host = client_host or "unknown"
    trusted = set(trusted_proxies or [])
    forwarded = headers.get("x-forwarded-for")
    if forwarded and host in trusted:
        host = forwarded.split(",")[0].strip() or host
    return f"ip:{host}"
