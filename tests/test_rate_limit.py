"""Tests for rate limiting backends and identity derivation."""

from starlette.requests import Request

from src.api.middleware import RateLimitMiddleware
from src.api.rate_limit import (
    InMemoryRateLimiter,
    create_rate_limiter,
    request_identity,
)


def _request(headers: dict[str, str], host: str) -> Request:
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/api/v1/ask",
        "raw_path": b"/api/v1/ask",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (host, 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    })


class TestInMemoryRateLimiter:
    def test_allows_up_to_limit_then_blocks(self):
        limiter = InMemoryRateLimiter(clock=lambda: 1000.0)
        first = limiter.hit("user", limit=2, window=60)
        second = limiter.hit("user", limit=2, window=60)
        third = limiter.hit("user", limit=2, window=60)

        assert first.allowed
        assert first.remaining == 1
        assert second.allowed
        assert second.remaining == 0
        assert not third.allowed
        assert third.remaining == 0
        assert 0 < third.reset_after <= 60

    def test_window_reset_allows_again(self):
        clock = [1000.0]
        limiter = InMemoryRateLimiter(clock=lambda: clock[0])
        assert limiter.hit("user", limit=1, window=60).allowed
        assert not limiter.hit("user", limit=1, window=60).allowed

        clock[0] += 60  # next fixed window
        assert limiter.hit("user", limit=1, window=60).allowed

    def test_keys_are_independent(self):
        limiter = InMemoryRateLimiter(clock=lambda: 500.0)
        assert limiter.hit("a", limit=1, window=60).allowed
        assert not limiter.hit("a", limit=1, window=60).allowed
        assert limiter.hit("b", limit=1, window=60).allowed

    def test_old_buckets_are_pruned(self):
        clock = [0.0]
        limiter = InMemoryRateLimiter(clock=lambda: clock[0])
        limiter.hit("a", limit=10, window=1)
        limiter.hit("b", limit=10, window=1)
        clock[0] = 10.0
        limiter.hit("c", limit=10, window=1)
        assert all(bucket >= 9 for _, bucket in limiter._counters)

    def test_reset_clears_counters(self):
        limiter = InMemoryRateLimiter(clock=lambda: 0.0)
        limiter.hit("a", limit=1, window=60)
        limiter.reset()
        assert limiter.hit("a", limit=1, window=60).allowed


class TestRateLimiterFactory:
    def test_falls_back_to_memory_when_redis_unreachable(self, monkeypatch):
        monkeypatch.setattr("src.api.rate_limit.settings.enable_cache", True)
        monkeypatch.setattr("src.api.rate_limit.settings.redis_url", "redis://127.0.0.1:1/0")
        limiter = create_rate_limiter()
        assert isinstance(limiter, InMemoryRateLimiter)

    def test_uses_memory_when_cache_disabled(self, monkeypatch):
        monkeypatch.setattr("src.api.rate_limit.settings.enable_cache", False)
        limiter = create_rate_limiter()
        assert isinstance(limiter, InMemoryRateLimiter)


class TestRequestIdentity:
    def test_api_key_is_hashed_not_leaked(self):
        identity = request_identity({"x-api-key": "super-secret-key"}, None)
        assert identity.startswith("key:")
        assert "super-secret" not in identity
        assert len(identity) == len("key:") + 16

    def test_bearer_token_is_supported(self):
        identity = request_identity({"authorization": "Bearer token-123"}, None)
        assert identity.startswith("key:")

    def test_falls_back_to_ip(self):
        assert request_identity({}, "203.0.113.9") == "ip:203.0.113.9"
        assert request_identity({}, None) == "ip:unknown"

    def test_forwarded_for_only_trusted(self):
        headers = {"x-forwarded-for": "198.51.100.7, 10.0.0.1"}
        assert (
            request_identity(headers, "10.0.0.1", trusted_proxies=["10.0.0.1"]) == "ip:198.51.100.7"
        )
        # Untrusted peer: ignore the header entirely (no spoofing).
        assert (
            request_identity(headers, "192.0.2.5", trusted_proxies=["10.0.0.1"]) == "ip:192.0.2.5"
        )


class TestMiddlewareIdentity:
    """The middleware must use the same rules (a wiring bug here is invisible
    to the unit tests above)."""

    def _middleware(self) -> RateLimitMiddleware:
        return RateLimitMiddleware(lambda scope, receive, send: None)

    def test_trusted_proxy_forwarded_for_is_used(self, monkeypatch):
        monkeypatch.setattr("src.api.middleware.settings.trusted_proxies", ["10.0.0.1"])
        middleware = self._middleware()
        request = _request({"X-Forwarded-For": "198.51.100.7, 10.0.0.1"}, "10.0.0.1")
        assert middleware._identity(request) == "ip:198.51.100.7"

    def test_untrusted_peer_cannot_spoof_forwarded_for(self, monkeypatch):
        monkeypatch.setattr("src.api.middleware.settings.trusted_proxies", ["10.0.0.1"])
        middleware = self._middleware()
        request = _request({"X-Forwarded-For": "198.51.100.7"}, "192.0.2.5")
        assert middleware._identity(request) == "ip:192.0.2.5"

    def test_invalid_api_key_does_not_become_identity(self, monkeypatch):
        monkeypatch.setattr("src.api.middleware.settings.api_key", "the-real-key")
        middleware = self._middleware()
        request = _request({"X-API-Key": "rotating-fake-key-1"}, "192.0.2.5")
        assert middleware._identity(request) == "ip:192.0.2.5"

    def test_valid_api_key_becomes_hashed_identity(self, monkeypatch):
        monkeypatch.setattr("src.api.middleware.settings.api_key", "the-real-key")
        middleware = self._middleware()
        request = _request({"X-API-Key": "the-real-key"}, "192.0.2.5")
        identity = middleware._identity(request)
        assert identity.startswith("key:")
        assert "the-real-key" not in identity
