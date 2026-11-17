"""HTTP middleware: request IDs, security headers, body limits, rate limiting."""

from __future__ import annotations

import uuid
from time import perf_counter
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.api.auth import key_identity
from src.api.rate_limit import (
    RateLimiter,
    RateLimitResult,
    get_rate_limiter,
    request_identity,
)
from src.api.responses import error_response
from src.core.config import settings
from src.core.logging import get_logger
from src.core.metrics import metrics
from src.core.security import constant_time_equals

log = get_logger(__name__)

DEFAULT_EXEMPT_PATHS = frozenset({
    "/health",
    "/healthz",
    "/ready",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
})

_REQUEST_ID_MAX_LENGTH = 64


def _clean_request_id(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > _REQUEST_ID_MAX_LENGTH:
        return None
    if not all(ch.isalnum() or ch in "-_." for ch in candidate):
        return None
    return candidate


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a request ID and record latency/throughput metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = _clean_request_id(request.headers.get("x-request-id")) or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()

        elapsed_ms = (perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        path = request.url.path
        metrics.increment(
            "http.requests",
            labels={"method": request.method, "path": path, "status": str(response.status_code)},
        )
        metrics.observe("http.latency", elapsed_ms, labels={"path": path})
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative security headers for every response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        if settings.app_env.lower() in {"production", "prod"}:
            headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared body exceeds ``MAX_REQUEST_BYTES``."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared = int(content_length)
            except ValueError:
                return error_response(
                    400,
                    "INVALID_CONTENT_LENGTH",
                    "Content-Length must be an integer",
                    request_id=getattr(request.state, "request_id", None),
                )
            if declared > settings.max_request_bytes:
                return error_response(
                    413,
                    "PAYLOAD_TOO_LARGE",
                    "Request body exceeds the configured limit",
                    details={
                        "content_length": declared,
                        "max_request_bytes": settings.max_request_bytes,
                    },
                    request_id=getattr(request.state, "request_id", None),
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit per identity (API key or IP) and per endpoint."""

    def __init__(
        self,
        app: Any,
        *,
        limiter: RateLimiter | None = None,
        exempt_paths: frozenset[str] | set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._exempt = frozenset(exempt_paths) if exempt_paths else DEFAULT_EXEMPT_PATHS

    def _identity(self, request: Request) -> str:
        presented = request.headers.get("x-api-key")
        if not presented:
            scheme, _, token = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() == "bearer":
                presented = token.strip() or None
        # Only a *valid* key becomes an identity; otherwise fall back to IP so
        # rotating fake keys cannot bypass the limit.
        if presented and settings.api_key and constant_time_equals(presented, settings.api_key):
            return key_identity(presented)
        client_host = request.client.host if request.client else None
        # Forward only the proxy header: passing the raw request headers would
        # let any client pick its own (unverified) API key as identity.
        forwarded = request.headers.get("x-forwarded-for")
        headers = {"x-forwarded-for": forwarded} if forwarded else {}
        return request_identity(headers, client_host, trusted_proxies=settings.trusted_proxies)

    @staticmethod
    def _apply_headers(response: Response, result: RateLimitResult) -> None:
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(max(1, int(result.reset_after)))

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if not settings.enable_rate_limit or request.method == "OPTIONS" or path in self._exempt:
            return await call_next(request)

        limiter = self._limiter or get_rate_limiter()
        identity = self._identity(request)
        key = f"{identity}:{request.method}:{path}"
        result = limiter.hit(
            key, limit=settings.rate_limit_requests, window=settings.rate_limit_window
        )

        if not result.allowed:
            metrics.increment("http.rate_limited", labels={"path": path})
            log.warning(
                "rate_limit_exceeded",
                identity=identity,
                path=path,
                limit=result.limit,
                window=settings.rate_limit_window,
            )
            limited = error_response(
                429,
                "RATE_LIMIT_EXCEEDED",
                "Rate limit exceeded; retry later",
                details={
                    "limit": result.limit,
                    "window_seconds": settings.rate_limit_window,
                    "retry_after": max(1, int(result.reset_after)),
                },
                request_id=getattr(request.state, "request_id", None),
                headers={"Retry-After": str(max(1, int(result.reset_after)))},
            )
            self._apply_headers(limited, result)
            return limited

        response = await call_next(request)
        self._apply_headers(response, result)
        return response
