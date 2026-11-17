"""API key authentication."""

from __future__ import annotations

import hashlib

from fastapi import Header, Request

from src.core.config import settings
from src.core.exceptions import AuthenticationError, ConfigurationError
from src.core.security import constant_time_equals


def _presented_key(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return None


def key_identity(key: str) -> str:
    """Non-reversible identity for a key (safe for logs/metrics/Redis keys)."""
    return "key:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> str:
    """FastAPI dependency enforcing the configured API key.

    Returns an identity string used by auditing. In development ``REQUIRE_AUTH``
    can be set to false, in which case requests are anonymous.
    """
    if not settings.require_auth:
        return "anonymous"

    presented = _presented_key(x_api_key, authorization)
    if not presented:
        raise AuthenticationError(
            "Missing API key; send 'X-API-Key' or 'Authorization: Bearer <key>'"
        )
    if not settings.api_key:
        raise ConfigurationError(
            "REQUIRE_AUTH is enabled but API_KEY is not configured",
            details={"hint": "set API_KEY or REQUIRE_AUTH=false for local development"},
        )
    if not constant_time_equals(presented, settings.api_key):
        raise AuthenticationError("Invalid API key")

    identity = key_identity(presented)
    request.state.identity = identity
    return identity
