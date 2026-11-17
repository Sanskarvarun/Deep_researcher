"""Security helpers: secret handling, config checks, and safe paths.

Nothing in this module reads secrets from anywhere except
:mod:`src.core.config` (which is backed by environment variables / ``.env``).
"""

from __future__ import annotations

import hmac
import re
from pathlib import Path
from typing import Any

from src.core.config import Settings, settings
from src.core.exceptions import ConfigurationError, ValidationError
from src.core.logging import get_logger

log = get_logger(__name__)

REDACTED = "***REDACTED***"

# Settings attributes that must never be logged or returned to clients.
SECRET_FIELDS = (
    "google_api_key",
    "openrouter_api_key",
    "groq_api_key",
    "hf_api_token",
    "qdrant_api_key",
    "cohere_api_key",
    "langsmith_api_key",
    "ragas_app_token",
    "api_key",
)

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|authorization|cookie|credential)",
    re.IGNORECASE,
)

_WEAK_API_KEYS = {"changeme", "change-me", "secret", "test", "dev", "password", "api_key"}
_MIN_API_KEY_LENGTH = 16


def redact(value: str | None) -> str:
    """Replace a secret value with a fixed marker (empty stays empty)."""
    if value is None or value == "":
        return ""
    return REDACTED


def mask_secret(value: str | None, *, visible: int = 4) -> str:
    """Return a log-safe preview such as ``abcd…wxyz``.

    Short values are fully redacted so masking can't leak them.
    """
    if not value:
        return ""
    if len(value) <= visible * 2:
        return REDACTED
    return f"{value[:visible]}…{value[-visible:]}"


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact values whose keys look sensitive."""
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if _SENSITIVE_KEY_PATTERN.search(str(key)):
            redacted[key] = REDACTED if value else value
        elif isinstance(value, dict):
            redacted[key] = redact_mapping(value)
        elif isinstance(value, list):
            redacted[key] = [
                redact_mapping(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            redacted[key] = value
    return redacted


def constant_time_equals(left: str, right: str) -> bool:
    """Compare two strings without leaking length/content via timing."""
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def secret_warnings(config: Settings | None = None) -> list[str]:
    """Non-fatal security observations (logged at startup)."""
    config = config or settings
    warnings: list[str] = []
    for field in SECRET_FIELDS:
        value = getattr(config, field, "")
        if isinstance(value, str) and value and value.lower() in _WEAK_API_KEYS:
            warnings.append(f"{field.upper()} uses a weak placeholder value")
    if config.api_key and len(config.api_key) < _MIN_API_KEY_LENGTH:
        warnings.append(
            f"API_KEY is shorter than {_MIN_API_KEY_LENGTH} characters; use a random token"
        )
    if "*" in config.cors_origins:
        warnings.append("CORS_ORIGINS allows every origin ('*')")
    return warnings


def validate_security_config(config: Settings | None = None) -> list[str]:
    """Validate security settings; raise in production on unsafe defaults.

    Returns the list of non-fatal warnings so callers can log them.
    """
    config = config or settings
    environment = config.app_env.lower()
    production = environment in {"production", "prod"}
    warnings = secret_warnings(config)
    errors: list[str] = []

    if config.require_auth and not config.api_key:
        (errors if production else warnings).append(
            "REQUIRE_AUTH is true but API_KEY is empty; all authenticated endpoints "
            "would reject requests"
        )
    if config.api_key and config.api_key.lower() in _WEAK_API_KEYS:
        if production:
            errors.append("API_KEY is a well-known placeholder value")
        else:
            warnings.append("API_KEY is a well-known placeholder value")
    if production:
        if config.debug:
            errors.append("DEBUG must be false in production")
        if "*" in config.cors_origins:
            errors.append("CORS_ORIGINS must not contain '*' in production")
        if not config.langsmith_api_key and config.enable_tracing:
            warnings.append("Tracing is enabled but LANGSMITH_API_KEY is empty")

    if errors:
        raise ConfigurationError(
            f"Unsafe configuration for environment '{config.app_env}': " + "; ".join(errors),
            details={"errors": errors, "warnings": warnings},
        )
    return warnings


def resolve_safe_path(
    raw_path: str,
    *,
    allowed_extensions: list[str] | None = None,
    base_dir: str | Path | None = None,
    max_size_mb: int | None = None,
) -> Path:
    """Resolve a user-supplied path, rejecting traversal and unsafe files.

    Args:
        raw_path: Untrusted path from a request.
        allowed_extensions: Optional allowlist such as ``[".pdf", ".txt"]``.
        base_dir: Optional root the resolved path must stay inside.
        max_size_mb: Optional size ceiling.

    Raises:
        ValidationError: If the path is unsafe or the file is unacceptable.
    """
    if not raw_path or not raw_path.strip():
        raise ValidationError("File path must not be empty", details={"path": raw_path})
    if "\x00" in raw_path:
        raise ValidationError("File path contains a null byte", details={"path": raw_path})

    candidate = Path(raw_path).expanduser()
    if ".." in candidate.parts:
        raise ValidationError("Path traversal is not allowed", details={"path": raw_path})

    resolved = candidate.resolve()
    if base_dir is not None:
        root = Path(base_dir).expanduser().resolve()
        if not resolved.is_relative_to(root):
            raise ValidationError(
                "Path is outside the allowed directory",
                details={"path": raw_path, "base_dir": str(root)},
            )

    if allowed_extensions is not None:
        suffix = resolved.suffix.lower()
        allowed = {ext.lower() for ext in allowed_extensions}
        if suffix not in allowed:
            raise ValidationError(
                "Unsupported file extension",
                details={"path": raw_path, "extension": suffix, "allowed": sorted(allowed)},
            )

    if not resolved.is_file():
        raise ValidationError("File does not exist", details={"path": str(resolved)})

    if max_size_mb is not None:
        size_mb = resolved.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(
                "File exceeds the maximum size",
                details={
                    "path": str(resolved),
                    "size_mb": round(size_mb, 2),
                    "max_size_mb": max_size_mb,
                },
            )

    return resolved
