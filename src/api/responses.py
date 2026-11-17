"""Shared response helpers."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from src.api.schemas import ErrorBody, ErrorResponse


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the stable ``{"error": {...}}`` envelope used by every failure."""
    body = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            details=details or {},
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        headers=headers or {},
    )
