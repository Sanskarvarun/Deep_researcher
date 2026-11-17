"""FastAPI application factory.

Middleware order (outermost first at runtime):
CORS -> request context -> security headers -> body limit -> rate limit.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from src.api.responses import error_response
from src.api.routes import API_VERSION, router
from src.core import DeepResearcherError
from src.core.config import settings
from src.core.logging import get_logger
from src.core.security import validate_security_config

log = get_logger(__name__)


def create_app() -> FastAPI:
    """Build the ASGI app with middleware, routes, and error handlers."""
    warnings = validate_security_config()
    for warning in warnings:
        log.warning("security_config_warning", warning=warning)

    production = settings.app_env.lower() in {"production", "prod"}
    app = FastAPI(
        title="DeepResearcher API",
        version=API_VERSION,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )

    # Added innermost first; Starlette wraps in reverse order.
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials="*" not in settings.cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
            expose_headers=[
                "X-Request-ID",
                "X-Response-Time-Ms",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
            ],
        )

    app.include_router(router)

    @app.exception_handler(DeepResearcherError)
    def _handle_app_error(request: Request, exc: DeepResearcherError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        if exc.status_code >= 500:
            log.error(
                "request_failed",
                code=exc.code,
                error=exc.message,
                request_id=request_id,
            )
        else:
            log.info(
                "request_rejected",
                code=exc.code,
                error=exc.message,
                request_id=request_id,
            )
        return error_response(
            exc.status_code,
            exc.code,
            exc.message,
            details=exc.details,
            request_id=request_id,
        )

    @app.exception_handler(StarletteHTTPException)
    def _handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Unknown routes and wrong methods bypass our AppError hierarchy; give
        # them the same envelope so clients never have to parse two shapes.
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        message = {
            404: "Resource not found",
            405: "Method not allowed for this resource",
        }.get(exc.status_code, str(exc.detail))
        headers = dict(getattr(exc, "headers", None) or {})
        return error_response(
            exc.status_code,
            code,
            message,
            request_id=getattr(request.state, "request_id", None),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg", "invalid value"),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        return error_response(
            422,
            "VALIDATION_ERROR",
            "Request validation failed",
            details={"errors": errors},
            request_id=getattr(request.state, "request_id", None),
        )

    @app.exception_handler(Exception)
    def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception(
            "unhandled_error",
            error=str(exc),
            request_id=getattr(request.state, "request_id", None),
        )
        return error_response(
            500,
            "INTERNAL_ERROR",
            "Internal server error",
            request_id=getattr(request.state, "request_id", None),
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )
