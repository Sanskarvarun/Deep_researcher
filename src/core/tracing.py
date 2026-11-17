"""Distributed tracing with OpenTelemetry.

Provides a module-level OpenTelemetry tracer and convenience span helpers
for the ``embed -> retrieve -> rerank -> generate`` pipeline, plus
correlation-ID helpers so every log record carries ``trace_id`` / ``span_id``.

LangSmith is the primary tracing backend: when ``LANGSMITH_TRACING=true`` and
``LANGSMITH_API_KEY`` is set, spans are exported over OTLP to
``LANGSMITH_ENDPOINT/otel/v1/traces``. Otherwise Phoenix (local, free) is used
when a server is reachable at ``PHOENIX_URL``; if neither is available, spans
are emitted to the console so they remain visible during development without
any external service. The OpenTelemetry exporter packages are imported lazily
so the application imports cleanly even when they are not installed.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from functools import wraps
from socket import socket
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import INVALID_SPAN_CONTEXT, format_span_id, format_trace_id

from src.core.config import settings

P = ParamSpec("P")
T = TypeVar("T")

_tracer: trace.Tracer | None = None


def _phoenix_reachable(phoenix_url: str, timeout: float = 1.0) -> bool:
    """Best-effort check that a Phoenix server is listening on the URL's port."""
    from urllib.parse import urlparse

    parsed = urlparse(phoenix_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6006
    with socket() as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def setup_tracing() -> trace.Tracer:
    """Configure the OpenTelemetry tracer provider.

    Returns a tracer bound to ``deepresearcher``. Safe to call multiple
    times -- subsequent calls return the already-configured tracer.
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    provider = TracerProvider()

    if settings.enable_tracing:
        exporter: Any
        if settings.langsmith_tracing and settings.langsmith_api_key:
            # LangSmith OTLP ingestion (primary backend -- no Phoenix needed).
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = settings.langsmith_endpoint.rstrip("/")
            exporter = OTLPSpanExporter(
                endpoint=f"{endpoint}/otel/v1/traces",
                headers={"x-api-key": settings.langsmith_api_key},
            )
        elif _phoenix_reachable(settings.phoenix_url):
            # Phoenix accepts OTLP traces at http://<host>:6006/v1/traces.
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            phoenix_url = settings.phoenix_url.rstrip("/")
            exporter = OTLPSpanExporter(endpoint=f"{phoenix_url}/v1/traces")
        else:
            # No backend reachable -- emit spans to the console so they remain
            # visible during development without any external service.
            exporter = ConsoleSpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("deepresearcher")
    return _tracer


def get_tracer(name: str = "deepresearcher") -> trace.Tracer:
    """Return the tracer, initializing it on first use."""
    global _tracer
    if _tracer is None:
        setup_tracing()
    return trace.get_tracer(name)


def current_span_context() -> dict[str, str]:
    """Return ``trace_id`` / ``span_id`` for the current OpenTelemetry span.

    Returns an empty dict when there is no active span (tracing disabled or
    outside an instrumented region). Used to correlate logs across service
    boundaries.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx == INVALID_SPAN_CONTEXT:
        return {}
    return {
        "trace_id": format_trace_id(ctx.trace_id),
        "span_id": format_span_id(ctx.span_id),
    }


@contextmanager
def trace_span(name: str, **attributes: Any) -> Generator[Any, None, None]:
    """Start a span with the given ``name`` and attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            span.set_attributes(attributes)  # type: ignore[attr-defined]
        yield span


def traced(
    name: str | None = None, **attributes: Any
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that wraps a function call in a traced span."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with trace_span(span_name, **attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Pipeline span helpers (embed -> retrieve -> rerank -> generate)
# ---------------------------------------------------------------------------


def trace_embed(**attributes: Any) -> AbstractContextManager[Any]:
    """Span for the embedding step."""
    return trace_span("embed", **attributes)


def trace_retrieve(**attributes: Any) -> AbstractContextManager[Any]:
    """Span for the retrieval step."""
    return trace_span("retrieve", **attributes)


def trace_rerank(**attributes: Any) -> AbstractContextManager[Any]:
    """Span for the reranking step.

    Wired into the pipeline when a reranker is introduced (Phase 4 /
    agentic retrieval). Available as a standalone helper until then.
    """
    return trace_span("rerank", **attributes)


def trace_generate(**attributes: Any) -> AbstractContextManager[Any]:
    """Span for the generation step."""
    return trace_span("generate", **attributes)
