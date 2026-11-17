"""Structured logging configuration with structlog."""

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from pythonjsonlogger import jsonlogger

from src.core.config import settings
from src.core.tracing import current_span_context


def _trace_correlation_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Inject the current OpenTelemetry ``trace_id`` / ``span_id`` into the
    event dict so logs are correlatable with traces across service
    boundaries (embed -> retrieve -> rerank -> generate)."""
    ctx = current_span_context()
    if ctx:
        event_dict.setdefault("trace_id", ctx["trace_id"])
        event_dict.setdefault("span_id", ctx["span_id"])
    return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _trace_correlation_processor,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            ),
            _json_renderer if settings.app_env == "production" else _console_renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _json_renderer(logger: Any, method_name: str, event_dict: MutableMapping[str, Any]) -> str:
    """JSON renderer for production."""
    return jsonlogger.JsonFormatter().format(
        logging.LogRecord(
            name=event_dict.get("logger", ""),
            level=logging.getLevelName(event_dict.get("level", "INFO")),
            pathname="",
            lineno=0,
            msg=event_dict.get("event", ""),
            args=(),
            exc_info=None,
        )
    )


def _console_renderer(logger: Any, method_name: str, event_dict: MutableMapping[str, Any]) -> str:
    """Pretty console renderer for development."""
    from rich.console import Console

    Console()

    # Extract key fields
    timestamp = event_dict.pop("timestamp", "")
    level = event_dict.pop("level", "info").upper()
    event = event_dict.pop("event", "")
    logger_name = event_dict.pop("logger", "")
    filename = event_dict.pop("filename", "")
    lineno = event_dict.pop("lineno", "")
    func_name = event_dict.pop("func_name", "")

    # Level styling
    level_styles = {
        "DEBUG": "dim",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }
    style = level_styles.get(level, "white")

    # Build output
    parts = [
        f"[dim]{timestamp}[/dim]",
        f"[{style}]{level}[/{style}]",
        f"[cyan]{logger_name}[/cyan]",
        f"[dim]{filename}:{lineno} in {func_name}[/dim]",
        event,
    ]

    # Add extra fields
    if event_dict:
        extra = " ".join(f"{k}={v}" for k, v in event_dict.items())
        parts.append(f"[dim]{extra}[/dim]")

    return " ".join(parts)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
