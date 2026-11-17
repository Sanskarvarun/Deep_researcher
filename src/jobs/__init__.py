"""Celery configuration and tasks for background ingestion jobs.

This module initializes the Celery app and registers signal handlers.
The actual task definitions are in `src.jobs.ingestion` and are imported
at the bottom to register them with the Celery app.

# ruff: noqa: PLC0415, PLC0411, RUF067, RUF068
"""

from __future__ import annotations

from typing import Any

from celery import Celery  # type: ignore[import-untyped]
from celery.signals import task_failure, task_revoked, task_success  # type: ignore[import-untyped]
from kombu import Queue  # type: ignore[import-untyped]

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)  # ruff: ignore[non-empty-init-module]

# Create Celery app
celery_app = Celery("deepresearcher")  # ruff: ignore[non-empty-init-module]

celery_app.conf.update(  # ruff: ignore[non-empty-init-module]
    broker_url=settings.redis_url,
    result_backend=settings.redis_url,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    # Queue configuration
    task_routes={
        "src.jobs.ingestion.*": {"queue": "ingestion"},
        "src.jobs.ingestion.ingest_file_task": {"queue": "ingestion"},
        "src.jobs.ingestion.ingest_batch_task": {"queue": "ingestion"},
    },
    task_create_missing_queues=True,
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    # Result expiration
    result_expires=86400,  # 24 hours
    # Retry policy
    task_autoretry_for=(Exception,),
    task_retry_backoff=True,
    task_retry_backoff_max=600,
    task_retry_jitter=True,
    task_max_retries=3,
)

# Define queues
celery_app.conf.task_queues = (  # ruff: ignore[non-empty-init-module]
    Queue("default", routing_key="default"),
    Queue("ingestion", routing_key="ingestion"),
)


@task_failure.connect  # ruff: ignore[non-empty-init-module]
def on_task_failure(
    sender: Any | None = None, exception: BaseException | None = None, **kwargs: object
) -> None:
    """Log task failures."""
    log.error(
        "celery_task_failed",
        task=getattr(sender, "name", "unknown") if sender else "unknown",
        error=str(exception),
        task_id=getattr(getattr(sender, "request", None), "id", "unknown"),
    )


@task_success.connect  # ruff: ignore[non-empty-init-module]
def on_task_success(
    sender: Any | None = None, result: object | None = None, **kwargs: object
) -> None:
    """Log task successes."""
    log.info(
        "celery_task_succeeded",
        task=getattr(sender, "name", "unknown") if sender else "unknown",
        task_id=getattr(getattr(sender, "request", None), "id", "unknown"),
    )


@task_revoked.connect  # ruff: ignore[non-empty-init-module]
def on_task_revoked(sender: Any | None = None, **kwargs: object) -> None:
    """Log task revocations."""
    log.warning(
        "celery_task_revoked",
        task=getattr(sender, "name", "unknown") if sender else "unknown",
        task_id=getattr(getattr(sender, "request", None), "id", "unknown"),
    )


# Import tasks to register them
from src.jobs import ingestion  # ruff: ignore[unused-import, module-import-not-at-top-of-file]

__all__ = ["celery_app"]
