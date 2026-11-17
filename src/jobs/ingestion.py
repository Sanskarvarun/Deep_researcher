"""Celery tasks for document ingestion."""

from __future__ import annotations

from celery import Task  # type: ignore[import-untyped]

from src.core.logging import get_logger
from src.ingestion.models import DocumentStatus
from src.ingestion.pipeline import IngestionPipeline
from src.jobs import celery_app

log = get_logger(__name__)


class IngestionTask(Task):  # type: ignore[misc]
    """Base task class with pipeline initialization."""

    _pipeline: IngestionPipeline | None = None

    @property
    def pipeline(self) -> IngestionPipeline:
        if self._pipeline is None:
            self._pipeline = IngestionPipeline()
        return self._pipeline


@celery_app.task(  # type: ignore[misc]
    bind=True,
    base=IngestionTask,
    name="src.jobs.ingestion.ingest_file_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
)
def ingest_file_task(
    self: IngestionTask,
    source: str,
    tenant_id: str | None = None,
    incremental: bool = False,
) -> dict:
    """Celery task to ingest a single file.

    Args:
        source: File path or URL to ingest.
        tenant_id: Tenant ID for multi-tenancy.
        incremental: If True, only ingest if file has changed.

    Returns:
        Dict with ingestion result status and metadata.
    """
    log.info("ingest_task_started", source=source, tenant_id=tenant_id, incremental=incremental)

    try:
        if incremental:
            result = self.pipeline.incremental_ingest(source, tenant_id)
        else:
            result = self.pipeline.ingest_file(source, tenant_id)

        return {
            "status": result.status.value,
            "document_id": str(result.document_id),
            "source": result.source,
            "chunks_created": result.chunks_created,
            "chunks_skipped": result.chunks_skipped,
            "error": result.error,
            "duration_ms": result.duration_ms,
            "metadata": result.metadata,
        }

    except Exception as e:
        log.error("ingest_task_failed", source=source, error=str(e))
        raise


@celery_app.task(
    bind=True,
    base=IngestionTask,
    name="src.jobs.ingestion.ingest_batch_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=2,
)  # type: ignore[misc]
def ingest_batch_task(
    self: IngestionTask,
    sources: list[str],
    tenant_id: str | None = None,
    incremental: bool = False,
) -> dict:
    """Celery task to ingest multiple files in batch.

    Args:
        sources: List of file paths or URLs to ingest.
        tenant_id: Tenant ID for multi-tenancy.
        incremental: If True, only ingest files that have changed.

    Returns:
        Dict with batch results summary.
    """
    log.info(
        "batch_ingest_started", count=len(sources), tenant_id=tenant_id, incremental=incremental
    )

    results = []
    completed = 0
    failed = 0

    for source in sources:
        try:
            if incremental:
                result = self.pipeline.incremental_ingest(source, tenant_id)
            else:
                result = self.pipeline.ingest_file(source, tenant_id)

            results.append({
                "source": source,
                "status": result.status.value,
                "document_id": str(result.document_id),
                "chunks_created": result.chunks_created,
                "chunks_skipped": result.chunks_skipped,
                "error": result.error,
            })

            if result.status == DocumentStatus.COMPLETED:
                completed += 1
            else:
                failed += 1

        except Exception as e:
            log.error("batch_item_failed", source=source, error=str(e))
            results.append({
                "source": source,
                "status": DocumentStatus.FAILED.value,
                "error": str(e),
            })
            failed += 1

    log.info("batch_ingest_completed", completed=completed, failed=failed, total=len(sources))

    return {
        "total": len(sources),
        "completed": completed,
        "failed": failed,
        "results": results,
    }


@celery_app.task(  # type: ignore[misc]
    bind=True,
    base=IngestionTask,
    name="src.jobs.ingestion.retry_dead_letter_task",
    max_retries=1,
)
def retry_dead_letter_task(
    self: IngestionTask,
    document_id: str,
    tenant_id: str | None = None,
) -> dict:
    """Retry a failed ingestion from the dead letter queue.

    Args:
        document_id: UUID of the document to retry.
        tenant_id: Tenant ID for multi-tenancy.

    Returns:
        Dict with retry result.
    """
    from src.ingestion.models import UUID, dead_letter_queue

    entry = dead_letter_queue.get(UUID(document_id))

    if not entry:
        return {"status": "not_found", "document_id": document_id}

    log.info("dead_letter_retry_started", document_id=document_id)

    try:
        # Retry the ingestion
        result = self.pipeline.ingest_file(entry.source, tenant_id or entry.tenant_id)

        if result.status == DocumentStatus.COMPLETED:
            dead_letter_queue.remove(UUID(document_id))
            return {
                "status": "retry_succeeded",
                "document_id": document_id,
                "chunks_created": result.chunks_created,
            }
        else:
            # Re-add to dead letter queue with incremented retry count
            entry.retry_count += 1
            return {
                "status": "retry_failed",
                "document_id": document_id,
                "error": result.error,
                "retry_count": entry.retry_count,
            }

    except Exception as e:
        log.error("dead_letter_retry_failed", document_id=document_id, error=str(e))
        entry.retry_count += 1
        return {
            "status": "retry_error",
            "document_id": document_id,
            "error": str(e),
            "retry_count": entry.retry_count,
        }


def get_task_status(task_id: str) -> dict:
    """Get the status of a Celery task.

    Args:
        task_id: Celery task ID.

    Returns:
        Dict with task status information.
    """
    result = celery_app.AsyncResult(task_id)

    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
        "traceback": result.traceback if result.failed() else None,
    }


def revoke_task(task_id: str, terminate: bool = True) -> dict:
    """Revoke a running Celery task.

    Args:
        task_id: Celery task ID.
        terminate: Whether to terminate the task process.

    Returns:
        Dict with revocation status.
    """
    celery_app.control.revoke(task_id, terminate=terminate)
    return {"task_id": task_id, "revoked": True}
