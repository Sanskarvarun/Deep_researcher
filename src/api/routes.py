"""HTTP routes for the RAG pipeline."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from src.api.auth import require_api_key
from src.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    HealthResponse,
    IngestBatchRequest,
    IngestBatchResponse,
    IngestBatchResult,
    IngestRequest,
    IngestResponse,
    TaskStatusResponse,
    UsageOut,
)
from src.core import DeepResearcherError, get_logger
from src.core.config import settings
from src.generation.chain import GenerationResult, RAGGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.jobs.ingestion import (
    celery_app,
    ingest_batch_task,
    ingest_file_task,
    retry_dead_letter_task,
)
from src.jobs.ingestion import get_task_status as _get_task_status
from src.jobs.ingestion import revoke_task as _revoke_task
from src.retrieval.models import MetadataFilter
from src.retrieval.retriever import Retriever

log = get_logger(__name__)
router = APIRouter()

API_VERSION = "0.1.0"


# WebSocket connection manager for progress tracking
class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, task_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[task_id] = websocket
        log.info("ws_connected", task_id=task_id, total=len(self.active_connections))

    def disconnect(self, task_id: str) -> None:
        if task_id in self.active_connections:
            del self.active_connections[task_id]
            log.info("ws_disconnected", task_id=task_id, total=len(self.active_connections))

    async def send_progress(self, task_id: str, data: dict) -> bool:
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_json(data)
                return True
            except Exception as e:
                log.warning("ws_send_failed", task_id=task_id, error=str(e))
                self.disconnect(task_id)
        return False


manager = ConnectionManager()


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide retriever (shares embedding client and cache)."""
    return Retriever()


@lru_cache(maxsize=1)
def get_generator() -> RAGGenerator:
    """Process-wide generator (shares LLM clients and cache)."""
    return RAGGenerator()


@lru_cache(maxsize=1)
def get_pipeline() -> IngestionPipeline:
    """Process-wide ingestion pipeline."""
    return IngestionPipeline()


def _ask_sync(payload: AskRequest) -> tuple[GenerationResult, float]:
    started = perf_counter()
    filters = (payload.filters or MetadataFilter()).model_copy(
        update={"tenant_id": payload.tenant_id}
    )
    results = get_retriever().search(
        payload.question,
        search_type=payload.search_type,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        filter=filters,
    )
    result = get_generator().generate(payload.question, results)
    return result, (perf_counter() - started) * 1000.0


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness probe (unauthenticated, rate-limit exempt)."""
    return HealthResponse(
        version=API_VERSION,
        environment=settings.app_env,
        tracing=settings.enable_tracing,
    )


@router.get("/debug/retrieve", tags=["debug"], dependencies=[Depends(require_api_key)])
async def debug_retrieve(q: str = "DeepResearcher", top_k: int = 5) -> dict:
    """Debug endpoint: show raw retrieval results."""
    from src.retrieval.retriever import Retriever

    r = Retriever()
    results = r.search(q, top_k=top_k)
    return {
        "query": q,
        "top_k": top_k,
        "results": [
            {
                "chunk_index": res.chunk_index,
                "document_id": str(res.document_id) if res.document_id else None,
                "score": res.score,
                "content_preview": res.content[:200],
            }
            for res in results
        ],
    }


@router.post(
    "/api/v1/ask",
    response_model=AskResponse,
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Retrieve relevant chunks and generate a cited answer."""
    result, latency_ms = await run_in_threadpool(_ask_sync, payload)
    return AskResponse(
        answer=result.answer,
        citations=[CitationOut.model_validate(citation) for citation in result.citations],
        sources_markdown=result.sources_markdown,
        provider=result.provider,
        model=result.model,
        usage=UsageOut(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        cost_usd=result.cost_usd,
        latency_ms=round(latency_ms, 1),
        request_id=getattr(request.state, "request_id", ""),
    )


def _ingest_sync(payload: IngestRequest) -> IngestResponse:
    result = get_pipeline().ingest_file(payload.path, tenant_id=payload.tenant_id)
    if result.error:
        raise DeepResearcherError(
            result.error,
            code="INGESTION_FAILED",
            details={"source": result.source},
            status_code=422,
        )
    return IngestResponse(
        document_id=str(result.document_id),
        chunks=result.chunks_created,
        tenant_id=payload.tenant_id,
    )


@router.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
async def ingest(payload: IngestRequest) -> IngestResponse:
    """Ingest a local file. Path is validated against traversal/size/type."""
    return await run_in_threadpool(_ingest_sync, payload)


# ---------------------------------------------------------------------------
# Async ingestion endpoints (Celery + WebSocket progress)
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/ingest/async",
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
async def ingest_async(payload: IngestRequest) -> dict:
    """Start async ingestion of a single file.

    Returns immediately with a task ID. Use the WebSocket endpoint
    /ws/ingest/{task_id} to track progress, or poll /api/v1/tasks/{task_id}.
    """
    task = ingest_file_task.delay(
        payload.path,
        tenant_id=payload.tenant_id,
        incremental=False,
    )
    return {"task_id": task.id, "status": "pending"}


@router.post(
    "/api/v1/ingest/async/incremental",
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
async def ingest_async_incremental(payload: IngestRequest) -> dict:
    """Start async incremental ingestion of a single file.

    Only processes the file if it has changed since last ingestion.
    """
    task = ingest_file_task.delay(
        payload.path,
        tenant_id=payload.tenant_id,
        incremental=True,
    )
    return {"task_id": task.id, "status": "pending"}


@router.post(
    "/api/v1/ingest/batch",
    response_model=IngestBatchResponse,
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
async def ingest_batch(payload: IngestBatchRequest) -> IngestBatchResponse:
    """Ingest multiple files synchronously."""
    results = []
    completed = 0
    failed = 0

    for path in payload.paths:
        result = get_pipeline().ingest_file(path, tenant_id=payload.tenant_id)
        results.append(
            IngestBatchResult(
                source=path,
                status=result.status.value,
                document_id=str(result.document_id),
                chunks_created=result.chunks_created,
                chunks_skipped=result.chunks_skipped,
                error=result.error,
            )
        )
        if result.status.value == "completed":
            completed += 1
        else:
            failed += 1

    return IngestBatchResponse(
        total=len(payload.paths),
        completed=completed,
        failed=failed,
        results=results,
    )


@router.post(
    "/api/v1/ingest/batch/async",
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
async def ingest_batch_async(payload: IngestBatchRequest) -> dict:
    """Start async batch ingestion of multiple files."""
    task = ingest_batch_task.delay(
        payload.paths,
        tenant_id=payload.tenant_id,
        incremental=payload.incremental,
    )
    return {"task_id": task.id, "status": "pending"}


@router.get(
    "/api/v1/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["tasks"],
    dependencies=[Depends(require_api_key)],
)
async def get_task(task_id: str) -> TaskStatusResponse:
    """Get the status of an async ingestion task."""

    status = _get_task_status(task_id)
    return TaskStatusResponse(**status)


@router.post(
    "/api/v1/tasks/{task_id}/revoke",
    tags=["tasks"],
    dependencies=[Depends(require_api_key)],
)
async def revoke_task_endpoint(task_id: str) -> dict:
    """Revoke a running task."""

    return _revoke_task(task_id)


@router.post(
    "/api/v1/dead-letter/{document_id}/retry",
    tags=["tasks"],
    dependencies=[Depends(require_api_key)],
)
async def retry_dead_letter(document_id: str, tenant_id: str | None = None) -> dict:
    """Retry a failed ingestion from the dead letter queue."""
    task = retry_dead_letter_task.delay(document_id, tenant_id)
    return {"task_id": task.id, "status": "pending"}


# ---------------------------------------------------------------------------
# WebSocket endpoint for real-time progress tracking
# ---------------------------------------------------------------------------


@router.websocket("/ws/ingest/{task_id}")
async def ws_ingest_progress(websocket: WebSocket, task_id: str) -> None:
    """WebSocket endpoint for real-time ingestion progress updates.

    The client connects to this endpoint and receives JSON messages with
    progress updates for the specified task.

    Message format:
    {
        "task_id": "uuid",
        "status": "pending|started|progress|success|failure|revoked",
        "progress": 0.0-1.0,
        "stage": "parsing|chunking|embedding|upserting|completed",
        "message": "Human-readable status message",
        "result": {...}  # Only present on completion
    }
    """
    from celery.result import AsyncResult  # type: ignore[import-untyped]

    await manager.connect(task_id, websocket)

    try:
        # Send initial status
        result = AsyncResult(task_id, app=celery_app)
        await websocket.send_json({
            "task_id": task_id,
            "status": result.status,
            "progress": 0.0,
            "stage": "pending",
            "message": "Task queued",
        })

        # Poll for updates
        last_status = None
        while True:
            await websocket.receive_text()  # Keep connection alive / wait for client ping

            result = AsyncResult(task_id, app=celery_app)
            current_status = result.status

            if current_status != last_status:
                # Status changed, send update
                progress_map = {
                    "PENDING": 0.0,
                    "STARTED": 0.1,
                    "PROGRESS": 0.5,
                    "SUCCESS": 1.0,
                    "FAILURE": 1.0,
                    "REVOKED": 1.0,
                }
                stage_map = {
                    "PENDING": "queued",
                    "STARTED": "started",
                    "PROGRESS": "processing",
                    "SUCCESS": "completed",
                    "FAILURE": "failed",
                    "REVOKED": "revoked",
                }

                progress = progress_map.get(current_status, 0.0)
                stage = stage_map.get(current_status, "unknown")

                message = {
                    "task_id": task_id,
                    "status": current_status.lower(),
                    "progress": progress,
                    "stage": stage,
                    "message": f"Task {current_status.lower()}",
                }

                if current_status == "SUCCESS" and result.result:
                    message["result"] = result.result

                await websocket.send_json(message)

                if current_status in ("SUCCESS", "FAILURE", "REVOKED"):
                    break

                last_status = current_status

    except WebSocketDisconnect:
        manager.disconnect(task_id)
    except Exception as e:
        log.error("ws_error", task_id=task_id, error=str(e))
        manager.disconnect(task_id)
