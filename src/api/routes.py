"""HTTP routes for the RAG pipeline."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from src.api.auth import require_api_key
from src.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    UsageOut,
)
from src.core import DeepResearcherError, get_logger
from src.core.config import settings
from src.generation.chain import GenerationResult, RAGGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.models import MetadataFilter
from src.retrieval.retriever import Retriever

log = get_logger(__name__)
router = APIRouter()

API_VERSION = "0.1.0"


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
