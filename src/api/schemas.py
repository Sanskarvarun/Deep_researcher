"""Pydantic request/response schemas for the HTTP API.

All client input is validated here (types, ranges, lengths, path safety) so
handlers can assume well-formed data and errors come back as 422 with a stable
envelope instead of raising deep inside the pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.config import settings
from src.core.exceptions import ValidationError as AppValidationError
from src.core.security import resolve_safe_path
from src.retrieval.models import MetadataFilter, SearchType

_TENANT_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"


class AskRequest(BaseModel):
    """Question for the RAG pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question to answer from the corpus.",
    )
    top_k: int = Field(default=5, ge=1, le=50)
    search_type: SearchType = SearchType.HYBRID
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    tenant_id: str = Field(default="default", pattern=_TENANT_PATTERN)
    filters: MetadataFilter | None = None

    @field_validator("question")
    @classmethod
    def _reject_control_characters(cls, value: str) -> str:
        if any(ord(ch) < 32 and ch not in "\n\t\r" for ch in value):
            raise ValueError("question must not contain control characters")
        return value


class CitationOut(BaseModel):
    """A citation attached to an answer."""

    model_config = ConfigDict(from_attributes=True)

    key: str
    chunk_id: str
    document_id: str | None = None
    content: str
    source: str | None = None
    section_header: str | None = None
    page_number: int | None = None
    score: float | None = None


class UsageOut(BaseModel):
    """Token usage for a generation call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    """Answer plus citations and provenance."""

    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    sources_markdown: str = ""
    provider: str
    model: str
    usage: UsageOut
    cost_usd: float = 0.0
    latency_ms: float
    request_id: str


class IngestRequest(BaseModel):
    """Ingest a single local file into the vector store."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(min_length=1, max_length=1024)
    tenant_id: str = Field(default="default", pattern=_TENANT_PATTERN)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        try:
            resolve_safe_path(
                value,
                allowed_extensions=settings.supported_extensions,
                base_dir=settings.ingest_root or None,
                max_size_mb=settings.max_file_size_mb,
            )
        except AppValidationError as exc:
            # Pydantic validators must raise ValueError to produce a 422.
            raise ValueError(exc.message) from exc
        return value


class IngestResponse(BaseModel):
    """Result of an ingestion request."""

    document_id: str
    chunks: int
    tenant_id: str


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str = "ok"
    version: str
    environment: str
    tracing: bool


class ErrorBody(BaseModel):
    """Stable error envelope returned by every failing endpoint."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Top-level error wrapper."""

    error: ErrorBody
