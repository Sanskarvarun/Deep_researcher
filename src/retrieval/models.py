"""Retrieval data models.

Defines the request/response shapes for the retrieval layer: search results,
metadata filters, and search-type selection.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SearchType(StrEnum):
    """Search mode: dense, sparse (BM25), or hybrid (RRF fusion)."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class MetadataFilter(BaseModel):
    """Optional metadata constraints applied to every search.

    Maps onto Qdrant payload fields (see `VectorStore.ensure_collection` for
    the indexed payload fields): tenant_id, document_id, source, source_type,
    section_header, tags, created_at.
    """

    tenant_id: str | None = None
    document_id: UUID | str | None = None
    source: str | None = None
    source_type: str | None = None
    section_header: str | None = None
    tags: list[str] | None = None  # match any of these tags
    created_after: datetime | None = None
    created_before: datetime | None = None


class SearchResult(BaseModel):
    """A single retrieval result.

    `score` is the raw relevance score from Qdrant (higher = more relevant).
    For hybrid search this is the Reciprocal Rank Fusion score.
    """

    id: str
    content: str
    score: float
    document_id: UUID | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None
    section_header: str | None = None
    page_number: int | None = None
    source: str | None = None
    source_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    tenant_id: str | None = None
    created_at: datetime | None = None
    embedding_model: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
