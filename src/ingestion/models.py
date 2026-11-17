"""Shared data models for the ingestion pipeline."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from qdrant_client.http import models as qmodels


class SourceType(StrEnum):
    """Supported document source types."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"
    YOUTUBE = "youtube"
    UNKNOWN = "unknown"


class DocumentStatus(StrEnum):
    """Document processing status."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    UPSERTING = "upserting"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    """A parsed document before chunking.

    This is the intermediate representation: raw text is extracted
    from the original file, metadata captured, and chunks are not
    yet created.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str
    source: str  # filename, URL, or path
    source_type: SourceType
    title: str | None = None
    author: str | None = None
    file_size_bytes: int | None = None
    page_count: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of normalized content (for deduplication)."""
        from src.ingestion.metadata import compute_content_hash

        return compute_content_hash(self.content)


class Chunk(BaseModel):
    """A chunk of a document ready for embedding.

    `content` is what gets embedded; `context` (if set) is the
    parent/sentence-window text returned to the LLM for generation
    while `content` is used for retrieval quality.
    """

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    context: str | None = None
    chunk_index: int
    chunk_count: int  # total chunks in document (set after chunking)
    section_header: str | None = None
    page_number: int | None = None
    metadata: dict = Field(default_factory=dict)

    # Populated after embedding
    embedding: list[float] | None = None
    embedding_model: str | None = None

    # Populated after upsert (Qdrant point id)
    point_id: str | None = None


class EmbeddedChunk(Chunk):
    """A chunk with its embedding vector attached."""

    embedding: list[float]
    embedding_model: str


class QdrantPoint(BaseModel):
    """A Qdrant point structure (payload + vectors)."""

    id: str
    dense_vector: list[float]
    sparse_vector: qmodels.SparseVector | None = None
    payload: dict


class IngestionResult(BaseModel):
    """Result of ingesting a single document."""

    document_id: UUID
    source: str
    status: DocumentStatus
    chunks_created: int = 0
    chunks_skipped: int = 0  # deduplicated
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict = Field(default_factory=dict)


class DeadLetterEntry(BaseModel):
    """A document that failed ingestion, stored for later retry/inspection."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    source: str
    source_type: SourceType
    content_hash: str
    error: str
    error_type: str  # e.g., "ParsingError", "EmbeddingError", "UpsertError"
    stage: str  # "parsing", "chunking", "embedding", "upserting"
    tenant_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
    last_retry_at: datetime | None = None
    payload: dict = Field(default_factory=dict)  # original document data for retry


class DeadLetterQueue(BaseModel):
    """In-memory dead letter queue (Redis-backed in production)."""

    entries: list[DeadLetterEntry] = Field(default_factory=list)
    max_size: int = 1000

    def add(self, entry: DeadLetterEntry) -> None:
        if len(self.entries) >= self.max_size:
            # Remove oldest entry
            self.entries.pop(0)
        self.entries.append(entry)

    def get(self, doc_id: UUID) -> DeadLetterEntry | None:
        for entry in self.entries:
            if entry.document_id == doc_id:
                return entry
        return None

    def retry(self, doc_id: UUID) -> DeadLetterEntry | None:
        """Mark an entry for retry."""
        entry = self.get(doc_id)
        if entry:
            entry.retry_count += 1
            entry.last_retry_at = datetime.now(UTC)
        return entry

    def remove(self, doc_id: UUID) -> bool:
        """Remove an entry after successful retry."""
        for i, entry in enumerate(self.entries):
            if entry.document_id == doc_id:
                self.entries.pop(i)
                return True
        return False

    def clear(self) -> int:
        count = len(self.entries)
        self.entries.clear()
        return count


# Global dead letter queue instance
dead_letter_queue = DeadLetterQueue()
