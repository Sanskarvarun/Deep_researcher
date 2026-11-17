"""Ingestion pipeline: orchestrates load → chunk → embed → upsert.

This is the single entry point for ingesting a document. It wires together
loaders, chunkers, the embedder, and the vector store, with status tracking
and error handling at every stage.

Usage:
    pipeline = IngestionPipeline()
    result = pipeline.ingest_file("data/docs/paper.pdf")
"""

import pathlib
import time
from dataclasses import dataclass
from uuid import UUID

from src.core import get_logger
from src.core.config import settings
from src.core.exceptions import DocumentProcessingError
from src.ingestion.chunkers import ChunkConfig, ChunkingStrategy, chunk_document
from src.ingestion.embedder import EmbeddingModel, create_embedder, embed_chunks
from src.ingestion.loaders import load_document
from src.ingestion.metadata import validate_source
from src.ingestion.models import Document, DocumentStatus, IngestionResult
from src.ingestion.vectorstore import VectorStore

log = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Aggregate ingestion configuration."""

    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk: ChunkConfig | None = None  # defaults from settings if None

    @classmethod
    def from_settings(cls) -> "PipelineConfig":
        return cls(
            strategy=ChunkingStrategy(settings.chunking_strategy),
            chunk=ChunkConfig(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                min_chunk_size=settings.chunk_min_size,
            ),
        )


class IngestionPipeline:
    """End-to-end document ingestion pipeline."""

    def __init__(
        self,
        embedder: EmbeddingModel | None = None,
        vectorstore: VectorStore | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig.from_settings()
        self._embedder = embedder
        self._vectorstore = vectorstore

    # Lazy properties so only construct what's needed
    @property
    def embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = create_embedder()
        return self._embedder

    @property
    def vectorstore(self) -> VectorStore:
        if self._vectorstore is None:
            self._vectorstore = VectorStore()
        return self._vectorstore

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def ingest_file(self, source: str, tenant_id: str | None = None) -> IngestionResult:
        """Ingest a single document (file path or URL).

        Args:
            source: File path or YouTube URL.
            tenant_id: Tenant for isolation and dedup scoping.

        Returns:
            IngestionResult with status, chunk counts, error (if any).
        """
        start = time.perf_counter()
        document = None  # Initialize to handle exceptions before document creation

        try:
            # 0. Validate
            validate_source(source, max_size_mb=settings.max_file_size_mb)

            # 1. Load
            document = load_document(source)

            # 2. Collection setup (determine dim from embedder)
            self.vectorstore.ensure_collection(dim=self.embedder.dim)

            # 3. Document-level dedup: skip if already ingested
            if self._document_exists(document.content_hash, tenant_id):
                log.info("document_already_ingested", source=source)
                return IngestionResult(
                    document_id=document.id,
                    source=source,
                    status=DocumentStatus.COMPLETED,
                    chunks_created=0,
                    chunks_skipped=-1,  # -1 signals "document-level skip"
                    duration_ms=(time.perf_counter() - start) * 1000,
                    metadata={"reason": "duplicate"},
                )

            # 4. Chunk
            chunks = chunk_document(
                document,
                config=self.config.chunk,
                strategy=self.config.strategy,
                embedder=self.embedder,
            )
            if not chunks:
                raise DocumentProcessingError("No chunks produced", details={"source": source})

            # Record the document source on every chunk so it can be used for
            # metadata filtering at retrieval time.
            for chunk in chunks:
                chunk.metadata.setdefault("source", document.source)

            # 5. Embed
            embedded = embed_chunks(chunks, self.embedder)

            # 6. Upsert (with chunk-level dedup)
            upserted = self.vectorstore.upsert_chunks(
                embedded,
                tenant_id=tenant_id,
                document_hash=document.content_hash,
            )

            duration_ms = (time.perf_counter() - start) * 1000
            log.info(
                "ingestion_complete",
                source=source,
                chunks=len(embedded),
                upserted=upserted,
                duration_ms=round(duration_ms, 1),
            )

            return IngestionResult(
                document_id=document.id,
                source=source,
                status=DocumentStatus.COMPLETED,
                chunks_created=upserted,
                chunks_skipped=len(embedded) - upserted,
                duration_ms=duration_ms,
                metadata={
                    "title": document.title,
                    "source_type": document.source_type.value,
                    "strategy": self.config.strategy.value,
                },
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            log.error("ingestion_failed", source=source, error=str(e))

            # Add to dead letter queue for later retry/inspection
            from src.ingestion.models import DeadLetterEntry, SourceType, dead_letter_queue

            doc_id = document.id if document is not None else UUID(int=0)
            content_hash = document.content_hash if document is not None else ""
            source_type = document.source_type if document is not None else SourceType.UNKNOWN

            dead_letter_queue.add(
                DeadLetterEntry(
                    document_id=doc_id,
                    source=source,
                    source_type=source_type,
                    content_hash=content_hash,
                    error=str(e),
                    error_type=type(e).__name__,
                    stage="unknown",
                    tenant_id=tenant_id,
                    payload={"source": source, "tenant_id": tenant_id},
                )
            )

            return IngestionResult(
                document_id=doc_id,
                source=source,
                status=DocumentStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms,
            )

    def ingest_files(
        self, sources: list[str], tenant_id: str | None = None
    ) -> list[IngestionResult]:
        """Ingest multiple documents; continues on failure.

        Returns:
            One IngestionResult per source (failures included).
        """
        results: list[IngestionResult] = []
        for source in sources:
            results.append(self.ingest_file(source, tenant_id=tenant_id))
        return results

    def ingest_document(self, document: Document, tenant_id: str | None = None) -> IngestionResult:
        """Ingest an already-parsed Document object (no file access)."""
        start = time.perf_counter()

        try:
            self.vectorstore.ensure_collection(dim=self.embedder.dim)

            if self._document_exists(document.content_hash, tenant_id):
                return IngestionResult(
                    document_id=document.id,
                    source=document.source,
                    status=DocumentStatus.COMPLETED,
                    chunks_skipped=-1,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    metadata={"reason": "duplicate"},
                )

            chunks = chunk_document(
                document,
                config=self.config.chunk,
                strategy=self.config.strategy,
                embedder=self.embedder,
            )
            if not chunks:
                raise DocumentProcessingError("No chunks produced")

            # Record the document source on every chunk so it can be used for
            # metadata filtering at retrieval time.
            for chunk in chunks:
                chunk.metadata.setdefault("source", document.source)

            embedded = embed_chunks(chunks, self.embedder)
            upserted = self.vectorstore.upsert_chunks(
                embedded,
                tenant_id=tenant_id,
                document_hash=document.content_hash,
            )

            return IngestionResult(
                document_id=document.id,
                source=document.source,
                status=DocumentStatus.COMPLETED,
                chunks_created=upserted,
                chunks_skipped=len(embedded) - upserted,
                duration_ms=(time.perf_counter() - start) * 1000,
                metadata={"title": document.title},
            )
        except Exception as e:
            log.error("ingestion_failed", source=document.source, error=str(e))
            from src.ingestion.models import DeadLetterEntry, dead_letter_queue

            dead_letter_queue.add(
                DeadLetterEntry(
                    document_id=document.id,
                    source=document.source,
                    source_type=document.source_type,
                    content_hash=document.content_hash,
                    error=str(e),
                    error_type=type(e).__name__,
                    stage="unknown",
                    tenant_id=tenant_id,
                    payload={
                        "source": document.source,
                        "tenant_id": tenant_id,
                        "content": document.content[:100],
                    },
                )
            )
            return IngestionResult(
                document_id=document.id,
                source=document.source,
                status=DocumentStatus.FAILED,
                error=str(e),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _document_exists(self, content_hash: str, tenant_id: str | None) -> bool:
        """Check if a document with this content hash is already ingested.

        Documents are identified by a hash stored on every chunk payload,
        so re-uploads are caught before any embedding calls are made.
        """
        return self.vectorstore.document_hash_exists(content_hash, tenant_id)

    def _get_file_fingerprint(self, path: str) -> tuple[int, int, float] | None:
        """Get file fingerprint: (size_bytes, mtime_ns, content_hash_prefix).

        Returns None if file doesn't exist.
        """
        from pathlib import Path

        try:
            stat = Path(path).stat()
            return (stat.st_size, int(stat.st_mtime_ns), 0)  # hash prefix computed lazily
        except OSError:
            return None

    def needs_update(self, source: str, tenant_id: str | None = None) -> tuple[bool, str]:
        """Check if a document needs re-ingestion.

        Args:
            source: File path or URL.
            tenant_id: Tenant for isolation.

        Returns:
            (needs_update, reason): True if document needs re-ingestion.
        """
        # Only works for local files

        if not pathlib.Path(source).exists():
            return True, "file_not_found"

        # Get current file fingerprint
        current_fp = self._get_file_fingerprint(source)
        if current_fp is None:
            return True, "cannot_stat"

        # Check if document exists with same content hash
        from src.ingestion.loaders import load_document

        document = load_document(source)
        if self._document_exists(document.content_hash, tenant_id):
            return False, "unchanged"

        return True, "changed_or_new"

    def incremental_ingest(self, source: str, tenant_id: str | None = None) -> IngestionResult:
        """Ingest a document only if it has changed since last ingestion.

        Args:
            source: File path or URL.
            tenant_id: Tenant for isolation.

        Returns:
            IngestionResult with status indicating if ingestion happened or was skipped.
        """
        needs_update, reason = self.needs_update(source, tenant_id)
        if not needs_update:
            log.info("incremental_ingest_skipped", source=source, reason=reason)
            # Load document to get its ID for the result
            from src.ingestion.loaders import load_document

            document = load_document(source)
            return IngestionResult(
                document_id=document.id,
                source=source,
                status=DocumentStatus.COMPLETED,
                chunks_created=0,
                chunks_skipped=-1,
                duration_ms=0.0,
                metadata={"reason": f"skipped: {reason}"},
            )

        log.info("incremental_ingest_started", source=source, reason=reason)
        return self.ingest_file(source, tenant_id)
