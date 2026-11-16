"""Qdrant vector store: collection setup, upsert with dedup, health checks.

Qdrant gives us native hybrid search (dense vectors + BM25 sparse vectors),
payload filtering (tenant isolation), and scalar quantization (int8) to keep
memory usage reasonable.
"""

from datetime import datetime
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.core import get_logger
from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.ingestion.metadata import compute_content_hash
from src.ingestion.models import EmbeddedChunk, QdrantPoint

log = get_logger(__name__)


class VectorStore:
    """Qdrant wrapper: setup, dedup-check, upsert."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection

        client_kwargs: dict = {"timeout": 30}
        if self.url == ":memory:":
            # In-memory mode — used by tests and smoke runs (no server needed)
            self.client = QdrantClient(location=":memory:", **client_kwargs)
        elif self.url.startswith(("http://", "https://")) or self.url.startswith("localhost"):
            self.client = QdrantClient(
                url=self.url,
                api_key=api_key or settings.qdrant_api_key,
                prefer_grpc=settings.qdrant_prefer_grpc,
                **client_kwargs,
            )
        else:
            # Treat as a local file path (embedded storage, persists to disk)
            self.client = QdrantClient(path=self.url, **client_kwargs)

        log.info("vectorstore_initialized", url=self.url, collection=self.collection)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        """Create the collection if missing (idempotent).

        Configures:
        - dense vectors (cosine, on-disk for large collections)
        - sparse BM25 vectors for hybrid search
        - int8 scalar quantization (memory savings, ~4x reduction)
        """
        exists = self.client.collection_exists(self.collection)

        if exists and not recreate:
            log.info("collection_exists", collection=self.collection)
            return

        if exists and recreate:
            log.warning("recreating_collection", collection=self.collection)
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                # Named vector "dense" — matches the upsert key
                "dense": qmodels.VectorParams(
                    size=dim,
                    distance=qmodels.Distance.COSINE,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                "bm25": qmodels.SparseVectorParams(index=qmodels.SparseIndexParams(on_disk=True))
            },
            quantization_config=qmodels.ScalarQuantization(
                scalar=qmodels.ScalarQuantizationConfig(
                    type=qmodels.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )

        # Payload indexes for fast filtering
        for field in ("tenant_id", "document_id", "source", "created_at", "source_type", "tags"):
            schema = (
                qmodels.PayloadSchemaType.DATETIME
                if field == "created_at"
                else qmodels.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=schema,
            )

        log.info("collection_created", collection=self.collection, dim=dim)

    def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            log.error("qdrant_health_check_failed", error=str(e))
            return False

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    def content_hash_exists(self, content_hash: str, tenant_id: str | None = None) -> bool:
        """Check if a content hash already exists in the store."""
        conditions: list[qmodels.Condition] = [
            qmodels.FieldCondition(key="content_hash", match=qmodels.MatchValue(value=content_hash))
        ]
        if tenant_id:
            conditions.append(
                qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=tenant_id))
            )

        results = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qmodels.Filter(must=conditions),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results[0]) > 0

    def document_hash_exists(self, document_hash: str, tenant_id: str | None = None) -> bool:
        """Check if a document (by content hash) is already ingested.

        Every chunk of a document carries its parent's hash, so one lookup
        at document level avoids re-embedding on re-upload.
        """
        conditions: list[qmodels.Condition] = [
            qmodels.FieldCondition(
                key="document_hash", match=qmodels.MatchValue(value=document_hash)
            )
        ]
        if tenant_id:
            conditions.append(
                qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=tenant_id))
            )

        results = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qmodels.Filter(must=conditions),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results[0]) > 0

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[EmbeddedChunk],
        tenant_id: str | None = None,
        document_hash: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Upsert embedded chunks into Qdrant.

        Deduplicates by content hash: chunks whose hash already exists
        (for this tenant) are skipped.

        Args:
            chunks: Embedded chunks to upsert.
            tenant_id: Tenant for isolation + dedup scope.
            document_hash: Parent document hash, stored on every chunk for
                document-level dedup on future uploads.
            batch_size: Points per upsert call.

        Returns:
            Number of chunks actually upserted (dedup-skips excluded).
        """
        if not chunks:
            return 0

        points: list[QdrantPoint] = []
        skipped = 0

        for chunk in chunks:
            if chunk.embedding is None:
                raise RetrievalError(f"Chunk {chunk.id} has no embedding")

            content_hash = compute_content_hash(chunk.content)

            if self.content_hash_exists(content_hash, tenant_id):
                skipped += 1
                log.debug("chunk_dedup_skipped", chunk_id=str(chunk.id), hash=content_hash[:12])
                continue

            payload = {
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "document_hash": document_hash,
                "content": chunk.content,
                "content_hash": content_hash,
                "chunk_index": chunk.chunk_index,
                "chunk_count": chunk.chunk_count,
                "section_header": chunk.section_header,
                "page_number": chunk.page_number,
                "embedding_model": chunk.embedding_model,
                "created_at": _now().isoformat(),
                "tags": [],
                **(chunk.metadata or {}),
            }
            if tenant_id:
                payload["tenant_id"] = tenant_id

            points.append(
                QdrantPoint(
                    id=str(chunk.id),
                    dense_vector=chunk.embedding,
                    sparse_vector=sparse_vector_for_text(chunk.content),
                    payload=payload,
                )
            )

        upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=qmodels.Batch(
                    ids=[p.id for p in batch],
                    vectors={
                        "dense": [p.dense_vector for p in batch],
                        "bm25": [
                            p.sparse_vector or qmodels.SparseVector(indices=[], values=[])
                            for p in batch
                        ],
                    },
                    payloads=[p.payload for p in batch],
                ),
            )
            upserted += len(batch)

        log.info(
            "chunks_upserted",
            upserted=upserted,
            skipped_dedup=skipped,
            collection=self.collection,
        )
        return upserted

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total points in the collection."""
        return self.client.count(collection_name=self.collection).count

    def delete_document(self, document_id: UUID, tenant_id: str | None = None) -> int:
        """Delete all chunks belonging to a document."""
        conditions: list[qmodels.Condition] = [
            qmodels.FieldCondition(
                key="document_id", match=qmodels.MatchValue(value=str(document_id))
            )
        ]
        if tenant_id:
            conditions.append(
                qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=tenant_id))
            )

        self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=conditions)),
        )
        log.info("document_deleted", document_id=str(document_id))
        return 1  # Qdrant delete is async; status not per-point


def _now() -> datetime:
    """UTC now."""
    from datetime import UTC, datetime

    return datetime.now(UTC)


def sparse_vector_for_text(text: str, vocab_size: int = 100_000) -> qmodels.SparseVector | None:
    """Convert text into a BM25-style sparse vector.

    Whitespace tokenizer with a stable MD5 hash -> index mapping; values are
    raw term frequencies. The same scheme is used at query time
    (see `src/retrieval/retriever.py`) so the dot product approximates
    term-overlap relevance.
    """
    from hashlib import md5

    freq: dict[int, float] = {}
    for raw in text.lower().split():
        term = raw.strip(".,!?;:\"'()[]{}<>/\\-—_")
        if not term:
            continue
        idx = int.from_bytes(md5(term.encode("utf-8")).digest()[:4], "little") % vocab_size
        freq[idx] = freq.get(idx, 0.0) + 1.0
    if not freq:
        return None
    items = sorted(freq.items())
    return qmodels.SparseVector(indices=[i for i, _ in items], values=[v for _, v in items])
