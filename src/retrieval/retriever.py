"""Basic retrieval: dense, sparse (BM25), and hybrid (RRF).

Builds on the Qdrant vector store (`src/ingestion/vectorstore.py`), which
stores per-chunk dense vectors (cosine similarity) and BM25 sparse vectors,
with payload indexes for metadata filtering and tenant isolation.

The `Retriever` exposes three search modes:

- **dense** — cosine similarity over the query's dense embedding.
- **sparse** — BM25-style keyword overlap via Qdrant sparse vectors.
- **hybrid** — dense + sparse results fused with Reciprocal Rank Fusion (RRF).

Every mode supports `top-k`, `score_threshold`, and metadata filtering
(tenant_id, document_id, source, source_type, section_header, tags, date).

Usage:
    retriever = Retriever()
    results = retriever.search("what is RAG?", top_k=5, score_threshold=0.5)
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from qdrant_client.http import models as qmodels

from src.core import cache_get, cache_set, get_logger, make_cache_key, metrics, trace_retrieve
from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.ingestion.vectorstore import VectorStore, sparse_vector_for_text
from src.retrieval.models import MetadataFilter, SearchResult, SearchType

if TYPE_CHECKING:
    from src.ingestion.embedder import EmbeddingModel

log = get_logger(__name__)

# RRF constant from the original Reciprocal Rank Fusion paper.
RRF_K = 60
# Candidates pulled per branch before fusing in hybrid search.
HYBRID_CANDIDATE_MULTIPLIER = 4


def _parse_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


class Retriever:
    """Query the vector store with dense, sparse, or hybrid search.

    Args:
        vectorstore: Qdrant wrapper (defaults to one built from settings).
        embedder: EmbeddingModel for query embedding (defaults to
            `create_embedder()` from settings).
    """

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        embedder: "EmbeddingModel | None" = None,
    ) -> None:
        self._vectorstore = vectorstore or VectorStore()
        self._embedder: EmbeddingModel | None = embedder

    @property
    def vectorstore(self) -> VectorStore:
        return self._vectorstore

    @property
    def embedder(self) -> "EmbeddingModel":
        embedder = self._embedder
        if embedder is None:
            from src.ingestion.embedder import create_embedder

            embedder = create_embedder()
            self._embedder = embedder
        return embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        search_type: SearchType | str = SearchType.HYBRID,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter: MetadataFilter | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[SearchResult]:
        """Search for chunks relevant to `query`.

        - dense: cosine similarity over the dense embedding.
        - sparse: BM25-style keyword overlap via Qdrant sparse vectors.
        - hybrid: dense + sparse results fused with Reciprocal Rank Fusion.

        Args:
            query: Natural-language query text.
            search_type: "dense", "sparse", or "hybrid" (default).
            top_k: Maximum number of results (defaults to settings.top_k).
            score_threshold: Minimum relevance score (dense/sparse only;
                hybrid uses RRF ranking and ignores this).
            filter: Optional metadata constraints.
            dense_weight / sparse_weight: RRF branch weights (default from
                settings.dense_weight / settings.bm25_weight).

        Returns:
            SearchResults ordered by relevance (highest score first).
        """
        top_k = top_k or settings.top_k
        if top_k <= 0:
            raise RetrievalError("top_k must be positive", details={"top_k": top_k})

        search_type = SearchType(search_type)
        qfilter = self._build_filter(filter)

        # Ensure the collection exists (normally created during ingestion).
        if not self._vectorstore.client.collection_exists(self._vectorstore.collection):
            self._vectorstore.ensure_collection(dim=self.embedder.dim)

        # Cache key based on corpus + query + all search parameters. The
        # collection and embedding model must be part of the key: the same
        # query against a different corpus/embedder is a different result set.
        filter_key = str(filter) if filter else "none"
        cache_key = make_cache_key(
            "ret",
            self._vectorstore.collection,
            getattr(self.embedder, "model_name", type(self.embedder).__name__),
            query,
            search_type.value,
            top_k,
            score_threshold,
            filter_key,
            dense_weight,
            sparse_weight,
        )
        cached = cache_get(cache_key)
        if cached is not None:
            log.debug("retrieve_cache_hit", query=query[:50])
            metrics.increment("retrieve.cache_hits")
            return cached  # type: ignore[no-any-return]

        labels = {"search_type": search_type.value}
        metrics.increment("retrieve.requests", labels=labels)
        with (
            trace_retrieve(
                query=query[:120],
                top_k=top_k,
                search_type=search_type.value,
            ),
            metrics.timer("retrieve.latency", labels=labels),
        ):
            if search_type == SearchType.DENSE:
                results = self._dense_search(query, top_k, score_threshold, qfilter)
            elif search_type == SearchType.SPARSE:
                results = self._sparse_search(query, top_k, score_threshold, qfilter)
            else:
                results = self._hybrid_search(
                    query,
                    top_k,
                    qfilter,
                    dense_weight=dense_weight or settings.dense_weight,
                    sparse_weight=sparse_weight or settings.bm25_weight,
                )

        # Cache results (TTL 1 hour)
        cache_set(cache_key, results, ttl=3600)
        return results

    def dense_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Dense (cosine) vector search."""
        return self.search(
            query,
            search_type=SearchType.DENSE,
            top_k=top_k,
            score_threshold=score_threshold,
            filter=filter,
        )

    def sparse_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Sparse (BM25 keyword) search."""
        return self.search(
            query,
            search_type=SearchType.SPARSE,
            top_k=top_k,
            score_threshold=score_threshold,
            filter=filter,
        )

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter: MetadataFilter | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[SearchResult]:
        """Hybrid search: dense + sparse fused with Reciprocal Rank Fusion."""
        return self.search(
            query,
            search_type=SearchType.HYBRID,
            top_k=top_k,
            score_threshold=score_threshold,
            filter=filter,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    # ------------------------------------------------------------------
    # Internal: single-branch search
    # ------------------------------------------------------------------

    def _dense_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float | None,
        qfilter: qmodels.Filter | None,
    ) -> list[SearchResult]:
        try:
            embedding = self.embedder.embed_texts([query])[0]
            response = self._vectorstore.client.query_points(
                collection_name=self._vectorstore.collection,
                query=embedding,
                using="dense",
                limit=top_k,
                query_filter=qfilter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            metrics.increment("retrieve.errors", labels={"search_type": "dense"})
            raise RetrievalError(f"Dense search failed: {e}", details={"query": query[:80]}) from e
        return self._to_results(response.points)

    def _sparse_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float | None,
        qfilter: qmodels.Filter | None,
    ) -> list[SearchResult]:
        sparse = sparse_vector_for_text(query)
        if sparse is None:
            return []
        try:
            response = self._vectorstore.client.query_points(
                collection_name=self._vectorstore.collection,
                query=qmodels.SparseVector(indices=sparse.indices, values=sparse.values),
                using="bm25",
                limit=top_k,
                query_filter=qfilter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            metrics.increment("retrieve.errors", labels={"search_type": "sparse"})
            raise RetrievalError(f"Sparse search failed: {e}", details={"query": query[:80]}) from e
        return self._to_results(response.points)

    # ------------------------------------------------------------------
    # Internal: hybrid fusion
    # ------------------------------------------------------------------

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        qfilter: qmodels.Filter | None,
        *,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[SearchResult]:
        candidate_limit = max(top_k * HYBRID_CANDIDATE_MULTIPLIER, 10)
        dense = self._dense_search(
            query, top_k=candidate_limit, score_threshold=None, qfilter=qfilter
        )
        sparse = self._sparse_search(
            query, top_k=candidate_limit, score_threshold=None, qfilter=qfilter
        )

        # Reciprocal Rank Fusion: score = sum(weight / (RRF_K + rank)).
        fused: dict[str, list] = {}
        for rank, result in enumerate(dense, start=1):
            entry = fused.setdefault(result.id, [0.0, result])
            entry[0] += dense_weight / (RRF_K + rank)
        for rank, result in enumerate(sparse, start=1):
            entry = fused.setdefault(result.id, [0.0, result])
            entry[0] += sparse_weight / (RRF_K + rank)

        ordered = sorted(fused.values(), key=lambda item: item[0], reverse=True)
        return [result for _, result in ordered[:top_k]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_filter(self, f: MetadataFilter | None) -> qmodels.Filter | None:
        """Convert a MetadataFilter into a Qdrant Filter (payload conditions)."""
        if f is None:
            return None
        conditions: list[qmodels.Condition] = []

        if f.tenant_id:
            conditions.append(
                qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=f.tenant_id))
            )
        if f.document_id is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="document_id", match=qmodels.MatchValue(value=str(f.document_id))
                )
            )
        if f.source:
            conditions.append(
                qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=f.source))
            )
        if f.source_type:
            conditions.append(
                qmodels.FieldCondition(
                    key="source_type", match=qmodels.MatchValue(value=f.source_type)
                )
            )
        if f.section_header:
            conditions.append(
                qmodels.FieldCondition(
                    key="section_header", match=qmodels.MatchValue(value=f.section_header)
                )
            )
        if f.tags:
            conditions.append(
                qmodels.FieldCondition(key="tags", match=qmodels.MatchAny(any=f.tags))
            )
        if f.created_after or f.created_before:
            range_kwargs: dict[str, datetime] = {}
            if f.created_after:
                range_kwargs["gte"] = f.created_after
            if f.created_before:
                range_kwargs["lt"] = f.created_before
            conditions.append(
                qmodels.FieldCondition(
                    key="created_at", range=qmodels.DatetimeRange(**range_kwargs)
                )
            )

        if not conditions:
            return None
        return qmodels.Filter(must=conditions)

    def _to_results(self, points: list[qmodels.ScoredPoint]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for p in points:
            payload = p.payload or {}
            results.append(
                SearchResult(
                    id=str(p.id),
                    content=payload.get("content", "") or "",
                    score=float(p.score),
                    document_id=_parse_uuid(payload.get("document_id")),
                    chunk_index=payload.get("chunk_index"),
                    chunk_count=payload.get("chunk_count"),
                    section_header=payload.get("section_header"),
                    page_number=payload.get("page_number"),
                    source=payload.get("source"),
                    source_type=payload.get("source_type"),
                    tags=list(payload.get("tags") or []),
                    tenant_id=payload.get("tenant_id"),
                    created_at=_parse_datetime(payload.get("created_at")),
                    embedding_model=payload.get("embedding_model"),
                    payload=payload,
                )
            )
        return results
