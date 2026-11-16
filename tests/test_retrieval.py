"""Tests for the retrieval module (dense, sparse BM25, hybrid RRF).

Uses an in-memory Qdrant (`VectorStore(url=":memory:")`) and the deterministic
`FakeEmbedder` so no network or API keys are required.
"""

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pytest

from src.ingestion.models import Chunk, DocumentStatus, EmbeddedChunk
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.vectorstore import VectorStore
from src.retrieval import MetadataFilter, Retriever, SearchType
from tests.factories import FakeEmbedder


def _upsert_chunks(
    store: VectorStore,
    embedder: FakeEmbedder,
    entries: list[tuple[str, str, str, str | None]],
    tenant_id: str = "t1",
) -> None:
    """Upsert chunks with controlled content.

    Each entry is (content, document_id, source, section_header).
    `document_id` is hashed to a deterministic UUID.
    """
    chunks: list[Chunk] = []
    for i, (content, document_id, source, section) in enumerate(entries):
        chunks.append(
            Chunk(
                id=uuid5(NAMESPACE_URL, f"{document_id}-{i}"),
                document_id=uuid5(NAMESPACE_URL, document_id),
                content=content,
                chunk_index=i,
                chunk_count=len(entries),
                section_header=section,
                metadata={"source": source},
            )
        )
    for chunk in chunks:
        chunk.embedding = embedder.embed_texts([chunk.content])[0]
        chunk.embedding_model = embedder.model_name
    embedded = [EmbeddedChunk(**chunk.model_dump()) for chunk in chunks]

    store.ensure_collection(dim=embedder.dim)
    store.upsert_chunks(embedded, tenant_id=tenant_id)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=64)


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(url=":memory:")


@pytest.fixture
def retriever(store: VectorStore, embedder: FakeEmbedder) -> Retriever:
    return Retriever(vectorstore=store, embedder=embedder)


@pytest.fixture
def sample_chunks(store: VectorStore, embedder: FakeEmbedder):
    """Three chunks with distinct content across two documents."""
    entries = [
        ("needle in a haystack", "doc-a", "a.txt", "Intro"),
        ("needle beta", "doc-b", "b.txt", "Cooking"),
        ("unrelated gamma delta", "doc-c", "c.txt", "Advanced"),
    ]
    _upsert_chunks(store, embedder, entries, tenant_id="t1")
    return entries


class TestDenseSearch:
    def test_returns_most_similar_first(self, retriever, sample_chunks):
        results = retriever.dense_search("needle in a haystack", top_k=3)

        assert len(results) == 3
        # Identical text -> identical vector -> cosine score of 1.0.
        assert results[0].content == "needle in a haystack"
        assert results[0].score == pytest.approx(1.0, abs=1e-6)
        assert results[0].document_id is not None

    def test_top_k_limits_results(self, retriever, sample_chunks):
        results = retriever.dense_search("needle", top_k=1)
        assert len(results) == 1

    def test_score_threshold_filters(self, retriever, sample_chunks):
        # No cosine score can exceed 1.0, so a threshold above that yields nothing.
        assert retriever.dense_search("needle", top_k=10, score_threshold=2.0) == []
        # Threshold of 0.0 returns everything (non-negative cosine scores).
        results = retriever.dense_search("needle", top_k=10, score_threshold=0.0)
        assert len(results) == 3

    def test_empty_store_creates_collection(self, store, embedder):
        # Collection does not exist yet; Retriever creates it lazily.
        retriever = Retriever(vectorstore=store, embedder=embedder)
        assert retriever.search("anything", top_k=5) == []


class TestSparseSearch:
    def test_bm25_matches_keyword(self, retriever, sample_chunks):
        results = retriever.sparse_search("beta", top_k=5, score_threshold=0.5)

        assert len(results) >= 1
        assert results[0].content == "needle beta"
        assert results[0].score > 0

    def test_no_terms_returns_empty(self, retriever):
        assert retriever.sparse_search("", top_k=5) == []


class TestHybridSearch:
    def test_rrf_fusion_ranks_needle_docs_first(self, retriever, sample_chunks):
        # The two "needle" docs match on both branches; the unrelated doc matches
        # neither well, so it ranks last.
        results = retriever.hybrid_search("needle", top_k=3)

        assert len(results) == 3
        assert all("needle" in r.content for r in results[:2])
        assert results[-1].content == "unrelated gamma delta"
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, retriever, sample_chunks):
        results = retriever.hybrid_search("needle", top_k=1)
        assert len(results) == 1

    def test_hybrid_ignores_score_threshold(self, retriever, sample_chunks):
        # RRF scores are not comparable to raw cosine/BM25 scores, so the
        # threshold is not applied to hybrid search.
        results = retriever.hybrid_search("needle", top_k=3, score_threshold=2.0)
        assert len(results) == 3

    def test_repeatable(self, retriever, sample_chunks):
        r1 = retriever.hybrid_search("needle", top_k=3)
        r2 = retriever.hybrid_search("needle", top_k=3)
        assert [r.id for r in r1] == [r.id for r in r2]


class TestMetadataFilter:
    def test_tenant_and_document(self, retriever):
        f = MetadataFilter(tenant_id="t1", document_id="doc-a")
        qf = retriever._build_filter(f)
        assert qf is not None
        conds = {c.key: c for c in qf.must}
        assert conds["tenant_id"].match.value == "t1"
        assert conds["document_id"].match.value == "doc-a"

    def test_tags_and_source(self, retriever):
        f = MetadataFilter(source="a.txt", tags=["Intro", "Advanced"])
        qf = retriever._build_filter(f)
        conds = {c.key: c for c in qf.must}
        assert conds["source"].match.value == "a.txt"
        assert conds["tags"].match.any == ["Intro", "Advanced"]

    def test_date_range(self, retriever):
        after = datetime(2026, 1, 1, tzinfo=UTC)
        before = datetime(2026, 2, 1, tzinfo=UTC)
        f = MetadataFilter(created_after=after, created_before=before)
        qf = retriever._build_filter(f)
        conds = {c.key: c for c in qf.must}
        rng = conds["created_at"].range
        assert rng.gte == after
        assert rng.lt == before

    def test_none_returns_none(self, retriever):
        assert retriever._build_filter(None) is None
        assert retriever._build_filter(MetadataFilter()) is None

    def test_search_type_coercion(self, retriever):
        assert isinstance(retriever.search("x", search_type="dense", top_k=1), list)
        assert isinstance(retriever.search("x", search_type=SearchType.SPARSE, top_k=1), list)


class TestSearchResultShape:
    def test_result_fields_populated(self, retriever, sample_chunks):
        results = retriever.dense_search("needle in a haystack", top_k=1)
        r = results[0]
        assert r.id
        assert r.content == "needle in a haystack"
        assert r.score == pytest.approx(1.0, abs=1e-6)
        assert r.payload.get("source") == "a.txt"
        assert r.payload.get("section_header") == "Intro"


class TestEndToEndViaPipeline:
    def test_pipeline_then_retrieve(self, tmp_path, store, embedder):
        """Ingest through the pipeline, then retrieve through the retriever."""
        doc = tmp_path / "needle.txt"
        doc.write_text("needle in a haystack", encoding="utf-8")

        pipeline = IngestionPipeline(
            embedder=embedder,
            vectorstore=store,  # type: ignore[arg-type]
        )
        result = pipeline.ingest_file(str(doc), tenant_id="t1")
        assert result.status == DocumentStatus.COMPLETED
        assert result.chunks_created > 0

        retriever = Retriever(vectorstore=store, embedder=embedder)
        results = retriever.dense_search("needle in a haystack", top_k=3)
        assert len(results) >= 1
        assert results[0].content == "needle in a haystack"
        assert results[0].payload.get("source") == str(doc)
