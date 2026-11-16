"""Integration test for the ingestion pipeline (fake embedder + qdrant client)."""

import pytest

from src.ingestion.chunkers import ChunkConfig, ChunkingStrategy
from src.ingestion.models import Document, DocumentStatus, EmbeddedChunk
from src.ingestion.pipeline import IngestionPipeline, PipelineConfig
from tests.factories import FakeEmbedder


class FakeVectorStore:
    """In-memory stand-in for Qdrant: tracks ensures, upserts, dedup."""

    def __init__(self) -> None:
        self.stored: list[EmbeddedChunk] = []
        self.ensure_calls: list[int] = []
        self.tenant_id_seen: list[str | None] = []
        self.deleted_documents: list[str] = []
        self.document_hashes_seen: list[str | None] = []
        self._existing_doc_hashes: set[str] = set()
        self._existing_chunk_hashes: set[str] = set()

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        self.ensure_calls.append(dim)

    def content_hash_exists(self, content_hash: str, tenant_id: str | None = None) -> bool:
        return content_hash in self._existing_chunk_hashes

    def document_hash_exists(self, document_hash: str, tenant_id: str | None = None) -> bool:
        return document_hash in self._existing_doc_hashes

    def upsert_chunks(
        self, chunks, tenant_id=None, document_hash: str | None = None, batch_size: int = 100
    ) -> int:
        self.tenant_id_seen.append(tenant_id)
        self.document_hashes_seen.append(document_hash)
        self.stored.extend(chunks)
        return len(chunks)

    def health_check(self) -> bool:
        return True

    def count(self) -> int:
        return len(self.stored)

    def delete_document(self, document_id, tenant_id=None) -> int:
        self.deleted_documents.append(str(document_id))
        return 1


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "rag_intro.txt"
    f.write_text(
        "RAG combines retrieval with generation. "
        "The system searches a knowledge base for relevant documents. "
        "Those documents are provided to a language model as context. "
        "This grounding reduces hallucination in generated answers. "
        "Chunking splits documents into embeddable pieces before indexing. "
        "Each chunk is embedded into a dense vector for similarity search. "
        "At query time the query is embedded and matched against chunks. "
        "The top matching chunks are retrieved and reranked. "
        "Reranking scores each chunk with a cross-encoder for precision. "
        "Hybrid search adds sparse keyword matching on top of dense vectors. "
        "Evaluation measures faithfulness, relevancy, precision, and recall. "
        "Production systems layer caching and observability on top. " * 3,
        encoding="utf-8",
    )
    return f


@pytest.fixture
def pipeline():
    store = FakeVectorStore()
    embedder = FakeEmbedder(dim=8)
    return IngestionPipeline(
        embedder=embedder,
        vectorstore=store,  # type: ignore[arg-type]
        config=PipelineConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk=ChunkConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=0),
        ),
    )


class TestIngestFile:
    def test_success(self, pipeline, sample_file):
        result = pipeline.ingest_file(str(sample_file), tenant_id="tenant_a")

        assert result.status == DocumentStatus.COMPLETED
        assert result.chunks_created > 0
        assert "rag" in result.metadata["title"].lower() or result.metadata["strategy"]
        # Collection created with embedder dim
        assert pipeline._vectorstore.ensure_calls == [8]
        # tenant_id propagated to store
        assert pipeline._vectorstore.tenant_id_seen == ["tenant_a"]
        # Chunks embedded
        assert all(c.embedding is not None for c in pipeline._vectorstore.stored)

    def test_duration_recorded(self, pipeline, sample_file):
        result = pipeline.ingest_file(str(sample_file))
        assert result.duration_ms > 0

    def test_dedup_skips_second_ingest(self, pipeline, sample_file):
        r1 = pipeline.ingest_file(str(sample_file))
        assert r1.status == DocumentStatus.COMPLETED
        assert r1.chunks_created > 0
        # Document hash propagated to the store
        doc_hash = pipeline._vectorstore.document_hashes_seen[0]
        assert doc_hash is not None

        # Simulate the doc already being stored (as the real store would see it)
        pipeline._vectorstore._existing_doc_hashes.add(doc_hash)

        r2 = pipeline.ingest_file(str(sample_file))
        assert r2.status == DocumentStatus.COMPLETED
        assert r2.metadata.get("reason") == "duplicate"
        assert r2.chunks_created == 0

    def test_failure_returns_result(self, pipeline, tmp_path):
        """Bad file → IngestionResult with FAILED status, not an exception."""
        bad = tmp_path / "bad.xyz"
        bad.write_text("x")

        result = pipeline.ingest_file(str(bad))
        assert result.status == DocumentStatus.FAILED
        assert result.error is not None
        assert "Unsupported" in result.error


class TestIngestFiles:
    def test_batch_continues_on_failure(self, pipeline, sample_file, tmp_path):
        bad = tmp_path / "nope.txt"
        results = pipeline.ingest_files([str(sample_file), str(bad)])

        assert len(results) == 2
        assert results[0].status == DocumentStatus.COMPLETED
        assert results[1].status == DocumentStatus.FAILED


class TestIngestDocument:
    def test_in_memory_document(self, pipeline):
        doc = Document(
            content="Context is everything in retrieval augmented generation. "
            "Chunking strategy determines retrieval precision and recall. "
            "Embedding models convert text into dense vector representations. "
            "Vector databases enable fast approximate nearest neighbor search. "
            "Reranking models rescore candidates for better final ordering. " * 5,
            source="memory",
            source_type="txt",
        )
        result = pipeline.ingest_document(doc)
        assert result.status == DocumentStatus.COMPLETED
        assert result.chunks_created > 0
        assert pipeline._vectorstore.tenant_id_seen == [None]


class TestPipelineConfig:
    def test_from_settings(self):
        cfg = PipelineConfig.from_settings()
        assert cfg.strategy in list(ChunkingStrategy)
        assert cfg.chunk is not None

    def test_semantic_uses_embedder(self):
        store = FakeVectorStore()
        embedder = FakeEmbedder(dim=8)
        pipeline = IngestionPipeline(
            embedder=embedder,
            vectorstore=store,  # type: ignore[arg-type]
            config=PipelineConfig(
                strategy=ChunkingStrategy.SEMANTIC,
                chunk=ChunkConfig(chunk_size=250),
            ),
        )
        doc = Document(
            content="Sentence one about topic A. Sentence two continues on A. "
            "Now we pivot to a completely different topic B entirely. "
            "Topic B continues here with more detail about it. " * 10,
            source="mem",
            source_type="txt",
        )
        result = pipeline.ingest_document(doc)
        assert result.status == DocumentStatus.COMPLETED
        assert embedder.call_count > 1  # semantic chunking + final embedding


class TestVectorStoreDedupLogic:
    """Unit tests of dedup helpers without Qdrant."""

    def test_doc_hash_tracked(self):
        store = FakeVectorStore()
        assert not store.document_hash_exists("doc1")
        store._existing_doc_hashes.add("doc1")
        assert store.document_hash_exists("doc1", tenant_id="t1")

    def test_chunk_hash_tracked(self):
        store = FakeVectorStore()
        assert not store.content_hash_exists("chunk1")
        store._existing_chunk_hashes.add("chunk1")
        assert store.content_hash_exists("chunk1")
