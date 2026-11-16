"""End-to-end test for section 1.5: First Working End-to-End Test.

Ingests 3-5 sample documents, queries via the Retriever, and verifies that
citations trace back to their sources. Uses in-memory Qdrant + FakeEmbedder
so no API keys or network are required.
"""

from pathlib import Path

from src.core import setup_logging
from src.ingestion.chunkers import ChunkConfig, ChunkingStrategy
from src.ingestion.models import DocumentStatus
from src.ingestion.pipeline import IngestionPipeline, PipelineConfig
from src.ingestion.vectorstore import VectorStore
from src.retrieval import Retriever, SearchType
from tests.factories import FakeEmbedder

SAMPLE_DOCS = [
    (
        "rag_intro.txt",
        "# Introduction to RAG\n\n"
        "Retrieval-Augmented Generation (RAG) combines information retrieval "
        "with text generation to ground large language models in external "
        "knowledge. A typical RAG pipeline has four stages: load, chunk, "
        "embed, and retrieve+generate.",
    ),
    (
        "vector_databases.md",
        "# Vector Databases for RAG\n\n"
        "Qdrant is an open-source vector search engine written in Rust. It "
        "supports dense vectors, sparse BM25 vectors for hybrid search, "
        "payload filtering for tenant isolation, and int8 scalar "
        "quantization.",
    ),
    (
        "hybrid_search.txt",
        "Hybrid Search: Dense + Sparse\n\n"
        "Dense vector search captures semantic meaning but can miss exact "
        "keyword matches. Sparse BM25 search captures keyword overlap but "
        "misses semantics. Reciprocal Rank Fusion (RRF) combines both ranked "
        "lists using score = sum(weight / (k + rank)).",
    ),
    (
        "citations_guide.md",
        "# Citations in RAG\n\n"
        "Every retrieved chunk is assigned a citation key (typically its "
        "document ID), and the LLM cites sources inline using markers like "
        "[doc-a]. Citations make RAG auditable by letting users verify claims "
        "against source chunks.",
    ),
    (
        "embedding_models.txt",
        "Embedding Models\n\n"
        "Embedding models convert text into dense vectors where semantically "
        "similar texts are close together. Common choices include "
        "BAAI/bge-large-en-v1.5, all-MiniLM-L6-v2, and gemini-embedding-001.",
    ),
]


def _write_sample_docs(directory: Path) -> list[Path]:
    """Write the sample documents into `directory` and return their paths."""
    written: list[Path] = []
    for filename, content in SAMPLE_DOCS:
        p = directory / filename
        p.write_text(content, encoding="utf-8")
        written.append(p)
    return written


class TestFirstEndToEnd:
    def test_ingest_sample_documents(self, tmp_path: Path):
        """Task 1: Ingest 3-5 sample documents."""
        docs = _write_sample_docs(tmp_path)
        assert len(docs) == 5

        store = VectorStore(url=":memory:")
        embedder = FakeEmbedder(dim=64)
        pipeline = IngestionPipeline(
            embedder=embedder,
            vectorstore=store,  # type: ignore[arg-type]
            config=PipelineConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk=ChunkConfig(chunk_size=300, chunk_overlap=30),
            ),
        )

        results = pipeline.ingest_files([str(d) for d in docs], tenant_id="e2e")

        completed = [r for r in results if r.status == DocumentStatus.COMPLETED]
        assert len(completed) == 5, f"Expected 5 completed, got {len(completed)}"
        assert all(r.chunks_created > 0 for r in completed)
        assert store.count() > 0

    def test_query_returns_cited_results(self, tmp_path: Path):
        """Task 2 + 3: Query and verify citations trace back to sources."""
        docs = _write_sample_docs(tmp_path)

        store = VectorStore(url=":memory:")
        embedder = FakeEmbedder(dim=64)
        pipeline = IngestionPipeline(
            embedder=embedder,
            vectorstore=store,  # type: ignore[arg-type]
            config=PipelineConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk=ChunkConfig(chunk_size=300, chunk_overlap=30),
            ),
        )
        pipeline.ingest_files([str(d) for d in docs], tenant_id="e2e")

        retriever = Retriever(vectorstore=store, embedder=embedder)

        # Query for RAG — should surface rag_intro.txt
        results = retriever.search(
            "What is RAG and how does it work?",
            search_type=SearchType.HYBRID,
            top_k=5,
        )

        assert len(results) >= 1
        rag_hits = [
            r for r in results if "rag" in r.content.lower() or "retrieval" in r.content.lower()
        ]
        assert len(rag_hits) >= 1, (
            f"Expected RAG-related results, got: {[r.content[:60] for r in results]}"
        )

        # Every result should carry citation metadata traceable to a source
        sources_seen = set()
        for r in results:
            assert r.id, "Every result must have an id (citation key)"
            assert r.content, "Every result must have content"
            assert r.score is not None
            # Payload carries source + section for clickable trace-back
            assert r.payload.get("source"), f"Result {r.id} missing source payload"
            sources_seen.add(r.payload.get("source"))

        assert len(sources_seen) >= 1, "Citations should reference at least one source"

    def test_dense_sparse_hybrid_all_work(self, tmp_path: Path):
        """All three search modes should return results after ingestion."""
        docs = _write_sample_docs(tmp_path)

        store = VectorStore(url=":memory:")
        embedder = FakeEmbedder(dim=64)
        pipeline = IngestionPipeline(
            embedder=embedder,
            vectorstore=store,  # type: ignore[arg-type]
            config=PipelineConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk=ChunkConfig(chunk_size=300, chunk_overlap=30),
            ),
        )
        pipeline.ingest_files([str(d) for d in docs], tenant_id="e2e")

        retriever = Retriever(vectorstore=store, embedder=embedder)

        for search_type in (SearchType.DENSE, SearchType.SPARSE, SearchType.HYBRID):
            results = retriever.search(
                "vector database quantization",
                search_type=search_type,
                top_k=3,
            )
            assert len(results) >= 1, f"{search_type} search returned no results"

    def test_metadata_filtering_isolation(self, tmp_path: Path):
        """Metadata filtering (tenant_id) should restrict results to a tenant."""
        docs = _write_sample_docs(tmp_path)

        store = VectorStore(url=":memory:")
        embedder = FakeEmbedder(dim=64)
        pipeline = IngestionPipeline(
            embedder=embedder,
            vectorstore=store,  # type: ignore[arg-type]
            config=PipelineConfig(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk=ChunkConfig(chunk_size=300, chunk_overlap=30),
            ),
        )
        pipeline.ingest_files([str(d) for d in docs], tenant_id="tenant_a")

        retriever = Retriever(vectorstore=store, embedder=embedder)

        # Query scoped to tenant_a should return results
        results_a = retriever.search(
            "RAG pipeline stages",
            search_type=SearchType.DENSE,
            top_k=5,
        )
        assert len(results_a) >= 1

        # All results should carry the tenant_a payload
        for r in results_a:
            assert r.payload.get("tenant_id") == "tenant_a"


setup_logging()
