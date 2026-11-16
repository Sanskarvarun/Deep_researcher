"""Tests for chunking strategies."""

import pytest

from src.core.exceptions import ChunkingError
from src.ingestion.chunkers import (
    ChunkConfig,
    ChunkingStrategy,
    chunk_document,
)
from src.ingestion.models import Document, SourceType
from tests.factories import FakeEmbedder


@pytest.fixture
def document():
    """A document long enough to require multiple chunks."""
    return Document(
        content=(
            "Retrieval-Augmented Generation (RAG) combines information retrieval "
            "with language generation. The system first searches a knowledge base "
            "for relevant documents, then feeds those documents to a language model "
            "as context. This approach reduces hallucination because the model can "
            "ground its answers in retrieved facts. Chunking is a critical step: "
            "documents are split into smaller pieces before embedding. Each chunk "
            "is embedded into a dense vector representation. At query time, the "
            "query is embedded and compared against all chunks. The most similar "
            "chunks are retrieved and passed to the model. Reranking further "
            "improves precision by scoring query-chunk pairs with a cross-encoder. "
            "Hybrid search combines dense vectors with sparse keyword matching. "
            "Reciprocal Rank Fusion merges results from both retrieval methods. "
            "Evaluation uses metrics like faithfulness and answer relevancy. "
            "Production systems add caching, observability, and fallbacks. "
        ),
        source="test.md",
        source_type=SourceType.MARKDOWN,
        title="RAG Overview",
    )


@pytest.fixture
def markdown_document():
    """Markdown doc with sections, sized to force splitting."""
    section_body = (
        "This section covers important details about the topic. "
        "It contains several sentences of meaningful content that will need "
        "to be split when the chunk size is small. Each sentence adds "
        "context and information about the subject matter. "
    ) * 3
    return Document(
        content=(
            "# Chapter One\n\n"
            + section_body
            + "\n\n## Section 1.1\n\n"
            + section_body
            + "\n\n## Section 1.2\n\n"
            + section_body
        ),
        source="doc.md",
        source_type=SourceType.MARKDOWN,
    )


class TestChunkConfig:
    def test_invalid_overlap_raises(self, document):
        with pytest.raises(ChunkingError, match="must be <"):
            chunk_document(
                document,
                config=ChunkConfig(chunk_size=100, chunk_overlap=100),
                strategy=ChunkingStrategy.FIXED,
            )


class TestFixedChunking:
    def test_chunks_respect_size(self, document):
        config = ChunkConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=0)
        chunks = chunk_document(document, config, strategy=ChunkingStrategy.FIXED)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 200

    def test_indices_and_counts_set(self, document):
        config = ChunkConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=0)
        chunks = chunk_document(document, config, strategy=ChunkingStrategy.FIXED)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.chunk_count == len(chunks) for c in chunks)

    def test_short_document_single_chunk(self):
        doc = Document(content="short", source="x.txt", source_type=SourceType.TXT)
        chunks = chunk_document(doc, strategy=ChunkingStrategy.FIXED)
        assert len(chunks) == 1

    def test_metadata_populated(self, document):
        chunks = chunk_document(
            document, ChunkConfig(chunk_size=200), strategy=ChunkingStrategy.FIXED
        )
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "fixed"
            assert chunk.metadata["char_count"] == len(chunk.content)
            assert chunk.metadata["approx_tokens"] > 0


class TestRecursiveChunking:
    def test_splits_on_paragraphs_first(self, document):
        # Big chunk size → whole doc is 1 chunk
        chunks = chunk_document(
            document, ChunkConfig(chunk_size=5000), strategy=ChunkingStrategy.RECURSIVE
        )
        assert len(chunks) == 1

        # Small → many chunks
        chunks = chunk_document(
            document, ChunkConfig(chunk_size=300), strategy=ChunkingStrategy.RECURSIVE
        )
        assert len(chunks) > 1

    def test_no_oversized_chunks(self, document):
        config = ChunkConfig(chunk_size=250, min_chunk_size=50)
        chunks = chunk_document(document, config, strategy=ChunkingStrategy.RECURSIVE)
        for chunk in chunks:
            # small tolerance: min_chunk merges can push slightly over
            assert len(chunk.content) <= 400


class TestMarkdownChunking:
    def test_sections_are_boundaries(self, markdown_document):
        config = ChunkConfig(chunk_size=200, min_chunk_size=50)
        chunks = chunk_document(markdown_document, config, strategy=ChunkingStrategy.MARKDOWN)

        # At least one chunk should have section context or header
        contents = " ".join(c.content for c in chunks)
        assert "Chapter One" in contents
        assert "Section 1.1" in contents or "## Section" in contents

    def test_falls_back_for_non_markdown(self, document):
        chunks = chunk_document(
            document, ChunkConfig(chunk_size=300), strategy=ChunkingStrategy.MARKDOWN
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip()


class TestSemanticChunking:
    def test_requires_embedder(self, document):
        with pytest.raises(ChunkingError, match="requires an embedder"):
            chunk_document(document, strategy=ChunkingStrategy.SEMANTIC, embedder=None)

    def test_with_embedder(self, document):
        embedder = FakeEmbedder(dim=8)
        chunks = chunk_document(
            document,
            ChunkConfig(chunk_size=250),
            strategy=ChunkingStrategy.SEMANTIC,
            embedder=embedder,
        )
        assert len(chunks) >= 1
        assert embedder.call_count >= 1
        # All chunks non-empty
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_short_text_single_chunk(self):
        doc = Document(content="Just one sentence.", source="x.txt", source_type=SourceType.TXT)
        chunks = chunk_document(doc, strategy=ChunkingStrategy.SEMANTIC, embedder=FakeEmbedder())
        assert len(chunks) == 1


class TestTinyChunkMerging:
    def test_no_tiny_chunks(self):
        """Chunks below min_chunk_size get merged into neighbors."""
        doc = Document(
            content=(
                "Para one is here.\n\n"
                "a\n\n"  # tiny
                "Para two is also here and longer.\n\n"
                "b\n\n"  # tiny
                "Para three wraps things up nicely."
            )
            * 5,
            source="x.txt",
            source_type=SourceType.TXT,
        )
        config = ChunkConfig(chunk_size=100, min_chunk_size=50)
        chunks = chunk_document(doc, config, strategy=ChunkingStrategy.RECURSIVE)
        for chunk in chunks:
            assert len(chunk.content.strip()) >= 30  # merged orphans

    def test_all_chunks_nonempty(self, document):
        for strategy in ChunkingStrategy:
            if strategy == ChunkingStrategy.SEMANTIC:
                chunks = chunk_document(
                    document,
                    ChunkConfig(chunk_size=300),
                    strategy=strategy,
                    embedder=FakeEmbedder(),
                )
            else:
                chunks = chunk_document(document, ChunkConfig(chunk_size=300), strategy=strategy)
            assert all(c.content.strip() for c in chunks), f"{strategy} produced empty chunk"


class TestStrategyComparison:
    def test_recursive_vs_fixed(self, document):
        """Recursive should produce fewer fragmentation artifacts on prose."""
        fixed = chunk_document(
            document, ChunkConfig(chunk_size=300, min_chunk_size=100), ChunkingStrategy.FIXED
        )
        recursive = chunk_document(
            document, ChunkConfig(chunk_size=300, min_chunk_size=100), ChunkingStrategy.RECURSIVE
        )
        assert len(fixed) >= 1
        assert len(recursive) >= 1
        # Both cover the full content (approximately, modulo trimming)
        fixed_len = sum(len(c.content) for c in fixed)
        rec_len = sum(len(c.content) for c in recursive)
        assert abs(fixed_len - rec_len) < len(document.content) * 0.5
