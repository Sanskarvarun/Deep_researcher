"""Tests for the embedder module (fake providers, batching, factory)."""

from typing import ClassVar
from uuid import uuid4

import pytest

from src.core.exceptions import EmbeddingError
from src.ingestion.embedder import (
    BATCH_SIZE,
    GeminiEmbedder,
    HuggingFaceEmbedder,
    create_embedder,
    embed_chunks,
)
from src.ingestion.models import Chunk
from tests.factories import FakeEmbedder


class TestHFEmbedder:
    def test_mean_pools_token_vectors(self, monkeypatch):
        """HF returns [[[tok1], [tok2], ...]] — mean-pooled to sentence vec."""
        embedder = HuggingFaceEmbedder(model="test", api_token="key", dim=3)

        class FakeResponse:
            status_code = 200

            def raise_for_status(self): ...

            def json(self):
                # One text, 3 tokens, each 3-dim: [texts][tokens][dim]
                return [
                    [
                        [1.0, 2.0, 4.0],
                        [3.0, 4.0, 6.0],
                        [5.0, 6.0, 8.0],
                    ]
                ]

        class FakeSession:
            headers: ClassVar[dict] = {}

            def post(self, *args, **kwargs):
                return FakeResponse()

        embedder._session = FakeSession()

        vectors = embedder.embed_texts(["hello"])
        assert vectors == [[3.0, 4.0, 6.0]]  # mean of the 3 token vectors

    def test_503_triggers_retry(self, monkeypatch):
        """Model-loading 503s should raise EmbeddingError (tenacity retries)."""
        embedder = HuggingFaceEmbedder(model="test", api_token="key")

        class FakeResponse:
            status_code = 503

            def raise_for_status(self):
                import requests

                raise requests.HTTPError("503 Service Unavailable", response=self)

            def json(self):
                return {"error": "loading"}

        class FakeSession:
            headers: ClassVar[dict] = {}

            def post(self, *args, **kwargs):
                return FakeResponse()

        embedder._session = FakeSession()
        embedder._embed_batch.retry.sleep = lambda *a, **k: None  # fast tests

        with pytest.raises(EmbeddingError):
            embedder._embed_batch(["hello"])

    def test_batches_requests(self, monkeypatch):
        """Texts exceeding BATCH_SIZE produce multiple API calls."""
        embedder = HuggingFaceEmbedder(model="test", api_token="key", dim=2)

        class FakeResponse:
            status_code = 200

            def raise_for_status(self): ...

            def json(self):
                return [[[1.0, 0.0]]] * BATCH_SIZE  # placeholder per call

        calls = {"count": 0}

        class FakeSession:
            headers: ClassVar[dict] = {}

            def post(self, *args, **kwargs):
                calls["count"] += 1
                # Return one pooled vector per input text
                n = len(kwargs["json"]["inputs"])
                return type(
                    "R",
                    (),
                    {
                        "status_code": 200,
                        "raise_for_status": lambda s: None,
                        "json": lambda s: [[[1.0, 0.0]]] * n,
                    },
                )()

        embedder._session = FakeSession()
        vectors = embedder.embed_texts(["a"] * (BATCH_SIZE * 2 + 3))
        assert calls["count"] == 3
        assert len(vectors) == BATCH_SIZE * 2 + 3


class TestEmbedChunks:
    def test_populates_embeddings(self):
        embedder = FakeEmbedder(dim=4)
        doc_id = uuid4()
        chunks = [
            Chunk(document_id=doc_id, content=f"text {i}", chunk_index=i, chunk_count=2)
            for i in range(2)
        ]
        embedded = embed_chunks(chunks, embedder)
        assert len(embedded) == 2
        for c in embedded:
            assert c.embedding is not None
            assert len(c.embedding) == 4
            assert c.embedding_model == "fake-embedder-v1"

    def test_empty_list(self):
        assert embed_chunks([], FakeEmbedder()) == []

    def test_batching_for_large_lists(self):
        embedder = FakeEmbedder()
        doc_id = uuid4()
        chunks = [
            Chunk(document_id=doc_id, content=f"t{i}", chunk_index=i, chunk_count=10)
            for i in range(10)
        ]
        embedded = embed_chunks(chunks, embedder)
        assert len(embedded) == 10
        assert embedder.call_count == 1  # all in one batch


class TestCreateEmbedder:
    def test_unknown_provider_raises(self, monkeypatch):
        with pytest.raises(EmbeddingError, match="Unknown embedding provider"):
            create_embedder("nope")

    def test_hf_without_token_raises(self, monkeypatch):
        from src.core.config import settings

        monkeypatch.setattr(settings, "hf_api_token", "")
        with pytest.raises(EmbeddingError, match="HF_API_TOKEN"):
            create_embedder("huggingface")

    def test_gemini_without_key_raises(self, monkeypatch):
        from src.core.config import settings

        monkeypatch.setattr(settings, "google_api_key", "")
        with pytest.raises(EmbeddingError, match="GOOGLE_API_KEY"):
            create_embedder("gemini")


class TestGeminiEmbedder:
    def test_handles_values_attribute(self, monkeypatch):
        """Gemini response embeddings: iterate .values safely."""
        embedder = GeminiEmbedder.__new__(GeminiEmbedder)
        embedder.model_name = "gemini-embedding-001"
        embedder.dim = 768

        class FakeEmbedding:
            values: ClassVar[list[float]] = [0.1, 0.2, 0.3]

        class FakeResult:
            embeddings: ClassVar[list[FakeEmbedding]] = [FakeEmbedding()]

        class FakeModels:
            def embed_content(self, **kwargs):
                return FakeResult()

        class FakeClient:
            models = FakeModels()

        embedder._client = FakeClient()
        vectors = embedder.embed_texts(["hello"])
        assert vectors == [[0.1, 0.2, 0.3]]

    def test_empty(self):
        embedder = GeminiEmbedder.__new__(GeminiEmbedder)
        assert embedder.embed_texts([]) == []
