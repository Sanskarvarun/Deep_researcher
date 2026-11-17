"""Embedding generation with batching, retry, and multi-provider support.

Providers:
- `huggingface`: HF Inference API (bge-large-en-v1.5) — free 30K req/month
- `local`: local sentence-transformers — unlimited, slower to load
- `gemini`: Gemini embeddings API — flexible 768-3072 dims

All providers implement `embed_texts(texts) -> list[list[float]]`.
Batching + retry logic lives here so callers get resilience for free.
"""

from typing import Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from src.core import (
    breakers,
    cache_get,
    cache_set,
    get_logger,
    make_cache_key,
    metrics,
    trace_embed,
)
from src.core.exceptions import EmbeddingError
from src.ingestion.models import EmbeddedChunk

log = get_logger(__name__)

BATCH_SIZE = 32  # texts per API call
MAX_RETRIES = 5


class EmbeddingModel(Protocol):
    """Interface all embedding providers implement."""

    model_name: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Provider 1: HuggingFace Inference API (default)
# ---------------------------------------------------------------------------


class HuggingFaceEmbedder:
    """Embeddings via HuggingFace Inference API (sentence-transformers host).

    Free tier: 30,000 requests/month, no credit card required.
    """

    def __init__(self, model: str, api_token: str, dim: int = 1024) -> None:
        import requests

        self.model_name = model
        self.dim = dim
        self._token = api_token
        self._url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_token}"})

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
        retry=retry_if_exception_type(EmbeddingError),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import requests

        try:
            response = self._session.post(
                self._url,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 503:
                # Model is loading — retry after warm-up
                raise EmbeddingError(f"Model loading (503): {e}") from e
            raise EmbeddingError(f"HF API error: {e}") from e
        except Exception as e:
            raise EmbeddingError(f"HF embedding request failed: {e}") from e

        # HF returns [[[...]]] (token embeddings) — mean-pool to sentence vector
        pooled: list[list[float]] = []
        for item in raw:
            if item and isinstance(item[0], list):
                token_vecs = item
                dim = len(token_vecs[0])
                mean = [sum(vec[i] for vec in token_vecs) / len(token_vecs) for i in range(dim)]
                pooled.append(mean)
            else:
                pooled.append(item)

        return pooled

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of BATCH_SIZE with per-text caching."""
        if not texts:
            return []

        results: list[list[float] | None] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = make_cache_key("emb", self.model_name, text)
            cached = cache_get(cache_key)
            if cached is not None:
                results.append(cached)  # type: ignore[arg-type]
            else:
                results.append(None)  # placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            with trace_embed(
                provider="huggingface", model=self.model_name, count=len(uncached_texts)
            ):
                for i in range(0, len(uncached_texts), BATCH_SIZE):
                    batch = uncached_texts[i : i + BATCH_SIZE]
                    batch_results = breakers.call("embed:huggingface", self._embed_batch, batch)
                    for j, emb in enumerate(batch_results):
                        idx = uncached_indices[i + j]
                        results[idx] = emb
                        cache_key = make_cache_key("emb", self.model_name, uncached_texts[i + j])
                        cache_set(cache_key, emb)
                    log.debug("embed_batch_done", model=self.model_name, batch_size=len(batch))

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Provider 2: Local sentence-transformers (unlimited)
# ---------------------------------------------------------------------------


class LocalEmbedder:
    """Embeddings via local sentence-transformers model. No API needed."""

    def __init__(self, model: str = "BAAI/bge-large-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        # bge models: first call downloads ~1.3GB, cached afterwards
        self._model = SentenceTransformer(model)
        embedding_dim = self._model.get_sentence_embedding_dimension()
        self.dim: int = embedding_dim if embedding_dim is not None else 768

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []
        for i, text in enumerate(texts):
            cache_key = make_cache_key("emb", self.model_name, text)
            cached = cache_get(cache_key)
            if cached is not None:
                results.append(cached)  # type: ignore[arg-type]
            else:
                results.append(None)  # placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        if uncached_texts:
            with trace_embed(provider="local", model=self.model_name, count=len(uncached_texts)):
                vectors = breakers.call(
                    "embed:local",
                    self._model.encode,
                    uncached_texts,
                    show_progress_bar=False,
                    convert_to_numpy=False,
                )
            for j, v in enumerate(vectors):
                emb = v.tolist()
                idx = uncached_indices[j]
                results[idx] = emb
                cache_key = make_cache_key("emb", self.model_name, uncached_texts[j])
                cache_set(cache_key, emb)
        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Provider 3: Gemini embeddings (flexible dims 128-3072)
# ---------------------------------------------------------------------------


class GeminiEmbedder:
    """Embeddings via Gemini API (gemini-embedding-001).

    Free tier available. Flexible output dims: 128-3072 (recommended 768/1536/3072).
    Supports multimodal (text+image) inputs.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
        dim: int = 768,
    ) -> None:
        from google import genai

        self.model_name = model
        self.dim = dim
        self._client = genai.Client(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        if not texts:
            return []

        results: list[list[float] | None] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []
        for i, text in enumerate(texts):
            cache_key = make_cache_key("emb", self.model_name, text)
            cached = cache_get(cache_key)
            if cached is not None:
                results.append(cached)  # type: ignore[arg-type]
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            with trace_embed(provider="gemini", model=self.model_name, count=len(uncached_texts)):
                result = breakers.call(
                    "embed:gemini",
                    self._client.models.embed_content,
                    model=self.model_name,
                    contents=list(uncached_texts),  # type: ignore[arg-type]
                    config=types.EmbedContentConfig(output_dimensionality=self.dim),
                )
            for j, embedding in enumerate(result.embeddings or []):
                values = getattr(embedding, "values", None)
                if values:
                    emb = [float(v) for v in values]
                    idx = uncached_indices[j]
                    results[idx] = emb
                    cache_key = make_cache_key("emb", self.model_name, uncached_texts[j])
                    cache_set(cache_key, emb)

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Factory + orchestration
# ---------------------------------------------------------------------------


def create_embedder(provider: str | None = None) -> EmbeddingModel:
    """Create an embedder from settings.

    Args:
        provider: "huggingface", "local", or "gemini". Defaults to
            EMBEDDING_PROVIDER env var.

    Returns:
        Configured embedder implementing EmbeddingModel.
    """
    from src.core.config import settings

    provider = provider or settings.embedding_provider

    if provider == "huggingface":
        if not settings.hf_api_token:
            raise EmbeddingError("HF_API_TOKEN not set (or use EMBEDDING_PROVIDER=local)")
        return HuggingFaceEmbedder(
            model=settings.embedding_model,
            api_token=settings.hf_api_token,
            dim=settings.embedding_dim,
        )
    elif provider == "local":
        return LocalEmbedder(model=settings.embedding_model)
    elif provider == "gemini":
        if not settings.google_api_key:
            raise EmbeddingError("GOOGLE_API_KEY not set")
        return GeminiEmbedder(
            model=settings.gemini_embedding_model,
            api_key=settings.google_api_key,
            dim=settings.gemini_embedding_dim,
        )
    else:
        raise EmbeddingError(
            f"Unknown embedding provider: {provider}",
            details={"supported": ["huggingface", "local", "gemini"]},
        )


def embed_chunks(
    chunks: list[EmbeddedChunk] | list, embedder: EmbeddingModel
) -> list[EmbeddedChunk]:
    """Embed chunk contents and return embedded chunks.

    Uses `chunk.context` when present (parent-document style: embed the
    short snippet, retrieve it, but preserve richer context for the LLM).
    """
    if not chunks:
        return []

    texts = [c.content for c in chunks]
    with (
        trace_embed(provider=embedder.model_name, model=embedder.model_name, count=len(chunks)),
        metrics.timer("embed.latency", labels={"provider": embedder.model_name}),
    ):
        embeddings = embedder.embed_texts(texts)
    metrics.increment("embed.requests", labels={"provider": embedder.model_name})

    embedded: list[EmbeddedChunk] = []
    for chunk, embedding in zip(chunks, embeddings, strict=False):
        chunk.embedding = embedding
        chunk.embedding_model = embedder.model_name
        embedded.append(EmbeddedChunk(**chunk.model_dump()))

    log.info("chunks_embedded", model=embedder.model_name, count=len(embedded))
    return embedded
