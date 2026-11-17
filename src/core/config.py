"""Core configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "deepresearcher"
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    debug: bool = Field(default=False, alias="DEBUG")

    # LLM Providers (primary: Gemini, fallback: OpenRouter, optional: Groq)
    # Primary LLM - Google Gemini
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-flash-latest", alias="GOOGLE_MODEL")
    google_temperature: float = Field(default=0.1, alias="GOOGLE_TEMPERATURE")
    google_max_tokens: int = Field(default=4096, alias="GOOGLE_MAX_TOKENS")

    # Fallback LLM - OpenRouter (OpenAI-compatible)
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free", alias="OPENROUTER_MODEL"
    )
    openrouter_temperature: float = Field(default=0.1, alias="OPENROUTER_TEMPERATURE")

    # Legacy LLM - Groq (optional, kept for future use)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-70b-versatile", alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.1, alias="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=4096, alias="GROQ_MAX_TOKENS")

    # Provider selection: which provider to try first
    llm_primary: str = Field(default="gemini", alias="LLM_PRIMARY")
    llm_fallbacks: list[str] = Field(default=["openrouter", "groq"], alias="LLM_FALLBACKS")

    # Embeddings (HuggingFace Inference API, local, or Gemini)
    # Options: "huggingface" (default), "local", "gemini"
    hf_api_token: str = Field(default="", alias="HF_API_TOKEN")
    embedding_provider: str = Field(default="huggingface", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    use_local_embeddings: bool = Field(default=False, alias="USE_LOCAL_EMBEDDINGS")
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL"
    )
    gemini_embedding_dim: int = Field(default=768, alias="GEMINI_EMBEDDING_DIM")

    # Vector Database (Qdrant)
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="documents", alias="QDRANT_COLLECTION")
    qdrant_prefer_grpc: bool = Field(default=False, alias="QDRANT_PREFER_GRPC")

    # Reranking
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")
    rerank_model: str = Field(default="rerank-english-v3.0", alias="RERANK_MODEL")
    use_local_reranker: bool = Field(default=True, alias="USE_LOCAL_RERANKER")
    local_reranker_model: str = Field(
        default="BAAI/bge-reranker-large", alias="LOCAL_RERANKER_MODEL"
    )

    # Retrieval
    top_k: int = Field(default=10, alias="TOP_K")
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")
    score_threshold: float = Field(default=0.5, alias="SCORE_THRESHOLD")
    enable_hybrid_search: bool = Field(default=True, alias="ENABLE_HYBRID_SEARCH")
    bm25_weight: float = Field(default=0.3, alias="BM25_WEIGHT")
    dense_weight: float = Field(default=0.7, alias="DENSE_WEIGHT")

    # Chunking
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    chunk_min_size: int = Field(default=100, alias="CHUNK_MIN_SIZE")
    chunking_strategy: str = Field(default="recursive", alias="CHUNKING_STRATEGY")
    semantic_chunking: bool = Field(default=False, alias="SEMANTIC_CHUNKING")

    # Caching (Redis)
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")

    # Observability
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="deepresearcher", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    phoenix_url: str = Field(default="http://localhost:6006", alias="PHOENIX_URL")
    enable_tracing: bool = Field(default=True, alias="ENABLE_TRACING")
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_workers: int = Field(default=1, alias="API_WORKERS")
    cors_origins: list[str] = Field(default=["*"], alias="CORS_ORIGINS")
    rate_limit_requests: int = Field(default=100, ge=1, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, ge=1, alias="RATE_LIMIT_WINDOW")
    enable_rate_limit: bool = Field(default=True, alias="ENABLE_RATE_LIMIT")

    # API security
    # When REQUIRE_AUTH=true, requests must send the API key via
    # `Authorization: Bearer <key>` or `X-API-Key: <key>`.
    api_key: str = Field(default="", alias="API_KEY")
    require_auth: bool = Field(default=True, alias="REQUIRE_AUTH")
    max_request_bytes: int = Field(default=1_048_576, ge=1024, alias="MAX_REQUEST_BYTES")
    trusted_proxies: list[str] = Field(default=[], alias="TRUSTED_PROXIES")
    # Optional root that API ingestion paths must stay inside. Empty = no root
    # restriction (still rejects traversal and unsupported extensions).
    ingest_root: str = Field(default="", alias="INGEST_ROOT")

    # Resilience (external services)
    retry_max_attempts: int = Field(default=3, ge=1, le=10, alias="RETRY_MAX_ATTEMPTS")
    retry_base_delay: float = Field(default=0.5, ge=0.0, alias="RETRY_BASE_DELAY")
    retry_max_delay: float = Field(default=8.0, ge=0.0, alias="RETRY_MAX_DELAY")
    # Per-request ceiling for provider SDK calls. Without this a provider that
    # stalls never returns, so neither retries nor the breaker can ever fire.
    llm_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0, alias="LLM_TIMEOUT_SECONDS")
    circuit_failure_threshold: int = Field(default=5, ge=1, alias="CIRCUIT_FAILURE_THRESHOLD")
    circuit_recovery_timeout: float = Field(default=30.0, ge=0.0, alias="CIRCUIT_RECOVERY_TIMEOUT")
    circuit_half_open_max_calls: int = Field(default=1, ge=1, alias="CIRCUIT_HALF_OPEN_MAX_CALLS")

    # Graceful degradation (large model -> small model on the same provider)
    degraded_model_enabled: bool = Field(default=True, alias="DEGRADED_MODEL_ENABLED")
    google_degraded_model: str = Field(
        default="gemini-flash-lite-latest", alias="GOOGLE_DEGRADED_MODEL"
    )
    openrouter_degraded_model: str = Field(
        default="meta-llama/llama-3.1-8b-instruct:free",
        alias="OPENROUTER_DEGRADED_MODEL",
    )
    groq_degraded_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_DEGRADED_MODEL")

    # Document Processing
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    supported_extensions: list[str] = Field(
        default=[".pdf", ".md", ".txt", ".html", ".htm"],
        alias="SUPPORTED_EXTENSIONS",
    )

    # Evaluation
    ragas_app_token: str = Field(default="", alias="RAGAS_APP_TOKEN")
    evaluation_dataset_path: str = Field(
        default="data/eval/golden_set.json", alias="EVALUATION_DATASET_PATH"
    )

    # Search API Keys (for WebSearchTool)
    serpapi_key: str = Field(default="", alias="SERPAPI_KEY")
    brave_search_key: str = Field(default="", alias="BRAVE_SEARCH_KEY")
    google_search_key: str = Field(default="", alias="GOOGLE_SEARCH_KEY")
    google_search_cx: str = Field(default="", alias="GOOGLE_SEARCH_CX")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
