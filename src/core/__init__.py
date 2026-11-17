"""Core module initialization."""

from src.core.cache import (
    CacheBackend,
    CacheConfig,
    InMemoryCache,
    RedisCache,
    cache_delete,
    cache_get,
    cache_set,
    close_cache,
    create_cache,
    get_cache,
    make_cache_key,
)
from src.core.config import Settings, get_settings, settings
from src.core.exceptions import (
    AgentError,
    AuthenticationError,
    AuthorizationError,
    CacheError,
    ChunkingError,
    ConfigurationError,
    DeepResearcherError,
    DocumentProcessingError,
    EmbeddingError,
    EvaluationError,
    ExternalServiceError,
    GenerationError,
    IngestionError,
    NotFoundError,
    RateLimitError,
    RerankError,
    RetrievalError,
    ValidationError,
)
from src.core.logging import get_logger, setup_logging
from src.core.metrics import MetricsRegistry, metrics, track_latency
from src.core.tracing import (
    current_span_context,
    get_tracer,
    setup_tracing,
    trace_embed,
    trace_generate,
    trace_rerank,
    trace_retrieve,
    trace_span,
    traced,
)

__all__ = [
    "AgentError",
    "AuthenticationError",
    "AuthorizationError",
    # Cache
    "CacheBackend",
    "CacheConfig",
    "CacheError",
    "ChunkingError",
    "ConfigurationError",
    # Exceptions
    "DeepResearcherError",
    "DocumentProcessingError",
    "EmbeddingError",
    "EvaluationError",
    "ExternalServiceError",
    "GenerationError",
    "InMemoryCache",
    "IngestionError",
    # Metrics
    "MetricsRegistry",
    "NotFoundError",
    "RateLimitError",
    "RedisCache",
    "RerankError",
    "RetrievalError",
    # Config
    "Settings",
    "ValidationError",
    "cache_delete",
    "cache_get",
    "cache_set",
    "close_cache",
    "create_cache",
    # Tracing / Observability
    "current_span_context",
    "get_cache",
    "get_logger",
    "get_settings",
    "get_tracer",
    "make_cache_key",
    "metrics",
    "settings",
    # Logging
    "setup_logging",
    "setup_tracing",
    "trace_embed",
    "trace_generate",
    "trace_rerank",
    "trace_retrieve",
    "trace_span",
    "traced",
    "track_latency",
]
