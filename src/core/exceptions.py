"""Custom exceptions for the application."""

from typing import Any


class DeepResearcherError(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
        status_code: int = 500,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.status_code = status_code


class ConfigurationError(DeepResearcherError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="CONFIGURATION_ERROR",
            details=details,
            status_code=500,
        )


class ValidationError(DeepResearcherError):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="VALIDATION_ERROR",
            details=details,
            status_code=400,
        )


class AuthenticationError(DeepResearcherError):
    """Raised when authentication fails."""

    def __init__(
        self, message: str = "Authentication failed", details: dict[str, Any] | None = None
    ):
        super().__init__(
            message,
            code="AUTHENTICATION_ERROR",
            details=details,
            status_code=401,
        )


class AuthorizationError(DeepResearcherError):
    """Raised when authorization fails."""

    def __init__(
        self, message: str = "Insufficient permissions", details: dict[str, Any] | None = None
    ):
        super().__init__(
            message,
            code="AUTHORIZATION_ERROR",
            details=details,
            status_code=403,
        )


class NotFoundError(DeepResearcherError):
    """Raised when a resource is not found."""

    def __init__(self, resource: str, identifier: str, details: dict[str, Any] | None = None):
        super().__init__(
            f"{resource} not found: {identifier}",
            code="NOT_FOUND",
            details={**details, "resource": resource, "identifier": identifier}
            if details
            else {"resource": resource, "identifier": identifier},
            status_code=404,
        )


class RateLimitError(DeepResearcherError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message,
            code="RATE_LIMIT_EXCEEDED",
            details={**details, "retry_after": retry_after}
            if details
            else {"retry_after": retry_after},
            status_code=429,
        )


class ExternalServiceError(DeepResearcherError):
    """Raised when an external service call fails."""

    def __init__(
        self,
        service: str,
        message: str,
        details: dict[str, Any] | None = None,
        status_code: int = 502,
    ):
        super().__init__(
            f"{service} error: {message}",
            code="EXTERNAL_SERVICE_ERROR",
            details={**details, "service": service} if details else {"service": service},
            status_code=status_code,
        )


class EmbeddingError(ExternalServiceError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("embedding", message, details)


class RetrievalError(ExternalServiceError):
    """Raised when document retrieval fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("retrieval", message, details)


class GenerationError(ExternalServiceError):
    """Raised when LLM generation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("generation", message, details)


class RerankError(ExternalServiceError):
    """Raised when reranking fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__("rerank", message, details)


class DocumentProcessingError(DeepResearcherError):
    """Raised when document processing fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="DOCUMENT_PROCESSING_ERROR",
            details=details,
            status_code=500,
        )


class IngestionError(DocumentProcessingError):
    """Raised when document ingestion fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details)


class ChunkingError(DocumentProcessingError):
    """Raised when text chunking fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, details)


class CacheError(DeepResearcherError):
    """Raised when cache operations fail."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="CACHE_ERROR",
            details=details,
            status_code=500,
        )


class EvaluationError(DeepResearcherError):
    """Raised when evaluation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="EVALUATION_ERROR",
            details=details,
            status_code=500,
        )


class AgentError(DeepResearcherError):
    """Raised when agent execution fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message,
            code="AGENT_ERROR",
            details=details,
            status_code=500,
        )
