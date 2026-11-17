"""Multi-provider LLM factory with fallback chain.

Supports:
- Gemini (primary): 1M context, function calling, structured output
- OpenRouter (fallback): 300+ models, `:free` variants, OpenAI-compatible
- Groq (optional): fast inference, Llama 3.1

Fallback chain configured via LLM_PRIMARY / LLM_FALLBACKS env vars.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from src.core import GenerationError, metrics
from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)


class LLMFactory:
    """Creates chat models for any provider with automatic fallback."""

    def __init__(self) -> None:
        self._clients: dict[str, BaseChatModel] = {}

    def get_model(self, provider: str | None = None, **overrides: Any) -> BaseChatModel:
        """Get a chat model for the given provider.

        Args:
            provider: "gemini", "openrouter", or "groq". Defaults to LLM_PRIMARY.
            **overrides: Override model, temperature, max_tokens.

        Returns:
            Configured BaseChatModel instance (cached).

        Raises:
            GenerationError: If provider is unknown or API key missing.
        """
        provider = provider or settings.llm_primary
        cache_key = f"{provider}:{overrides.get('model', '')}:{overrides.get('temperature', '')}"

        if cache_key in self._clients:
            return self._clients[cache_key]

        client = self._build(provider, **overrides)
        self._clients[cache_key] = client
        log.info("llm_client_created", provider=provider, model=overrides.get("model"))
        return client

    def stream(self, provider: str | None = None, **overrides: Any) -> BaseChatModel:
        """Get a chat model for the given provider, configured for streaming.

        Returns the same cached client as :meth:`get_model`; streaming is
        performed by calling ``.stream()`` / ``.astream()`` on the client.
        """
        return self.get_model(provider, **overrides)

    def _build(self, provider: str, **overrides: Any) -> BaseChatModel:
        """Build a chat model for a specific provider."""
        builders = {
            "gemini": self._build_gemini,
            "openrouter": self._build_openrouter,
            "groq": self._build_groq,
        }

        if provider not in builders:
            raise GenerationError(f"Unknown LLM provider: {provider}")

        try:
            return builders[provider](**overrides)
        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"Failed to build {provider} client: {e}") from e

    def _build_gemini(self, **overrides: Any) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise GenerationError("GOOGLE_API_KEY not set")

        return ChatGoogleGenerativeAI(
            model=overrides.get("model", settings.google_model),
            temperature=overrides.get("temperature", settings.google_temperature),
            max_output_tokens=overrides.get("max_tokens", settings.google_max_tokens),
            google_api_key=settings.google_api_key,
        )

    def _build_openrouter(self, **overrides: Any) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        if not settings.openrouter_api_key:
            raise GenerationError("OPENROUTER_API_KEY not set")

        return ChatOpenAI(
            model=overrides.get("model", settings.openrouter_model),
            temperature=overrides.get("temperature", settings.openrouter_temperature),
            api_key=SecretStr(settings.openrouter_api_key),
            base_url="https://openrouter.ai/api/v1",
        )

    def _build_groq(self, **overrides: Any) -> BaseChatModel:
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise GenerationError("GROQ_API_KEY not set")

        return ChatGroq(
            model=overrides.get("model", settings.groq_model),
            temperature=overrides.get("temperature", settings.groq_temperature),
            max_tokens=overrides.get("max_tokens", settings.groq_max_tokens),
            api_key=SecretStr(settings.groq_api_key),
        )

    def get_fallback_chain(self) -> list[str]:
        """Return ordered provider list: primary first, then fallbacks."""
        return [settings.llm_primary, *settings.llm_fallbacks]


# Retry decorator for transient LLM errors
_llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(0, 1),
    retry=retry_if_exception_type(GenerationError),
    reraise=True,
)


@retry(
    stop=stop_after_attempt(len(settings.llm_fallbacks) + 1),
    wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(0, 1),
    reraise=True,
)
def invoke_with_fallback(factory: LLMFactory, messages: list, **kwargs: Any) -> Any:
    """Invoke LLM with automatic provider fallback.

    Tries primary provider, then each fallback in order.

    Args:
        factory: LLMFactory instance.
        messages: List of LangChain messages.
        **kwargs: Passed to model.invoke().

    Returns:
        Model response.

    Raises:
        GenerationError: If all providers fail.
    """
    chain = factory.get_fallback_chain()
    errors: list[str] = []

    for provider in chain:
        try:
            log.info("llm_invoke_attempt", provider=provider)
            model = factory.get_model(provider)
            response = model.invoke(messages, **kwargs)
            log.info("llm_invoke_success", provider=provider)
            return response
        except Exception as e:
            error_msg = f"{provider}: {e}"
            errors.append(error_msg)
            log.warning("llm_invoke_failed", provider=provider, error=str(e))

    metrics.increment("generate.errors", labels={"provider": "all_failed"})
    raise GenerationError(f"All LLM providers failed: {errors}")


def stream_with_fallback(factory: LLMFactory, messages: list, **kwargs: Any) -> Iterator[Any]:
    """Stream LLM response chunks with automatic provider fallback.

    Yields chunks from the first provider that successfully begins streaming.
    If a provider fails before yielding any chunk, the next fallback is tried.

    Args:
        factory: LLMFactory instance.
        messages: List of LangChain messages (or dicts convertible to them).
        **kwargs: Passed to model.stream().

    Yields:
        Streamed response chunks.

    Raises:
        GenerationError: If all providers fail to stream.
    """
    from langchain_core.messages import convert_to_messages

    lc_messages = convert_to_messages(messages)
    chain = factory.get_fallback_chain()
    errors: list[str] = []

    for provider in chain:
        try:
            log.info("llm_stream_attempt", provider=provider)
            model = factory.stream(provider)
            yield from model.stream(lc_messages, **kwargs)
            log.info("llm_stream_success", provider=provider)
            return
        except Exception as e:
            # Drain any partially yielded chunk from the generator above.
            error_msg = f"{provider}: {e}"
            errors.append(error_msg)
            log.warning("llm_stream_failed", provider=provider, error=str(e))

    raise GenerationError(f"All LLM providers failed to stream: {errors}")


async def astream_with_fallback(
    factory: LLMFactory, messages: list, **kwargs: Any
) -> AsyncIterator[Any]:
    """Async-stream LLM response chunks with automatic provider fallback.

    Mirrors :func:`stream_with_fallback` but uses ``model.astream()``.
    """
    from langchain_core.messages import convert_to_messages

    lc_messages = convert_to_messages(messages)
    chain = factory.get_fallback_chain()
    errors: list[str] = []

    for provider in chain:
        try:
            log.info("llm_astream_attempt", provider=provider)
            model = factory.stream(provider)
            async for chunk in model.astream(lc_messages, **kwargs):
                yield chunk
            log.info("llm_astream_success", provider=provider)
            return
        except Exception as e:
            error_msg = f"{provider}: {e}"
            errors.append(error_msg)
            log.warning("llm_astream_failed", provider=provider, error=str(e))

    raise GenerationError(f"All LLM providers failed to stream: {errors}")


# Module-level singleton
llm_factory = LLMFactory()
