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
from langchain_core.messages import convert_to_messages
from pydantic import SecretStr

from src.core import (
    CircuitOpenError,
    GenerationError,
    breakers,
    call_with_resilience,
    metrics,
)
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

    def default_model(self, provider: str) -> str:
        """Configured (full-size) model for a provider."""
        return {
            "gemini": settings.google_model,
            "openrouter": settings.openrouter_model,
            "groq": settings.groq_model,
        }.get(provider, "")

    def degraded_model(self, provider: str) -> str:
        """Smaller fallback model for a provider (graceful degradation)."""
        return {
            "gemini": settings.google_degraded_model,
            "openrouter": settings.openrouter_degraded_model,
            "groq": settings.groq_degraded_model,
        }.get(provider, "")


def _attempt_plan(
    factory: LLMFactory, provider: str | None = None
) -> list[tuple[str, str | None, bool]]:
    """Build the ordered attempt list as ``(provider, model_override, degraded)``.

    Full-size models are tried across every provider first; only then are the
    smaller degraded models tried (e.g. Llama 70B -> 8B, Gemini flash ->
    flash-lite). This preserves answer quality while still degrading
    gracefully when an entire model tier is unavailable.
    """
    chain = factory.get_fallback_chain()
    if provider:
        chain = [provider, *[p for p in chain if p != provider]]

    plan: list[tuple[str, str | None, bool]] = [(p, None, False) for p in chain]
    if settings.degraded_model_enabled:
        for p in chain:
            degraded = factory.degraded_model(p)
            if degraded and degraded != factory.default_model(p):
                plan.append((p, degraded, True))
    return plan


def _invoke_once(
    factory: LLMFactory,
    provider: str,
    messages: list,
    *,
    model: str | None,
    temperature: float | None,
    kwargs: dict[str, Any],
) -> Any:
    """Invoke one provider, protected by retry + circuit breaker."""
    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if temperature is not None:
        overrides["temperature"] = temperature

    service = f"llm:{provider}"

    def _call() -> Any:
        chat_model = factory.get_model(provider, **overrides)
        return chat_model.invoke(messages, **kwargs)

    return call_with_resilience(service, _call)


def invoke_with_fallback(
    factory: LLMFactory,
    messages: list,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    outcome: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Invoke the LLM with retries, breaker protection, and graceful fallback.

    Tries the requested provider first (default: ``LLM_PRIMARY``), then the
    remaining providers, then smaller degraded models. Providers whose circuit
    breaker is open are skipped instead of being called again.

    Args:
        factory: LLMFactory instance.
        messages: List of LangChain messages.
        provider: Preferred provider; defaults to the configured primary.
        model: Optional model override for the first attempt.
        temperature: Optional temperature override.
        outcome: Optional dict populated with the winning provider/model and
            whether the degraded tier was used.
        **kwargs: Passed to ``model.invoke()``.

    Returns:
        Model response.

    Raises:
        GenerationError: If every provider and degraded model fails.
    """
    errors: list[str] = []
    plan = _attempt_plan(factory, provider)
    if model and plan:
        # Caller-supplied model replaces the first attempt's model but is not a
        # degraded tier (it was explicitly requested at full quality).
        plan[0] = (plan[0][0], model, False)

    for attempt_provider, attempt_model, degraded in plan:
        service = f"llm:{attempt_provider}"
        try:
            log.info(
                "llm_invoke_attempt",
                provider=attempt_provider,
                model=attempt_model or "default",
                degraded=degraded,
            )
            response = _invoke_once(
                factory,
                attempt_provider,
                messages,
                model=attempt_model,
                temperature=temperature,
                kwargs=kwargs,
            )
            log.info("llm_invoke_success", provider=attempt_provider, degraded=degraded)
            if outcome is not None:
                outcome.update(
                    provider=attempt_provider,
                    model=attempt_model or factory.default_model(attempt_provider),
                    degraded=degraded,
                )
            if degraded:
                metrics.increment("generate.degraded", labels={"provider": attempt_provider})
            return response
        except CircuitOpenError as e:
            errors.append(f"{service}: {e.message}")
            metrics.increment("generate.circuit_rejected", labels={"provider": attempt_provider})
            log.warning("llm_circuit_open", provider=attempt_provider)
        except Exception as e:
            errors.append(f"{attempt_provider}: {e}")
            log.warning(
                "llm_invoke_failed",
                provider=attempt_provider,
                model=attempt_model or "default",
                error=str(e),
            )

    metrics.increment("generate.errors", labels={"provider": "all_failed"})
    log.error("llm_all_providers_failed", attempts=len(plan))
    raise GenerationError(f"All LLM providers failed: {errors}")


def stream_with_fallback(
    factory: LLMFactory,
    messages: list,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """Stream LLM response chunks with retries and provider fallback.

    Yields chunks from the first provider that successfully begins streaming.
    If a provider fails before yielding any chunk, the next fallback is tried.

    Args:
        factory: LLMFactory instance.
        messages: List of LangChain messages (or dicts convertible to them).
        provider: Preferred provider; defaults to the configured primary.
        **kwargs: Passed to model.stream().

    Yields:
        Streamed response chunks.

    Raises:
        GenerationError: If all providers fail to stream.
    """
    lc_messages = convert_to_messages(messages)
    errors: list[str] = []

    for attempt_provider, attempt_model, degraded in _attempt_plan(factory, provider):
        breaker = breakers.get(f"llm:{attempt_provider}")
        try:
            breaker.before_call()
        except CircuitOpenError as e:
            errors.append(f"llm:{attempt_provider}: {e.message}")
            log.warning("llm_circuit_open", provider=attempt_provider)
            continue

        try:
            log.info(
                "llm_stream_attempt",
                provider=attempt_provider,
                model=attempt_model or "default",
                degraded=degraded,
            )
            overrides = {"model": attempt_model} if attempt_model else {}
            model_obj = factory.stream(attempt_provider, **overrides)
            yield from model_obj.stream(lc_messages, **kwargs)
        except Exception as e:
            breaker.record_failure()
            errors.append(f"{attempt_provider}: {e}")
            log.warning("llm_stream_failed", provider=attempt_provider, error=str(e))
            continue
        breaker.record_success()
        log.info("llm_stream_success", provider=attempt_provider, degraded=degraded)
        return

    raise GenerationError(f"All LLM providers failed to stream: {errors}")


async def astream_with_fallback(
    factory: LLMFactory,
    messages: list,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Async-stream LLM chunks with retries and provider fallback.

    Mirrors :func:`stream_with_fallback` but uses ``model.astream()``.
    """
    lc_messages = convert_to_messages(messages)
    errors: list[str] = []

    for attempt_provider, attempt_model, degraded in _attempt_plan(factory, provider):
        breaker = breakers.get(f"llm:{attempt_provider}")
        try:
            breaker.before_call()
        except CircuitOpenError as e:
            errors.append(f"llm:{attempt_provider}: {e.message}")
            log.warning("llm_circuit_open", provider=attempt_provider)
            continue

        try:
            log.info(
                "llm_astream_attempt",
                provider=attempt_provider,
                model=attempt_model or "default",
                degraded=degraded,
            )
            overrides = {"model": attempt_model} if attempt_model else {}
            model_obj = factory.stream(attempt_provider, **overrides)
            async for chunk in model_obj.astream(lc_messages, **kwargs):
                yield chunk
        except Exception as e:
            breaker.record_failure()
            errors.append(f"{attempt_provider}: {e}")
            log.warning("llm_astream_failed", provider=attempt_provider, error=str(e))
            continue
        breaker.record_success()
        log.info("llm_astream_success", provider=attempt_provider, degraded=degraded)
        return

    raise GenerationError(f"All LLM providers failed to stream: {errors}")


# Module-level singleton
llm_factory = LLMFactory()
