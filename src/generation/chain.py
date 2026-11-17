"""RAG generation chain.

Wraps retrieval results, prompt templates, the LLM (with fallback),
citation tracking, and token counting into a single ``generate`` call that
returns a formatted answer with inline citations and a sources section.

Usage::

    generator = RAGGenerator()
    result = generator.generate(
        question="What is RAG?",
        results=retriever.search("what is RAG?", top_k=5),
    )
    print(result.answer)
    print(result.sources_markdown)
"""

from threading import Event, Lock
from typing import Any, TypeVar, cast

from langchain_core.messages import BaseMessage, convert_to_messages
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field

from src.core import GenerationError, cache_get, cache_set, make_cache_key, metrics, traced
from src.core.config import settings
from src.core.logging import get_logger
from src.generation.citations import Citation, CitationTracker
from src.generation.llm import LLMFactory, invoke_with_fallback, llm_factory
from src.generation.prompts import RAGPromptBuilder
from src.generation.tokens import TokenUsage, count_message_tokens, count_tokens, estimate_cost

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def _response_text(response: Any) -> str:
    """Extract plain text from an LLM response (AIMessage or str)."""
    if isinstance(response, BaseMessage):
        content = response.content
        if isinstance(content, list):
            return "".join(str(part) for part in content)
        return content or ""
    if isinstance(response, str):
        return response
    return str(response)


class GenerationResult(BaseModel):
    """Result of a RAG generation call."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    usage: TokenUsage
    cost_usd: float
    provider: str
    model: str
    sources_markdown: str = ""

    @property
    def answer_with_citations(self) -> str:
        """The full answer including inline citations and the sources list."""
        parts = [self.answer]
        if self.sources_markdown:
            parts.append("")
            parts.append(self.sources_markdown)
        return "\n".join(parts)


class RAGGenerator:
    """Generate cited answers from retrieved chunks."""

    def __init__(
        self,
        factory: LLMFactory | None = None,
        prompt_builder: RAGPromptBuilder | None = None,
        citation_tracker: CitationTracker | None = None,
    ) -> None:
        self.factory = factory or llm_factory
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.citation_tracker = citation_tracker or CitationTracker()
        self._inflight_locks: dict[str, Event] = {}
        self._inflight_lock = Lock()

    @traced(name="generate")
    def generate(
        self,
        question: str,
        results: list,
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        output_schema: type[T] | None = None,
    ) -> GenerationResult | T:
        """Generate an answer for ``question`` grounded in ``results``.

        Args:
            question: The user's question.
            results: Retrieval results (SearchResult-like objects).
            provider: LLM provider override ("gemini", "openrouter", "groq").
            model: Model name override.
            temperature: Temperature override.
            output_schema: Optional Pydantic model for structured output. When
                set, the primary provider is used with
                ``.with_structured_output(output_schema)`` and a structured
                object is returned instead of a ``GenerationResult``.

        Returns:
            A ``GenerationResult`` (or the structured object when
            ``output_schema`` is given).
        """
        messages = self.prompt_builder.build_messages(question, results)
        prompt_tokens = count_message_tokens(messages)

        labels = {
            "provider": provider or settings.llm_primary,
            "schema": "structured" if output_schema else "text",
        }
        metrics.increment("generate.requests", labels=labels)

        # Skip cache for structured output
        if output_schema is not None:
            return self._generate_structured(
                question, results, messages, prompt_tokens, output_schema, model_instance=model
            )

        # Cache key based on prompt + generation parameters (stable across processes)
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        cache_key = make_cache_key(
            "llm",
            prompt_text,
            provider or settings.llm_primary,
            model or "",
            temperature or 0.0,
        )
        cached = cache_get(cache_key)
        if cached is not None:
            log.debug("generate_cache_hit", provider=provider or settings.llm_primary)
            metrics.increment("generate.cache_hits")
            return cached  # type: ignore[no-any-return]

        # In-flight request deduplication (coalescing).
        # If another thread is already generating this same request, wait for
        # it to fill the cache instead of paying for a duplicate LLM call.
        with self._inflight_lock:
            existing_event = self._inflight_locks.get(cache_key)
            if existing_event is None:
                owner = True
                existing_event = Event()
                self._inflight_locks[cache_key] = existing_event
            else:
                owner = False
        event = existing_event

        if not owner:
            # Wait up to 30s for the owner, then fall back to generating.
            event.wait(timeout=30)
            cached = cache_get(cache_key)
            if cached is not None:
                log.debug("generate_coalesced_hit", provider=provider or settings.llm_primary)
                metrics.increment("generate.coalesced_hits")
                return cached  # type: ignore[no-any-return]

        try:
            # Resolve provider/model. A ``model`` instance passed directly takes
            # precedence (useful for testing without API keys); otherwise the
            # factory/fallback chain is used.
            provider_used: str
            model_used: str
            if model is not None and not isinstance(model, str):
                chat_model = model
                try:
                    response = chat_model.invoke(convert_to_messages(messages))
                except Exception as e:
                    metrics.increment("generate.errors", labels={"provider": provider or "custom"})
                    raise GenerationError(
                        f"Model invocation failed: {e}", details={"model": type(model).__name__}
                    ) from e
                provider_used = provider or "custom"
                model_used = getattr(chat_model, "model_name", "custom")
            else:
                provider = provider or settings.llm_primary
                model_name = model or (
                    settings.google_model if provider == "gemini" else settings.openrouter_model
                )
                temperature = settings.google_temperature if temperature is None else temperature
                chat_model = self.factory.get_model(
                    provider, model=model_name, temperature=temperature
                )
                try:
                    log.info("llm_generate_attempt", provider=provider, model=model_name)
                    response = chat_model.invoke(convert_to_messages(messages))
                    log.info("llm_generate_success", provider=provider)
                    provider_used = provider
                    model_used = model_name
                except Exception as e:
                    metrics.increment("generate.errors", labels={"provider": provider})
                    log.warning("llm_generate_failed_primary", provider=provider, error=str(e))
                    # Fall back to the multi-provider chain (settings defaults).
                    response = invoke_with_fallback(
                        self.factory,
                        convert_to_messages(messages),
                    )
                    provider_used = getattr(response, "response_metadata", {}).get(
                        "provider", provider
                    )
                    model_used = getattr(response, "response_metadata", {}).get(
                        "model_name", model_name
                    )

            otel_trace.get_current_span().set_attributes({
                "provider": provider_used,
                "model": model_used,
                "question": question[:120],
            })

            answer = _response_text(response)
            completion_tokens = count_tokens(answer)
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            # Register citations and rewrite the answer to use inline markers.
            tracker = CitationTracker()
            for result in results:
                tracker.add(result)
            sources_markdown = tracker.format_references()

            result = GenerationResult(
                answer=answer,
                citations=tracker.reference_list(),
                usage=usage,
                cost_usd=estimate_cost(usage.prompt_tokens, usage.completion_tokens, model_used),
                provider=provider_used,
                model=model_used,
                sources_markdown=sources_markdown,
            )
            log.info(
                "llm_generate_complete",
                provider=provider_used,
                model=model_used,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=round(result.cost_usd, 6),
                citations=len(result.citations),
            )
            # Cache the result (TTL 24 hours)
            cache_set(cache_key, result, ttl=86400)
            return result
        finally:
            if owner:
                with self._inflight_lock:
                    self._inflight_locks.pop(cache_key, None)
                event.set()

    def _generate_structured(
        self,
        question: str,
        results: list,
        messages: list[dict[str, str]],
        prompt_tokens: int,
        output_schema: type[T],
        model_instance: Any = None,
    ) -> T:
        """Generate a structured (Pydantic) output using the primary provider."""
        from langchain_core.messages import convert_to_messages

        if model_instance is not None and not isinstance(model_instance, str):
            structured_model = model_instance.with_structured_output(output_schema)
        else:
            chat_model = self.factory.get_model(settings.llm_primary)
            structured_model = chat_model.with_structured_output(output_schema)
        try:
            response = structured_model.invoke(convert_to_messages(messages))
        except Exception as e:
            raise GenerationError(
                f"Structured generation failed: {e}", details={"schema": output_schema.__name__}
            ) from e

        # Still track citations for bookkeeping even though output is structured.
        tracker = CitationTracker()
        for result in results:
            tracker.add(result)
        log.info(
            "llm_generate_structured",
            schema=output_schema.__name__,
            citations=len(tracker.reference_list()),
        )
        return cast(T, response)
