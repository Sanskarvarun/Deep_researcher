"""Generation module - LLM calls, prompt templates, streaming, citations."""

from src.generation.chain import GenerationResult, RAGGenerator
from src.generation.citations import Citation, CitationTracker
from src.generation.llm import (
    LLMFactory,
    astream_with_fallback,
    invoke_with_fallback,
    llm_factory,
    stream_with_fallback,
)
from src.generation.prompts import PromptTemplate, PromptType, RAGPromptBuilder
from src.generation.tokens import (
    MODEL_COSTS,
    TokenUsage,
    count_message_tokens,
    count_tokens,
    estimate_cost,
)

__all__ = [
    "MODEL_COSTS",
    "Citation",
    "CitationTracker",
    "GenerationResult",
    "LLMFactory",
    "PromptTemplate",
    "PromptType",
    "RAGGenerator",
    "RAGPromptBuilder",
    "TokenUsage",
    "astream_with_fallback",
    "count_message_tokens",
    "count_tokens",
    "estimate_cost",
    "invoke_with_fallback",
    "llm_factory",
    "stream_with_fallback",
]
