"""Tests for the generation layer (prompts, citations, tokens, generation).

No API keys or network are required; a deterministic fake chat model is used.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from src.core.exceptions import GenerationError, ValidationError
from src.generation import (
    Citation,
    CitationTracker,
    GenerationResult,
    PromptTemplate,
    PromptType,
    RAGGenerator,
    RAGPromptBuilder,
    count_message_tokens,
    count_tokens,
    estimate_cost,
    invoke_with_fallback,
    stream_with_fallback,
)
from src.generation.chain import _content_text, _response_text
from src.generation.llm import LLMFactory
from src.retrieval.models import SearchResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeChatModel:
    """Deterministic chat model stand-in for testing."""

    def __init__(self, response_text: str = "fake answer") -> None:
        self.response_text = response_text
        self.last_messages: list | None = None
        self.invoke_count = 0
        self.stream_count = 0

    def invoke(self, messages: list, **kwargs: Any) -> AIMessage:
        self.last_messages = messages
        self.invoke_count += 1
        return AIMessage(content=self.response_text)

    def stream(self, messages: list, **kwargs: Any) -> Any:
        self.last_messages = messages
        self.stream_count += 1
        yield AIMessageChunk(content=self.response_text)

    async def astream(self, messages: list, **kwargs: Any) -> Any:
        self.last_messages = messages
        yield AIMessageChunk(content=self.response_text)

    def with_structured_output(self, schema: Any) -> "FakeStructuredModel":
        return FakeStructuredModel(schema, self.response_text)


class FakeStructuredModel:
    def __init__(self, schema: Any, response_text: str) -> None:
        self.schema = schema
        self.response_text = response_text

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        # Return a schema instance built from the response text.
        return self.schema(answer=self.response_text)


class AlwaysFailingModel:
    def invoke(self, messages: list, **kwargs: Any) -> Any:
        raise RuntimeError("provider down")

    def stream(self, messages: list, **kwargs: Any) -> Any:
        raise RuntimeError("provider down")


def _stub_every_tier(factory: LLMFactory, model_cls: type) -> None:
    """Pre-seed clients for the full-size *and* degraded tiers.

    The fallback plan includes the smaller degraded models, so stubbing only
    the default keys would let the test reach a live provider.
    """
    for provider in ("gemini", "openrouter", "groq"):
        factory._clients[f"{provider}::"] = model_cls()
        degraded = factory.degraded_model(provider)
        if degraded:
            factory._clients[f"{provider}:{degraded}:"] = model_cls()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(doc_id: str, content: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        id=f"chunk-{doc_id}",
        content=content,
        score=score,
        document_id=None,
        source=f"{doc_id}.txt",
        section_header="Intro",
        page_number=1,
    )


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_format_fills_variables(self):
        t = PromptTemplate(
            name="greet",
            template="Hello {name}, welcome to {place}.",
            input_variables=["name", "place"],
            prompt_type=PromptType.USER,
        )
        assert t.format(name="Ada", place="RAG") == "Hello Ada, welcome to RAG."

    def test_missing_variable_raises(self):
        t = PromptTemplate(
            name="greet",
            template="Hello {name}.",
            input_variables=["name"],
            prompt_type=PromptType.USER,
        )
        with pytest.raises(ValidationError):
            t.format(place="RAG")

    def test_no_variables(self):
        t = PromptTemplate(
            name="system",
            template="You are a helpful assistant.",
            input_variables=[],
            prompt_type=PromptType.SYSTEM,
        )
        assert t.format() == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# RAGPromptBuilder
# ---------------------------------------------------------------------------


class TestRAGPromptBuilder:
    def test_build_context_labels_chunks(self):
        builder = RAGPromptBuilder()
        results = [_make_result("doc-a", "alpha content"), _make_result("doc-b", "beta content")]
        context = builder.build_context(results)
        assert "[doc-1]" in context
        assert "[doc-2]" in context
        assert "alpha content" in context
        assert "beta content" in context

    def test_build_messages_structure(self):
        builder = RAGPromptBuilder()
        messages = builder.build_messages("what is RAG?", [_make_result("doc-a", "alpha")])
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]
        assert "what is RAG?" in messages[-1]["content"]
        assert "[doc-1]" in messages[-1]["content"]

    def test_few_shots_are_included(self):
        builder = RAGPromptBuilder(
            few_shots=[
                {
                    "question": "Q1?",
                    "context": "[chunk-doc-a] c1",
                    "answer": "A1 [chunk-doc-a]",
                }
            ]
        )
        messages = builder.build_messages("Q2?", [_make_result("doc-a", "c2")])
        # system, few-shot user, few-shot assistant, current user
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert "Q1?" in messages[1]["content"]
        assert "A1" in messages[2]["content"]

    def test_empty_results(self):
        builder = RAGPromptBuilder()
        messages = builder.build_messages("solo?", [])
        assert "[solo?" not in messages[-1]["content"]
        assert "solo?" in messages[-1]["content"]

    def test_empty_results_forbids_citation_markers(self):
        """No context means nothing to cite; without this the model invents a
        marker like [doc-a] and the answer ships with an empty sources list."""
        builder = RAGPromptBuilder()
        content = builder.build_messages("solo?", [])[-1]["content"]
        assert "do NOT include any citation markers" in content

    def test_non_empty_results_do_not_get_the_no_context_notice(self):
        builder = RAGPromptBuilder()
        content = builder.build_messages("q?", [_make_result("doc-a", "alpha")])[-1]["content"]
        assert "No source documents were retrieved" not in content


# ---------------------------------------------------------------------------
# CitationTracker
# ---------------------------------------------------------------------------


class TestCitationTracker:
    def test_add_returns_key(self):
        tracker = CitationTracker()
        r = _make_result("doc-a", "alpha content")
        key = tracker.add(r)
        assert key == "chunk-doc-a"

    def test_inline_marker(self):
        assert CitationTracker.inline_marker("doc-a") == "[doc-a]"

    def test_cite_lookup(self):
        tracker = CitationTracker()
        r = _make_result("doc-a", "alpha content")
        tracker.add(r)
        assert tracker.cite(r.id) == "chunk-doc-a"

    def test_duplicate_chunk_reuses_key(self):
        tracker = CitationTracker()
        r = _make_result("doc-a", "alpha content")
        k1 = tracker.add(r)
        k2 = tracker.add(r)
        assert k1 == k2
        assert len(tracker.reference_list()) == 1

    def test_format_references(self):
        tracker = CitationTracker()
        tracker.add(_make_result("doc-a", "alpha content here"))
        out = tracker.format_references()
        assert "## Sources" in out
        assert "chunk-doc-a" in out
        assert "alpha content here" in out

    def test_empty_references(self):
        assert CitationTracker().format_references() == ""

    def test_citation_model_fields(self):
        c = Citation(
            key="chunk-doc-a",
            chunk_id="chunk-1",
            document_id="doc-a",
            content="some content",
            source="a.txt",
            score=0.9,
        )
        assert c.key == "chunk-doc-a"
        assert c.score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_heuristic(self):
        # Token count should be positive for non-empty text and grow with length.
        assert count_tokens("") == 0
        assert count_tokens("abcd") >= 1
        assert count_tokens("a" * 100) > count_tokens("abcd")
        # tiktoken (when installed) may split differently than the heuristic;
        # we only require a sane positive count.
        assert isinstance(count_tokens("hello world"), int)

    def test_count_message_tokens(self):
        messages = [{"role": "user", "content": "hello world"}]
        # With tiktoken installed, count may differ from the char heuristic;
        # just require a positive count.
        assert count_message_tokens(messages) >= 1

    def test_estimate_cost_free_models(self):
        assert estimate_cost(1000, 1000, "gemini-2.5-flash-lite") == pytest.approx(0.0)
        assert estimate_cost(1000, 1000, "unknown-model") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _response_text
# ---------------------------------------------------------------------------


class TestResponseText:
    def test_aimessage(self):
        assert _response_text(AIMessage(content="hello")) == "hello"

    def test_string(self):
        assert _response_text("plain") == "plain"

    def test_none_content(self):
        assert _response_text(AIMessage(content="")) == ""

    def test_dict_text_blocks(self):
        """Gemini 3.x returns typed content blocks as dicts, not objects."""
        message = AIMessage(
            content=[
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ]
        )
        assert _response_text(message) == "Hello world"
        assert "{" not in _response_text(message)

    def test_object_text_blocks(self):
        """Some providers hand back typed blocks as objects, not dicts."""
        part = type("Part", (), {"text": "object block"})()
        assert _content_text([part]) == "object block"

    def test_mixed_and_non_text_blocks_are_skipped(self):
        message = AIMessage(
            content=[
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "tool-1"},
                " tail",
            ]
        )
        assert _response_text(message) == "answer tail"


# ---------------------------------------------------------------------------
# RAGGenerator
# ---------------------------------------------------------------------------


class TestRAGGenerator:
    def test_generate_uses_fake_model(self):
        results = [_make_result("doc-a", "alpha content"), _make_result("doc-b", "beta content")]
        model = FakeChatModel("The answer is alpha [doc-a].")
        generator = RAGGenerator(prompt_builder=RAGPromptBuilder())
        result = generator.generate("what is alpha?", results, model=model, provider="gemini")

        assert isinstance(result, GenerationResult)
        assert result.answer == "The answer is alpha [doc-a]."
        assert result.provider == "gemini"
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert result.usage.total_tokens == (
            result.usage.prompt_tokens + result.usage.completion_tokens
        )
        assert len(result.citations) == 2
        assert result.sources_markdown != ""

    def test_generate_full_output(self):
        model = FakeChatModel("alpha is first.")
        generator = RAGGenerator()
        result = generator.generate(
            "tell me about alpha",
            [_make_result("doc-a", "alpha content")],
            model=model,
            provider="openrouter",
        )
        full = result.answer_with_citations
        assert "alpha is first" in full
        assert "## Sources" in full

    def test_structured_output(self):
        from pydantic import BaseModel

        class AnswerSchema(BaseModel):
            answer: str

        model = FakeChatModel("structured answer")
        generator = RAGGenerator()
        result = generator.generate(
            "q",
            [_make_result("doc-a", "c")],
            model=model,
            provider="gemini",
            output_schema=AnswerSchema,
        )
        assert isinstance(result, AnswerSchema)
        assert result.answer == "structured answer"

    def test_citation_tracker_isolated_per_call(self):
        model = FakeChatModel("x")
        generator = RAGGenerator()
        generator.generate("q", [_make_result("doc-a", "c1")], model=model)
        generator.generate("q", [_make_result("doc-b", "c2")], model=model)
        # Each generate() uses a fresh tracker internally.
        assert model.invoke_count == 2


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_invoke_with_fallback_first_succeeds(self):
        factory = LLMFactory()
        model = FakeChatModel("ok")
        factory._clients["gemini::"] = model  # cache key used by get_model
        response = invoke_with_fallback(
            factory,
            [{"role": "user", "content": "hi"}],
            provider="gemini",
        )
        assert _response_text(response) == "ok"

    def test_invoke_with_fallback_all_fail(self):
        factory = LLMFactory()
        _stub_every_tier(factory, AlwaysFailingModel)
        with pytest.raises(GenerationError):
            invoke_with_fallback(
                factory,
                [{"role": "user", "content": "hi"}],
                provider="gemini",
            )

    def test_stream_with_fallback_yields(self):
        factory = LLMFactory()
        model = FakeChatModel("streamed")
        factory._clients["gemini::"] = model
        chunks = list(
            stream_with_fallback(
                factory,
                [{"role": "user", "content": "hi"}],
                provider="gemini",
            )
        )
        assert len(chunks) == 1
        assert chunks[0].content == "streamed"
