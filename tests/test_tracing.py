"""Tests for OpenTelemetry tracing / correlation IDs (Phase 2.1 Observability)."""

from src.core.tracing import (
    current_span_context,
    setup_tracing,
    trace_embed,
    trace_generate,
    trace_rerank,
    trace_retrieve,
    trace_span,
    traced,
)


class TestCurrentSpanContext:
    def test_empty_outside_span(self):
        assert current_span_context() == {}

    def test_populated_inside_span(self):
        setup_tracing()
        with trace_span("retrieve", query="what is RAG", top_k=5):
            ctx = current_span_context()
        assert "trace_id" in ctx
        assert ctx["trace_id"]
        assert ctx.get("span_id")

    def test_context_cleared_after_span(self):
        setup_tracing()
        with trace_span("generate"):
            pass
        assert current_span_context() == {}


class TestSpanHelpers:
    def test_pipeline_helpers_exist(self):
        # All four pipeline stages are available as span helpers.
        assert callable(trace_embed)
        assert callable(trace_retrieve)
        assert callable(trace_rerank)
        assert callable(trace_generate)

    def test_helpers_create_spans(self):
        setup_tracing()
        with trace_embed(model="bge-large", count=3):
            ctx = current_span_context()
            assert ctx["trace_id"]
        with trace_retrieve(query="x", top_k=5):
            assert current_span_context()["trace_id"]
        with trace_rerank(model="bge-reranker"):
            assert current_span_context()["trace_id"]
        with trace_generate(provider="gemini"):
            assert current_span_context()["trace_id"]


class TestTracedDecorator:
    def test_decorator_creates_span(self):
        setup_tracing()

        @traced(name="custom.op", version=1)
        def run(x: int) -> int:
            return x * 2

        assert run(21) == 42
        # The decorator spans the whole call; context is cleared after.
        assert current_span_context() == {}
