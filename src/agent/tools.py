"""Tools for the LangGraph agent."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.generation.chain import RAGGenerator
from src.retrieval.retriever import Retriever

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Retrieve Tool
# ---------------------------------------------------------------------------


class RetrieveInput(BaseModel):
    """Input for the retrieve tool."""

    query: str = Field(description="The search query")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    search_type: str = Field(default="hybrid", description="Search type: dense, sparse, or hybrid")
    score_threshold: float | None = Field(default=None, description="Minimum score threshold")
    tenant_id: str = Field(default="default", description="Tenant ID for isolation")


class RetrieveTool(BaseTool):
    """Retrieve relevant documents from the vector store."""

    name: str = "retrieve"
    description: str = (
        "Search the knowledge base for documents relevant to the query. "
        "Returns a list of document chunks with scores and metadata."
    )
    args_schema: type[BaseModel] = RetrieveInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._retriever = Retriever()

    def _run(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        score_threshold: float | None = None,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        from src.retrieval.models import MetadataFilter, SearchType

        log.info("tool_retrieve", query=query, top_k=top_k, search_type=search_type)

        stype = SearchType(search_type)
        filter_obj = MetadataFilter(tenant_id=tenant_id)

        results = self._retriever.search(
            query,
            search_type=stype,
            top_k=top_k,
            score_threshold=score_threshold,
            filter=filter_obj,
        )

        return [
            {
                "content": r.content,
                "score": r.score,
                "document_id": str(r.document_id) if r.document_id else None,
                "chunk_id": r.chunk_index,
                "source": r.source,
                "section_header": r.section_header,
                "page_number": r.page_number,
                "metadata": r.payload,
            }
            for r in results
        ]

    async def _arun(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Rerank Tool
# ---------------------------------------------------------------------------


class RerankInput(BaseModel):
    """Input for the rerank tool."""

    query: str = Field(description="The search query")
    documents: list[dict[str, Any]] = Field(description="Documents to rerank")
    top_k: int = Field(default=5, description="Number of top documents to return")


class RerankTool(BaseTool):
    """Rerank documents using a cross-encoder model."""

    name: str = "rerank"
    description: str = (
        "Rerank a list of documents by relevance to the query using a cross-encoder. "
        "Returns the top-k most relevant documents."
    )
    args_schema: type[BaseModel] = RerankInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._reranker = None
        self._init_reranker()

    def _init_reranker(self) -> None:
        """Initialize the reranker."""
        try:
            from src.rerank.reranker import Reranker  # type: ignore[import-not-found]

            self._reranker = Reranker()
        except Exception as e:
            log.warning("reranker_unavailable", error=str(e))
            self._reranker = None

    def _run(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not self._reranker:
            log.warning("reranker_not_available_returning_original")
            return documents[:top_k]

        log.info("tool_rerank", query=query, doc_count=len(documents), top_k=top_k)

        contents = [d["content"] for d in documents]
        scores = self._reranker.rerank(query, contents, top_k=top_k)

        # Combine documents with scores
        reranked = []
        for idx, score in scores:
            doc = documents[idx].copy()
            doc["rerank_score"] = score
            reranked.append(doc)

        return reranked

    async def _arun(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Generate Tool
# ---------------------------------------------------------------------------


class GenerateInput(BaseModel):
    """Input for the generate tool."""

    question: str = Field(description="The question to answer")
    context: list[dict[str, Any]] = Field(description="Context documents with content and metadata")
    provider: str | None = Field(default=None, description="LLM provider: gemini, openrouter, groq")
    model: str | None = Field(default=None, description="Model name")
    temperature: float | None = Field(default=None, description="Temperature for generation")


class GenerateTool(BaseTool):
    """Generate an answer using the RAG pipeline."""

    name: str = "generate"
    description: str = (
        "Generate an answer to the question using the provided context documents. "
        "Returns the answer with citations."
    )
    args_schema: type[BaseModel] = GenerateInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._generator = RAGGenerator()

    def _run(
        self,
        question: str,
        context: list[dict[str, Any]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        log.info("tool_generate", question=question, context_count=len(context))

        # Convert context to SearchResult-like objects for the generator
        from uuid import UUID

        from src.retrieval.models import SearchResult

        search_results = []
        for i, ctx in enumerate(context):
            score = ctx.get("score")
            if score is not None:
                score = float(score)
            search_results.append(
                SearchResult(
                    id=ctx.get("chunk_id", f"chunk-{i}"),
                    document_id=UUID(ctx["document_id"]) if ctx.get("document_id") else None,
                    content=ctx["content"],
                    score=0.0,
                    source=ctx.get("source"),
                    section_header=ctx.get("section_header"),
                    page_number=ctx.get("page_number"),
                    payload=ctx.get("metadata", {}),
                )
            )

        result = self._generator.generate(
            question,
            search_results,
            provider=provider,
            model=model,
            temperature=temperature,
        )

        return {
            "answer": result.answer,
            "citations": [
                {
                    "key": c.key,
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "content": c.content,
                    "source": c.source,
                    "section_header": c.section_header,
                    "page_number": c.page_number,
                    "score": c.score,
                }
                for c in result.citations
            ],
            "sources_markdown": result.sources_markdown,
            "provider": result.provider,
            "model": result.model,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            "cost_usd": result.cost_usd,
        }

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Reflect Tool
# ---------------------------------------------------------------------------


class ReflectInput(BaseModel):
    """Input for the reflect tool."""

    question: str = Field(description="The original question")
    answer: str = Field(description="The answer to reflect on")
    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Citations in the answer",
    )
    context: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Context used for the answer",
    )


class ReflectTool(BaseTool):
    """Reflect on the quality of an answer and suggest improvements."""

    name: str = "reflect"
    description: str = (
        "Reflect on the quality and accuracy of an answer. "
        "Identifies potential issues like hallucinations, missing citations, "
        "or incomplete answers. Returns a reflection with suggested improvements."
    )
    args_schema: type[BaseModel] = ReflectInput

    def _run(
        self,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        log.info("tool_reflect", question=question, answer_length=len(answer))

        # Heuristic checks
        issues = []
        suggestions = []

        # Check for empty answer
        if not answer.strip():
            issues.append("Answer is empty")
            suggestions.append("Generate a substantive answer based on the context")

        # Check for citations
        if not citations:
            issues.append("No citations provided")
            suggestions.append("Add inline citations [doc-X] for all factual claims")

        # Check if answer addresses the question
        question_keywords = set(question.lower().split())
        answer_keywords = set(answer.lower().split())
        overlap = len(question_keywords & answer_keywords) / max(len(question_keywords), 1)
        if overlap < 0.2:
            issues.append("Answer may not address the question")
            suggestions.append("Ensure the answer directly addresses the user's question")

        # Check for "I don't know" patterns
        uncertain_phrases = [
            "i don't know",
            "i cannot",
            "unable to",
            "not enough information",
            "insufficient context",
        ]
        if any(phrase in answer.lower() for phrase in uncertain_phrases):
            issues.append("Answer expresses uncertainty")
            suggestions.append("Consider retrieving more context or rephrasing the query")

        # Check citation quality
        if citations:
            cited_docs = {c.get("document_id") for c in citations if c.get("document_id")}
            if len(cited_docs) == 1 and len(citations) > 3:
                suggestions.append("Consider citing multiple sources for better credibility")

        return {
            "quality_score": max(0.0, 1.0 - len(issues) * 0.2),
            "issues": issues,
            "suggestions": suggestions,
            "needs_improvement": len(issues) > 0,
            "citation_count": len(citations),
            "unique_sources_cited": len(cited_docs),
        }

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Calculator Tool
# ---------------------------------------------------------------------------


class CalculatorInput(BaseModel):
    """Input for the calculator tool."""

    expression: str = Field(description="Mathematical expression to evaluate")


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely."""

    name: str = "calculator"
    description: str = (
        "Evaluate a mathematical expression. Supports +, -, *, /, **, (), and basic functions."
    )
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> dict[str, Any]:
        import ast
        import operator

        log.info("tool_calculator", expression=expression)

        # Safe evaluation using AST
        allowed_operators: dict[type, Any] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def eval_node(node: ast.AST) -> float:
            if isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, (int, float)):
                    return float(value)
                raise ValueError(f"Unsupported constant type: {type(value)}")
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError(f"Unsupported operator: {op_type}")
                return cast(float, allowed_operators[op_type](left, right))
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                unary_op_type: type = type(node.op)
                if unary_op_type not in allowed_operators:
                    raise ValueError(f"Unsupported unary operator: {unary_op_type}")
                return cast(float, allowed_operators[unary_op_type](operand))
            else:
                raise ValueError(f"Unsupported AST node: {type(node)}")

        try:
            tree = ast.parse(expression, mode="eval")
            result = eval_node(tree.body)
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e), "expression": expression}

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

_tools_cache: list[BaseTool] | None = None


def get_all_tools() -> list[BaseTool]:
    """Get all available tools for the agent (lazy initialization)."""
    global _tools_cache
    if _tools_cache is None:
        _tools_cache = [
            RetrieveTool(),
            RerankTool(),
            GenerateTool(),
            ReflectTool(),
            CalculatorTool(),
        ]
    return _tools_cache


def get_tool_by_name(name: str) -> BaseTool | None:
    """Get a tool by name."""
    for tool in get_all_tools():
        if tool.name == name:
            return tool
    return None
