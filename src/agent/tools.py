"""Tools for the LangGraph agent."""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
from typing import Any, cast

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.core.config import settings
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
# Summarize Tool
# ---------------------------------------------------------------------------


class SummarizeInput(BaseModel):
    """Input for the summarize tool."""

    text: str = Field(description="Text to summarize")
    max_length: int = Field(default=200, description="Maximum length of summary")
    focus: str | None = Field(default=None, description="Focus area for summarization")


class SummarizeTool(BaseTool):
    """Summarize text using an LLM."""

    name: str = "summarize"
    description: str = (
        "Summarize a long text into a concise summary. "
        "Uses an LLM for abstractive summarization with optional focus area."
    )
    args_schema: type[BaseModel] = SummarizeInput

    def _run(
        self,
        text: str,
        max_length: int = 200,
        focus: str | None = None,
    ) -> dict[str, Any]:
        log.info("tool_summarize", text_length=len(text), max_length=max_length, focus=focus)

        if len(text) <= max_length:
            return {"summary": text, "original_length": len(text), "summary_length": len(text)}

        # Use LLM for abstractive summarization
        from src.generation.chain import RAGGenerator

        generator = RAGGenerator()
        prompt = f"""Summarize the following text in at most {max_length} characters.
{f"Focus on: {focus}." if focus else ""}

Text:
{text[:8000]}  # Limit input length

Summary:"""

        response = generator.generate(
            prompt,
            [],  # No retrieval needed
            temperature=0.3,
        )

        summary = response.answer
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
        }

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Web Search Tool
# ---------------------------------------------------------------------------


class WebSearchInput(BaseModel):
    """Input for the web search tool."""

    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results")
    source: str = Field(default="auto", description="Search source: auto, serpapi, brave, google")


class WebSearchTool(BaseTool):
    """Search the web for current information using multiple providers.

    Supports SerpAPI, Brave Search, and Google Custom Search.
    Automatically falls back between providers if one fails.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for current information not in the knowledge base. "
        "Returns search results with snippets and URLs. "
        "Supports SerpAPI, Brave Search, and Google Custom Search."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _search_serpapi(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Search using SerpAPI."""
        if not settings.serpapi_key:
            return []
        try:
            params: dict[str, str | int] = {
                "q": query,
                "api_key": settings.serpapi_key,
                "num": max_results,
                "engine": "google",
            }
            response = await self.client.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()
            organic = data.get("organic_results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "url": r.get("link", ""),
                    "source": "serpapi",
                }
                for r in organic[:max_results]
            ]
        except Exception as e:
            log.warning("serpapi_search_failed", error=str(e))
            return []

    async def _search_brave(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Search using Brave Search API."""
        if not settings.brave_search_key:
            return []
        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": settings.brave_search_key,
            }
            params: dict[str, str | int] = {"q": query, "count": max_results}
            response = await self.client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            web = data.get("web", {}).get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("description", ""),
                    "url": r.get("url", ""),
                    "source": "brave",
                }
                for r in web[:max_results]
            ]
        except Exception as e:
            log.warning("brave_search_failed", error=str(e))
            return []

    async def _search_google(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Search using Google Custom Search API."""
        if not settings.google_search_key or not settings.google_search_cx:
            return []
        try:
            params: dict[str, str | int] = {
                "key": settings.google_search_key,
                "cx": settings.google_search_cx,
                "q": query,
                "num": min(max_results, 10),
            }
            response = await self.client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "url": r.get("link", ""),
                    "source": "google",
                }
                for r in items[:max_results]
            ]
        except Exception as e:
            log.warning("google_search_failed", error=str(e))
            return []

    async def _run_async(
        self,
        query: str,
        max_results: int = 5,
        source: str = "auto",
    ) -> dict[str, Any]:
        log.info("tool_web_search", query=query, max_results=max_results, source=source)

        # Determine which providers to use
        providers = []
        if source == "auto":
            providers = ["serpapi", "brave", "google"]
        elif source == "serpapi":
            providers = ["serpapi"]
        elif source == "brave":
            providers = ["brave"]
        elif source == "google":
            providers = ["google"]
        else:
            providers = ["serpapi", "brave", "google"]

        all_results: list[dict[str, Any]] = []

        for provider in providers:
            if provider == "serpapi":
                results = await self._search_serpapi(query, max_results)
            elif provider == "brave":
                results = await self._search_brave(query, max_results)
            elif provider == "google":
                results = await self._search_google(query, max_results)
            else:
                continue

            if results:
                all_results.extend(results)
                if len(all_results) >= max_results:
                    break

            # Small delay between providers to be respectful
            await asyncio.sleep(0.1)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        return {
            "query": query,
            "results": unique_results[:max_results],
            "sources_used": list({r["source"] for r in unique_results}),
        }

    def _run(
        self,
        query: str,
        max_results: int = 5,
        source: str = "auto",
    ) -> dict[str, Any]:
        """Sync wrapper for async search."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._run_async(query, max_results, source))

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        source: str = "auto",
    ) -> dict[str, Any]:
        return await self._run_async(query, max_results, source)


# ---------------------------------------------------------------------------
# Code Execution Tool
# ---------------------------------------------------------------------------


class CodeExecutionInput(BaseModel):
    """Input for the code execution tool."""

    code: str = Field(description="Python code to execute")
    timeout: int = Field(default=30, description="Execution timeout in seconds")


class CodeExecutionTool(BaseTool):
    """Execute Python code in a sandboxed environment."""

    name: str = "code_execution"
    description: str = (
        "Execute Python code and return the result. "
        "Useful for calculations, data processing, or algorithmic tasks. "
        "Execution is sandboxed and timed out after 30 seconds."
    )
    args_schema: type[BaseModel] = CodeExecutionInput

    def _run(self, code: str, timeout: int = 30) -> dict[str, Any]:
        import subprocess
        import sys
        import tempfile

        log.info("tool_code_execution", code_length=len(code))

        # Create a sandboxed execution environment
        # Use a restricted Python environment
        sandbox_code = f"""
import sys
import builtins

# Restrict builtins for safety
_allowed_builtins = {{
    'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
    'sum', 'min', 'max', 'abs', 'round', 'sorted', 'reversed',
    'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool',
    'isinstance', 'hasattr', 'getattr', 'setattr', 'type',
    'Exception', 'ValueError', 'TypeError', 'IndexError', 'KeyError',
    'StopIteration', 'ArithmeticError', 'ZeroDivisionError',
}}

# Replace builtins
import builtins
original_builtins = dict(builtins.__dict__)
for name in list(builtins.__dict__.keys()):
    if name not in _allowed_builtins:
        delattr(builtins, name)

# Execute user code
try:
    exec('''{code}''', {{}})
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
finally:
    # Restore builtins
    builtins.__dict__.clear()
    builtins.__dict__.update(original_builtins)
"""

        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sandbox_code)
            temp_path = f.name

        try:
            # Run with timeout
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Execution timed out after {timeout} seconds", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
        finally:
            # Clean up temp file
            with contextlib.suppress(OSError):
                pathlib.Path(temp_path).unlink()

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
            SummarizeTool(),
            WebSearchTool(),
            CodeExecutionTool(),
        ]
    return _tools_cache


def get_tool_by_name(name: str) -> BaseTool | None:
    """Get a tool by name."""
    for tool in get_all_tools():
        if tool.name == name:
            return tool
    return None


# ---------------------------------------------------------------------------
# Dynamic Tool Selection
# ---------------------------------------------------------------------------


def select_tools_for_query(
    query: str, available_tools: list[BaseTool] | None = None
) -> list[BaseTool]:
    """Select relevant tools based on the query.

    Uses simple keyword matching to determine which tools are relevant.
    In production, this could use an LLM for more sophisticated selection.
    """
    if available_tools is None:
        available_tools = get_all_tools()

    selected = []

    # Always include retrieve and generate for RAG queries
    retrieve_tool = next((t for t in available_tools if t.name == "retrieve"), None)
    generate_tool = next((t for t in available_tools if t.name == "generate"), None)
    if retrieve_tool:
        selected.append(retrieve_tool)
    if generate_tool:
        selected.append(generate_tool)

    # Add rerank for complex queries
    complex_words = ["compare", "analyze", "evaluate", "rank", "best", "top"]
    if any(word in query.lower() for word in complex_words):
        rerank_tool = next((t for t in available_tools if t.name == "rerank"), None)
        if rerank_tool:
            selected.append(rerank_tool)

    # Add reflect for complex queries
    if "?" in query and len(query) > 50:
        reflect_tool = next((t for t in available_tools if t.name == "reflect"), None)
        if reflect_tool:
            selected.append(reflect_tool)

    # Add calculator for math queries
    math_ops = ["+", "-", "*", "/", "=", "calculate", "compute", "math"]
    if any(op in query for op in math_ops):
        calc_tool = next((t for t in available_tools if t.name == "calculator"), None)
        if calc_tool:
            selected.append(calc_tool)

    # Add summarize for long content requests
    summarize_words = ["summarize", "summary", "tl;dr", "brief"]
    if any(word in query.lower() for word in summarize_words):
        summarize_tool = next((t for t in available_tools if t.name == "summarize"), None)
        if summarize_tool:
            selected.append(summarize_tool)

    # Add web search for current events or external info
    current_words = ["latest", "recent", "news", "current", "2024", "2025"]
    if any(word in query.lower() for word in current_words):
        web_search_tool = next((t for t in available_tools if t.name == "web_search"), None)
        if web_search_tool:
            selected.append(web_search_tool)

    # Add code execution for programming tasks
    code_words = ["code", "python", "script", "algorithm", "function", "debug"]
    if any(word in query.lower() for word in code_words):
        code_tool = next((t for t in available_tools if t.name == "code_execution"), None)
        if code_tool:
            selected.append(code_tool)

    # Remove duplicates while preserving order
    seen = set()
    unique_selected = []
    for tool in selected:
        if tool.name not in seen:
            seen.add(tool.name)
            unique_selected.append(tool)

    return unique_selected


# ---------------------------------------------------------------------------
# LLM-based Dynamic Tool Selection (Production-ready)
# ---------------------------------------------------------------------------


def select_tools_for_query_llm(
    query: str,
    available_tools: list[BaseTool] | None = None,
) -> list[BaseTool]:
    """Select relevant tools using an LLM for intelligent tool selection.

    This is the production-ready version that uses an LLM to intelligently
    select tools based on the query, rather than simple keyword matching.
    """
    if available_tools is None:
        available_tools = get_all_tools()

    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in available_tools)

    prompt = f"""Given the user query, select the most relevant tools from those available.

User Query: {query}

Available Tools:
{tool_descriptions}

Select the most relevant tools (return tool names only, one per line).
Consider:
- Does the query need external information? (web_search)
- Does it need calculations? (calculator)
- Does it need code execution? (code_execution)
- Does it need document retrieval? (retrieve)
- Does it need reranking? (rerank)
- Does it need summarization? (summarize)
- Does it need reflection/quality check? (reflect)
- Does it need answer generation? (generate)

Tool names (one per line):"""

    from src.generation.chain import RAGGenerator

    generator = RAGGenerator()
    response = generator.generate(
        prompt,
        [],
        temperature=0.1,
    )

    # Parse selected tool names
    selected_names = [
        line.strip()
        for line in response.answer.split("\n")
        if line.strip() and line.strip() in [t.name for t in available_tools]
    ]

    # Map names to tools
    tool_map = {t.name: t for t in available_tools}
    selected = [tool_map[name] for name in selected_names if name in tool_map]

    # Always include retrieve and generate for RAG
    for tool in available_tools:
        if tool.name in ("retrieve", "generate") and tool not in selected:
            selected.insert(0, tool)  # Put at front

    return selected
