"""Agent state definition for the LangGraph agent."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from typing_extensions import TypedDict

from src.generation.citations import Citation
from src.retrieval.models import SearchResult


class PlanStep(BaseModel):
    """A single step in the execution plan."""

    step: int
    action: str  # "retrieve", "rerank", "generate", "reflect", "tool_use"
    description: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    result: Any = None
    error: str | None = None


class AgentState(TypedDict):
    """State for the LangGraph agent.

    Attributes:
        messages: Conversation history (user/assistant messages)
        question: The original user question
        retrieved_docs: Documents retrieved from the vector store
        reranked_docs: Documents after reranking
        citations: Citations generated during generation
        plan: Execution plan with steps
        current_step: Index of the current step being executed
        tool_calls: Pending tool calls
        tool_results: Results from tool executions
        reflection: Reflection on the current answer
        final_answer: The final generated answer
        metadata: Additional metadata (token usage, latency, etc.)
        error: Any error that occurred
    """

    messages: list[BaseMessage]
    question: str
    retrieved_docs: list[SearchResult]
    reranked_docs: list[SearchResult]
    citations: list[Citation]
    plan: list[PlanStep]
    current_step: int
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    reflection: str | None
    final_answer: str | None
    metadata: dict[str, Any]
    error: str | None


def create_initial_state(question: str, messages: list[BaseMessage] | None = None) -> AgentState:
    """Create an initial agent state."""
    return AgentState(
        messages=messages or [],
        question=question,
        retrieved_docs=[],
        reranked_docs=[],
        citations=[],
        plan=[],
        current_step=0,
        tool_calls=[],
        tool_results=[],
        reflection=None,
        final_answer=None,
        metadata={},
        error=None,
    )
