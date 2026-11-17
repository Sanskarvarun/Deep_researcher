"""Nodes for the LangGraph agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from src.agent.state import AgentState, PlanStep, create_initial_state
from src.agent.tools import (
    get_all_tools,
    get_tool_by_name,
)
from src.core.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool Node (lazy initialization)
# ---------------------------------------------------------------------------

_tool_node: ToolNode | None = None


def get_tool_node() -> ToolNode:
    """Get the tool node (lazy initialization)."""
    global _tool_node
    if _tool_node is None:
        _tool_node = ToolNode(get_all_tools())
    return _tool_node


# ---------------------------------------------------------------------------
# Plan Node - Create execution plan
# ---------------------------------------------------------------------------


def plan_node(state: AgentState) -> dict[str, Any]:
    """Create an execution plan based on the question."""
    question = state["question"]

    log.info("node_plan", question=question)

    # Simple heuristic plan based on question analysis
    plan = []

    # Always start with retrieval
    plan.append(
        PlanStep(
            step=1,
            action="retrieve",
            description="Retrieve relevant documents for the question",
            tool="retrieve",
            args={"query": state["question"], "top_k": 5},
        )
    )

    # Add rerank if complex query
    if len(question.split()) > 10 or any(
        word in question.lower() for word in ["compare", "analyze", "evaluate", "rank"]
    ):
        plan.append(
            PlanStep(
                step=2,
                action="rerank",
                description="Rerank retrieved documents for better relevance",
                tool="rerank",
                args={"query": state["question"], "top_k": 5},
            )
        )

    # Add generate
    next_step = len(plan) + 1
    plan.append(
        PlanStep(
            step=next_step,
            action="generate",
            description="Generate answer from retrieved context",
            tool="generate",
            args={"question": state["question"]},
        )
    )

    # Add reflection for complex queries
    if "?" in question and len(question) > 50:
        plan.append(
            PlanStep(
                step=next_step + 1,
                action="reflect",
                description="Reflect on answer quality and completeness",
                tool="reflect",
                args={"question": state["question"]},
            )
        )

    return {"plan": plan, "current_step": 0}


# ---------------------------------------------------------------------------
# Retrieve Node
# ---------------------------------------------------------------------------


def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Execute the retrieve tool."""
    log.info("node_retrieve")

    current_plan = state["plan"][state["current_step"]]
    args = current_plan.args or {}

    tool = get_tool_by_name("retrieve")
    if not tool:
        return {"error": "Retrieve tool not available"}

    result = tool.invoke(args)
    current_plan.status = "completed"
    current_plan.result = result

    return {
        "retrieved_docs": result,
        "plan": state["plan"],
        "current_step": state["current_step"] + 1,
    }


# ---------------------------------------------------------------------------
# Rerank Node
# ---------------------------------------------------------------------------


def rerank_node(state: AgentState) -> dict[str, Any]:
    """Execute the rerank tool."""
    log.info("node_rerank")

    current_plan = state["plan"][state["current_step"]]
    args = current_plan.args or {}
    args["documents"] = state["retrieved_docs"]

    tool = get_tool_by_name("rerank")
    if not tool:
        return {"error": "Rerank tool not available"}

    result = tool.invoke(args)
    current_plan.status = "completed"
    current_plan.result = result

    return {
        "reranked_docs": result,
        "plan": state["plan"],
        "current_step": state["current_step"] + 1,
    }


# ---------------------------------------------------------------------------
# Generate Node
# ---------------------------------------------------------------------------


def generate_node(state: AgentState) -> dict[str, Any]:
    """Execute the generate tool."""
    log.info("node_generate")

    current_plan = state["plan"][state["current_step"]]
    args = current_plan.args or {}

    # Use reranked docs if available, otherwise retrieved
    context = state["reranked_docs"] if state["reranked_docs"] else state["retrieved_docs"]
    args["context"] = context

    tool = get_tool_by_name("generate")
    if not tool:
        return {"error": "Generate tool not available"}

    result = tool.invoke(args)
    current_plan.status = "completed"
    current_plan.result = result

    return {
        "final_answer": result.get("answer"),
        "citations": result.get("citations", []),
        "metadata": {
            **state["metadata"],
            "generation": {
                "provider": result.get("provider"),
                "model": result.get("model"),
                "usage": result.get("usage"),
                "cost_usd": result.get("cost_usd"),
            },
        },
        "plan": state["plan"],
        "current_step": state["current_step"] + 1,
    }


# ---------------------------------------------------------------------------
# Reflect Node
# ---------------------------------------------------------------------------


def reflect_node(state: AgentState) -> dict[str, Any]:
    """Execute the reflect tool."""
    log.info("node_reflect")

    current_plan = state["plan"][state["current_step"]]
    args = current_plan.args or {}
    args["answer"] = state["final_answer"] or ""
    args["citations"] = state["citations"]
    args["context"] = state["reranked_docs"] if state["reranked_docs"] else state["retrieved_docs"]

    tool = get_tool_by_name("reflect")
    if not tool:
        return {"error": "Reflect tool not available"}

    result = tool.invoke(args)
    current_plan.status = "completed"
    current_plan.result = result

    # If reflection suggests improvement, we could loop back
    # For now, just record the reflection
    return {
        "reflection": result,
        "plan": state["plan"],
        "current_step": state["current_step"] + 1,
    }


# ---------------------------------------------------------------------------
# Tool Use Node (for Calculator and other tools)
# ---------------------------------------------------------------------------


def tool_use_node(state: AgentState) -> dict[str, Any]:
    """Execute a tool call from the LLM."""
    log.info("node_tool_use")

    # This node is called when the LLM decides to use a tool
    # The tool calls are in state["tool_calls"]
    tool_calls = state.get("tool_calls", [])

    if not tool_calls:
        return {}

    results = []
    for call in tool_calls:
        tool_name: str | None = call.get("name")
        if not tool_name:
            results.append({"error": "Tool call missing name", "call": call})
            continue
        args = call.get("args", {})

        tool = get_tool_by_name(tool_name)
        if not tool:
            results.append({"error": f"Tool {tool_name} not found", "call": call})
            continue

        try:
            result = tool.invoke(args)
            results.append({"tool": tool_name, "result": result, "call": call})
        except Exception as e:
            results.append({"tool": tool_name, "error": str(e), "call": call})

    return {
        "tool_results": results,
        "tool_calls": [],  # Clear after execution
    }


# ---------------------------------------------------------------------------
# Conditional Routing
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> str:
    """Decide whether to continue to the next step or end."""
    if state.get("error"):
        return END

    current_step = state["current_step"]
    plan = state["plan"]

    if current_step >= len(plan):
        return END

    # Check if current step is a tool that needs to be executed
    if current_step < len(plan):
        next_action = plan[current_step].action
        if next_action in ["retrieve", "rerank", "generate", "reflect", "tool_use"]:
            return next_action

    return END


def route_to_tool(state: AgentState) -> str:
    """Route to the appropriate tool node based on the current plan step."""
    if state.get("error"):
        return END

    current_step = state["current_step"]
    plan = state["plan"]

    if current_step >= len(plan):
        return END

    return plan[current_step].action


# ---------------------------------------------------------------------------
# Human Review Node (for Human-in-the-Loop)
# ---------------------------------------------------------------------------


def human_review_node(state: AgentState) -> dict[str, Any]:
    """Human review node - pauses execution for human approval.

    This node interrupts the graph execution and waits for human input.
    The human can approve, reject, or modify the generated answer.
    """
    log.info("node_human_review")

    # The interrupt will pause execution and wait for human input
    # The human can provide feedback via the interrupt value

    # Present the current answer for review
    review_data = {
        "question": state["question"],
        "answer": state.get("final_answer"),
        "citations": state.get("citations", []),
        "plan": state.get("plan", []),
    }

    # Interrupt and wait for human response
    human_response = interrupt(review_data)

    # Process human response
    action = human_response.get("action", "approve")  # approve, reject, modify
    feedback = human_response.get("feedback", "")
    modified_answer = human_response.get("modified_answer")

    if action == "approve":
        log.info("human_review_approved")
        return {"human_feedback": "approved", "final_answer": state["final_answer"]}
    elif action == "modify" and modified_answer:
        log.info("human_review_modified")
        return {"final_answer": modified_answer, "human_feedback": feedback}
    else:
        log.info("human_review_rejected", feedback=feedback)
        # Reject - send back to generate with feedback
        return {
            "human_feedback": f"rejected: {feedback}",
            "tool_calls": [
                {
                    "name": "generate",
                    "args": {"question": state["question"] + f"\n\nHuman feedback: {feedback}"},
                }
            ],
        }


def route_after_human_review(state: AgentState) -> str:
    """Route after human review - either back to generate or end."""
    feedback = str(state.get("human_feedback", ""))
    if feedback == "approved":
        return END
    elif feedback.startswith("rejected"):
        return "generate"
    return END


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------


def create_agent_graph(checkpointer: Any = None) -> Any:
    """Create the LangGraph agent graph with optional checkpointing."""
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]
    from langgraph.graph import StateGraph

    # Use SqliteSaver for checkpointing if not provided
    if checkpointer is None:
        checkpointer = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("tool_use", tool_use_node)
    workflow.add_node("human_review", human_review_node)

    # Set entry point
    workflow.set_entry_point("plan")

    # Add conditional edges
    workflow.add_conditional_edges(
        "plan",
        route_to_tool,
        {
            "retrieve": "retrieve",
            "rerank": "rerank",
            "generate": "generate",
            "reflect": "reflect",
            "tool_use": "tool_use",
            END: END,
        },
    )

    # Add edges from tool nodes back to routing
    workflow.add_conditional_edges(
        "retrieve",
        should_continue,
        {
            "rerank": "rerank",
            "generate": "generate",
            "reflect": "reflect",
            "tool_use": "tool_use",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "rerank",
        should_continue,
        {
            "generate": "generate",
            "reflect": "reflect",
            "tool_use": "tool_use",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "generate",
        should_continue,
        {
            "reflect": "reflect",
            "tool_use": "tool_use",
            "human_review": "human_review",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "rerank": "rerank",  # Could re-retrieve based on reflection
            "generate": "generate",  # Could regenerate
            "tool_use": "tool_use",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "tool_use",
        should_continue,
        {
            "retrieve": "retrieve",
            "rerank": "rerank",
            "generate": "generate",
            "reflect": "reflect",
            "tool_use": "tool_use",
            END: END,
        },
    )

    # Human review node goes to end or back to generate
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "generate": "generate",
            END: END,
        },
    )

    return workflow.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Agent Runner
# ---------------------------------------------------------------------------


class AgentRunner:
    """High-level runner for the agent."""

    def __init__(self) -> None:
        self.graph = create_agent_graph()

    def run(self, question: str, messages: list[Any] | None = None) -> dict[str, Any]:
        """Run the agent on a question."""
        initial_state = create_initial_state(question, messages)

        log.info("agent_run_started", question=question)

        try:
            final_state = self.graph.invoke(initial_state)

            return {
                "answer": final_state.get("final_answer"),
                "citations": final_state.get("citations", []),
                "reflection": final_state.get("reflection"),
                "plan": final_state.get("plan", []),
                "retrieved_docs": final_state.get("retrieved_docs", []),
                "reranked_docs": final_state.get("reranked_docs", []),
                "metadata": final_state.get("metadata", {}),
                "error": final_state.get("error"),
            }
        except Exception as e:
            log.error("agent_run_failed", question=question, error=str(e))
            return {
                "answer": None,
                "error": str(e),
                "citations": [],
                "plan": [],
            }

    async def arun(self, question: str, messages: list[Any] | None = None) -> dict[str, Any]:
        """Async version of run."""
        initial_state = create_initial_state(question, messages)

        log.info("agent_run_started", question=question)

        try:
            final_state = await self.graph.ainvoke(initial_state)

            return {
                "answer": final_state.get("final_answer"),
                "citations": final_state.get("citations", []),
                "reflection": final_state.get("reflection"),
                "plan": final_state.get("plan", []),
                "retrieved_docs": final_state.get("retrieved_docs", []),
                "reranked_docs": final_state.get("reranked_docs", []),
                "metadata": final_state.get("metadata", {}),
                "error": final_state.get("error"),
            }
        except Exception as e:
            log.error("agent_run_failed", question=question, error=str(e))
            return {
                "answer": None,
                "error": str(e),
                "citations": [],
                "plan": [],
            }


# Global instance
agent_runner = AgentRunner()
