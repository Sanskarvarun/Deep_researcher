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


def _decompose_question(question: str) -> list[str]:
    """Decompose a complex question into sub-questions using an LLM.

    Returns a list of sub-questions (the original question is always first).
    Falls back to the original question alone when decomposition fails or the
    question is already atomic.
    """
    if len(question) < 40 or not any(
        marker in question for marker in [" and ", " then ", " after ", ","]
    ):
        return [question]

    prompt = f"""Break this question into the smallest set of factually answerable sub-questions.

Question: {question}

Rules:
- Only decompose when sub-questions have separate answers that must be combined
- Keep each sub-question self-contained and answerable on its own
- If the question is already atomic, return it unchanged
- Output one sub-question per line, no numbering or extra text
"""
    try:
        from src.generation.chain import RAGGenerator

        response = RAGGenerator().generate(prompt, [], temperature=0.0)
        lines = [ln.strip(" -•\t") for ln in response.answer.split("\n")]
        candidates = [
            ln for ln in lines if ln and not ln.lower().startswith(("sub-question", "answer:"))
        ]
        subs = candidates[:6]
        return subs or [question]
    except Exception as e:
        log.warning("question_decomposition_failed", error=str(e))
        return [question]


def _synthesize_plan(question: str, sub_questions: list[str]) -> list[PlanStep]:
    """Build an execution plan from the question and its sub-questions."""
    plan: list[PlanStep] = []

    for i, sub in enumerate(sub_questions):
        plan.append(
            PlanStep(
                step=i + 1,
                action="retrieve",
                description=f"Retrieve evidence for: {sub[:80]}",
                tool="retrieve",
                args={"query": sub, "top_k": 5},
            )
        )

    plan.append(
        PlanStep(
            step=len(plan) + 1,
            action="rerank",
            description="Rerank accumulated evidence",
            tool="rerank",
            args={"query": question, "top_k": 5},
        )
    )
    plan.append(
        PlanStep(
            step=len(plan) + 1,
            action="generate",
            description="Generate grounded answer from combined evidence",
            tool="generate",
            args={"question": question},
        )
    )
    plan.append(
        PlanStep(
            step=len(plan) + 1,
            action="reflect",
            description="Reflect on answer quality and evidence coverage",
            tool="reflect",
            args={"question": question},
        )
    )
    return plan


def plan_node(state: AgentState) -> dict[str, Any]:
    """Create an execution plan: decompose the question, then synthesize steps."""
    question = state["question"]

    log.info("node_plan", question=question)

    # LLM-driven decomposition into sub-questions, then plan synthesis.
    sub_questions = _decompose_question(question)
    log.info("question_decomposed", count=len(sub_questions))
    plan = _synthesize_plan(question, sub_questions)

    return {
        "plan": plan,
        "current_step": 0,
        "sub_questions": [
            {"id": f"sq-{i + 1}", "question": sub, "status": "pending"}
            for i, sub in enumerate(sub_questions)
        ],
        "hop": 0,
        "metadata": {
            **state.get("metadata", {}),
            "planning": {
                "sub_questions": sub_questions,
                "steps": len(plan),
                "llm_driven": True,
            },
        },
    }


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
    reflection_result = result

    # Check if multi-hop reasoning should continue
    hop = state.get("hop", 0)
    max_hops = state.get("max_hops", 3)
    reflection_data = result if isinstance(result, dict) else {}
    needs_more_info = hop < max_hops - 1 and reflection_data.get("needs_improvement", False)

    if needs_more_info:
        # Generate follow-up sub-questions based on reflection
        follow_up_subs = reflection_data.get("suggestions", [])
        if follow_up_subs:
            new_subs = [
                {
                    "id": f"sq-{len(state.get('sub_questions', [])) + i + 1}",
                    "question": sub,
                    "status": "pending",
                }
                for i, sub in enumerate(follow_up_subs[:3])
            ]
            # Extend the plan with new retrieval steps for follow-ups
            new_plan_steps = [
                PlanStep(
                    step=len(state.get("plan", [])) + i + 1,
                    action="retrieve",
                    description=f"Follow-up: {sub['question'][:80]}",
                    tool="retrieve",
                    args={"query": sub["question"], "top_k": 5},
                )
                for i, sub in enumerate(follow_up_subs[:3])
            ]

            return {
                "reflection": reflection_result,
                "plan": state["plan"] + new_plan_steps,
                "current_step": state["current_step"] + 1,
                "sub_questions": state.get("sub_questions", []) + new_subs,
                "hop": hop + 1,
            }

    return {
        "reflection": reflection_result,
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
# Multi-Hop Reasoning Node
# ---------------------------------------------------------------------------


def multi_hop_node(state: AgentState) -> dict[str, Any]:
    """Multi-hop reasoning node - evaluates if more hops are needed.

    This node is called when the plan includes a "multi_hop" action.
    It evaluates the current state and decides whether to continue with
    another hop (retrieve more info) or proceed to the next step.
    """
    log.info("node_multi_hop", hop=state.get("hop", 0))

    hop = state.get("hop", 0)
    accumulated_docs = state.get("accumulated_docs", [])

    # If we've reached max hops, proceed to next step
    if hop >= state.get("max_hops", 3):
        log.info("multi_hop_max_reached", hop=hop)
        return {"current_step": state["current_step"] + 1}

    # If we have accumulated docs, add them to the state
    if state.get("retrieved_docs"):
        accumulated_docs = state.get("accumulated_docs", []) + state["retrieved_docs"]
        # Deduplicate by content
        seen = set()
        unique_docs = []
        for doc in accumulated_docs:
            key = doc.content[:100] if hasattr(doc, "content") else str(doc)[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return {
            "accumulated_docs": unique_docs,
            "hop": state.get("hop", 0) + 1,
            "current_step": state["current_step"] + 1,
        }

    return {"current_step": state["current_step"] + 1}


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
        if next_action in ["retrieve", "rerank", "generate", "reflect", "tool_use", "multi_hop"]:
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
    workflow.add_node("multi_hop", multi_hop_node)
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
            "multi_hop": "multi_hop",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "multi_hop",
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
