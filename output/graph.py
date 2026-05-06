from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from final_practice_guided.output.nodes import (
    context_agent_node,
    qa_agent_node,
    quality_agent_node,
    human_review_node,
    publish_node,
)
from final_practice_guided.output.state import QAState


def quality_router(state: QAState) -> str:
    """Route after quality_agent: pass or retry cap → human_review, else → qa_agent."""
    if state.get("quality_passed") or state.get("retry_count", 0) >= 3:
        return "human_review"
    return "qa_agent"


def review_router(state: QAState) -> str:
    """Route after human_review: approved → publish, else → qa_agent."""
    if state.get("human_approved"):
        return "publish"
    return "qa_agent"


def build_graph():
    checkpointer = MemorySaver()
    graph = StateGraph(QAState)

    # Register nodes
    graph.add_node("context_agent", context_agent_node)
    graph.add_node("qa_agent", qa_agent_node)
    graph.add_node("quality_agent", quality_agent_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("publish", publish_node)

    # Entry point
    graph.set_entry_point("context_agent")

    # Direct edges
    graph.add_edge("context_agent", "qa_agent")
    graph.add_edge("qa_agent", "quality_agent")
    graph.add_edge("publish", END)

    # Conditional edges
    graph.add_conditional_edges(
        "quality_agent",
        quality_router,
        {"human_review": "human_review", "qa_agent": "qa_agent"},
    )
    graph.add_conditional_edges(
        "human_review",
        review_router,
        {"publish": "publish", "qa_agent": "qa_agent"},
    )

    return graph.compile(checkpointer=checkpointer)
